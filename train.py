import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import (MuZeroNetwork, MCTS, select_action, MinMaxStats, Node,
                   ReplayBuffer, MuZeroConfig, scalar_to_support_loss)
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, ResizeObservation
import numpy as np
import os
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Game:
    """Stores a single episode of interaction data."""
    def __init__(self, action_space_size: int, discount: float):
        self.environment_steps = 0
        self.actions: List[int] = []
        self.observations: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []   # Root value estimates from MCTS
        self.policies: List[np.ndarray] = []  # MCTS policy distributions
        self.done = False
        self.action_space_size = action_space_size
        self.discount = discount

    def store_search_stats(self, root: Node, action_probs: Dict[int, float]):
        """Store MCTS search results."""
        self.values.append(root.value())
        policy = np.zeros(self.action_space_size)
        for a, p in action_probs.items():
            policy[a] = p
        self.policies.append(policy)

    def compute_target_value(self, state_index: int, td_steps: int) -> float:
        """Compute n-step target value."""
        value = 0.0
        discount = 1.0
        for i in range(td_steps):
            idx = state_index + i
            if idx < len(self.rewards):
                value += self.rewards[idx] * discount
                discount *= self.discount
            else:
                break
        if state_index + td_steps < len(self.values):
            value += self.values[state_index + td_steps] * discount
        return value

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        """Create training targets for a segment of the game."""
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            value = self.compute_target_value(current_index, td_steps)
            if current_index < len(self.rewards):
                reward = self.rewards[current_index]
                policy = self.policies[current_index]
            else:
                reward = 0
                policy = np.zeros(self.action_space_size)
            targets.append((value, reward, policy))
        return targets

    def store_transition(self, action: Optional[int], observation: np.ndarray, reward: float):
        """Store a single environment step."""
        self.actions.append(action if action is not None else 0)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.environment_steps += 1

    def get_observation(self, index: int) -> np.ndarray:
        return self.observations[index]


class MuZeroTrainer:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.checkpoint_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )

        self.network = MuZeroNetwork(config).to(device)
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=config.lr_init,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_rate
        )
        self.replay_buffer = ReplayBuffer(config)

        self.training_step = 0
        self.num_played_games = 0
        self.num_played_steps = 0
        self.total_rewards = []

        self.writer = SummaryWriter(os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S')))

    def train(self):
        try:
            while self.training_step < self.config.training_steps:
                # Generate self-play if buffer not ready
                if len(self.replay_buffer) < self.config.batch_size:
                    self.self_play()

                # If we have enough data, do a training step
                if len(self.replay_buffer) >= self.config.minimum_games_in_buffer:
                    self.training_step += 1
                    self.update_weights()
                    if self.training_step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint()

                # Periodic evaluation
                if self.training_step % self.config.eval_interval == 0:
                    self.evaluate()

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(is_final=True)

    def self_play(self):
        logging.info(f"Starting self-play. Buffer size: {len(self.replay_buffer)}")

        with ThreadPoolExecutor(max_workers=self.config.num_actors) as executor:
            games = list(executor.map(self.play_game, range(self.config.num_actors)))

        for game in games:
            priorities = self.compute_priorities(game)
            self.replay_buffer.save_game(game, priorities)
            self.total_rewards.append(sum(game.rewards))
            self.num_played_games += 1
            self.num_played_steps += game.environment_steps

        self.log_self_play_metrics(games)

    def play_game(self, thread_id: int) -> Game:
        """Execute one self-play game."""
        env = self.make_env()
        game = Game(self.config.action_space_size, self.config.discount)
        observation = env.reset()
        game.store_transition(None, observation, 0)
        done = False

        while not done and len(game.actions) < self.config.max_moves:
            # Prepare observation (32 frames stacked in FrameStack)
            observation_tensor = self.prepare_observation(observation)

            with torch.no_grad():
                network_output = self.network.initial_inference(observation_tensor)
                root = Node(0, None)
                root.expand(list(range(self.config.action_space_size)), network_output)

            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction
            )

            mcts = MCTS(self.config)
            min_max_stats = MinMaxStats()
            action_probs = mcts.run(root, self.config.action_space_size, self.network, min_max_stats)

            game.store_search_stats(root, action_probs)
            temperature = self.config.visit_softmax_temperature_fn(self.training_step)
            action = select_action(self.config, root, temperature)

            observation, reward, done, _ = env.step(action)
            game.store_transition(action, observation, reward)

        return game

    def update_weights(self):
        batch, indices, weights = self.replay_buffer.sample_batch(
            self.config.num_unroll_steps,
            self.config.td_steps
        )

        value_loss, reward_loss, policy_loss, total_loss_per_sample = self.compute_losses(batch, weights)
        total_loss_scalar = total_loss_per_sample.mean()

        self.optimizer.zero_grad()
        total_loss_scalar.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        priorities = total_loss_per_sample.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        self.log_training_metrics(value_loss.item(), reward_loss.item(), policy_loss.item(), total_loss_scalar.item())

    def compute_losses(self, batch, weights):
        observations, actions_list, targets_list = zip(*batch)
        batch_size = len(observations)
        observations_tensor = torch.stack([self.prepare_observation(obs) for obs in observations]).to(device)
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        network_output = self.network.initial_inference(observations_tensor)
        hidden_state = network_output.hidden_state

        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        total_loss_per_sample = 0

        for t in range(self.config.num_unroll_steps + 1):
            target_values = []
            target_rewards = []
            target_policies = []
            actions = []
            for i in range(batch_size):
                if t < len(targets_list[i]):
                    v, r, p = targets_list[i][t]
                    target_values.append(v)
                    target_rewards.append(r)
                    target_policies.append(p)
                else:
                    # Past end of episode, absorbing state
                    target_values.append(0)
                    target_rewards.append(0)
                    target_policies.append(np.zeros(self.config.action_space_size))

                if t > 0 and (t-1) < len(actions_list[i]):
                    actions.append(actions_list[i][t-1])
                else:
                    actions.append(0)

            target_values = torch.tensor(target_values, device=device, dtype=torch.float32)
            target_rewards = torch.tensor(target_rewards, device=device, dtype=torch.float32)
            target_policies = torch.tensor(target_policies, device=device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(-1)

            value_loss = scalar_to_support_loss(network_output.value, target_values, self.config.support_size)
            policy_loss = -(target_policies * torch.log_softmax(network_output.policy_logits, dim=1)).sum(1)

            if t > 0:
                reward_loss = scalar_to_support_loss(network_output.reward, target_rewards, self.config.support_size)
            else:
                reward_loss = torch.zeros(batch_size, device=device)

            total_loss = (
                self.config.value_loss_weight * value_loss +
                self.config.reward_loss_weight * reward_loss +
                self.config.policy_loss_weight * policy_loss
            )
            total_loss_per_sample += total_loss * weights

            if t < self.config.num_unroll_steps:
                # Recurrent inference for next step
                network_output = self.network.recurrent_inference(hidden_state, actions_tensor)
                hidden_state = network_output.hidden_state

            total_value_loss += value_loss.mean()
            total_reward_loss += reward_loss.mean()
            total_policy_loss += policy_loss.mean()

        total_value_loss /= (self.config.num_unroll_steps + 1)
        total_policy_loss /= (self.config.num_unroll_steps + 1)
        if self.config.num_unroll_steps > 0:
            total_reward_loss /= self.config.num_unroll_steps
        else:
            total_reward_loss = torch.zeros(1, device=device)

        total_loss_per_sample = total_loss_per_sample / (self.config.num_unroll_steps + 1)
        return total_value_loss, total_reward_loss, total_policy_loss, total_loss_per_sample

    def evaluate(self):
        logging.info("Starting evaluation")
        rewards = []
        env = self.make_env()

        for _ in range(self.config.eval_episodes):
            observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                observation_tensor = self.prepare_observation(observation)
                with torch.no_grad():
                    network_output = self.network.initial_inference(observation_tensor)
                    root = Node(0, None)
                    root.expand(list(range(self.config.action_space_size)), network_output)

                    mcts = MCTS(self.config)
                    min_max_stats = MinMaxStats()
                    action_probs = mcts.run(root, self.config.action_space_size, self.network, min_max_stats)
                    action = select_action(self.config, root, temperature=0)

                observation, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)

        self.log_eval_metrics(rewards)

    def save_checkpoint(self, is_final: bool = False):
        checkpoint = {
            'training_step': self.training_step,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_{self.training_step}.pt' if not is_final else 'final_checkpoint.pt'
        )
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")

    def make_env(self):
        env = gym.make(self.config.env_name)
        # Follow a similar setting to MuZero on Atari: no terminal on life loss, no scaling
        env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=False, scale_obs=False)
        env = ResizeObservation(env, (96, 96))
        env = FrameStack(env, num_stack=32)  # Stack 32 frames
        return env

    def prepare_observation(self, observation: np.ndarray) -> torch.Tensor:
        # observation shape: (H=96, W=96, C=96) after stacking 32 RGB frames (32*3=96 channels)
        obs = np.array(observation)
        obs = np.transpose(obs, (2, 0, 1))  # (C,H,W)
        obs = obs / 255.0
        return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    def compute_priorities(self, game: Game) -> np.ndarray:
        priorities = []
        for i in range(len(game.values)):
            target_value = game.compute_target_value(i, self.config.td_steps)
            priority = abs(game.values[i] - target_value) + 1e-6
            priorities.append(priority)
        return np.array(priorities)

    def log_self_play_metrics(self, games: List[Game]):
        rewards = [sum(game.rewards) for game in games]
        lengths = [game.environment_steps for game in games]

        self.writer.add_scalar('SelfPlay/AverageReward', np.mean(rewards), self.training_step)
        self.writer.add_scalar('SelfPlay/AverageLength', np.mean(lengths), self.training_step)
        self.writer.add_scalar('SelfPlay/GamesPlayed', self.num_played_games, self.training_step)
        self.writer.add_scalar('SelfPlay/TotalSteps', self.num_played_steps, self.training_step)

    def log_training_metrics(self, value_loss: float, reward_loss: float,
                             policy_loss: float, total_loss: float):
        self.writer.add_scalar('Loss/Value', value_loss, self.training_step)
        self.writer.add_scalar('Loss/Reward', reward_loss, self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss, self.training_step)
        self.writer.add_scalar('Loss/Total', total_loss, self.training_step)
        self.writer.add_scalar('Training/LearningRate', self.optimizer.param_groups[0]['lr'], self.training_step)
        self.writer.add_scalar('Training/ReplayBufferSize', len(self.replay_buffer), self.training_step)

    def log_eval_metrics(self, rewards: List[float]):
        avg_reward = np.mean(rewards)
        self.writer.add_scalar('Eval/AverageReward', avg_reward, self.training_step)
        self.writer.add_histogram('Eval/Rewards', np.array(rewards), self.training_step)
        logging.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")


def main():
    # Example on how to start training:
    config = MuZeroConfig()
    # Customize config as needed, for example:
    config.env_name = "ALE/Pong-v5"
    env = gym.make(config.env_name)
    config.action_space_size = env.action_space.n
    config.observation_shape = (96, 96, 96)  # (C,H,W) = 96 channels from 32 RGB frames

    # Start training
    trainer = MuZeroTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
