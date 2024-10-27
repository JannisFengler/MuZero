import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import (MuZeroNetwork, MCTS, select_action, MinMaxStats, Node,
                   ReplayBuffer, MuZeroConfig, transform_value, scalar_to_support_loss)
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
    """Class to store game history."""
    def __init__(self, action_space_size: int, discount: float):
        self.environment_steps = 0
        self.actions: List[int] = []
        self.observations: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []  # Root values from MCTS
        self.policies: List[np.ndarray] = []  # MCTS policies
        self.done = False
        self.action_space_size = action_space_size
        self.discount = discount

    def store_search_stats(self, root: Node, action_probs: Dict[int, float]):
        """Store MCTS search statistics."""
        self.values.append(root.value())
        policy = np.zeros(self.action_space_size)
        for action, prob in action_probs.items():
            policy[action] = prob
        self.policies.append(policy)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int) -> List[Tuple[float, float, np.ndarray]]:
        """Create target values for training."""
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

    def store_transition(self, action: int, observation: np.ndarray, reward: float):
        """Store a single transition."""
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.environment_steps += 1

    def get_observation(self, index: int) -> np.ndarray:
        """Retrieve the observation at a given index."""
        return self.observations[index]

class MuZeroTrainer:
    """Main training class for MuZero."""
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.checkpoint_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )

        # Initialize network and training components
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

        # Training metrics
        self.training_step = 0
        self.num_played_games = 0
        self.num_played_steps = 0
        self.total_rewards = []

        # Initialize TensorBoard
        self.writer = SummaryWriter(os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S')))

    def train(self):
        """Main training loop."""
        try:
            while self.training_step < self.config.training_steps:
                # Generate self-play games if needed
                if len(self.replay_buffer) < self.config.batch_size:
                    self.self_play()

                # Training step
                if len(self.replay_buffer) >= self.config.minimum_games_in_buffer:
                    self.training_step += 1
                    self.update_weights()
                    if self.training_step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint()

                # Evaluation
                if self.training_step % self.config.eval_interval == 0:
                    self.evaluate()

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(is_final=True)

    def self_play(self):
        """Generate self-play games."""
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
        """Play a single game."""
        env = self.make_env()
        game = Game(self.config.action_space_size, self.config.discount)

        observation = env.reset()
        game.store_transition(None, observation, 0)  # Initial transition with no action

        done = False
        while not done and len(game.actions) < self.config.max_moves:
            # Prepare history
            history = game.actions[-32:]  # Last 32 actions
            observation_tensor = self.prepare_observation(observation, history)

            # MCTS
            root = Node(0, None)

            with torch.no_grad():
                network_output = self.network.initial_inference(observation_tensor)
                root.expand(range(self.config.action_space_size), network_output)

            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction
            )

            # Run MCTS
            mcts = MCTS(self.config)
            min_max_stats = MinMaxStats()
            action_probs = mcts.run(root, self.config.action_space_size, self.network, min_max_stats)

            # Store MCTS statistics
            game.store_search_stats(root, action_probs)

            # Select action
            temperature = self.config.visit_softmax_temperature_fn(self.training_step)
            action = select_action(self.config, root, temperature)

            # Execute action
            observation, reward, done, _ = env.step(action)

            # Store transition
            game.store_transition(action, observation, reward)

        return game

    def update_weights(self):
        """Update network weights using replay buffer."""
        batch, indices, weights = self.replay_buffer.sample_batch(
            self.config.num_unroll_steps,
            self.config.td_steps
        )

        # Compute losses
        value_loss, reward_loss, policy_loss, total_loss_per_sample = self.compute_losses(batch, weights)
        total_loss_scalar = total_loss_per_sample.mean()

        # Optimize
        self.optimizer.zero_grad()
        total_loss_scalar.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Update replay priorities
        priorities = total_loss_per_sample.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # Log metrics
        self.log_training_metrics(value_loss.item(), reward_loss.item(), policy_loss.item(), total_loss_scalar.item())

    def compute_losses(self, batch, weights) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training losses."""
        observations, actions_list, targets_list = zip(*batch)
        batch_size = len(observations)

        # Prepare tensors
        observations_tensor = torch.stack([self.prepare_observation(obs, []) for obs in observations]).to(device)
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        # Initial inference
        network_output = self.network.initial_inference(observations_tensor)

        # Initialize accumulators
        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        total_loss_per_sample = 0

        # Unroll the dynamics
        hidden_state = network_output.hidden_state
        for t in range(self.config.num_unroll_steps + 1):
            target_values = []
            target_rewards = []
            target_policies = []
            actions = []
            for i in range(batch_size):
                if t < len(targets_list[i]):
                    target_values.append(targets_list[i][t][0])
                    target_rewards.append(targets_list[i][t][1])
                    target_policies.append(targets_list[i][t][2])
                else:
                    target_values.append(0)
                    target_rewards.append(0)
                    target_policies.append(np.zeros(self.config.action_space_size))
                if t > 0 and t - 1 < len(actions_list[i]):
                    actions.append(actions_list[i][t - 1])
                else:
                    actions.append(0)  # Dummy action

            target_values = torch.tensor(target_values, device=device, dtype=torch.float32)
            target_rewards = torch.tensor(target_rewards, device=device, dtype=torch.float32)
            target_policies = torch.tensor(target_policies, device=device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(-1)

            # Value loss
            value_loss = scalar_to_support_loss(network_output.value.squeeze(-1), target_values, self.config.support_size)
            total_value_loss += value_loss

            # Policy loss
            policy_loss = -(torch.log_softmax(network_output.policy_logits, dim=1) * target_policies).sum(1)
            total_policy_loss += policy_loss

            if t > 0:
                # Reward loss
                reward_loss = scalar_to_support_loss(network_output.reward.squeeze(-1), target_rewards, self.config.support_size)
                total_reward_loss += reward_loss
            else:
                reward_loss = torch.zeros(batch_size, device=device)

            # Total loss per sample
            total_loss = (
                self.config.value_loss_weight * value_loss +
                self.config.reward_loss_weight * reward_loss +
                self.config.policy_loss_weight * policy_loss
            )
            total_loss_per_sample += total_loss * weights

            # Recurrent inference
            if t < self.config.num_unroll_steps:
                network_output = self.network.recurrent_inference(hidden_state, actions_tensor)
                hidden_state = network_output.hidden_state

        # Mean over batch
        total_loss_per_sample = total_loss_per_sample / (self.config.num_unroll_steps + 1)

        # Scale losses
        total_value_loss = (total_value_loss / (self.config.num_unroll_steps + 1)).mean()
        total_reward_loss = (total_reward_loss / self.config.num_unroll_steps).mean()
        total_policy_loss = (total_policy_loss / (self.config.num_unroll_steps + 1)).mean()

        return total_value_loss, total_reward_loss, total_policy_loss, total_loss_per_sample

    def evaluate(self):
        """Evaluate current network."""
        logging.info("Starting evaluation")
        rewards = []
        env = self.make_env()

        for _ in range(self.config.eval_episodes):
            observation = env.reset()
            done = False
            total_reward = 0

            while not done:
                observation_tensor = self.prepare_observation(observation, [])

                with torch.no_grad():
                    network_output = self.network.initial_inference(observation_tensor)
                    root = Node(0, None)
                    root.expand(range(self.config.action_space_size), network_output)

                    mcts = MCTS(self.config)
                    min_max_stats = MinMaxStats()
                    action_probs = mcts.run(
                        root,
                        self.config.action_space_size,
                        self.network,
                        min_max_stats
                    )

                    action = select_action(self.config, root, temperature=0)

                observation, reward, done, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        self.log_eval_metrics(rewards)

    def save_checkpoint(self, is_final: bool = False):
        """Save network checkpoint."""
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
        """Create environment with proper wrappers."""
        env = gym.make(self.config.env_name)
        env = AtariPreprocessing(
            env,
            frame_skip=1,
            grayscale_obs=False,  # Use RGB frames
            scale_obs=False,
            terminal_on_life_loss=False  # Depending on your preference
        )
        env = ResizeObservation(env, (96, 96))
        env = FrameStack(env, num_stack=32)
        return env

    def prepare_observation(self, observation: np.ndarray, actions: List[int]) -> torch.Tensor:
        """Prepare observation and action planes for network input."""
        if isinstance(observation, (gym.wrappers.frame_stack.LazyFrames, np.ndarray)):
            observation = np.array(observation)
        observation = np.transpose(observation, (2, 0, 1))  # CHW format

        # Normalize observation to [0, 1]
        observation = observation / 255.0

        # Encode actions as planes
        action_planes = np.zeros((self.config.action_space_size, *observation.shape[1:]), dtype=np.float32)
        for i, action in enumerate(actions[-32:]):
            action_planes[action] = 1.0  # Set action plane to 1

        # Concatenate observation and action planes
        observation = np.concatenate([observation, action_planes], axis=0)
        return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    def compute_priorities(self, game: Game) -> np.ndarray:
        """Compute priorities for replay buffer."""
        priorities = []
        for i in range(len(game.values)):
            target_value = game.compute_target_value(i, self.config.td_steps)
            priority = abs(game.values[i] - target_value) + 1e-6
            priorities.append(priority)
        return np.array(priorities)

    def log_self_play_metrics(self, games: List[Game]):
        """Log self-play metrics to TensorBoard."""
        rewards = [sum(game.rewards) for game in games]
        lengths = [game.environment_steps for game in games]

        self.writer.add_scalar('SelfPlay/AverageReward', np.mean(rewards), self.training_step)
        self.writer.add_scalar('SelfPlay/AverageLength', np.mean(lengths), self.training_step)
        self.writer.add_scalar('SelfPlay/GamesPlayed', self.num_played_games, self.training_step)
        self.writer.add_scalar('SelfPlay/TotalSteps', self.num_played_steps, self.training_step)

    def log_training_metrics(self, value_loss: float, reward_loss: float,
                             policy_loss: float, total_loss: float):
        """Log training metrics to TensorBoard."""
        self.writer.add_scalar('Loss/Value', value_loss, self.training_step)
        self.writer.add_scalar('Loss/Reward', reward_loss, self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss, self.training_step)
        self.writer.add_scalar('Loss/Total', total_loss, self.training_step)
        self.writer.add_scalar('Training/LearningRate', self.optimizer.param_groups[0]['lr'], self.training_step)
        self.writer.add_scalar('Training/ReplayBufferSize', len(self.replay_buffer), self.training_step)

    def log_eval_metrics(self, rewards: List[float]):
        """Log evaluation metrics to TensorBoard."""
        avg_reward = np.mean(rewards)
        self.writer.add_scalar('Eval/AverageReward', avg_reward, self.training_step)
        self.writer.add_histogram('Eval/Rewards', np.array(rewards), self.training_step)
        logging.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")

def scalar_to_support_loss(logits: torch.Tensor, targets: torch.Tensor, support_size: int) -> torch.Tensor:
    """Compute cross-entropy loss with soft targets."""
    target_support = scalar_to_support(targets, support_size)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_support * log_probs).sum(1)
    return loss

def create_default_config() -> MuZeroConfig:
    """Create default configuration for training."""
    config = MuZeroConfig()

    # Environment
    config.env_name = "ALE/Pong-v5"
    config.action_space_size = gym.make(config.env_name).action_space.n
    config.observation_shape = (32 * 3 + config.action_space_size, 96, 96)  # 32 RGB frames + action planes

    # Network
    config.support_size = 300  # Results in 601 categorical bins
    config.num_channels = 256
    config.num_blocks = 16
    config.downsample_steps = [(128, 2), (256, 3)]  # (channels, num_blocks)

    # Training
    config.training_steps = int(1000e3)
    config.batch_size = 1024
    config.num_unroll_steps = 5
    config.td_steps = 10
    config.num_actors = 8
    config.num_simulations = 50

    # Replay Buffer
    config.replay_buffer_size = int(1e6)
    config.minimum_games_in_buffer = 100
    config.priority_alpha = 1.0
    config.priority_beta = 1.0

    # Optimization
    config.lr_init = 0.05
    config.lr_decay_rate = 0.1
    config.lr_decay_steps = int(350e3)
    config.momentum = 0.9
    config.weight_decay = 1e-4
    config.max_grad_norm = 10.0  # Gradient clipping

    # Loss weights
    config.value_loss_weight = 0.25
    config.reward_loss_weight = 1.0
    config.policy_loss_weight = 1.0

    # MCTS
    config.max_moves = 27000  # 30 minutes of gameplay at 15 FPS
    config.root_dirichlet_alpha = 0.3
    config.root_exploration_fraction = 0.25
    config.pb_c_base = 19652
    config.pb_c_init = 1.25

    # Evaluation
    config.eval_interval = 1000
    config.eval_episodes = 32
    config.checkpoint_interval = 1000

    return config

def main():
    """Main training entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('muzero_training.log'),
            logging.StreamHandler()
        ]
    )

    # Initialize config and check GPU availability
    config = create_default_config()
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        # Create and run trainer
        trainer = MuZeroTrainer(config)
        logging.info("Starting training")
        trainer.train()

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.exception("Training failed with error")
        raise
    finally:
        # Final cleanup
        if 'trainer' in locals():
            trainer.save_checkpoint(is_final=True)
            trainer.writer.close()

if __name__ == "__main__":
    main()
