import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import (MuZeroNetwork, MCTS, select_action, MinMaxStats, Node, 
                  ReplayBuffer, MuZeroConfig, transform_value, NetworkOutput)
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from collections import deque
import time
import os
from datetime import datetime
import gc
import psutil
from typing import List, Tuple, Dict, Optional
import logging
import json
from concurrent.futures import ThreadPoolExecutor

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
            bootstrap_index = current_index + td_steps

            if bootstrap_index < len(self.values):
                value = self.compute_target_value(current_index, bootstrap_index)
            else:
                value = 0

            if current_index < len(self.rewards):
                targets.append((
                    value,
                    self.rewards[current_index],
                    self.policies[current_index]
                ))
            else:
                # States past the end of games are treated as absorbing states
                targets.append((0, 0, np.zeros(self.action_space_size)))

        return targets

    def compute_target_value(self, state_index: int, bootstrap_index: int) -> float:
        """Compute n-step target value."""
        value = self.values[bootstrap_index] * self.discount ** (bootstrap_index - state_index)

        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount ** i

        return value

    def store_transition(self, action: int, observation: np.ndarray, reward: float):
        """Store a single transition."""
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.environment_steps += 1

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
        game.observations.append(observation)

        done = False
        while not done and len(game.actions) < self.config.max_moves:
            # MCTS
            root = Node(0, None)
            observation_tensor = self.prepare_observation(observation)

            with torch.no_grad():
                network_output = self.network.initial_inference(observation_tensor)
                root.expand(range(self.config.action_space_size), network_output)

            if len(game.actions) < self.config.num_simulations:
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
        self.log_training_metrics(value_loss.mean(), reward_loss.mean(), policy_loss.mean(), total_loss_scalar)

    def compute_losses(self, batch, weights) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training losses."""
        observations, actions, targets = zip(*batch)
        observations_tensor = torch.stack([self.prepare_observation(obs) for obs in observations])

        # Initial inference
        network_output = self.network.initial_inference(observations_tensor)

        # Value loss
        target_values = torch.tensor([target[0] for target in targets], device=device)
        value_loss = self.scalar_to_support_loss(
            network_output.value.squeeze(-1),
            target_values
        )  # Per-sample loss

        # Policy loss
        target_policies = torch.tensor([target[2] for target in targets], device=device)
        policy_loss = -(
            torch.log_softmax(network_output.policy_logits, dim=1) *
            target_policies
        ).sum(1)  # Per-sample loss

        # Recurrent inferences
        reward_loss = torch.zeros(len(batch), device=device)  # Initialize per-sample reward_loss

        hidden_state = network_output.hidden_state
        for unroll_step in range(self.config.num_unroll_steps):
            actions_tensor = torch.tensor(
                [actions[b][unroll_step] for b in range(len(actions))],
                device=device
            ).unsqueeze(-1)  # Shape (batch_size, 1)

            network_output = self.network.recurrent_inference(hidden_state, actions_tensor)
            hidden_state = network_output.hidden_state

            # Reward loss
            target_rewards = torch.tensor(
                [targets[b][unroll_step + 1][1] if unroll_step + 1 < len(targets[b]) else 0 for b in range(len(targets))],
                device=device
            )

            step_reward_loss = self.scalar_to_support_loss(
                network_output.reward.squeeze(-1),
                target_rewards
            )  # Per-sample loss

            reward_loss += step_reward_loss

        # Apply weights
        value_loss = value_loss * weights
        reward_loss = reward_loss * weights
        policy_loss = policy_loss * weights

        # Mean over batch
        value_loss_scalar = value_loss.mean()
        reward_loss_scalar = reward_loss.mean()
        policy_loss_scalar = policy_loss.mean()

        total_loss_per_sample = (
            self.config.value_loss_weight * value_loss +
            self.config.reward_loss_weight * reward_loss +
            self.config.policy_loss_weight * policy_loss
        )

        total_loss_scalar = total_loss_per_sample.mean()

        return (
            value_loss_scalar,
            reward_loss_scalar,
            policy_loss_scalar,
            total_loss_per_sample
        )

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
                observation_tensor = self.prepare_observation(observation)

                with torch.no_grad():
                    network_output = self.network.initial_inference(observation_tensor)
                    root = Node(0, None)
                    root.expand(range(self.config.action_space_size), network_output)

                    mcts = MCTS(self.config)
                    action_probs = mcts.run(
                        root,
                        self.config.action_space_size,
                        self.network,
                        MinMaxStats()
                    )

                    action = select_action(self.config, root, temperature=0)

                observation, reward, done, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        self.log_eval_metrics(rewards)

    def save_checkpoint(self, is_final: bool = False):
        """Save network checkpoint."""
        if self.training_step % self.config.checkpoint_interval == 0 or is_final:
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
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=True
        )
        env = FrameStack(env, num_stack=4)
        return env

    def prepare_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Prepare observation for network input."""
        if isinstance(observation, (gym.wrappers.frame_stack.LazyFrames, np.ndarray)):
            observation = np.array(observation)
        observation = np.transpose(observation, (2, 0, 1))  # CHW format
        return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    def compute_priorities(self, game: Game) -> np.ndarray:
        """Compute priorities for replay buffer."""
        return np.array([abs(value) + 1e-6 for value in game.values])

    def log_self_play_metrics(self, games: List[Game]):
        """Log self-play metrics to TensorBoard."""
        rewards = [sum(game.rewards) for game in games]
        lengths = [game.environment_steps for game in games]

        self.writer.add_scalar('SelfPlay/AverageReward', np.mean(rewards), self.training_step)
        self.writer.add_scalar('SelfPlay/AverageLength', np.mean(lengths), self.training_step)
        self.writer.add_scalar('SelfPlay/GamesPlayed', self.num_played_games, self.training_step)
        self.writer.add_scalar('SelfPlay/TotalSteps', self.num_played_steps, self.training_step)

    def log_training_metrics(self, value_loss: torch.Tensor, reward_loss: torch.Tensor,
                             policy_loss: torch.Tensor, total_loss: torch.Tensor):
        """Log training metrics to TensorBoard."""
        self.writer.add_scalar('Loss/Value', value_loss.item(), self.training_step)
        self.writer.add_scalar('Loss/Reward', reward_loss.item(), self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.training_step)
        self.writer.add_scalar('Loss/Total', total_loss.item(), self.training_step)
        self.writer.add_scalar('Training/LearningRate', self.optimizer.param_groups[0]['lr'], self.training_step)
        self.writer.add_scalar('Training/ReplayBufferSize', len(self.replay_buffer), self.training_step)

    def log_eval_metrics(self, rewards: List[float]):
        """Log evaluation metrics to TensorBoard."""
        avg_reward = np.mean(rewards)
        self.writer.add_scalar('Eval/AverageReward', avg_reward, self.training_step)
        self.writer.add_histogram('Eval/Rewards', np.array(rewards), self.training_step)
        logging.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")

    def scalar_to_support_loss(self, logits: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """Convert scalar targets to categorical and compute cross-entropy loss."""
        scalar = transform_value(scalar)  # Apply value transformation
        targets = self.scalar_to_categorical(scalar, self.config.support_size)
        return torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    def scalar_to_categorical(self, x: torch.Tensor, support_size: int) -> torch.Tensor:
        """Transform scalar to categorical representation."""
        x = torch.clamp(x, -support_size, support_size)
        floor = x.floor()
        prob = x - floor
        logits = torch.zeros(x.shape[0], 2 * support_size + 1, device=x.device)
        index = floor + support_size
        index = index.long()
        logits.scatter_(1, index.unsqueeze(-1), 1 - prob.unsqueeze(-1))
        logits.scatter_(1, (index + 1).unsqueeze(-1), prob.unsqueeze(-1))
        return logits

def create_default_config() -> MuZeroConfig:
    """Create default configuration for training."""
    config = MuZeroConfig()

    # Environment
    config.env_name = "ALE/Pong-v5"
    config.action_space_size = 6  # Pong-specific
    config.observation_shape = (4, 84, 84)  # 4 stacked frames

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
    config.max_grad_norm = 10.0  # Added this

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

    # Visit softmax temperature function
    config.visit_softmax_temperature_fn = lambda training_step: 1.0

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

# Helper functions for debugging and visualization

def visualize_game(game: Game, save_dir: str = 'game_visualizations'):
    """Visualize and save game states and statistics."""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(game.rewards)
    plt.title('Game Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(save_dir, 'rewards.png'))
    plt.close()

    # Plot value estimates
    plt.figure(figsize=(10, 5))
    plt.plot(game.values)
    plt.title('MCTS Value Estimates')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.savefig(os.path.join(save_dir, 'values.png'))
    plt.close()

    # Plot policy entropy
    entropies = [-np.sum(p * np.log(p + 1e-8)) for p in game.policies]
    plt.figure(figsize=(10, 5))
    plt.plot(entropies)
    plt.title('Policy Entropy')
    plt.xlabel('Step')
    plt.ylabel('Entropy')
    plt.savefig(os.path.join(save_dir, 'policy_entropy.png'))
    plt.close()

def profile_performance(trainer: MuZeroTrainer, num_steps: int = 100):
    """Profile training performance."""
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    # Run training steps
    for _ in range(num_steps):
        trainer.training_step += 1
        trainer.update_weights()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(50)  # Print top 50 time-consuming operations

def debug_network_outputs(trainer: MuZeroTrainer, observation: np.ndarray):
    """Debug network predictions for a given observation."""
    observation_tensor = trainer.prepare_observation(observation)

    with torch.no_grad():
        # Initial inference
        initial_output = trainer.network.initial_inference(observation_tensor)

        print("Initial Inference:")
        print(f"Value: {initial_output.scalar_value():.3f}")
        print(f"Policy logits shape: {initial_output.policy_logits.shape}")
        print(f"Policy distribution:\n{torch.softmax(initial_output.policy_logits, dim=1).cpu().numpy()}")

        # Recurrent inference
        action = torch.tensor([[0]], device=device)  # Test with action 0
        recurrent_output = trainer.network.recurrent_inference(initial_output.hidden_state, action)

        print("\nRecurrent Inference:")
        print(f"Value: {recurrent_output.scalar_value():.3f}")
        print(f"Reward: {recurrent_output.scalar_reward():.3f}")
        print(f"Policy distribution:\n{torch.softmax(recurrent_output.policy_logits, dim=1).cpu().numpy()}")

def export_model_architecture(network: MuZeroNetwork, save_path: str = 'model_architecture.txt'):
    """Export detailed model architecture."""
    with open(save_path, 'w') as f:
        f.write(str(network))

        # Calculate number of parameters
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

        f.write(f"\n\nTotal parameters: {total_params:,}")
        f.write(f"\nTrainable parameters: {trainable_params:,}")

        # Memory usage estimation
        param_size = sum(p.nelement() * p.element_size() for p in network.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in network.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024

        f.write(f"\nEstimated model size: {size_mb:.2f} MB")
