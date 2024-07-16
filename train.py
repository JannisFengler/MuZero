import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import MuZeroNetwork, MCTS, select_action, update_weights, MuZeroConfig, MinMaxStats, Node
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from collections import deque
import time
import os
from datetime import datetime
import gc
import psutil

class GameHistory:
    def __init__(self, config):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        self.config = config

    def store_search_statistics(self, root, action_space):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(action_space)
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index):
        return self.observation_history[state_index]

    def make_target(self, state_index, num_unroll_steps, td_steps, to_play):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.compute_target_value(current_index, bootstrap_index)
            else:
                value = 0

            if current_index < len(self.root_values):
                targets.append((value, self.reward_history[current_index], self.child_visits[current_index]))
            else:
                targets.append((0, 0, []))

        return targets

    def compute_target_value(self, current_index, bootstrap_index):
        bootstrap_value = self.root_values[bootstrap_index]
        value = bootstrap_value * self.config.discount ** (bootstrap_index - current_index)
        for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
            value += reward * self.config.discount ** i
        return value

class ReplayBuffer:
    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = deque([], maxlen=self.window_size)

    def save_game(self, game):
        if len(self.buffer) == self.window_size:
            self.buffer.popleft()
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps, td_steps):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]

        return [(g.make_image(i), g.action_history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play_history[i]))
                for (g, i) in game_pos]

    def sample_game(self):
        return np.random.choice(self.buffer)

    def sample_position(self, game):
        return np.random.randint(len(game.action_history))

    def __len__(self):
        return len(self.buffer)

def play_game(config, network, device):
    game = GameHistory(config)
    env = make_atari_env(config.env_name)
    observation = env.reset()
    game.observation_history.append(observation)
    done = False

    print("Starting new game")
    step = 0
    while not done and step < config.max_moves:
        step += 1
        
        observation_tensor = process_observation(observation, device)

        with torch.no_grad():
            initial_inference = network.initial_inference(observation_tensor.to(torch.bfloat16))

        root = Node(prior=0, hidden_state=initial_inference.hidden_state)

        mcts = MCTS(config)
        mcts.run(root, config.action_space_size, network, MinMaxStats())

        game.store_search_statistics(root, config.action_space_size)

        action = select_action(config, root, step)

        try:
            next_observation, reward, done, info = env.step(action)
        except ValueError:
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        game.observation_history.append(next_observation)
        game.action_history.append(action)
        game.reward_history.append(reward)
        game.to_play_history.append(1)  # Assuming single-player game

        observation = next_observation

    print(f"Game completed. Game length: {step}, Total reward: {sum(game.reward_history)}")
    
    return game

def play_multiple_games(config, network, device, num_games):
    games = []
    for game_id in range(num_games):
        print(f"Starting game {game_id + 1}/{num_games}")
        game = play_game(config, network, device)
        games.append(game)
        print(f"Game {game_id + 1} completed. Total reward: {sum(game.reward_history)}, Length: {len(game.action_history)}")
    return games

def train_muzero(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = MuZeroNetwork(
        config.observation_shape,
        config.action_space_size,
        config.num_blocks,
        config.num_channels,
        config.support_size
    ).to(device, dtype=torch.bfloat16)
    
    optimizer = optim.SGD(
        network.parameters(),
        lr=config.lr_init,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_rate)
    
    replay_buffer = ReplayBuffer(config)
    
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be written to: {log_dir}")

    total_games = 0
    total_steps = 0

    for training_step in range(config.training_steps):
        print(f"Starting training step {training_step}")
        
        # Self-play
        if len(replay_buffer) < config.batch_size * config.games_per_training_step:
            print(f"Starting self-play at step {training_step}")
            games = play_multiple_games(config, network, device, config.games_per_training_step)
            for game in games:
                replay_buffer.save_game(game)
                total_games += 1
                total_steps += len(game.action_history)
                writer.add_scalar('Game/TotalReward', sum(game.reward_history), total_games)
                writer.add_scalar('Game/Length', len(game.action_history), total_games)
            writer.add_scalar('Training/TotalGames', total_games, training_step)
            writer.add_scalar('Training/TotalGameSteps', total_steps, training_step)
            print(f"Finished self-play. Replay buffer size: {len(replay_buffer)}")

        # Training
        if len(replay_buffer) >= config.batch_size:
            print(f"Sampling batch from replay buffer. Buffer size: {len(replay_buffer)}")
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            losses = update_weights(optimizer, network, batch, config)
            log_losses(writer, losses, training_step)
            print(f"Updated weights. Losses: {losses}")

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/LearningRate', current_lr, training_step)
            print(f"Updated learning rate. Current LR: {current_lr}")

        # Logging
        if training_step % config.log_interval == 0:
            log_metrics(training_step, network, replay_buffer, writer, config, optimizer)
            log_memory_usage(writer, training_step)
            print(f"Logged metrics at step {training_step}")

        # Evaluation
        if training_step % config.eval_interval == 0:
            print(f"Starting evaluation at step {training_step}")
            avg_reward, avg_length = evaluate_model(network, writer, training_step, config, device)
            print(f"Evaluation complete. Average reward: {avg_reward}, Average length: {avg_length}")

        # Save network
        if training_step % config.checkpoint_interval == 0:
            save_checkpoint(network, optimizer, training_step, config)
            print(f"Saved checkpoint at step {training_step}")

        print(f"Completed training step {training_step}")
        writer.flush()  # Ensure all events are written to disk

        # Force garbage collection every 10 steps
        if training_step % 10 == 0:
            gc.collect()

    writer.close()
    print("Training completed")

def evaluate_model(network, writer, training_step, config, device):
    env = make_atari_env(config.env_name)
    rewards = []
    episode_lengths = []

    for episode in range(config.eval_episodes):
        observation = env.reset()
        done = False
        reward_sum = 0
        step = 0

        print(f"Starting evaluation episode {episode + 1}/{config.eval_episodes}")

        while not done and step < config.max_moves:
            step += 1
            observation_tensor = process_observation(observation, device)
            
            with torch.no_grad():
                initial_inference = network.initial_inference(observation_tensor.to(torch.bfloat16))
            
            root = Node(prior=0, hidden_state=initial_inference.hidden_state)
            
            mcts = MCTS(config)
            mcts.run(root, config.action_space_size, network, MinMaxStats())
            
            action = select_action(config, root, temperature=0)  # Greedy action for evaluation
            
            try:
                observation, reward, done, _ = env.step(action)
            except ValueError:
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            reward_sum += reward

            if step % 100 == 0:
                print(f"Evaluation episode {episode + 1}, step {step}, current reward: {reward_sum}")

        rewards.append(reward_sum)
        episode_lengths.append(step)
        print(f"Evaluation episode {episode + 1} completed. Total reward: {reward_sum}, Length: {step}")

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    
    writer.add_scalar('Evaluation/AverageReward', avg_reward, training_step)
    writer.add_scalar('Evaluation/AverageEpisodeLength', avg_length, training_step)
    writer.add_histogram('Evaluation/EpisodeRewards', np.array(rewards), training_step)
    writer.add_histogram('Evaluation/EpisodeLengths', np.array(episode_lengths), training_step)
    
    print(f"Evaluation complete. Average reward: {avg_reward}, Average length: {avg_length}")
    return avg_reward, avg_length

def process_observation(observation, device):
    if isinstance(observation, gym.wrappers.frame_stack.LazyFrames):
        observation_array = np.array(observation)
    elif isinstance(observation, np.ndarray):
        observation_array = observation
    elif isinstance(observation, tuple) and isinstance(observation[0], gym.wrappers.frame_stack.LazyFrames):
        observation_array = np.array(observation[0])
    else:
        raise ValueError(f"Unexpected observation type: {type(observation)}")
    
    if len(observation_array.shape) == 3:
        if observation_array.shape[-1] == 3 or observation_array.shape[-1] == 4:
            observation_array = np.transpose(observation_array, (2, 0, 1))
    elif len(observation_array.shape) == 4:
        pass
    else:
        raise ValueError(f"Unexpected observation shape: {observation_array.shape}")
    
    return torch.FloatTensor(observation_array).unsqueeze(0).to(device, dtype=torch.bfloat16)

def log_metrics(training_step, network, replay_buffer, writer, config, optimizer):
    writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], training_step)
    writer.add_scalar('Training/ReplayBufferSize', len(replay_buffer), training_step)

    for name, param in network.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param.to(torch.float32), training_step)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad.to(torch.float32), training_step)


def log_losses(writer, losses, training_step):
    total_loss, value_loss, reward_loss, policy_loss = losses
    writer.add_scalar('Loss/Total', total_loss, training_step)
    writer.add_scalar('Loss/Value', value_loss, training_step)
    writer.add_scalar('Loss/Reward', reward_loss, training_step)
    writer.add_scalar('Loss/Policy', policy_loss, training_step)

def log_memory_usage(writer, step):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    writer.add_scalar('System/MemoryUsage', mem, step)
    print(f"Memory usage: {mem:.2f} MB")

def make_atari_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, scale_obs=True, frame_skip=1)
    env = FrameStack(env, 4)
    return env

def save_checkpoint(network, optimizer, training_step, config):
    checkpoint_path = os.path.join(config.checkpoint_dir, f'muzero_checkpoint_{training_step}.pth')
    torch.save({
        'step': training_step,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == '__main__':
    config = MuZeroConfig()
    config.set_game('atari')
    config.env_name = 'PongNoFrameskip-v4'
    config.action_space_size = 6  # Pong-specific
    config.observation_shape = (4, 84, 84)  # 4 stacked frames, each 84x84
    config.support_size = 300  # This results in 601 categories (2 * 300 + 1)
    config.max_moves = 1000  # Max moves per episode
    config.discount = 0.997  # Discount factor
    config.num_simulations = 50  # Number of future moves self-played
    config.batch_size = 128
    config.td_steps = 10
    config.num_unroll_steps = 5
    config.pb_c_base = 19652
    config.pb_c_init = 1.25
    config.root_dirichlet_alpha = 0.25
    config.root_exploration_fraction = 0.25
    config.training_steps = 100000
    config.checkpoint_interval = 1000
    config.self_play_interval = 100
    config.log_interval = 100
    config.eval_interval = 1000
    config.eval_episodes = 2
    config.window_size = 10000  # Number of self-play games to keep in the replay buffer
    config.num_actors = 1
    config.lr_init = 0.05
    config.lr_decay_rate = 0.1
    config.lr_decay_steps = 350000
    config.momentum = 0.9
    config.weight_decay = 1e-4
    config.num_channels = 128  # Number of channels in convolutional layers
    config.num_blocks = 16  # Number of residual blocks in the representation network

    # Network architecture (you may need to adjust these based on your MuZeroNetwork implementation)
    config.encoding_size = 128
    config.hidden_size = 256
    config.games_per_training_step = 10  # Number of games to play before each training step
    
    config.checkpoint_dir = './checkpoints'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    train_muzero(config)
