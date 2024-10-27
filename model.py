import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_value(value: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Transform value according to paper's scaling function."""
    return torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1 + eps * value)

def inverse_transform_value(transformed_value: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Inverse of the value transformation function."""
    return torch.sign(transformed_value) * (((torch.sqrt(1 + 4 * eps * (torch.abs(transformed_value) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)

class MuZeroConfig:
    def __init__(self):
        # Network architecture
        self.observation_shape: Tuple[int, int, int] = None
        self.action_space_size: int = None
        self.num_players: int = None
        self.support_size: int = 300  # Value/reward support size (paper uses 601 total bins)
        self.num_blocks: int = 16  # Number of residual blocks
        self.num_channels: int = 256  # Number of channels in conv layers
        self.reduced_channels_reward: int = 256  # Channels in reward head
        self.reduced_channels_value: int = 256  # Channels in value head
        self.reduced_channels_policy: int = 256  # Channels in policy head
        self.fc_reward_layers: List[int] = [256]  # Hidden layers in reward head
        self.fc_value_layers: List[int] = [256]  # Hidden layers in value head
        self.fc_policy_layers: List[int] = [256]  # Hidden layers in policy head
        self.downsample_steps: List[Tuple[int, int]] = [(128, 2), (256, 3)]  # Channels and blocks per downsample

        # Training
        self.training_steps: int = int(1000e3)
        self.checkpoint_interval: int = int(1e3)
        self.batch_size: int = 1024
        self.num_unroll_steps: int = 5
        self.td_steps: int = 10
        self.num_actors: int = 8
        self.max_moves: int = 27000  # Approximately 30 minutes of gameplay at 15 FPS
        self.num_simulations: int = 50  # For Atari (800 for board games)

        # Replay Buffer
        self.replay_buffer_size: int = int(1e6)
        self.num_workers: int = 4
        self.minimum_games_in_buffer: int = 100
        self.priority_alpha: float = 1.0
        self.priority_beta: float = 1.0

        # MCTS
        self.pb_c_base: float = 19652
        self.pb_c_init: float = 1.25
        self.root_dirichlet_alpha: float = 0.3
        self.root_exploration_fraction: float = 0.25
        self.discount: float = 0.997

        # Optimization
        self.weight_decay: float = 1e-4
        self.momentum: float = 0.9
        self.lr_init: float = 0.05
        self.lr_decay_rate: float = 0.1
        self.lr_decay_steps: int = int(350e3)
        self.value_loss_weight: float = 0.25
        self.reward_loss_weight: float = 1.0
        self.policy_loss_weight: float = 1.0
        self.max_grad_norm: float = 10.0  # Gradient clipping

        # Evaluation
        self.eval_interval: int = 1000
        self.eval_episodes: int = 32

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Returns temperature for action selection."""
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name: str):
        """Configures for specific environment."""
        if env_name.startswith('ALE/'):
            self.action_space_size = gym.make(env_name).action_space.n
            self.num_players = 1
            self.observation_shape = (96, 96, 96)  # 32 RGB frames at 96x96 resolution
        else:
            raise ValueError(f"Unknown game: {env_name}")

class NetworkOutput:
    """Holds network predictions."""
    def __init__(self, value: torch.Tensor, reward: Optional[torch.Tensor],
                 policy_logits: torch.Tensor, hidden_state: torch.Tensor):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state

    def scalar_value(self) -> float:
        """Returns value as scalar."""
        return support_to_scalar(self.value, self.value.shape[-1] // 2).item()

    def scalar_reward(self) -> Optional[float]:
        """Returns reward as scalar."""
        if self.reward is None:
            return None
        return support_to_scalar(self.reward, self.reward.shape[-1] // 2).item()

class DownsampleBlock(nn.Module):
    """Downsampling block as per paper."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.blocks = nn.ModuleList([ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        return x

class ResidualBlock(nn.Module):
    """Standard residual block."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class RepresentationNetwork(nn.Module):
    """Initial representation network that processes raw observations."""
    def __init__(self, observation_shape: Tuple[int, int, int], num_channels: int,
                 downsample_steps: List[Tuple[int, int]]):
        super().__init__()

        self.input_channels = observation_shape[0]
        # Initial convolution
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Downsampling path
        layers = []
        current_channels = num_channels
        for out_channels, num_blocks in downsample_steps:
            layers.append(DownsampleBlock(current_channels, out_channels, num_blocks))
            current_channels = out_channels

        self.downsample = nn.Sequential(*layers)

        # Final residual blocks
        self.resblocks = nn.ModuleList([ResidualBlock(current_channels) for _ in range(6)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.downsample(x)
        for block in self.resblocks:
            x = block(x)
        # Scale hidden state to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)
        return x

class DynamicsNetwork(nn.Module):
    """Predicts next hidden state and immediate reward."""
    def __init__(self, num_channels: int, num_blocks: int, action_space_size: int,
                 reduced_channels_reward: int, fc_reward_layers: List[int], support_size: int):
        super().__init__()

        self.action_space_size = action_space_size

        # Process action and combine with state
        self.conv_action = nn.Conv2d(action_space_size, num_channels, kernel_size=1)
        self.conv_state = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

        # Reward head
        reward_layers = []
        current_channels = num_channels
        for hidden_size in fc_reward_layers:
            reward_layers.extend([
                nn.Linear(current_channels, hidden_size),
                nn.ReLU(),
            ])
            current_channels = hidden_size
        reward_layers.append(nn.Linear(current_channels, 2 * support_size + 1))
        self.reward_head = nn.Sequential(*reward_layers)

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_state.size(0)

        # Create action planes and combine with hidden state
        action_planes = torch.zeros(batch_size, self.action_space_size,
                                    hidden_state.size(2), hidden_state.size(3),
                                    device=hidden_state.device, dtype=hidden_state.dtype)

        action_idx = action.view(batch_size, 1, 1, 1).expand(-1, -1,
                                                              hidden_state.size(2),
                                                              hidden_state.size(3))
        action_planes.scatter_(1, action_idx, 1)

        # Process action and combine with state
        x = hidden_state + self.conv_action(action_planes)
        x = F.relu(self.bn(self.conv_state(x)))

        # Apply residual blocks
        for block in self.resblocks:
            x = block(x)

        # Scale hidden state to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)

        # Compute reward
        reward = self.reward_head(x.mean(dim=(2, 3)))

        return x, reward

class PredictionNetwork(nn.Module):
    """Predicts policy and value from hidden state."""
    def __init__(self, num_channels: int, action_space_size: int,
                 reduced_channels_policy: int, reduced_channels_value: int,
                 fc_policy_layers: List[int], fc_value_layers: List[int],
                 support_size: int):
        super().__init__()

        # Policy head
        policy_layers = []
        current_channels = num_channels
        for hidden_size in fc_policy_layers:
            policy_layers.extend([
                nn.Linear(current_channels, hidden_size),
                nn.ReLU(),
            ])
            current_channels = hidden_size
        policy_layers.append(nn.Linear(current_channels, action_space_size))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value head
        value_layers = []
        current_channels = num_channels
        for hidden_size in fc_value_layers:
            value_layers.extend([
                nn.Linear(current_channels, hidden_size),
                nn.ReLU(),
            ])
            current_channels = hidden_size
        value_layers.append(nn.Linear(current_channels, 2 * support_size + 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Global average pooling
        x = x.mean(dim=(2, 3))

        # Compute policy and value
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

class MuZeroNetwork(nn.Module):
    """Full MuZero network combining representation, dynamics and prediction."""
    def __init__(self, config: MuZeroConfig):
        super().__init__()

        self.config = config

        self.representation = RepresentationNetwork(
            config.observation_shape,
            config.num_channels,
            config.downsample_steps
        )

        self.dynamics = DynamicsNetwork(
            config.num_channels,
            config.num_blocks,
            config.action_space_size,
            config.reduced_channels_reward,
            config.fc_reward_layers,
            config.support_size
        )

        self.prediction = PredictionNetwork(
            config.num_channels,
            config.action_space_size,
            config.reduced_channels_policy,
            config.reduced_channels_value,
            config.fc_policy_layers,
            config.fc_value_layers,
            config.support_size
        )

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        """First inference from observation."""
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, None, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        """Next state inference from hidden state and action."""
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return NetworkOutput(value, reward, policy_logits, next_hidden_state)

def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert categorical distribution to scalar."""
    probabilities = torch.softmax(logits, dim=1)
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=logits.dtype)
    x = torch.sum(support * probabilities, dim=1)
    x = inverse_transform_value(x)
    return x

def scalar_to_support(scalar: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert scalar to categorical distribution."""
    scalar = transform_value(scalar)
    scalar = torch.clamp(scalar, -support_size, support_size)
    floor = scalar.floor()
    prob_upper = scalar - floor
    prob_lower = 1 - prob_upper
    support = torch.zeros(scalar.size(0), 2 * support_size + 1, device=scalar.device)
    lower_idx = (floor + support_size).long()
    upper_idx = (lower_idx + 1).clamp(0, 2 * support_size)
    lower_idx = lower_idx.clamp(0, 2 * support_size)

    support.scatter_(1, lower_idx.unsqueeze(1), prob_lower.unsqueeze(1))
    support.scatter_(1, upper_idx.unsqueeze(1), prob_upper.unsqueeze(1))

    return support

class MinMaxStats:
    """Maintains max-min statistics for value normalization."""
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        else:
            return 0.5  # If all values are the same, return 0.5

class Node:
    """Node in the MCTS tree."""
    def __init__(self, prior: float, hidden_state: Optional[torch.Tensor]):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[int, 'Node'] = {}
        self.hidden_state = hidden_state
        self.reward = 0
        self.expanded = False

    def is_expanded(self) -> bool:
        return self.expanded

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions: List[int], network_output: NetworkOutput):
        """Expand node with predictions from neural network."""
        self.expanded = True
        policy = torch.softmax(network_output.policy_logits, dim=1).cpu().numpy()[0]
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.scalar_reward()
        for action in actions:
            self.children[action] = Node(
                prior=policy[action],
                hidden_state=None
            )

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """Add exploration noise to prior probabilities."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - exploration_fraction) + n * exploration_fraction

class MCTS:
    """Monte Carlo Tree Search implementation."""
    def __init__(self, config: MuZeroConfig):
        self.config = config

    def run(self, root: Node, action_space_size: int, network: MuZeroNetwork, min_max_stats: MinMaxStats) -> Dict[int, float]:
        """Run MCTS simulations and return action probabilities."""
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            while node.is_expanded():
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            action = list(parent.children.keys())[list(parent.children.values()).index(node)]

            # Expand leaf node
            network_output = network.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]], device=parent.hidden_state.device)
            )
            node.expand(list(range(action_space_size)), network_output)
            value = network_output.scalar_value()

            self.backpropagate(search_path, value, self.config.discount, min_max_stats)

        # Return normalized visit counts as action probabilities
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        visit_counts.sort(reverse=True)  # Sort by visit count
        actions = [action for _, action in visit_counts]
        counts = [count for count, _ in visit_counts]
        sum_visits = sum(counts)
        action_probs = {action: count / sum_visits for action, count in zip(actions, counts)}
        return action_probs

    def select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        """Select child based on UCB score."""
        max_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child, min_max_stats)
            if score > max_score:
                max_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """Calculate UCB score for a child."""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
        """Backpropagate value through search path."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + discount * value

def select_action(config: MuZeroConfig, root: Node, temperature: float = 1.0) -> int:
    """Select action based on visit counts and temperature."""
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
    else:
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
        action = np.random.choice(actions, p=visit_count_distribution)

    return action

class ReplayBuffer:
    """Prioritized replay buffer implementation."""
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.replay_buffer_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.priorities = []
        self.config = config
        self.alpha = config.priority_alpha
        self.beta = config.priority_beta

    def save_game(self, game: 'Game', priorities: Optional[np.ndarray] = None):
        """Save game history to buffer with priorities."""
        if priorities is None:
            max_prio = max(self.priorities) if self.priorities else 1
            priorities = np.full(len(game.rewards), max_prio)

        # Append game history
        self.buffer.append(game)
        self.priorities.extend(priorities)

        # Remove old games if buffer is full
        while len(self.buffer) > self.window_size:
            removed_game = self.buffer.pop(0)
            self.priorities = self.priorities[len(removed_game.rewards):]

    def sample_batch(self, num_unroll_steps: int, td_steps: int) -> Tuple[List[Tuple[np.ndarray, List[int], List[Tuple[float, float, np.ndarray]]]], List[int], np.ndarray]:
        """Sample batch of training data."""
        total_priorities = sum(self.priorities)
        probs = [p / total_priorities for p in self.priorities]
        indices = np.random.choice(len(self.priorities), size=self.batch_size, p=probs)
        weights = (len(self.priorities) * np.array(probs)[indices]) ** -self.beta
        weights = weights / weights.max()

        batch = []
        batch_indices = []
        for idx in indices:
            game_index = 0
            while idx >= len(self.buffer[game_index].rewards):
                idx -= len(self.buffer[game_index].rewards)
                game_index += 1
            game = self.buffer[game_index]
            game_pos = idx
            batch.append((
                game.get_observation(game_pos),
                game.actions[game_pos:game_pos + num_unroll_steps],
                game.make_target(game_pos, num_unroll_steps, td_steps)
            ))
            batch_indices.append(idx)

        return batch, batch_indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)
