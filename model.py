import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
from collections import deque
import gym
from gym.wrappers import AtariPreprocessing, FrameStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MuZeroConfig:
    def __init__(self):
        self.action_space_size: int = None
        self.num_players: int = None

        self.observation_shape: Tuple[int, int, int] = None
        self.support_size: int = 300
        self.num_blocks: int = 2
        self.num_channels: int = 128

        self.num_actors: int = 1
        self.max_moves: int = 512
        self.num_simulations: int = 20

        self.root_dirichlet_alpha: float = 0.3
        self.root_exploration_fraction: float = 0.25

        self.pb_c_base: float = 19652
        self.pb_c_init: float = 1.25

        self.training_steps: int = int(1000e3)
        self.checkpoint_interval: int = int(1e3)
        self.batch_size: int = 1024
        self.num_unroll_steps: int = 5
        self.td_steps: int = 10
        self.num_mcts_workers: int = 16

        self.replay_buffer_size: int = int(1e6)
        self.num_workers: int = 4

        self.weight_decay: float = 1e-4
        self.momentum: float = 0.9
        self.lr_init: float = 0.05

        self.lr_decay_rate: float = 0.1
        self.lr_decay_steps: int = 350e3

        self.discount: float = 0.997
        self.value_loss_weight: float = 0.25
        self.reward_loss_weight: float = 1.0
        self.policy_loss_weight: float = 1.0
        self.max_grad_norm: float = 5.0

        self.num_envs: int = 8

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name: str):
        if env_name == 'atari':
            self.action_space_size = 6
            self.num_players = 1
            self.observation_shape = (4, 84, 84)
        else:
            raise ValueError(f"Unknown game: {env_name}")


class NetworkOutput:
    def __init__(self, value: torch.Tensor, reward: torch.Tensor, policy_logits: torch.Tensor, hidden_state: torch.Tensor):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state

    def scalar_value(self) -> float:
        return support_to_scalar(self.value, self.value.shape[-1] // 2).item()

    def scalar_reward(self) -> float:
        return support_to_scalar(self.reward, self.reward.shape[-1] // 2).item() if self.reward is not None else None


class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], num_blocks: int, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device, dtype=torch.bfloat16)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)
        return x


class MuZeroNetwork(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], action_space_size: int, num_blocks: int, num_channels: int, support_size: int):
        super().__init__()
        self.action_space_size = action_space_size
        self.num_channels = num_channels
        self.support_size = support_size

        self.representation = RepresentationNetwork(observation_shape, num_blocks, num_channels).to(device, dtype=torch.bfloat16)
        self.dynamics = DynamicsNetwork(num_channels, num_blocks, action_space_size, support_size).to(device, dtype=torch.bfloat16)
        self.prediction = PredictionNetwork(num_channels, action_space_size, support_size).to(device, dtype=torch.bfloat16)

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, None, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return NetworkOutput(value, reward, policy_logits, next_hidden_state)


class PredictionNetwork(nn.Module):
    def __init__(self, num_channels: int, action_space_size: int, support_size: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.fc_policy = nn.Linear(num_channels, action_space_size)
        self.fc_value = nn.Linear(num_channels, 2 * support_size + 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(device, dtype=torch.bfloat16)
        x = F.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))  # Global average pooling
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value


class DynamicsNetwork(nn.Module):
    def __init__(self, num_channels: int, num_blocks: int, action_space_size: int, support_size: int):
        super().__init__()
        self.action_space_size = action_space_size
        self.conv = nn.Conv2d(num_channels + action_space_size, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.reward_head = nn.Linear(num_channels, 2 * support_size + 1)

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = hidden_state.to(device, dtype=torch.bfloat16)
        batch_size = hidden_state.size(0)
        if action is not None:
            action = action.view(batch_size, 1)
            action_one_hot = torch.zeros(batch_size, self.action_space_size, hidden_state.size(2), hidden_state.size(3), device=device, dtype=torch.bfloat16)
            action_one_hot.scatter_(1, action.view(batch_size, 1, 1, 1).expand(-1, -1, hidden_state.size(2), hidden_state.size(3)), 1)
        else:
            action_one_hot = torch.zeros(batch_size, self.action_space_size, hidden_state.size(2), hidden_state.size(3), device=device, dtype=torch.bfloat16)
        
        x = torch.cat([hidden_state, action_one_hot], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
        reward = self.reward_head(x.mean(dim=(2, 3)))
        return x, reward


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device, dtype=torch.bfloat16)
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class Node:
    def __init__(self, prior: float, hidden_state: torch.Tensor):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = hidden_state
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def update(self, value: float):
        self.value_sum += value
        self.visit_count += 1


class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTS:
    def __init__(self, config: MuZeroConfig):
        self.config = config

    def run(self, root: Node, action_space_size: int, network: MuZeroNetwork, min_max_stats: MinMaxStats) -> Node:
        for _ in range(self.config.num_simulations):
            self.run_single_simulation(root, action_space_size, network, min_max_stats)
        return root

    def run_single_simulation(self, root: Node, action_space_size: int, network: MuZeroNetwork, min_max_stats: MinMaxStats):
        node = root
        search_path = [node]
        current_tree_depth = 0

        while node.expanded():
            current_tree_depth += 1
            action, node = self.select_child(node, min_max_stats)
            search_path.append(node)

        parent = search_path[-1]
        with torch.no_grad():
            network_output = network.recurrent_inference(parent.hidden_state, None)

        self.expand_node(parent, action_space_size, network_output)

        value = network_output.scalar_value()
        reward = network_output.scalar_reward()
        self.backpropagate(search_path, value, reward if reward is not None else 0, min_max_stats)

    def select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        _, action, child = max((self.ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())
        return prior_score + value_score

    def expand_node(self, node: Node, action_space_size: int, network_output: NetworkOutput):
        node.hidden_state = network_output.hidden_state
        reward = network_output.scalar_reward()
        for i in range(action_space_size):
            child_node = Node(prior=0, hidden_state=network_output.hidden_state)
            child_node.reward = reward if reward is not None else 0
            node.children[i] = child_node

    def backpropagate(self, search_path: List[Node], value: float, reward: float, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.update(value)
            min_max_stats.update(node.value())
            value = node.reward + self.config.discount * value


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=torch.bfloat16)
    x = torch.sum(support * probabilities, dim=-1)
    return x


def select_action(config: MuZeroConfig, root: Node, temperature: float) -> int:
    if not root.children:
        return np.random.randint(config.action_space_size)

    visit_counts = [child.visit_count for child in root.children.values()]
    actions = [action for action in root.children.keys()]
    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
    elif temperature == float('inf'):
        action = np.random.choice(actions)
    else:
        visit_count_distribution = np.array(visit_counts) ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
        action = np.random.choice(actions, p=visit_count_distribution)

    return action


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(*x.shape, 2 * support_size + 1, device=x.device, dtype=torch.bfloat16)
    logits.scatter_(-1, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(indexes > 2 * support_size, 0.0)
    indexes = indexes.masked_fill_(indexes > 2 * support_size, 0.0)
    logits.scatter_(-1, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def update_weights(
    optimizer: torch.optim.Optimizer,
    network: MuZeroNetwork,
    batch: List[Tuple[torch.Tensor, List[int], List[Tuple[float, float, List[float]]]]],
    config: MuZeroConfig
) -> Tuple[float, float, float, float]:
    total_value_loss, total_reward_loss, total_policy_loss = 0.0, 0.0, 0.0
    total_loss = 0.0

    for image, actions, targets in batch:
        optimizer.zero_grad()  # Zero gradients for each item in the batch
        
        image = image.to(device, dtype=torch.bfloat16)
        network_output = network.initial_inference(image)
        hidden_state = network_output.hidden_state
        predictions = [(1.0, network_output)]

        for action in actions:
            action = torch.tensor([action], device=device)
            network_output = network.recurrent_inference(hidden_state, action)
            hidden_state = network_output.hidden_state
            predictions.append((1.0 / len(actions), network_output))

        batch_loss = 0.0
        for prediction, target in zip(predictions, targets):
            gradient_scale, network_output = prediction
            target_value, target_reward, target_policy = target

            value_loss = config.value_loss_weight * gradient_scale * scalar_to_support_loss(
                network_output.value, torch.tensor(target_value, device=device, dtype=torch.bfloat16), config.support_size
            )
            total_value_loss += value_loss.item()

            if network_output.reward is not None:
                reward_loss = config.reward_loss_weight * gradient_scale * scalar_to_support_loss(
                    network_output.reward, torch.tensor(target_reward, device=device, dtype=torch.bfloat16), config.support_size
                )
                total_reward_loss += reward_loss.item()
            else:
                reward_loss = 0

            target_policy_tensor = torch.tensor(target_policy, device=device, dtype=torch.float32)
            target_policy_tensor = target_policy_tensor.unsqueeze(0).expand_as(network_output.policy_logits)

            policy_loss = config.policy_loss_weight * gradient_scale * F.cross_entropy(
                network_output.policy_logits,
                target_policy_tensor
            )
            total_policy_loss += policy_loss.item()

            batch_loss += value_loss + reward_loss + policy_loss

        if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
            print("NaN or inf detected in loss. Skipping this update.")
            continue

        batch_loss.backward()  # Backward pass for each item in the batch
        total_loss += batch_loss.item()

    # Clip gradients and update weights after processing all items in the batch
    torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
    optimizer.step()

    return total_loss, total_value_loss, total_reward_loss, total_policy_loss

def scalar_to_support_loss(logits: torch.Tensor, scalar: torch.Tensor, support_size: int) -> torch.Tensor:
    scalar_support = scalar_to_support(scalar.unsqueeze(-1), support_size).squeeze(-1)
    return F.cross_entropy(logits, torch.argmax(scalar_support, dim=1))

class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.replay_buffer_size
        self.batch_size = config.batch_size
        self.buffer = deque(maxlen=self.window_size)

    def save_game(self, game: 'GameHistory'):
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int) -> List[Tuple[torch.Tensor, List[int], List[Tuple[float, float, List[float]]]]]:
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]

        return [(self.process_observation(g.make_image(i)),
                 g.action_history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play_history[i]))
                for (g, i) in game_pos]

    def sample_game(self) -> 'GameHistory':
        return np.random.choice(self.buffer)

    def sample_position(self, game: 'GameHistory') -> int:
        return np.random.randint(len(game.action_history))

    def __len__(self) -> int:
        return len(self.buffer)

    def process_observation(self, observation: np.ndarray) -> torch.Tensor:
        if isinstance(observation, (gym.wrappers.frame_stack.LazyFrames, np.ndarray)):
            observation_array = np.array(observation)
        elif isinstance(observation, tuple):
            if isinstance(observation[0], gym.wrappers.frame_stack.LazyFrames):
                observation_array = np.array(observation[0])
            elif isinstance(observation[0], np.ndarray):
                observation_array = observation[0]
            else:
                raise ValueError(f"Unexpected observation type inside tuple: {type(observation[0])}")
        else:
            raise ValueError(f"Unexpected observation type: {type(observation)}")

        if observation_array.ndim == 4:  # Batched observations
            pass
        elif observation_array.ndim == 3:  # Single observation
            observation_array = np.expand_dims(observation_array, axis=0)
        else:
            raise ValueError(f"Unexpected observation shape: {observation_array.shape}")

        return torch.tensor(observation_array, dtype=torch.bfloat16, device=device)
