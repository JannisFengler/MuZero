# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state

    def scalar_value(self):
        return support_to_scalar(self.value, self.value.shape[-1] // 2).item()

    def scalar_reward(self):
        return support_to_scalar(self.reward, self.reward.shape[-1] // 2) if self.reward is not None else None

class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = x.to(device, dtype=torch.bfloat16)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)
        return x

class MuZeroNetwork(nn.Module):
    def __init__(self, observation_shape, action_space_size, num_blocks, num_channels, support_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.num_channels = num_channels
        self.support_size = support_size

        self.representation = RepresentationNetwork(observation_shape, num_blocks, num_channels).to(device, dtype=torch.bfloat16)
        self.dynamics = DynamicsNetwork(num_channels, num_blocks, action_space_size, support_size).to(device, dtype=torch.bfloat16)
        self.prediction = PredictionNetwork(num_channels, action_space_size, support_size).to(device, dtype=torch.bfloat16)

    def initial_inference(self, observation) -> NetworkOutput:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, None, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action=None) -> NetworkOutput:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return NetworkOutput(value, reward, policy_logits, next_hidden_state)

class PredictionNetwork(nn.Module):
    def __init__(self, num_channels, action_space_size, support_size):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.fc_policy = nn.Linear(num_channels, action_space_size)
        self.fc_value = nn.Linear(num_channels, 2 * support_size + 1)

    def forward(self, x):
        x = x.to(device, dtype=torch.bfloat16)
        x = F.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))  # Global average pooling
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

class DynamicsNetwork(nn.Module):
    def __init__(self, num_channels, num_blocks, action_space_size, support_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.conv = nn.Conv2d(num_channels + action_space_size, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.reward_head = nn.Linear(num_channels, 2 * support_size + 1)

    def forward(self, hidden_state, action=None):
        hidden_state = hidden_state.to(device, dtype=torch.bfloat16)
        batch_size = hidden_state.size(0)
        if action is not None:
            action_one_hot = torch.zeros(batch_size, self.action_space_size, hidden_state.size(2), hidden_state.size(3)).to(hidden_state.device, dtype=torch.bfloat16)
            action_one_hot.scatter_(1, action.view(batch_size, 1, 1, 1).expand(-1, -1, hidden_state.size(2), hidden_state.size(3)), 1)
        else:
            action_one_hot = torch.zeros(batch_size, self.action_space_size, hidden_state.size(2), hidden_state.size(3)).to(hidden_state.device, dtype=torch.bfloat16)
        
        x = torch.cat([hidden_state, action_one_hot], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
        reward = self.reward_head(x.mean(dim=(2, 3)))
        return x, reward

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = x.to(device, dtype=torch.bfloat16)
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Node:
    def __init__(self, prior, hidden_state):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = hidden_state
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def update(self, value):
        self.value_sum += value
        self.visit_count += 1

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def save_game(self, game):
        if len(self.buffer) < self.capacity:
            self.buffer.append(game)
        else:
            self.buffer[self.position] = game
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, num_unroll_steps: int, td_steps: int) -> List:
        games = self.sample_games(num_unroll_steps + td_steps)
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_games(self, num_unroll_steps):
        return np.random.choice(self.buffer, num_unroll_steps)

    def sample_position(self, game) -> int:
        return np.random.randint(len(game.history))

    def __len__(self) -> int:
        return len(self.buffer)

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
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()

    def run(self, root: Node, action_space_size: int, network: MuZeroNetwork, min_max_stats: MinMaxStats) -> Node:
        if root.hidden_state is None:
            raise ValueError("Root node must have a hidden state")

        with ThreadPoolExecutor(max_workers=self.config.num_mcts_workers) as executor:
            for _ in range(0, self.config.num_simulations, self.config.num_mcts_workers):
                futures = [executor.submit(self.run_single_simulation, root, action_space_size, network, min_max_stats) for _ in range(self.config.num_mcts_workers)]
                for future in futures:
                    future.result()

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
            network_output = network.recurrent_inference(parent.hidden_state)

        with self.lock:
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
        rewards = network_output.scalar_reward()
        if rewards is not None:
            for i, reward in enumerate(rewards):
                child_node = Node(prior=0, hidden_state=network_output.hidden_state)
                child_node.reward = reward.item()  # Convert the reward tensor to a scalar value
                node.children[i] = child_node

    def backpropagate(self, search_path: List[Node], value: float, reward: float, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            with self.lock:
                node.update(value)
            min_max_stats.update(node.value())
            value = node.reward + self.config.discount * value

def support_to_scalar(logits, support_size):
    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(-support_size, support_size + 1).to(logits.device, dtype=torch.bfloat16)
    x = torch.sum(support * probabilities, dim=-1)
    return x.item()

def select_action(config, root: Node, temperature: float) -> int:
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

def scalar_to_support(x, support_size):
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

def support_to_scalar(logits, support_size):
    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(-support_size, support_size + 1).to(logits.device, dtype=torch.bfloat16)
    x = torch.sum(support * probabilities, dim=-1)
    return x

def update_weights(optimizer: torch.optim.Optimizer, network: MuZeroNetwork, batch, config):
    loss = 0
    value_loss, reward_loss, policy_loss = 0, 0, 0

    for image, actions, targets in batch:
        image = image.to(device, dtype=torch.bfloat16)
        network_output = network.initial_inference(image)
        hidden_state = network_output.hidden_state
        predictions = [(1.0, network_output)]

        for action in actions:
            network_output = network.recurrent_inference(hidden_state, action)
            hidden_state = network_output.hidden_state
            predictions.append((1.0 / len(actions), network_output))

        for prediction, target in zip(predictions, targets):
            gradient_scale, network_output = prediction
            target_value, target_reward, target_policy = target

            value_loss += config.value_loss_weight * gradient_scale * scalar_to_support_loss(
                network_output.value, target_value, config.support_size
            )

            if network_output.reward is not None:
                reward_loss += config.reward_loss_weight * gradient_scale * scalar_to_support_loss(
                    network_output.reward, target_reward, config.support_size
                )

            policy_loss += config.policy_loss_weight * gradient_scale * F.cross_entropy(
                network_output.policy_logits,
                torch.tensor(target_policy).to(network_output.policy_logits.device)
            )

    loss = value_loss + reward_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
    optimizer.step()

    return (loss.item(), value_loss.item(), reward_loss.item(), policy_loss.item())

def scalar_to_support_loss(logits, scalar, support_size):
    scalar_support = scalar_to_support(scalar.unsqueeze(-1), support_size).squeeze(-1)
    return F.cross_entropy(logits, torch.argmax(scalar_support, dim=1))

class MuZeroConfig:
    def __init__(self):
        self.action_space_size = None
        self.num_players = None

        self.observation_shape = None
        self.support_size = 300
        self.num_blocks = 6
        self.num_channels = 128

        self.num_actors = 1
        self.max_moves = 512
        self.num_simulations = 20

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.batch_size = 1024
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.num_mcts_workers = 16

        self.replay_buffer_size = int(1e6)
        self.num_workers = 4

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.05

        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 350e3

        self.discount = 0.997
        self.value_loss_weight = 0.25
        self.reward_loss_weight = 1.0
        self.policy_loss_weight = 1.0
        self.max_grad_norm = 5

        self.num_envs = 8

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name):
        if env_name == 'atari':
            self.action_space_size = 6
            self.num_players = 1
            self.observation_shape = (4, 84, 84)
        else:
            raise ValueError(f"Unknown game: {env_name}")
