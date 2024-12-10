import math
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MuZeroConfig:
    """Configuration class for MuZero hyperparameters."""

    def __init__(self):
        # Environment
        self.env_name: str = "ALE/Pong-v5"
        self.observation_shape: Tuple[int, int, int] = None
        self.action_space_size: int = None
        self.num_players: int = 1
        self.support_size: int = 300  # 601 bins total

        # Network
        self.num_blocks: int = 2  # 16
        self.num_channels: int = 256
        self.reduced_channels_reward: int = 256
        self.reduced_channels_value: int = 256
        self.reduced_channels_policy: int = 256
        self.fc_reward_layers: List[int] = [256]
        self.fc_value_layers: List[int] = [256]
        self.fc_policy_layers: List[int] = [256]
        self.downsample_steps: List[Tuple[int, int]] = [(128, 2), (256, 3)]

        # Training
        self.training_steps: int = int(1000e3)
        self.checkpoint_interval: int = 1000
        self.batch_size: int = 2  # 1024
        self.num_unroll_steps: int = 5
        self.td_steps: int = 10
        self.num_actors: int = 2
        self.max_moves: int = 27000
        self.num_simulations: int = 5  # 50

        # Replay Buffer
        self.replay_buffer_size: int = int(1e6)
        self.num_workers: int = 2
        self.minimum_games_in_buffer: int = 2
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
        self.max_grad_norm: float = 10.0

        # Evaluation
        self.eval_interval: int = 1000
        self.eval_episodes: int = 32

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Return the temperature for visit softmax selection."""
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


def transform_value(value: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    return torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1 +
                                eps * value)


def inverse_transform_value(
    transformed_value: torch.Tensor,
    eps: float = 0.001
) -> torch.Tensor:
    return (torch.sign(transformed_value) *
            (((torch.sqrt(1 + 4 * eps * (torch.abs(transformed_value) + 1 +
              eps)) - 1) / (2 * eps)) ** 2 - 1))


def support_to_scalar(
    logits: torch.Tensor,
    support_size: int
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    support = torch.arange(-support_size,
                           support_size + 1,
                           device=logits.device,
                           dtype=logits.dtype)
    x = torch.sum(support * probabilities, dim=1)
    x = inverse_transform_value(x)
    return x


def scalar_to_support(
    scalar: torch.Tensor,
    support_size: int
) -> torch.Tensor:
    scalar = transform_value(scalar)
    scalar = torch.clamp(scalar, -support_size, support_size)
    floor = scalar.floor()
    prob_upper = scalar - floor
    prob_lower = 1 - prob_upper
    support = torch.zeros(scalar.size(0),
                          2 * support_size + 1,
                          device=scalar.device)
    lower_idx = (floor + support_size).long()
    upper_idx = (lower_idx + 1).clamp_(0, 2 * support_size)
    lower_idx = lower_idx.clamp_(0, 2 * support_size)
    support.scatter_(1, lower_idx.unsqueeze(1), prob_lower.unsqueeze(1))
    support.scatter_(1, upper_idx.unsqueeze(1), prob_upper.unsqueeze(1))
    return support


def scalar_to_support_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    support_size: int
) -> torch.Tensor:
    target_support = scalar_to_support(targets, support_size)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_support * log_probs).sum(1)
    return loss


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        logging.info(f"Initialized ResidualBlock with {num_channels} channels.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.blocks = nn.ModuleList([ResidualBlock(out_channels) for _ in range(num_blocks)])

        logging.info(f"Initialized DownsampleBlock from {in_channels} to {out_channels} with {num_blocks} blocks.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        return x


class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], num_channels: int, downsample_steps: List[Tuple[int, int]]):
        super().__init__()
        self.input_channels = observation_shape[0]
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        layers = []
        current_channels = num_channels
        for out_channels, num_blocks in downsample_steps:
            layers.append(DownsampleBlock(current_channels, out_channels, num_blocks))
            current_channels = out_channels
        self.downsample = nn.Sequential(*layers)
        self.resblocks = nn.ModuleList([ResidualBlock(current_channels) for _ in range(6)])

        logging.info(f"Initialized RepresentationNetwork with input_channels={self.input_channels}, num_channels={num_channels}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # logging.info(f"RepresentationNetwork forward called with input shape={x.shape}.")
        x = F.relu(self.bn1(self.conv1(x)))
       # logging.info(f"After initial conv: {x.shape}")
        x = self.downsample(x)
       # logging.info(f"After downsample: {x.shape}")
        for i, block in enumerate(self.resblocks):
            x = block(x)
            #logging.info(f"After resblock {i+1}: {x.shape}")
        return x


class DynamicsNetwork(nn.Module):
    def __init__(self, num_channels: int, num_blocks: int, action_space_size: int,
                 reduced_channels_reward: int, fc_reward_layers: List[int], support_size: int):
        super().__init__()
        self.action_space_size = action_space_size
        self.conv_action = nn.Conv2d(action_space_size, num_channels, kernel_size=1)
        self.conv_state = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

        reward_layers = []
        current_channels = num_channels
        for hidden_size in fc_reward_layers:
            reward_layers.append(nn.Linear(current_channels, hidden_size))
            reward_layers.append(nn.ReLU())
            current_channels = hidden_size
        reward_layers.append(nn.Linear(current_channels, 2 * support_size + 1))
        self.reward_head = nn.Sequential(*reward_layers)

      #  logging.info(f"Initialized DynamicsNetwork with num_channels={num_channels}, action_space_size={action_space_size}.")

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #logging.info(f"DynamicsNetwork forward called with hidden_state shape={hidden_state.shape}, action shape={action.shape}.")
        batch_size = hidden_state.size(0)
        action_planes = torch.zeros(batch_size, self.action_space_size, hidden_state.size(2), hidden_state.size(3),
                                    device=hidden_state.device, dtype=hidden_state.dtype)
        action_idx = action.view(batch_size, 1, 1, 1).expand(-1, -1, hidden_state.size(2), hidden_state.size(3))
        action_planes.scatter_(1, action_idx, 1)

        x = hidden_state + self.conv_action(action_planes)
        x = F.relu(self.bn(self.conv_state(x)))
       # logging.info(f"DynamicsNetwork after conv_state: {x.shape}")
        for i, block in enumerate(self.resblocks):
            x = block(x)
           # logging.info(f"DynamicsNetwork after resblock {i+1}: {x.shape}")

        reward = self.reward_head(x.mean(dim=(2, 3)))
        #logging.info(f"DynamicsNetwork reward head output shape: {reward.shape}")
        return x, reward


class PredictionNetwork(nn.Module):
    def __init__(self, num_channels: int, action_space_size: int,
                 reduced_channels_policy: int, reduced_channels_value: int,
                 fc_policy_layers: List[int], fc_value_layers: List[int], support_size: int):
        super().__init__()
        current_channels = num_channels
        policy_layers = []
        for hidden_size in fc_policy_layers:
            policy_layers.append(nn.Linear(current_channels, hidden_size))
            policy_layers.append(nn.ReLU())
            current_channels = hidden_size
        policy_layers.append(nn.Linear(current_channels, action_space_size))
        self.policy_head = nn.Sequential(*policy_layers)

        current_channels = num_channels
        value_layers = []
        for hidden_size in fc_value_layers:
            value_layers.append(nn.Linear(current_channels, hidden_size))
            value_layers.append(nn.ReLU())
            current_channels = hidden_size
        value_layers.append(nn.Linear(current_channels, 2 * support_size + 1))
        self.value_head = nn.Sequential(*value_layers)

        logging.info(f"Initialized PredictionNetwork with action_space_size={action_space_size}.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      #  logging.info(f"PredictionNetwork forward called with input shape={x.shape}.")
        x_mean = x.mean(dim=(2, 3))
       # logging.info(f"PredictionNetwork x_mean shape: {x_mean.shape}")
        policy_logits = self.policy_head(x_mean)
       # logging.info(f"PredictionNetwork policy_logits shape: {policy_logits.shape}")
        value = self.value_head(x_mean)
       # logging.info(f"PredictionNetwork value head output shape: {value.shape}")
        return policy_logits, value


class NetworkOutput:
    def __init__(self, value: torch.Tensor, reward: Optional[torch.Tensor],
                 policy_logits: torch.Tensor, hidden_state: torch.Tensor):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state
        """
        logging.info(
            f"Created NetworkOutput with value shape={value.shape}, "
            f"reward shape={reward.shape if reward is not None else None}, "
            f"policy_logits shape={policy_logits.shape}, hidden_state shape={hidden_state.shape}"
        )
        """
    def scalar_value(self) -> float:
        return support_to_scalar(self.value, self.value.shape[-1] // 2).item()

    def scalar_reward(self) -> Optional[float]:
        if self.reward is None:
            return None
        return support_to_scalar(self.reward, self.reward.shape[-1] // 2).item()


class MuZeroNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.representation = RepresentationNetwork(
            config.observation_shape,
            config.num_channels,
            config.downsample_steps
        )
        self.dynamics = DynamicsNetwork(config.num_channels,
                                        config.num_blocks,
                                        config.action_space_size,
                                        config.reduced_channels_reward,
                                        config.fc_reward_layers,
                                        config.support_size)
        self.prediction = PredictionNetwork(config.num_channels,
                                            config.action_space_size,
                                            config.reduced_channels_policy,
                                            config.reduced_channels_value,
                                            config.fc_policy_layers,
                                            config.fc_value_layers,
                                            config.support_size)

        logging.info("Initialized MuZeroNetwork.")

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        #logging.info(f"initial_inference called with observation shape={observation.shape}.")
        hidden_state = self.representation(observation)
      #  logging.info(f"initial_inference hidden_state shape={hidden_state.shape}.")
        policy_logits, value = self.prediction(hidden_state)
       # logging.info(f"initial_inference policy_logits shape={policy_logits.shape}, value shape={value.shape}.")
        return NetworkOutput(value, None, policy_logits, hidden_state)

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> NetworkOutput:
       # logging.info(f"recurrent_inference called with hidden_state shape={hidden_state.shape}, action shape={action.shape}.")
        next_hidden_state, reward = self.dynamics(hidden_state, action)
       # logging.info(f"recurrent_inference next_hidden_state shape={next_hidden_state.shape}, reward shape={reward.shape}.")
        policy_logits, value = self.prediction(next_hidden_state)
       # logging.info(f"recurrent_inference policy_logits shape={policy_logits.shape}, value shape={value.shape}.")
        return NetworkOutput(value, reward, policy_logits, next_hidden_state)


class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        #logging.info("Initialized MinMaxStats.")

    def update(self, value: float) -> None:
        previous_max = self.maximum
        previous_min = self.minimum
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        #logging.debug(f"MinMaxStats updated: previous_max={previous_max}, previous_min={previous_min}, new_max={self.maximum}, new_min={self.minimum}")

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            normalized = (value - self.minimum) / (self.maximum - self.minimum)
        else:
            normalized = 0.5
        #logging.debug(f"MinMaxStats normalize: value={value}, normalized={normalized}")
        return normalized


class Node:
    def __init__(self, prior: float, hidden_state: Optional[torch.Tensor]):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, 'Node'] = {}
        self.hidden_state = hidden_state
        self.reward = 0.0
        self.expanded_flag = False

    def is_expanded(self) -> bool:
        return self.expanded_flag

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        actions: List[int],
        network_output: 'NetworkOutput'
    ) -> None:
        self.expanded_flag = True
        policy = torch.softmax(network_output.policy_logits, dim=1).detach().cpu().numpy()[0]
        self.hidden_state = network_output.hidden_state
        self.reward = (network_output.scalar_reward() if network_output.reward is not None else 0)
        #logging.info(f"Node expanded with {len(actions)} actions.")
        for action in actions:
            self.children[action] = Node(prior=policy[action], hidden_state=None)
        #logging.info(f"Children actions: {list(self.children.keys())}")

    def add_exploration_noise(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float
    ) -> None:
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            original_prior = self.children[a].prior
            self.children[a].prior = (self.children[a].prior * (1 - exploration_fraction) +
                                      n * exploration_fraction)
            #logging.info(f"Added exploration noise to action {a}: original_prior={original_prior}, new_prior={self.children[a].prior}")


class MCTS:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        #logging.info("Initialized MCTS with given MuZeroConfig.")

    def run(
        self,
        root: Node,
        action_space_size: int,
        network: MuZeroNetwork,
        min_max_stats: MinMaxStats
    ) -> Dict[int, float]:
        #logging.info(f"MCTS run started with {self.config.num_simulations} simulations.")
        expansions = []
        for sim in range(self.config.num_simulations):
            node = root
            search_path = [node]
            actions_path = []

            # Descend the tree
            while node.is_expanded():
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)
                actions_path.append(action)

            parent = search_path[-2]
            action = actions_path[-1]

            expansions.append((parent.hidden_state, action, search_path))
            #logging.info(f"MCTS simulation {sim+1}: Expanded action {action}.")

        if expansions:
            # Batch inference
            hidden_states = torch.cat([hs for (hs, _, _) in expansions], dim=0)
            actions = torch.tensor([a for (_, a, _) in expansions], device=hidden_states.device).unsqueeze(1)
            #logging.info(f"MCTS batch inference with hidden_states shape={hidden_states.shape} and actions shape={actions.shape}.")
            network_output_batch = network.recurrent_inference(hidden_states, actions)
            #logging.info(f"MCTS received network_output_batch.")

            # Assign expansions and backpropagate
            for i, (_, _, search_path) in enumerate(expansions):
                node = search_path[-1]
                node.expand(list(range(action_space_size)), NetworkOutput(
                    network_output_batch.value[i:i+1],
                    network_output_batch.reward[i:i+1],
                    network_output_batch.policy_logits[i:i+1],
                    network_output_batch.hidden_state[i:i+1]
                ))
                value = support_to_scalar(network_output_batch.value[i:i+1], self.config.support_size).item()
                #logging.info(f"MCTS simulation {i+1}: Backpropagating value {value}.")
                self.backpropagate(search_path, value, self.config.discount, min_max_stats)

        # Calculate visit counts
        visit_counts = [(child.visit_count, a) for a, child in root.children.items()]
        sum_visits = sum([vc for vc, _ in visit_counts])
        if sum_visits == 0:
            #logging.warning("Sum of visit counts is zero. Returning uniform probabilities.")
            action_probs = {a: 1.0 / len(root.children) for _, a in visit_counts}
        else:
            action_probs = {a: vc / sum_visits for vc, a in visit_counts}
        #logging.info(f"MCTS run completed with action_probs: {action_probs}")
        return action_probs

    def select_child(
        self,
        node: Node,
        min_max_stats: MinMaxStats
    ) -> Tuple[int, Node]:
        max_score = -float('inf')
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            score = self.ucb_score(node, child, min_max_stats)
            if score > max_score:
                max_score = score
                best_action = action
                best_child = child
        #logging.debug(f"Selected child action {best_action} with score {max_score}.")
        return best_action, best_child

    def ucb_score(
        self,
        parent: Node,
        child: Node,
        min_max_stats: MinMaxStats
    ) -> float:
        pb_c = (math.log((parent.visit_count + self.config.pb_c_base + 1) /
                         self.config.pb_c_base) + self.config.pb_c_init)
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def backpropagate(
        self,
        search_path: List[Node],
        value: float,
        discount: float,
        min_max_stats: MinMaxStats
    ) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + discount * value
            #logging.info(f"Backpropagated value {value} to node. Current visit_count={node.visit_count}, value_sum={node.value_sum}.")


def select_action(
    config: MuZeroConfig,
    root: Node,
    temperature: float = 1.0
) -> int:
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
        #logging.info(f"Action selected with temperature=0: {action}")
    else:
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution /= visit_count_distribution.sum()
        action = np.random.choice(actions, p=visit_count_distribution)
        #logging.info(f"Action selected with temperature={temperature}: {action}")
    return action


class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.replay_buffer_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.priorities = []
        self.config = config
        self.alpha = config.priority_alpha
        self.beta = config.priority_beta

        #logging.info("Initialized ReplayBuffer.")

    def save_game(self, game: 'Game', priorities: Optional[np.ndarray] = None):
        if priorities is None:
            max_prio = max(self.priorities) if self.priorities else 1
            priorities = np.full(len(game.rewards), max_prio)

        self.buffer.append(game)
        self.priorities.extend(priorities)
        logging.info(f"Saved game to ReplayBuffer. Buffer size: {len(self.buffer)}, Priorities size: {len(self.priorities)}")

        while len(self.buffer) > self.window_size:
            removed_game = self.buffer.pop(0)
            self.priorities = self.priorities[len(removed_game.rewards):]
            logging.info(f"Removed oldest game from ReplayBuffer. Buffer size: {len(self.buffer)}, Priorities size: {len(self.priorities)}")

    def sample_batch(
        self,
        num_unroll_steps: int,
        td_steps: int
    ) -> Tuple[List[Tuple[np.ndarray, List[int], List[Tuple[float, float, np.ndarray]]]], List[int], np.ndarray]:
        total_priorities = sum(self.priorities)
        if total_priorities == 0:
            logging.warning("Total priorities sum to zero. Sampling uniformly.")
            probs = [1 / len(self.priorities) for _ in self.priorities]
        else:
            probs = [p / total_priorities for p in self.priorities]
        indices = np.random.choice(len(self.priorities), size=self.batch_size, p=probs)
        weights = (len(self.priorities) * np.array(probs)[indices]) ** -self.beta
        weights = weights / weights.max()

        batch = []
        batch_indices = []
        for idx in indices:
            game_index = 0
            while game_index < len(self.buffer) and idx >= len(self.buffer[game_index].rewards):
                idx -= len(self.buffer[game_index].rewards)
                game_index += 1
            if game_index >= len(self.buffer):
                logging.warning(f"Index {idx} out of bounds. Skipping.")
                continue
            game = self.buffer[game_index]
            game_pos = idx
            batch.append((
                game.get_observation(game_pos),
                game.actions[game_pos:game_pos + num_unroll_steps],
                game.make_target(game_pos, num_unroll_steps, td_steps)
            ))
            batch_indices.append(idx)

        logging.info(f"Sampled batch from ReplayBuffer. Batch size: {len(batch)}, Sampled indices: {batch_indices}")
        return batch, batch_indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
        logging.info(f"Updated priorities for indices: {indices}")

    def __len__(self) -> int:
        return len(self.buffer)
