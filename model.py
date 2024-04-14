import torch
import torch.nn as nn
import math
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """Buffer to store gameplay experiences."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def size(self):
        return len(self.buffer)

class RepresentationNetwork(nn.Module):
    """Neural network to encode the observation into a hidden state."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Assuming the input has one channel
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # Example input to determine output size
        example_input = torch.zeros((1, 1, 84, 84))  # Update the size according to your input dimension
        example_output = self.conv(example_input)
        output_size = example_output.numel()
        self.fc = nn.Linear(output_size, 256)

    def forward(self, x):
        x = x.unsqueeze(1)  # Assuming x is missing the channel dimension
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embedding = nn.Embedding(10, 10)
        self.transition = nn.Sequential(
            nn.Linear(256 + 10, 256),
            nn.ReLU()
        )
        self.reward_predictor = nn.Linear(256, 1)
        self.terminal_predictor = nn.Linear(256, 1)

    def forward(self, state, action):
        # Flatten the state tensor
        state_flat = state.view(state.size(0), -1)

        # Embedding the action
        action_emb = self.action_embedding(action)

        # Concatenating along the feature dimension
        x = torch.cat([state_flat, action_emb], dim=1)

        next_state = self.transition(x)
        reward = self.reward_predictor(next_state)
        terminal = torch.sigmoid(self.terminal_predictor(next_state))
        return next_state, reward, terminal > 0.5


class PredictionNetwork(nn.Module):
    """Neural network to predict the policy and value."""
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(256, 10)
        self.value = nn.Linear(256, 1)

    def forward(self, state):
        policy_logits = self.policy(state)
        value_estimate = self.value(state)
        return policy_logits.squeeze(1), value_estimate.squeeze(1)

class Node:
    """A node in the MCTS tree."""
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def expand(self, action_priors):
        for i, prior in enumerate(action_priors.tolist()):
            if i not in self.children:
                self.children[i] = Node(prior)

    def select(self, total_visits):
        c1 = 1.25
        c2 = 19652
        log_denom = math.log((total_visits + c2 + 1) / c2)
        best_action = max(self.children.items(),
                          key=lambda item: item[1].value() + item[1].prior * 
                          (math.sqrt(total_visits) / (1 + item[1].visit_count)) * 
                          (c1 + log_denom))
        return best_action[0]

def mcts(root, model, state, num_simulations, buffer):
    """Perform the Monte Carlo Tree Search algorithm."""
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        initial_state = state.clone()

        while node.children:
            total_visits = sum(child.visit_count for child in node.children.values())
            action = node.select(total_visits)
            next_state, reward, is_terminal = simulate(model, state, action)
            buffer.add((state.clone(), action, reward, next_state.clone(), is_terminal))
            state = next_state

            if is_terminal:
                node = node.children.get(action)
                if node is None:
                    node = Node(0.0)
                    root.children[action] = node
                search_path.append(node)
                break

            node = node.children.get(action)
            if node is None:
                policy, _ = model['prediction'](state)
                node = Node(0.0)
                node.expand(policy.squeeze(0))
                root.children[action] = node
            search_path.append(node)

        state = initial_state

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += reward

    # Return the action with the highest visit count
    return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

def simulate(model, state, action):
    """Simulate the next state and reward based on the current state and action."""
    action_tensor = torch.tensor([action], dtype=torch.long)
    return model['dynamics'](state, action_tensor)
