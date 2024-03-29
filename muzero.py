import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from collections import namedtuple

# Hyperparameters
epochs = 10000
learning_rate = 0.01
gamma = 0.99

# Define the environment
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done', 'mcts_policy'))

class ReplayBuffer:
    def __init__(self, capacity=3000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, trajectory):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.fc_policy = nn.Linear(hidden_dim, action_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def policy_value_reward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        reward = self.fc_reward(x)
        return policy_logits, value, reward

class Node:
    def __init__(self, prior):
        self.prior = prior
        self.children = {}
        self.value_sum = 0
        self.visits = 0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

def ucb_score(parent, child):
    pb_c = np.log((parent.visits + 1) / parent.visits) + np.log(2)
    pb_c = np.sqrt(pb_c)
    return -child.value + pb_c * child.prior

class MuZero:
    def __init__(self, net, num_simulations=100):
        self.net = net
        self.num_simulations = num_simulations

    def select_action(self, obs):
        root = Node(0)
        state = torch.tensor(obs, dtype=torch.float32)
        mcts_policy = self.run_mcts(root, state.detach(), False)
        action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return action, mcts_policy

    def run_mcts(self, node, state, done):
        if done:
            return 0
        if not node.children:
            policy_logits, value, _ = self.net.policy_value_reward(state.unsqueeze(0))
            policy = F.softmax(policy_logits, dim=-1)
            node.children = {a: Node(p.item()) for a, p in enumerate(policy[0])}
            return value.item()

        action, child = max(node.children.items(), key=lambda item: ucb_score(node, item[1]))
        _, _, reward = self.net.policy_value_reward(state)
        value = reward.item() + self.run_mcts(child, state, done)

        child.value_sum += value
        child.visits += 1
        return value

    def update(self, replay_buffer, batch_size, optimizer):
        if len(replay_buffer) < batch_size:
            return 0, 0, 0
        trajectories = replay_buffer.sample(batch_size)
        for trajectory in trajectories:
            obses, actions, rewards, next_obses, dones, mcts_policies = zip(*trajectory)

            obses = torch.tensor(obses, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            mcts_policies = torch.tensor(mcts_policies, dtype=torch.float32)

            policy_logits, values, predicted_rewards = self.net.policy_value_reward(obses)
            value_loss = F.mse_loss(values.squeeze(), rewards + gamma * torch.roll(rewards, -1))
            policy_loss = F.cross_entropy(policy_logits, actions)
            reward_loss = F.mse_loss(predicted_rewards.squeeze(), rewards)

            loss = value_loss + policy_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return value_loss.item(), policy_loss.item(), reward_loss.item()

def evaluate(muzero, num_episodes=10):
    total_rewards = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = muzero.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    return total_rewards / num_episodes

def train():
    net = Net()
    muzero = MuZero(net)
    replay_buffer = ReplayBuffer()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        obs = env.reset()
        total_reward = 0
        done = False
        trajectory = []
        while not done:
            action, mcts_policy = muzero.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            trajectory.append(Transition(obs, action, reward, next_obs, done, mcts_policy))
            obs = next_obs
        replay_buffer.push(trajectory)
        
        value_loss, policy_loss, reward_loss = muzero.update(replay_buffer, 64, optimizer)
        
        if (epoch + 1) % 500 == 0:
            eval_reward = evaluate(muzero)
            print(f"Epoch {epoch + 1}, Total Reward: {total_reward}, Eval Reward: {eval_reward}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Reward Loss: {reward_loss}")

train()
