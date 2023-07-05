import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.autograd.set_detect_anomaly(True)

class MuZeroNetwork(nn.Module):
    def __init__(self, input_shape, action_space):
        super(MuZeroNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_space = action_space
        self.state_space = input_shape[0]
        self.discount_factor = 0.99

        self.representation = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU()
        )

        self.dynamics = nn.Sequential(
            nn.Linear(512 + self.action_space, 512),
            nn.ReLU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(512, self.action_space)
        )

        self.value = nn.Sequential(
            nn.Linear(512, 1)
        )

    def initial_state(self, x):
        return self.representation(x)

    def recurrent_state(self, s, a):
        a = a.to(device).long()
        a = a.view(-1, 1)
        one_hot_action = torch.nn.functional.one_hot(a.squeeze().long(), num_classes=self.action_space).to(device)

        s = s.repeat(one_hot_action.size(0), 1)
        x = torch.cat((s, one_hot_action), dim=1)

        return self.dynamics(x)

    def value_and_reward(self, s):
        value = self.value(s).squeeze()
        policy = self.prediction(s).squeeze()
        return value, policy


    def generate_training_data(self, trajectories):
        training_data = []
        for obs, state, action, reward, done in trajectories:
            target_action = torch.zeros(self.action_space, device=device)
            target_action[action] = 1.0
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_representation = self.representation(obs)

            value, _ = self.value_and_reward(next_state_representation)
            target_value = reward + (1 - done) * self.discount_factor * value

            # Ensure target_value is a tensor with at least one dimension
            target_value = target_value.unsqueeze(0)

            training_data.append((state, target_action, target_value))

        return training_data


    def update(self, training_data, batch_size):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        batches = [
            training_data[i: i + batch_size]
            for i in range(0, len(training_data), batch_size)
        ]

        for batch in batches:
            states, target_actions, target_values = zip(*batch)
            states = torch.stack(states)
            target_actions = torch.stack(target_actions).squeeze(1) # Remove unnecessary dimension
            target_values = torch.cat(target_values).unsqueeze(1)

            optimizer.zero_grad()

            value, policy = self.value_and_reward(states)

            value_loss = nn.MSELoss()(value, target_values)
            policy_loss = nn.BCEWithLogitsLoss()(policy, target_actions)

            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()


env = gym.make("CartPole-v1")

input_shape = (4,)
n_actions = 2
muzero_net = MuZeroNetwork(input_shape, n_actions).to(device)

num_episodes = 5000
num_simulations = 50
batch_size = 64

muzero = muzero_net

for episode in range(num_episodes):
    print(f"Starting Episode {episode}")

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
    state = muzero.initial_state(obs)
    done = False
    episode_reward = 0
    trajectories = []

    while not done:
        value, policy = muzero.value_and_reward(state)
        action = torch.distributions.Categorical(logits=policy).sample().item()

        next_obs, reward, done, _ = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = muzero.initial_state(next_obs)

        trajectories.append((obs, state, action, reward, done))

        obs = next_obs
        state = next_state
        episode_reward += reward

    print(f"Episode {episode}, Reward: {episode_reward}")

    training_data = muzero.generate_training_data(trajectories)
    muzero.update(training_data, batch_size)

    gc.collect()
