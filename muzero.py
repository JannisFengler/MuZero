import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.dynamics_state = nn.Sequential(
            nn.Linear(512 + self.action_space, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.dynamics_reward = nn.Sequential(
            nn.Linear(512 + self.action_space, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space)
        )

        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def initial_state(self, x):
        return self.representation(x)
    
    def next_state(self, s, a):
        a = a.to(device).long()  # Convert action to long tensor
        a = a.view(-1, 1)  # Reshape action tensor
        one_hot_action = torch.nn.functional.one_hot(a.squeeze().long(), num_classes=self.action_space).unsqueeze(0).to(device)

        x = torch.cat((s, one_hot_action), dim=1)
        return self.dynamics_state(x)

    def reward(self, s, a):
        a = a.to(device).long()  # Convert action to long tensor
        a = a.view(-1, 1)  # Reshape action tensor
        one_hot_action = torch.nn.functional.one_hot(a.squeeze().long(), num_classes=self.action_space).unsqueeze(0).to(device)




        x = torch.cat((s, one_hot_action), dim=1)


        return self.dynamics_reward(x)
    
    def policy_value(self, s):
        policy_logits = self.policy(s)  # Use 's' directly
        value = self.value(s)  # Use 's' directly

        return policy_logits, value.squeeze(0)

    def generate_training_data(self, trajectories):
        """
        Converts trajectories into training data.
        """
        training_data = []
        for obs, state, action, reward, done in trajectories:  # Add 'state' to the loop variables
            target_action = torch.zeros(self.action_space, device=device)
            target_action[action] = 1.0
            obs = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)  # Convert 'obs' to a tensor
            next_state_representation = self.representation(obs)  # Use 'obs' instead of 'state'
            target_value = (reward + (1 - done) * self.discount_factor * self.value(next_state_representation)).squeeze()

            training_data.append((state, target_action, target_value))  # Use 'state' instead of 'obs'
        return training_data

    def update(self, training_data, batch_size):
        """
        Update the network using the training data.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Split training data into batches
        batches = [
            training_data[i : i + batch_size]
            for i in range(0, len(training_data), batch_size)
        ]

        for batch in batches:
            states, target_actions, target_values = zip(*batch)
            states = torch.stack(states)
            target_actions = torch.stack(target_actions)
            target_values = torch.stack(target_values).view(-1, 1, 1)


            optimizer.zero_grad()

            # Compute the policy and value predictions
            policy, value = self.policy_value(states)

            # Compute the loss
            action_loss = nn.BCEWithLogitsLoss()(policy.squeeze(1), target_actions)

            value_loss = nn.MSELoss()(value, target_values)
            loss = action_loss + value_loss
            # Backpropagate the loss
            loss.backward()
            # Update the network weights
            optimizer.step()

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network, num_simulations, discount_factor=0.99, exploration_weight=1.0):
        self.network = network     
        self.num_simulations = num_simulations
        self.discount_factor = discount_factor
        self.exploration_weight = exploration_weight

    def run(self, state):
        root = Node(0)

        for _ in range(self.num_simulations):
            node, action_history = self.select(root, state)
            value = self.expand_and_evaluate(node, action_history, state)
            self.backpropagate(node, value, action_history)

        return root  # Return the root object

    def select(self, node, state):
        action_history = []

        while True:
            if not node.expanded():  # Move the if-statement here
                break
            action, node = self.select_child(node)
            action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
            action_tensor = action_tensor.unsqueeze(0)
            state = self.network.next_state(state, action_tensor)  # Pass the 'state' variable to the 'next_state' function
            action_history.append(action)

        return node, action_history


    def select_child(self, node):
        max_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.exploration_weight * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            q = child.value()
            score = q + u

            if score > max_score:
                max_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def legal_actions(self, action_history):
        return list(range(self.network.action_space))

    def expand_and_evaluate(self, node, action_history, state):  # Add state as an input
        next_hidden_state, reward, done = self.apply_action(state, action_history)
        policy_logits, value = self.network.policy_value(next_hidden_state)
        legal_actions = self.legal_actions(action_history)
        policy_logits = policy_logits.view(-1, self.network.action_space)
        policy = {a: np.exp(policy_logits[0, a].cpu().detach().numpy()) for a in legal_actions}

        policy_sum = sum(policy.values())
        for a in legal_actions:
            policy[a] /= policy_sum

        for action, prob in policy.items():
            node.children[action] = Node(prob)

        return value


    def backpropagate(self, node, value, action_history):
        for action in reversed(action_history):
            if action in node.children:
                node.value_sum += value
                node.visit_count += 1
                value *= self.discount_factor
                node = node.children[action]


    def apply_action(self, hidden_state, action_history):  # Add env as an input
        total_reward = 0
        done = False
        for action in action_history:
            action_one_hot = torch.tensor([action], dtype=torch.float32, device=device)
            action_one_hot = action_one_hot.unsqueeze(0)

            hidden_state = self.network.next_state(hidden_state, action_one_hot)  # Update hidden_state
            reward = self.network.reward(hidden_state, action_one_hot)
            total_reward += reward.item()

        return hidden_state, total_reward, done


    def select_action(self, root, temperature=1.0):  # Add the temperature parameter
        visit_counts = np.array(
    [child.visit_count for child in root.children.values()], dtype=np.float32
)

        visit_counts = visit_counts**(1/temperature)  # Apply temperature scaling
        visit_counts /= visit_counts.sum()  # Normalize visit_counts to get probabilities
        best_action = np.random.choice(list(root.children.keys()), p=visit_counts)

        return best_action



env = gym.make("CartPole-v1")

input_shape = (4,)  
n_actions = 2
muzero_net = MuZeroNetwork(input_shape, n_actions).to(device)

num_episodes = 5000
num_simulations = 50
temperature = 1.0
batch_size = 64

# Initialize MuZero network and MCTS
muzero = muzero_net
mcts = MCTS(muzero, num_simulations)  # Add 'env' as an argument


# Training loop
for episode in range(num_episodes):
    print(f"Starting Episode {episode}")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
    print("obs.shape:", obs.shape)
    state = muzero.representation(obs)
    done = False
    episode_reward = 0
    trajectories = []

    while not done:
        root_node = mcts.run(state)
        action = mcts.select_action(root_node, temperature)
        obs, reward, done, _, _ = env.step(action)
        next_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = muzero.representation(next_obs)
        trajectories.append((obs, state, action, reward, done))  # Store both 'obs' and 'state'
        #print(done)
        state = next_state
        episode_reward += reward


    print(f"Episode {episode}, Reward: {episode_reward}")

    # Train the network with collected trajectories
    training_data = muzero.generate_training_data(trajectories)
    muzero.update(training_data, batch_size)

    # Decay the temperature parameter over time (optional)
    temperature *= 0.99
