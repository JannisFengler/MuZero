import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import RepresentationNetwork, DynamicsNetwork, PredictionNetwork, ReplayBuffer, Node, mcts, simulate
import gym
from gym.wrappers import AtariPreprocessing
import numpy as np

def train(model, buffer, optimizer, batch_size=64, writer=None, global_step=None):
    if buffer.size() < batch_size:
        print("Buffer size too small for sampling:", buffer.size())
        return

    experiences = buffer.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = zip(*experiences)

    batch_state = torch.stack(batch_state).detach()
    batch_next_state = torch.stack(batch_next_state).detach()
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).detach()
    batch_action = torch.tensor(batch_action, dtype=torch.long).detach()
    batch_terminal = torch.tensor(batch_terminal, dtype=torch.float32).detach()

    current_policy, current_value = model['prediction'](batch_state)
    _, next_value = model['prediction'](batch_next_state)

    expected_value = batch_reward + (1 - batch_terminal) * 0.99 * next_value.squeeze(-1)
    value_loss = F.mse_loss(current_value.squeeze(-1), expected_value.detach())

    _, predicted_reward, _ = model['dynamics'](batch_state, batch_action)
    reward_loss = F.mse_loss(predicted_reward.squeeze(-1), batch_reward)

    policy_probs = F.softmax(current_policy, dim=1)
    log_prob = torch.log(policy_probs.gather(1, batch_action.unsqueeze(1)) + 1e-8)
    policy_loss = -log_prob.mean()

    loss = value_loss + reward_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if writer and global_step is not None:
        writer.add_scalar('Loss/Total', loss.item(), global_step)
        writer.add_scalar('Loss/Value', value_loss.item(), global_step)
        writer.add_scalar('Loss/Reward', reward_loss.item(), global_step)
        writer.add_scalar('Loss/Policy', policy_loss.item(), global_step)

    print(f"Step {global_step}: Training Loss: {loss.item()} | Value Loss: {value_loss.item()} | Reward Loss: {reward_loss.item()} | Policy Loss: {policy_loss.item()}")

def main():
    env_name = 'PongNoFrameskip-v4'  # Use the 'NoFrameskip' version of the environment
    env = gym.make(env_name)
    env = AtariPreprocessing(env)  # Preprocess Atari frames

    representation = RepresentationNetwork()
    dynamics = DynamicsNetwork()
    prediction = PredictionNetwork()
    buffer = ReplayBuffer()

    model = nn.ModuleDict({
        'representation': representation,
        'dynamics': dynamics,
        'prediction': prediction
    })

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    num_episodes = 10000
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Correct handling of the state
        encoded_state = representation(torch.from_numpy(state).unsqueeze(0).float())
        root = Node(0.0)
        root.expand(torch.rand(env.action_space.n))

        done = False
        score = 0  # Initialize the score for the episode
        while not done:
            action = mcts(root, model, encoded_state, 50, buffer)
            output = env.step(action)  # Perform an environment step

            if isinstance(output, tuple) and len(output) == 5:
                next_state, reward, done, truncated, info = output  # Unpacking the step output
                done = done or truncated  # Consider the episode as done if either done or truncated is True
                score += reward  # Update the score
            else:
                raise ValueError("Unexpected number of return values from environment step function.")

            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            next_encoded_state = representation(torch.from_numpy(next_state).unsqueeze(0).float())
            buffer.add((encoded_state.clone(), action, reward, next_encoded_state.clone(), done))
            encoded_state = next_encoded_state

            if done:
                break

        # Log the score for the episode
        writer.add_scalar('Score', score, episode)

        train(model, buffer, optimizer, writer=writer, global_step=episode)

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
