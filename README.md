# MuZero Implementation for Atari Games

This repository contains a Python implementation of the MuZero algorithm for Atari games. The implementation is based on the paper "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" by Schrittwieser et al.

## Project Overview

MuZero is a reinforcement learning algorithm developed by DeepMind that can learn to master games without prior knowledge of their rules. This implementation focuses on applying MuZero to Atari games, providing a framework for further research and experimentation in this domain.

This implementation is designed to be hackable and consists of only two main files, making it easy to understand, modify, and extend for your research needs.

## Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/JannisFengler/MuZero.git
   cd MuZero
   ```

2. Install the required dependencies:
   ```
   pip install torch numpy gym[atari] tensorboard psutil
   ```

## Usage

1. Configure the environment:
   Open `train.py` and set the desired Atari game in the `config.env_name` variable. The default is set to Pong.

2. Start the training process:
   ```
   python train.py
   ```

3. Monitor the training progress using TensorBoard:
   ```
   tensorboard --logdir=runs
   ```

## Project Structure

The project consists of only two main files:

- `train.py`: Main script that initializes the environment and runs the training loop.
- `model.py`: Contains the neural network architecture and MCTS implementation.

This minimal structure makes the project highly hackable and easy to modify for your specific research needs.

## Notes

- Training MuZero can be computationally intensive. A GPU is highly recommended for reasonable training times.
- Reinforcement learning results can be volatile. Multiple runs may be necessary to achieve consistent performance.
- Hyperparameters in `MuZeroConfig` may need tuning for optimal performance on different games.
- The project's simplicity (two files) allows for easy experimentation and modification.

## Future Work

- Implement parallel self-play for improved training efficiency
- Extend support to a broader range of Atari games
- Enhance logging and visualization of training metrics
- Develop a comprehensive test suite for model components

Thank you for your interest in this research project.
