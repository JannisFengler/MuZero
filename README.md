# MuZero Implementation for Atari Games

This is a BasmentAGI project.

This repository contains a Python implementation of the MuZero algorithm for Atari games. The implementation is based on the paper *"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"* by Schrittwieser et al. (2020).

## Project Overview

MuZero is a reinforcement learning algorithm developed by DeepMind that can learn to master games without prior knowledge of their rules. This implementation focuses on applying MuZero to Atari games using the ALE (Arcade Learning Environment) interface. It provides a framework for further research and experimentation.

This implementation is designed to be simple, hackable, and consists of only two main files, making it easy to understand, modify, and extend.

## Setup

### Prerequisites
- Python 3.8 or higher (required by ale-py)
- CUDA-compatible GPU (recommended for faster training)
- ALE (Arcade Learning Environment) for Atari games

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JannisFengler/MuZero.git
   cd MuZero
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy gymnasium tensorboard psutil ale-py "gymnasium[atari,accept-rom-license]"
   ```
   This installs:
   * `gymnasium` and Atari extras, providing Atari environments
   * `ale-py`, the official Python interface to the Atari Learning Environment
   * `torch` for neural network acceleration
   * `tensorboard` for monitoring training progress
   * `psutil` for resource utilization metrics

3. Register the ALE environments with Gymnasium:
   ```python
   import gymnasium as gym
   import ale_py
   gym.register_envs(ale_py)
   ```
   This step is done in the code (see `train.py`), so no additional actions are required on your part.

## Usage

1. Configure the environment:
   Open `train.py` and set the desired Atari game environment in `config.env_name`. For example:
   ```python
   config.env_name = "ALE/Pong-v5"
   ```
   Make sure that the chosen environment (like `"ALE/Pong-v5"`) is available and supported by `ale-py`.

2. Start the training process:
   ```bash
   python train.py
   ```

3. Monitor the training progress using TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```
   Open the displayed link in your browser to view training curves and other metrics.

## Project Structure

The project consists of only two main files for core functionality:
* `train.py`: The main script that sets up the environment, runs MuZero training (self-play, optimization), and orchestrates the learning loop.
* `model.py`: Contains the neural network architectures, the MuZero-specific functions (representation, dynamics, prediction), and the MCTS implementation.

This minimal structure allows for easy understanding and modification for research purposes.

## Notes

* Training MuZero on Atari is computationally intensive. A GPU is highly recommended to achieve reasonable training times.
* Reinforcement learning results can be stochastic. Multiple runs and hyperparameter tuning may be necessary to achieve the best performance.
* The hyperparameters in `MuZeroConfig` can be adjusted to improve performance on specific games.
* The code uses a custom frame stacking wrapper (`CustomFrameStack`) to stack multiple frames, enabling the model to understand temporal dynamics of the Atari environment.

## Future Work

* Implement parallel or distributed self-play to speed up data generation
* Extend support to a wider variety of Atari games or other ALE-supported ROMs
* Enhance logging, debugging, and visualization for better insights into the training process
* Add a comprehensive test suite for unit testing components of the implementation

Thank you for your interest in this research project. We hope this codebase serves as a starting point for your explorations into MuZero and Atari game mastering.
