# General AlphaZero
## Overview

This project implements a game-playing agent based on the AlphaZero algorithm, inspired by the DeepMind paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm". The agent is designed to play games with itself and learn through reinforcement learning. The agent generates data based on modified MCTS tree search algorithm and trains itself on the generated data.

## Research Paper
- [Google DeepMind](https://deepmind.google/)
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)

## Features

- Self-play: The agent plays games against itself to generate training data.
- Neural Network: Implements a neural network for the game-playing policy and value estimation.
- Reinforcement Learning: Utilizes reinforcement learning techniques for training the agent.
- Game Environment: Support for multiple game environments (TicTacToe, ConnectFour, etc.).
- Data Creation: Create and Save game data for model to be trained on. 

## Installation
#### Prerequisites:
- Make sure you are running python version of 3.8 - 3.11.

#### Install Dependencies:
1. Clone the repository: `git clone https://github.com/amanmoon/general_alpha_zero.git`
2. Navigate to the project directory: `cd general_alpha_zero`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

  ###  Activate Virtual Environment:
  #### MacOS / Linux
  1. Navigate to the project directory: `cd general_alpha_zero`
  2. Create Virtual Environment: `python3.10 -m venv <venv Name>`
  3. Activate Virtual Environment:`source <venv Name>/bin/activate`
  4. Deactivate Virtual Environment: `deactivate`
  ### Train the Model:
  1. Choose Game you wish to Train Model for and import right Classes inside the Train.py file.
  2. Choose appropriate hyperparameters in args.
  3. To run the Train Script: `python3 Train.py`
  ### Playing against a Model:
  1. Choose player you wish to play as and modify search parameters in args inside the Play.py file.
  2. Import Correct Model.
  3. To run the Play Script: `python3 Play.py`
  ### Bet two Models:
  1. Choose Models you wish to bet against each other inside the Arena file.
  2. To run the Arena Script: `python3 Arena.py`
  
## Contact
For questions or suggestions, feel free to reach out at amanmoon099@gmail.com.
