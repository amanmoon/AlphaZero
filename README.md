<div align="center">
  
# General AlphaZero

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

*A generalized, high-performance reinforcement learning agent based on DeepMind's AlphaZero algorithm.*

---
</div>

## Overview

This repository provides a clean, modular, and highly parallelized implementation of the **AlphaZero** algorithm. Originally introduced by DeepMind in their groundbreaking paper *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*, this implementation represents a generalized game-playing agent capable of mastering symmetric zero-sum games solely through self-play and reinforcement learning.

By combining **Monte Carlo Tree Search (MCTS)** with a deep neural network that evaluates board states and policy distributions, this agent iteratively improves without any human data or domain-specific heuristics.

## Key Features

- ** Generalized Self-Play:** The agent plays against itself, dynamically generating high-quality training data.
- ** Parallelized Execution:** Leverages multiprocessing for tree search and self-play (`Alpha_MCTS_Parallel.py`, `Alpha_Zero_Parallel.py`), massively accelerating data generation.
- ** Multi-Game Support:** Pluggable game environment architecture. Easily extensible to various board games (e.g., TicTacToe, ConnectFour).
- ** Intuitive Web Interface:** Includes a fully-featured **Streamlit Web UI** for seamless interaction, configuration, and game monitoring.
- ** Automated Checkpointing:** Built-in utilities (`save_games.py`, `train_from_saved_games.py`) for caching game histories and resuming training runs seamlessly.
- ** Competitive Arena:** Pit different neural network models against each other in `Arena.py` to evaluate ELO and true model performance.

## Foundational Research

The implementation is heavily inspired by the following foundational research papers:
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## Installation

### Prerequisites
- **Python:** Version `3.12` or higher is strictly required for optimal performance.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amanmoon/general_alpha_zero.git
   cd general_alpha_zero
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create a virtual environment natively with Python 3.12+
   python3 -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### Run the Streamlit Application UI

We provide a beautiful, interactive web interface to interact with the models and environments.

1. Ensure your virtual environment is active and all dependencies are installed.
2. Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open the provided `localhost` URL in your browser to start exploring!

### Train a New Model

1. Open `Train.py`. 
2. Import the desired game class and configure the hyperparameters dictionary (`args`).
3. Execute the training script:
   ```bash
   python3 Train.py
   ```
   *Tip: Use `train_from_saved_games.py` to bootstrap learning from previously serialized MCTS explorations!*

### Play Against the Agent

Want to test your skills against the neural network?

1. Open `Play.py`.
2. Select your desired model checkpoint and configure the MCTS simulation count for the AI's turns.
3. Run the script:
   ```bash
   python3 Play.py
   ```

### Model Arena (Evaluating Checkpoints)

To definitively prove which iteration of your neural network is superior, let them battle:

1. Open `Arena.py` and assign the respective paths to the models you wish to evaluate.
2. Run the tournament:
   ```bash
   python3 Arena.py
   ```

## Repository Architecture

```text
├── Games/                     # Game logic implementations (TicTacToe, Connect4, etc.)
├── Alpha_MCTS.py              # Core Monte Carlo Tree Search logic
├── Alpha_MCTS_Parallel.py     # Asynchronous MCTS implementation
├── Alpha_Zero.py              # Self-play and neural network training loops
├── Alpha_Zero_Parallel.py     # High-performance parallelized self-play
├── Arena.py                   # Model vs Model evaluation environment
├── Play.py                    # Human vs AI interactive script
├── Train.py                   # Entry point for training from scratch
├── app.py                     # Streamlit frontend application
└── requirements.txt           # Python dependency specifications
```

## Contact

If you are a recruiter, researcher, or just an enthusiast who wants to discuss reinforcement learning, AI architecture, or optimal search algorithms, I'd love to connect!

**Email:** [amanmoon099@gmail.com](mailto:amanmoon099@gmail.com)
