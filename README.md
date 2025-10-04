# AI-Powered Pac-Man (Deep Q-Learning Agent)
**Project Overview**

This project develops a complete, functional Pac-Man game environment using Pygame and integrates a Deep Q-Network (DQN) agent that learns the optimal policy to navigate the maze, maximize pellet collection, and avoid the ghost. This showcases proficiency in both classical software architecture and advanced Reinforcement Learning (RL) techniques.

# Features & Technologies
**Key Features**
* **Reinforcement Learning (DQN)**: Implements a Deep Q-Network using PyTorch to model the agent's decision-making process based on game state, reward, and Q-values.

* **Experience Replay Buffer**: Utilizes a replay memory buffer (deque) to stabilize and enhance the agent's off-policy learning process.

* **Pathfinding**: Ghost movement is controlled by the A* Search Algorithm to ensure a consistently challenging and rational adversary.

* **Software Architecture**: Built with Object-Oriented Programming (OOP) principles using Pygame to manage game components (Pacman, Ghost, Pellets, Grid).

# Tech Stack
* Language- Python 3.x

* AI/ML/RL- PyTorch, Deep Q-Learning (DQN), Reinforcement Learning

* Game Engine- Pygame

* Algorithms- A* Search (Pathfinding), Linear Layers (nn.Linear)

# Learning Metrics
The agent demonstrates successful policy learning over iterations, proving effective RL implementation.

* Initial Performance (Random)- Collects <5 pellets per episode.

* Learned Policy (After training)- Consistently collects >50 pellets, demonstrating effective maze navigation and avoidance.

# Setup (Requires Pygame & PyTorch)
**Local Installation**
* Clone the repository: git clone [https://github.com/sagarika789/dqn-pacman-game.git](https://github.com/sagarika789/dqn-pacman-game.git)
cd dqn-pacman-game

* Install dependencies: pip install -r requirements.txt

* Execute the game and begin training:

**Note: Training runs in the terminal. The script generates the learning curve plot upon termination.**
python chapacgame.py

Visuals
Results: The script generates the learning_curve.png plot showing the increase in collected pellets over iterations, which confirms the agent's learning success.
