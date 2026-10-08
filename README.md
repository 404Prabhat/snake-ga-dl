# Snake AI: Genetic Algorithm vs Deep Learning

Train AI agents to play the classic Snake game using two approaches:

1. **Genetic Algorithm (GA)** — a small MLP brain whose weights evolve through selection, crossover, and mutation (pure NumPy).
2. **Deep Q-Network (DQN)** — a PyTorch neural net trained with experience replay and a target network.

Both agents share the same game environment (`env/snake_env.py`) and use the same 11-feature state
(danger ahead/right/left, current direction, food location).

## Project structure

```
├── env/snake_env.py     # Snake game environment
├── ga/train_ga.py       # Genetic Algorithm trainer + player
├── dl/train_dqn.py      # DQN trainer + player
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Genetic Algorithm:
```bash
python ga/train_ga.py train   # evolve agents, saves ga/best_genome.pkl
python ga/train_ga.py play    # watch the best genome play
```

Deep Learning (DQN):
```bash
python dl/train_dqn.py train --episodes 400   # saves dl/dqn.pt
python dl/train_dqn.py play                   # watch the trained model play
```

## How it works

- **GA**: population of 60 genomes (MLP weight vectors), fitness = food eaten + survival bonus,
  top 4 elites kept, tournament selection, 70% crossover, Gaussian mutation.
- **DQN**: 11→128→128→3 MLP, epsilon-greedy exploration, replay buffer of 50k,
  target network synced every 10 episodes, Adam optimizer.

## License

MIT
