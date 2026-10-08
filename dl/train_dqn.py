"""Deep Q-Network agent for Snake (PyTorch)."""
import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.snake_env import SnakeEnv


class DQN(nn.Module):
    def __init__(self, n_in=11, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train(episodes=400, gamma=0.9, lr=1e-3, batch=64, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SnakeEnv(seed=seed)
    policy = DQN()
    target = DQN()
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=lr)
    memory = deque(maxlen=50000)
    eps, eps_min, eps_decay = 1.0, 0.02, 0.995
    recent_scores = deque(maxlen=20)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done, total_reward = False, 0.0
        while not done:
            if random.random() < eps:
                action = random.randrange(3)
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor(state, dtype=torch.float32)).argmax())
            nstate, reward, done, info = env.step(action)
            memory.append((state, action, reward, nstate, float(done)))
            state = nstate
            total_reward += reward

            if len(memory) >= batch:
                batch_data = random.sample(memory, batch)
                s, a, r, s2, d = map(np.array, zip(*batch_data))
                s = torch.tensor(s, dtype=torch.float32)
                s2 = torch.tensor(s2, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.long)
                r = torch.tensor(r, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)

                q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(s2).max(1).values
                q_target = r + gamma * q_next * (1 - d)
                loss = nn.functional.mse_loss(q, q_target)
                opt.zero_grad()
                loss.backward()
                opt.step()

        eps = max(eps_min, eps * eps_decay)
        recent_scores.append(info["score"])
        if ep % 10 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"Ep {ep:4d} | eps {eps:.3f} | last score {info['score']:2d} | "
                  f"avg20 {np.mean(recent_scores):.2f} | reward {total_reward:.1f}")

    out = Path(__file__).parent / "dqn.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved model to {out}")


def play(games=5, render=False):
    env = SnakeEnv()
    model = DQN()
    model.load_state_dict(torch.load(Path(__file__).parent / "dqn.pt", weights_only=True))
    model.eval()
    scores = []
    for g in range(games):
        state, done = env.reset(), False
        while not done:
            with torch.no_grad():
                action = int(model(torch.tensor(state, dtype=torch.float32)).argmax())
            state, _, done, info = env.step(action)
            if render:
                print(env.render(), end="\r\n\n")
        scores.append(info["score"])
    print(f"Scores over {games} games: {scores} (avg {np.mean(scores):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "play"])
    ap.add_argument("--episodes", type=int, default=400)
    args = ap.parse_args()
    if args.mode == "train":
        train(episodes=args.episodes)
    else:
        play()
