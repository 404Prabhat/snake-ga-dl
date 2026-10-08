"""Genetic Algorithm agent: a small MLP whose weights are the genome."""
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.snake_env import SnakeEnv

INPUT = 11
HIDDEN = 16
OUTPUT = 3
GENOME_SIZE = INPUT * HIDDEN + HIDDEN + HIDDEN * OUTPUT + OUTPUT


def forward(genome, state):
    w1 = genome[: INPUT * HIDDEN].reshape(INPUT, HIDDEN)
    b1 = genome[INPUT * HIDDEN : INPUT * HIDDEN + HIDDEN]
    w2 = genome[INPUT * HIDDEN + HIDDEN : INPUT * HIDDEN + HIDDEN + HIDDEN * OUTPUT].reshape(HIDDEN, OUTPUT)
    b2 = genome[INPUT * HIDDEN + HIDDEN + HIDDEN * OUTPUT :]
    h = np.tanh(np.array(state) @ w1 + b1)
    return h @ w2 + b2


def evaluate(genome, games=3, size=10):
    total = 0.0
    for g in range(games):
        env = SnakeEnv(size=size, seed=g)
        state = env.reset()
        done = False
        while not done:
            action = int(np.argmax(forward(genome, state)))
            state, reward, done, info = env.step(action)
        total += info["score"] * 10 + env.steps * 0.01
    return total / games


def train(pop_size=60, generations=80, elite=4, mutation_rate=0.08,
          mutation_scale=0.5, crossover_rate=0.7, seed=42):
    rng = np.random.default_rng(seed)
    pop = [rng.standard_normal(GENOME_SIZE) * 0.5 for _ in range(pop_size)]

    for gen in range(generations):
        fitness = [evaluate(g) for g in pop]
        ranked = sorted(zip(fitness, pop), key=lambda x: x[0], reverse=True)
        best_fit, best = ranked[0]
        print(f"Gen {gen+1:3d} | best fitness {best_fit:8.2f} | avg fitness {np.mean(fitness):8.2f}")

        new_pop = [g.copy() for _, g in ranked[:elite]]
        while len(new_pop) < pop_size:
            def pick():
                a, b = rng.integers(0, len(ranked) // 2, size=2)
                return ranked[a][1] if ranked[a][0] >= ranked[b][0] else ranked[b][1]

            p1, p2 = pick(), pick()
            if rng.random() < crossover_rate:
                cut = rng.integers(1, GENOME_SIZE - 1)
                child = np.concatenate([p1[:cut], p2[cut:]])
            else:
                child = p1.copy()
            noise = rng.random(GENOME_SIZE) < mutation_rate
            child = child + noise * rng.standard_normal(GENOME_SIZE) * mutation_scale
            new_pop.append(child)
        pop = new_pop

    fitness = [evaluate(g) for g in pop]
    best = pop[int(np.argmax(fitness))]
    out = Path(__file__).parent / "best_genome.pkl"
    with open(out, "wb") as f:
        pickle.dump(best, f)
    print(f"Saved best genome to {out}")
    return best


def play(path=None, games=5, render=False):
    path = path or Path(__file__).parent / "best_genome.pkl"
    with open(path, "rb") as f:
        genome = pickle.load(f)
    scores = []
    for g in range(games):
        env = SnakeEnv(seed=1000 + g)
        state, done = env.reset(), False
        while not done:
            action = int(np.argmax(forward(genome, state)))
            state, _, done, info = env.step(action)
            if render:
                print(env.render(), end="\r\n\n")
        scores.append(info["score"])
    print(f"Scores over {games} games: {scores} (avg {np.mean(scores):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "play"])
    args = ap.parse_args()
    if args.mode == "train":
        train()
    else:
        play()
