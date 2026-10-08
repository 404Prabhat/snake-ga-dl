"""Self-contained Snake game environment shared by GA and DL agents."""
import random
from collections import deque

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


class SnakeEnv:
    def __init__(self, size=10, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        cx, cy = self.size // 2, self.size // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.dir = RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.max_steps = self.size * self.size * 4
        return self.get_state()

    def _spawn_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.snake]
        return self.rng.choice(free) if free else None

    def step(self, action):
        """action: 0=straight,1=right,2=left relative to current direction."""
        order = [UP, RIGHT, DOWN, LEFT]
        idx = order.index(self.dir)
        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4
        self.dir = order[idx]

        hx, hy = self.snake[0]
        dx, dy = DELTAS[self.dir]
        head = (hx + dx, hy + dy)
        self.steps += 1

        reward = -0.01
        done = False

        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size) or head in list(self.snake)[:-1]:
            return self.get_state(), -10.0, True, {"score": self.score}

        self.snake.appendleft(head)
        if head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._spawn_food()
            if self.food is None:
                done = True
        else:
            self.snake.pop()

        if self.steps >= self.max_steps:
            done = True
            reward -= 5.0

        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self):
        order = [UP, RIGHT, DOWN, LEFT]
        idx = order.index(self.dir)
        straight = order[idx]
        right = order[(idx + 1) % 4]
        left = order[(idx - 1) % 4]

        def danger(d):
            pt = self._next_in(d)
            return 1.0 if (not (0 <= pt[0] < self.size and 0 <= pt[1] < self.size) or pt in self.snake) else 0.0

        fx, fy = self.food
        hx, hy = self.snake[0]
        state = [
            danger(straight),
            danger(right),
            danger(left),
            float(self.dir == LEFT),
            float(self.dir == RIGHT),
            float(self.dir == UP),
            float(self.dir == DOWN),
            float(fx < hx),
            float(fx > hx),
            float(fy < hy),
            float(fy > hy),
        ]
        return state

    def _next_in(self, d):
        hx, hy = self.snake[0]
        dx, dy = DELTAS[d]
        return (hx + dx, hy + dy)

    def render(self):
        grid = [["."] * self.size for _ in range(self.size)]
        for i, (x, y) in enumerate(self.snake):
            grid[y][x] = "O" if i == 0 else "o"
        if self.food:
            grid[self.food[1]][self.food[0]] = "F"
        return "\n".join(" ".join(row) for row in grid)
