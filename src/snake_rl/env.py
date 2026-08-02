import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gymnasium only needed for training/AI play, not manual play
    gym = None
    spaces = None

from .core import SnakeGame
from .direction import Direction

# Fixed action order shared by SnakeEnv and main.py's AI overlay.
ACTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def build_observation(game: SnakeGame) -> np.ndarray:
    """12-dim relative observation: danger flags, food direction, current
    heading. Relative (not full-grid) so training difficulty doesn't scale
    with grid_size, and it's the single source both SnakeEnv (training) and
    main.py (AI-play overlay) feed to the policy.
    """
    head_x, head_y = game.snake[0]
    occupied = set(game.snake) - {game.snake[-1]}  # tail vacates this tick, same as step()
    hazards = occupied | game.obstacles

    def blocked(dx: int, dy: int) -> float:
        x, y = head_x + dx, head_y + dy
        if not (0 <= x < game.grid_size and 0 <= y < game.grid_size):
            return 1.0
        return 1.0 if (x, y) in hazards else 0.0

    danger = [
        blocked(0, -1),  # up
        blocked(0, 1),  # down
        blocked(-1, 0),  # left
        blocked(1, 0),  # right
    ]

    food_x, food_y = game.food
    food_dir = [
        1.0 if food_y < head_y else 0.0,  # food up
        1.0 if food_y > head_y else 0.0,  # food down
        1.0 if food_x < head_x else 0.0,  # food left
        1.0 if food_x > head_x else 0.0,  # food right
    ]

    heading = [1.0 if game.direction is d else 0.0 for d in ACTIONS]

    return np.array(danger + food_dir + heading, dtype=np.float32)


if gym is not None:

    class SnakeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, grid_size: int = 100, max_steps: Optional[int] = None):
            super().__init__()
            self.game = SnakeGame(grid_size=grid_size)
            self.max_steps = max_steps or grid_size * grid_size * 2
            self._steps = 0
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
            self.action_space = spaces.Discrete(4)

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            self.game.reset(seed=seed)
            self._steps = 0
            return build_observation(self.game), {}

        def step(self, action: int):
            direction = ACTIONS[action]
            prev_dist = self._food_distance()
            prev_score = self.game.score

            alive = self.game.step(direction)
            self._steps += 1
            terminated = not alive
            truncated = self._steps >= self.max_steps

            if not alive:
                reward = -10.0
            else:
                reward = -0.01  # per-step cost: discourages stalling
                if self.game.score > prev_score:
                    reward += 10.0
                else:
                    new_dist = self._food_distance()
                    reward += 0.1 if new_dist < prev_dist else -0.1

            return build_observation(self.game), reward, terminated, truncated, {"score": self.game.score}

        def _food_distance(self) -> int:
            hx, hy = self.game.snake[0]
            fx, fy = self.game.food
            return abs(hx - fx) + abs(hy - fy)

    class HumanFeedbackWrapper(gym.Wrapper):
        """Applies logged human +1/-1 feedback (from main.py's in-game
        reward/punish keys) as a reward bonus during training.

        Observations are exact binary flag vectors (danger/food-dir/heading),
        so an (obs, action) exact-match lookup is enough -- no fuzzy nearest-
        neighbor matching needed.
        """

        def __init__(self, env, feedback_path: Union[str, Path], bonus_scale: float = 5.0):
            super().__init__(env)
            self.bonus_scale = bonus_scale
            self.table = _load_feedback_table(feedback_path)
            self._last_obs = None

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            return obs, info

        def step(self, action):
            key = (tuple(self._last_obs.tolist()), int(action))
            bonus = self.table.get(key, 0.0) * self.bonus_scale
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._last_obs = obs
            return obs, reward + bonus, terminated, truncated, info


def _load_feedback_table(feedback_path: Union[str, Path]) -> dict:
    """Manual feedback wins over auto on a colliding (obs, action) key: the
    observation is a coarse 12-bit binary vector, so unrelated situations
    collide on the same key often, and a deliberate human judgment call
    shouldn't get averaged away by the blunt auto food/death signal.
    Entries with no "source" predate the auto-feedback feature and were all
    logged by hand, so they default to manual.
    """
    path = Path(feedback_path)
    if not path.exists():
        return {}
    buckets: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (tuple(entry["obs"]), int(entry["action"]))
            source = entry.get("source", "manual")
            buckets.setdefault(key, {"manual": [], "auto": []})[source].append(entry["reward"])
    table = {}
    for key, sources in buckets.items():
        vals = sources["manual"] or sources["auto"]
        table[key] = sum(vals) / len(vals)
    return table
