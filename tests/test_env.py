import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.core import SnakeGame
from snake_rl.direction import Direction
from snake_rl.env import ACTIONS, HumanFeedbackWrapper, SnakeEnv, build_observation

# observation vector layout: [danger_up, danger_down, danger_left, danger_right,
#                              food_up, food_down, food_left, food_right,
#                              heading_up, heading_down, heading_left, heading_right]


def test_observation_danger_flags_wall_self_obstacle():
    game = SnakeGame(grid_size=10, seed=1)
    game.snake = deque([(0, 5), (1, 5), (2, 5)])  # head against the left wall
    game.obstacles = {(0, 4)}  # directly above the head
    obs = build_observation(game)
    assert obs[0] == 1.0, "obstacle above should read as danger"
    assert obs[2] == 1.0, "wall to the left should read as danger"
    assert obs[3] == 1.0, "own neck to the right should read as danger"
    assert obs[1] == 0.0, "clear cell below should not read as danger"


def test_observation_food_direction_flags():
    game = SnakeGame(grid_size=10, seed=1)
    head_x, head_y = game.snake[0]
    game.food = (head_x + 2, head_y - 2)  # up and to the right
    obs = build_observation(game)
    assert obs[4] == 1.0 and obs[5] == 0.0  # food up, not down
    assert obs[6] == 0.0 and obs[7] == 1.0  # not left, food right


def test_observation_heading_one_hot():
    game = SnakeGame(grid_size=10, seed=1)
    game.direction = Direction.UP
    obs = build_observation(game)
    assert list(obs[8:12]) == [1.0, 0.0, 0.0, 0.0]


def test_snake_env_reset_and_step_shapes():
    env = SnakeEnv(grid_size=10)
    obs, info = env.reset(seed=1)
    assert obs.shape == (12,)
    assert info == {}
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "score" in info


def test_snake_env_death_reward_is_minus_ten():
    env = SnakeEnv(grid_size=5)
    env.reset(seed=1)
    env.game.snake = deque([(4, 2), (3, 2), (2, 2)])
    env.game.direction = Direction.RIGHT
    action = ACTIONS.index(Direction.RIGHT)
    _, reward, terminated, _, _ = env.step(action)
    assert terminated
    assert reward == -10.0


def test_snake_env_food_reward_includes_plus_ten():
    env = SnakeEnv(grid_size=10)
    obs, _ = env.reset(seed=1)
    head_x, head_y = env.game.snake[0]
    env.game.food = (head_x + 1, head_y)
    action = ACTIONS.index(Direction.RIGHT)
    _, reward, terminated, _, info = env.step(action)
    assert not terminated
    assert reward > 9.0  # +10 food - 0.01 step cost
    assert info["score"] == 1


def test_human_feedback_wrapper_applies_bonus():
    env = SnakeEnv(grid_size=10)
    obs, _ = env.reset(seed=1)

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(json.dumps({"obs": [float(v) for v in obs], "action": 0, "reward": -1.0, "source": "manual"}) + "\n")

        wrapped = HumanFeedbackWrapper(SnakeEnv(grid_size=10), path, bonus_scale=5.0)
        wobs, _ = wrapped.reset(seed=1)
        assert (wobs == obs).all()

        plain_env = SnakeEnv(grid_size=10)
        plain_env.reset(seed=1)
        _, plain_reward, *_ = plain_env.step(0)

        _, bonus_reward, *_ = wrapped.step(0)
        assert abs(bonus_reward - (plain_reward - 5.0)) < 1e-6
    finally:
        os.unlink(path)


def test_human_feedback_manual_overrides_auto_on_same_key():
    env = SnakeEnv(grid_size=10)
    obs, _ = env.reset(seed=1)

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        entries = [
            {"obs": [float(v) for v in obs], "action": 0, "reward": -1.0, "source": "auto"},
            {"obs": [float(v) for v in obs], "action": 0, "reward": -1.0, "source": "auto"},
            {"obs": [float(v) for v in obs], "action": 0, "reward": 1.0, "source": "manual"},
        ]
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        wrapped = HumanFeedbackWrapper(SnakeEnv(grid_size=10), path, bonus_scale=5.0)
        key = (tuple(obs.tolist()), 0)
        # manual (+1.0) should win outright, not average with the two auto -1.0 entries
        assert wrapped.table[key] == 1.0
    finally:
        os.unlink(path)


def test_gymnasium_check_env():
    from gymnasium.utils.env_checker import check_env

    check_env(SnakeEnv(grid_size=10), skip_render_check=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
