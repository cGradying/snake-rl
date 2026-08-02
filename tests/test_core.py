import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.core import SnakeGame
from snake_rl.direction import Direction


def test_move_survives():
    game = SnakeGame(grid_size=10, seed=1)
    head_before = game.snake[0]
    alive = game.step(Direction.RIGHT)
    assert alive and game.alive
    assert game.snake[0] == (head_before[0] + 1, head_before[1])


def test_eating_food_grows_and_scores():
    game = SnakeGame(grid_size=10, seed=1)
    length_before = len(game.snake)
    game.food = (game.snake[0][0] + 1, game.snake[0][1])
    game.step(Direction.RIGHT)
    assert game.score == 1
    assert len(game.snake) == length_before + 1


def test_wall_collision_ends_game():
    game = SnakeGame(grid_size=5, seed=1)
    game.snake = deque([(4, 2), (3, 2), (2, 2)])
    game.direction = Direction.RIGHT
    alive = game.step(Direction.RIGHT)
    assert not alive and not game.alive


def test_self_collision_ends_game():
    game = SnakeGame(grid_size=10, seed=1)
    # head, neck, body, tail -- moving DOWN drives the head into the neck
    game.snake = deque([(5, 5), (5, 6), (6, 6), (6, 5)])
    game.direction = Direction.LEFT
    alive = game.step(Direction.DOWN)
    assert not alive and not game.alive


def test_cannot_reverse_into_self():
    game = SnakeGame(grid_size=10, seed=1)
    game.direction = Direction.RIGHT
    # request the opposite direction; game should keep moving RIGHT instead
    alive = game.step(Direction.LEFT)
    assert alive
    assert game.direction == Direction.RIGHT


def test_non_growing_move_does_not_falsely_collide():
    # regression check: moving forward normally must not treat the
    # about-to-vacate tail cell as a collision
    game = SnakeGame(grid_size=10, seed=1)
    game.food = (0, 0)  # keep food away so this move doesn't grow
    for _ in range(3):
        assert game.step(Direction.RIGHT)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
