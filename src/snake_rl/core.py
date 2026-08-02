import random
from collections import deque
from typing import Optional, Set, Tuple

from .direction import Direction

# One more obstacle appears on the board for every this-many points scored.
OBSTACLE_INTERVAL = 5


class SnakeGame:
    """Pure game logic, no rendering. step() is the seam a future
    Gymnasium env wraps directly."""

    def __init__(self, grid_size: int = 100, seed: Optional[int] = None, obstacles_enabled: bool = True):
        self.grid_size = grid_size
        self.obstacles_enabled = obstacles_enabled
        self._rng = random.Random(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid - 1, mid), (mid - 2, mid)])
        self.direction = Direction.RIGHT
        self.score = 0
        self.alive = True
        self.obstacles: Set[Tuple[int, int]] = set()
        self._place_food()

    def _random_free_cell(self, exclude: Set[Tuple[int, int]]) -> Tuple[int, int]:
        # rejection sampling degrades as the grid fills up; cap attempts and
        # fall back to an exhaustive scan rather than spinning indefinitely
        for _ in range(1000):
            cell = (
                self._rng.randrange(self.grid_size),
                self._rng.randrange(self.grid_size),
            )
            if cell not in exclude:
                return cell

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in exclude:
                    return (x, y)

        raise RuntimeError("no free cell left on the grid")

    def _place_food(self) -> None:
        self.food = self._random_free_cell(set(self.snake) | self.obstacles)

    def _maybe_add_obstacle(self) -> None:
        if not self.obstacles_enabled:
            return
        target_count = self.score // OBSTACLE_INTERVAL
        if len(self.obstacles) < target_count:
            cell = self._random_free_cell(set(self.snake) | self.obstacles | {self.food})
            self.obstacles.add(cell)

    def step(self, direction: Direction) -> bool:
        """Advance one tick. Returns alive (False on collision)."""
        if not self.alive:
            return False

        if direction != self.direction.opposite():
            self.direction = direction

        dx, dy = self.direction.value
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.alive = False
            return False

        growing = new_head == self.food
        # tail cell vacates this tick unless the snake is growing into it
        body_blocked = set(self.snake) if growing else set(self.snake) - {self.snake[-1]}
        if new_head in body_blocked or new_head in self.obstacles:
            self.alive = False
            return False

        self.snake.appendleft(new_head)
        if growing:
            self.score += 1
            self._maybe_add_obstacle()
            self._place_food()
        else:
            self.snake.pop()

        return True
