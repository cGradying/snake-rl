import random
from collections import deque
from typing import Optional

from .direction import Direction


class SnakeGame:
    """Pure game logic, no rendering. step() is the seam a future
    Gymnasium env wraps directly."""

    def __init__(self, grid_size: int = 100, seed: Optional[int] = None):
        self.grid_size = grid_size
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
        self._place_food()

    def _place_food(self) -> None:
        occupied = set(self.snake)
        while True:
            cell = (
                self._rng.randrange(self.grid_size),
                self._rng.randrange(self.grid_size),
            )
            if cell not in occupied:
                self.food = cell
                return

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
        blocked = set(self.snake) if growing else set(self.snake) - {self.snake[-1]}
        if new_head in blocked:
            self.alive = False
            return False

        self.snake.appendleft(new_head)
        if growing:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return True
