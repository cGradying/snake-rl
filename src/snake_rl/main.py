import sys

import pygame

from .core import SnakeGame
from .direction import Direction

GRID_SIZE = 100
CELL_SIZE = 6
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 15

BG_COLOR = (17, 17, 17)
SNAKE_COLOR = (80, 220, 120)
HEAD_COLOR = (140, 240, 170)
FOOD_COLOR = (220, 80, 80)
TEXT_COLOR = (230, 230, 230)

KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_w: Direction.UP,
    pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT,
    pygame.K_d: Direction.RIGHT,
}


def draw_cell(surface, x, y, color):
    rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Snake 100x100")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    game = SnakeGame(grid_size=GRID_SIZE)
    pending_direction = game.direction

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_DIRECTION:
                    pending_direction = KEY_TO_DIRECTION[event.key]
                elif event.key == pygame.K_r and not game.alive:
                    game.reset()
                    pending_direction = game.direction
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if game.alive:
            game.step(pending_direction)

        screen.fill(BG_COLOR)
        for i, (x, y) in enumerate(game.snake):
            draw_cell(screen, x, y, HEAD_COLOR if i == 0 else SNAKE_COLOR)
        draw_cell(screen, *game.food, FOOD_COLOR)

        score_surf = font.render(f"Score: {game.score}", True, TEXT_COLOR)
        screen.blit(score_surf, (8, 8))

        if not game.alive:
            msg = font.render("Game Over — press R to restart", True, TEXT_COLOR)
            screen.blit(msg, (WINDOW_SIZE // 2 - msg.get_width() // 2, WINDOW_SIZE // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
