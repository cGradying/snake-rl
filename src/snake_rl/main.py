import sys
from pathlib import Path

import pygame

from .core import SnakeGame
from .direction import Direction
from .env import ACTIONS, build_observation

GRID_SIZE = 100
CELL_SIZE = 6
GRID_PIXELS = GRID_SIZE * CELL_SIZE
PANEL_WIDTH = 240
WINDOW_W = GRID_PIXELS + PANEL_WIDTH
WINDOW_H = GRID_PIXELS

# speed ramps with score: harder as you go, capped so it stays playable
FPS_BASE = 12
FPS_MAX = 45
FPS_PER_POINT = 0.6

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "ppo_snake.zip"

# astra moon palette
BG_COLOR = (15, 23, 42)  # bg_bottom
PANEL_BG_COLOR = (17, 26, 46)  # panel
BORDER_COLOR = (30, 41, 59)  # border
SNAKE_COLOR = (16, 185, 129)  # emerald
HEAD_COLOR = (52, 211, 153)  # emerald_light
FOOD_COLOR = (239, 68, 68)  # red
OBSTACLE_COLOR = (125, 141, 161)  # dim
TEXT_COLOR = (201, 209, 217)  # text
DIM_TEXT_COLOR = (125, 141, 161)  # dim
BAR_COLOR = (16, 185, 129)  # emerald
BAR_ACTIVE_COLOR = (110, 231, 183)  # emerald_pale


def current_fps(score: int) -> float:
    return min(FPS_MAX, FPS_BASE + score * FPS_PER_POINT)


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

ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT"]


class Agent:
    """Lazily loads the trained PPO model so manual play never needs
    stable-baselines3/torch installed."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.error = None

    def load(self) -> None:
        if self.model is not None or self.error is not None:
            return
        try:
            from stable_baselines3 import PPO
        except ImportError:
            self.error = "stable-baselines3 not installed — pip install -r requirements-ml.txt"
            return
        if not self.model_path.exists():
            self.error = "no trained model — run: python -m snake_rl.train"
            return
        self.model = PPO.load(self.model_path)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        probs = self.model.policy.get_distribution(obs_tensor).distribution.probs[0]
        return int(action), probs.detach().cpu().numpy()


def draw_cell(surface, x, y, color):
    rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def draw_panel(surface, font, small_font, ai_mode, agent, action_probs, chosen_action, game):
    panel_rect = (GRID_PIXELS, 0, PANEL_WIDTH, WINDOW_H)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect)

    x = GRID_PIXELS + 16
    y = 16
    mode_text = "AI: ON" if ai_mode else "AI: OFF"
    surface.blit(font.render(mode_text, True, TEXT_COLOR), (x, y))
    y += 34
    surface.blit(small_font.render("press M to toggle", True, DIM_TEXT_COLOR), (x, y))
    y += 34

    speed_line = f"speed {current_fps(game.score):.0f} fps · {len(game.obstacles)} obstacles"
    surface.blit(small_font.render(speed_line, True, DIM_TEXT_COLOR), (x, y))
    y += 34

    if not ai_mode:
        surface.blit(small_font.render("manual control", True, DIM_TEXT_COLOR), (x, y))
        return

    if agent.error:
        for i, line in enumerate(_wrap(agent.error, 26)):
            surface.blit(small_font.render(line, True, DIM_TEXT_COLOR), (x, y + i * 20))
        return

    surface.blit(small_font.render("policy confidence", True, DIM_TEXT_COLOR), (x, y))
    y += 28

    bar_max_w = PANEL_WIDTH - 32
    for i, label in enumerate(ACTION_LABELS):
        prob = float(action_probs[i]) if action_probs is not None else 0.0
        active = i == chosen_action
        color = BAR_ACTIVE_COLOR if active else BAR_COLOR

        surface.blit(small_font.render(label, True, TEXT_COLOR), (x, y))
        bar_y = y + 22
        pygame.draw.rect(surface, BORDER_COLOR, (x, bar_y, bar_max_w, 14))
        pygame.draw.rect(surface, color, (x, bar_y, int(bar_max_w * prob), 14))
        pct = small_font.render(f"{prob * 100:.0f}%", True, DIM_TEXT_COLOR)
        surface.blit(pct, (x + bar_max_w - pct.get_width(), y))
        y += 50


def _wrap(text, width):
    words = text.split(" ")
    lines, current = [], ""
    for w in words:
        trial = f"{current} {w}".strip()
        if len(trial) > width:
            lines.append(current)
            current = w
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Snake 100x100")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 22)

    game = SnakeGame(grid_size=GRID_SIZE)
    pending_direction = game.direction
    ai_mode = False
    agent = Agent(MODEL_PATH)
    action_probs = None
    chosen_action = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    ai_mode = not ai_mode
                    if ai_mode:
                        agent.load()
                elif not ai_mode and event.key in KEY_TO_DIRECTION:
                    pending_direction = KEY_TO_DIRECTION[event.key]
                elif event.key == pygame.K_r and not game.alive:
                    game.reset()
                    pending_direction = game.direction
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if ai_mode and agent.model is not None and game.alive:
            obs = build_observation(game)
            chosen_action, action_probs = agent.predict(obs)
            pending_direction = ACTIONS[chosen_action]

        if game.alive:
            game.step(pending_direction)

        screen.fill(BG_COLOR)
        for ox, oy in game.obstacles:
            draw_cell(screen, ox, oy, OBSTACLE_COLOR)
        for i, (x, y) in enumerate(game.snake):
            draw_cell(screen, x, y, HEAD_COLOR if i == 0 else SNAKE_COLOR)
        draw_cell(screen, *game.food, FOOD_COLOR)

        score_surf = font.render(f"Score: {game.score}", True, TEXT_COLOR)
        screen.blit(score_surf, (8, 8))

        if not game.alive:
            msg = font.render("Game Over — press R to restart", True, TEXT_COLOR)
            screen.blit(msg, (GRID_PIXELS // 2 - msg.get_width() // 2, WINDOW_H // 2))

        draw_panel(screen, font, small_font, ai_mode, agent, action_probs, chosen_action, game)

        pygame.display.flip()
        clock.tick(current_fps(game.score))

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
