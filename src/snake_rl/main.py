import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from .core import SnakeGame
from .direction import Direction
from .env import ACTIONS, build_observation

PANEL_WIDTH = 240
MAX_PLAY_PX = 720  # play-area pixel budget; cell size derives from grid_size to fit this

DEFAULT_GRID_SIZE = 100
GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 150
GRID_SIZE_STEP = 10

# speed ramps with score: harder as you go, capped so it stays playable
FPS_BASE = 12
FPS_MAX = 45
FPS_PER_POINT = 0.6

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "ppo_snake.zip"
FEEDBACK_PATH = Path(__file__).resolve().parents[2] / "feedback.jsonl"

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
REWARD_COLOR = (110, 231, 183)  # emerald_pale
PUNISH_COLOR = (248, 113, 113)  # red_light

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

REWARD_KEYS = {pygame.K_EQUALS, pygame.K_KP_PLUS}
PUNISH_KEYS = {pygame.K_MINUS, pygame.K_KP_MINUS}

ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT"]


@dataclass
class Layout:
    grid_size: int
    cell_size: int

    @property
    def grid_pixels(self) -> int:
        return self.grid_size * self.cell_size

    @property
    def window_w(self) -> int:
        return self.grid_pixels + PANEL_WIDTH

    @property
    def window_h(self) -> int:
        return self.grid_pixels


def compute_cell_size(grid_size: int) -> int:
    return max(1, MAX_PLAY_PX // grid_size)


class Agent:
    """Lazily loads the trained PPO model so manual play never needs
    stable-baselines3/torch installed."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.error = None

    def load(self) -> None:
        if self.model is not None:
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
        self.error = None

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        probs = self.model.policy.get_distribution(obs_tensor).distribution.probs[0]
        return int(action), probs.detach().cpu().numpy()


def log_feedback(obs, action: int, reward: float, source: str) -> None:
    entry = {"obs": [float(v) for v in obs], "action": int(action), "reward": reward, "source": source}
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def current_fps(score: int) -> float:
    return min(FPS_MAX, FPS_BASE + score * FPS_PER_POINT)


def draw_cell(surface, x, y, cell_size, color):
    rect = (x * cell_size, y * cell_size, cell_size, cell_size)
    pygame.draw.rect(surface, color, rect)


def draw_panel(surface, font, small_font, layout, ai_mode, agent, action_probs, chosen_action, game, toast):
    panel_rect = (layout.grid_pixels, 0, PANEL_WIDTH, layout.window_h)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect)

    x = layout.grid_pixels + 16
    y = 16
    mode_text = "AI: ON" if ai_mode else "AI: OFF"
    surface.blit(font.render(mode_text, True, TEXT_COLOR), (x, y))
    y += 34
    surface.blit(small_font.render("M: toggle AI", True, DIM_TEXT_COLOR), (x, y))
    y += 24
    surface.blit(small_font.render("+/-: reward/punish its last move", True, DIM_TEXT_COLOR), (x, y))
    y += 24
    surface.blit(small_font.render("auto: +1 on food, -1 on death", True, DIM_TEXT_COLOR), (x, y))
    y += 34

    speed_line = f"speed {current_fps(game.score):.0f} fps · {len(game.obstacles)} obstacles"
    surface.blit(small_font.render(speed_line, True, DIM_TEXT_COLOR), (x, y))
    y += 34

    if toast:
        text, color, _ = toast
        surface.blit(small_font.render(text, True, color), (x, y))
    y += 30

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


def settings_menu(clock, font, small_font) -> int:
    """Pre-game screen: pick the grid size ('the box'). Returns grid_size."""
    menu_w, menu_h = 480, 260
    screen = pygame.display.set_mode((menu_w, menu_h))
    pygame.display.set_caption("Snake RL — Setup")
    grid_size = DEFAULT_GRID_SIZE

    choosing = True
    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    grid_size = max(GRID_SIZE_MIN, grid_size - GRID_SIZE_STEP)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    grid_size = min(GRID_SIZE_MAX, grid_size + GRID_SIZE_STEP)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    choosing = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        screen.fill(BG_COLOR)
        title = font.render("Snake RL — Setup", True, TEXT_COLOR)
        screen.blit(title, (menu_w // 2 - title.get_width() // 2, 40))
        label = font.render(f"Grid size: {grid_size} x {grid_size}", True, HEAD_COLOR)
        screen.blit(label, (menu_w // 2 - label.get_width() // 2, 110))
        hint = small_font.render("Left/Right (or A/D) to adjust · Enter to start", True, DIM_TEXT_COLOR)
        screen.blit(hint, (menu_w // 2 - hint.get_width() // 2, 160))
        hint2 = small_font.render(f"range {GRID_SIZE_MIN}-{GRID_SIZE_MAX}, step {GRID_SIZE_STEP}", True, DIM_TEXT_COLOR)
        screen.blit(hint2, (menu_w // 2 - hint2.get_width() // 2, 188))
        pygame.display.flip()
        clock.tick(30)

    return grid_size


def run() -> None:
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 22)

    grid_size = settings_menu(clock, font, small_font)
    layout = Layout(grid_size=grid_size, cell_size=compute_cell_size(grid_size))
    screen = pygame.display.set_mode((layout.window_w, layout.window_h))
    pygame.display.set_caption(f"Snake RL — {grid_size}x{grid_size}")

    game = SnakeGame(grid_size=grid_size)
    pending_direction = game.direction
    ai_mode = False
    agent = Agent(MODEL_PATH)
    action_probs = None
    chosen_action = None
    last_obs = None
    toast = None  # (text, color, expires_at)

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
                elif ai_mode and last_obs is not None and event.key in REWARD_KEYS:
                    log_feedback(last_obs, chosen_action, 1.0, source="manual")
                    toast = ("+1 logged", REWARD_COLOR, time.time() + 1.5)
                elif ai_mode and last_obs is not None and event.key in PUNISH_KEYS:
                    log_feedback(last_obs, chosen_action, -1.0, source="manual")
                    toast = ("-1 logged", PUNISH_COLOR, time.time() + 1.5)

        if ai_mode and agent.model is not None and game.alive:
            obs = build_observation(game)
            chosen_action, action_probs = agent.predict(obs)
            pending_direction = ACTIONS[chosen_action]
            last_obs = obs

        prev_score = game.score
        prev_alive = game.alive

        if game.alive:
            game.step(pending_direction)

        if ai_mode and agent.model is not None and last_obs is not None:
            if game.score > prev_score:
                log_feedback(last_obs, chosen_action, 1.0, source="auto")
                toast = ("auto +1 · ate food", REWARD_COLOR, time.time() + 1.2)
            elif prev_alive and not game.alive:
                log_feedback(last_obs, chosen_action, -1.0, source="auto")
                toast = ("auto -1 · died", PUNISH_COLOR, time.time() + 1.8)

        if toast and time.time() > toast[2]:
            toast = None

        screen.fill(BG_COLOR)
        for ox, oy in game.obstacles:
            draw_cell(screen, ox, oy, layout.cell_size, OBSTACLE_COLOR)
        for i, (x, y) in enumerate(game.snake):
            draw_cell(screen, x, y, layout.cell_size, HEAD_COLOR if i == 0 else SNAKE_COLOR)
        draw_cell(screen, *game.food, layout.cell_size, FOOD_COLOR)

        score_surf = font.render(f"Score: {game.score}", True, TEXT_COLOR)
        screen.blit(score_surf, (8, 8))

        if not game.alive:
            msg = font.render("Game Over — press R to restart", True, TEXT_COLOR)
            screen.blit(msg, (layout.grid_pixels // 2 - msg.get_width() // 2, layout.window_h // 2))

        draw_panel(screen, font, small_font, layout, ai_mode, agent, action_probs, chosen_action, game, toast)

        pygame.display.flip()
        clock.tick(current_fps(game.score))

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
