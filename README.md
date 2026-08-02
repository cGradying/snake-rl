# snake-rl

Classic snake on a 100x100 grid, built with pygame. Game logic lives in a
pure, rendering-free `SnakeGame` class so it can later be wrapped as a
reinforcement-learning environment without touching the core rules.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m snake_rl.main
```

Controls: arrow keys / WASD to steer, `R` to restart after game over, `Esc` to quit.

## Structure

- `src/snake_rl/core.py` — `SnakeGame`: `reset()`, `step(direction) -> alive`. No pygame import.
- `src/snake_rl/direction.py` — `Direction` enum with `opposite()` (blocks instant reversal).
- `src/snake_rl/main.py` — pygame window, input, rendering. The only file that imports pygame.
- `tests/test_core.py` — assert-based self-check for the game rules (`python tests/test_core.py`).

## Future: ML

`SnakeGame.step()` already has the shape a Gymnasium environment needs:

1. Wrap it in a `gymnasium.Env` subclass — observation = grid/snake state, action = one of 4 directions, reward = +1 on food, -1 (or small negative) on death, small per-step penalty to encourage efficiency.
2. Train an agent with `stable-baselines3` (DQN or PPO) against the wrapped env.
3. Swap `main.py`'s keyboard input for the trained policy's action to watch it play, and iterate on reward shaping / observation design (e.g. relative food direction, danger-ahead flags) to get it from "survives" to "masters" the game.

None of this is implemented yet — this repo is the game only.
