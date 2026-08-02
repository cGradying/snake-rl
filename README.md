# snake-rl

Classic snake on a 100x100 grid, built with pygame. Game logic lives in a
pure, rendering-free `SnakeGame` class, wrapped as a Gymnasium environment
and trained with PPO (`stable-baselines3`) to play it.

## Run (manual play)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m snake_rl.main
```

Controls: arrow keys / WASD to steer, `R` to restart after game over, `M` to
toggle AI control, `Esc` to quit.

## AI mode

Press `M` in-game to hand control to the trained policy. A right-side panel
shows the policy's confidence for each of the 4 moves (bar = probability,
brightest bar = the move it's about to make) — a live view into what it's
"thinking" each tick, not just what it does.

Needs the ML extras and a trained model:

```bash
.venv/bin/pip install -r requirements-ml.txt
PYTHONPATH=src .venv/bin/python -m snake_rl.train --timesteps 300000
```

Trains PPO against `SnakeEnv` and saves to `models/ppo_snake.zip`, which
`main.py` loads automatically when you press `M`. Without this step, `M`
still toggles AI mode but the panel explains what's missing instead of
crashing — manual play never requires the ML deps at all.

## Structure

- `src/snake_rl/core.py` — `SnakeGame`: `reset(seed=None)`, `step(direction) -> alive`. No pygame import.
- `src/snake_rl/direction.py` — `Direction` enum with `opposite()` (blocks instant reversal).
- `src/snake_rl/env.py` — `build_observation(game)` (shared by training and the AI-play overlay) and `SnakeEnv(gymnasium.Env)`.
- `src/snake_rl/train.py` — PPO training script, saves `models/ppo_snake.zip`.
- `src/snake_rl/main.py` — pygame window, input, rendering, AI toggle + confidence-bar panel. The only file that imports pygame.
- `tests/test_core.py` — assert-based self-check for the game rules (`python tests/test_core.py`).

## How the observation works

The policy doesn't see the raw 100x100 grid — it sees 12 numbers relative to
the snake's head: 4 danger flags (wall/self immediately up/down/left/right),
4 food-direction flags, 4 one-hot current-heading flags. This keeps training
difficulty independent of grid size, at the cost of the agent having no
long-range awareness of its own body — it can still trap itself. Reward is
+10 food / -10 death / -0.01 per step / ±0.1 for moving toward vs. away from
food (the distance-shaping term is what makes a 100x100 grid tractable to
learn on at all — pure sparse +10/-10 reward is far too rare a signal at
this scale).

"Mastering" the game is an ongoing tuning problem, not a one-shot solve —
longer training, and likely a richer observation (e.g. lookahead further
than one cell) to get past the self-trapping failure mode as the snake gets
long.
