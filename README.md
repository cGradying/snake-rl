# snake-rl

Snake on a customizable grid, built with pygame. Game logic lives in a
pure, rendering-free `SnakeGame` class, wrapped as a Gymnasium environment
and trained with PPO (`stable-baselines3`) to play it.

## Run (manual play)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m snake_rl.main
```

A setup screen appears first — Left/Right (or A/D) picks the grid size (20 to
150, default 100), Enter starts. Controls in-game: arrow keys / WASD to
steer, `R` to restart after game over, `M` to toggle AI control, `+`/`-` to
reward/punish the AI's last move (AI mode only), `Esc` to quit.

## Challenge scaling

Speed and hazards both ramp with score, no separate difficulty setting:

- **Speed**: tick rate climbs from 12 FPS toward a 45 FPS cap as score rises (`current_fps()` in `main.py`).
- **Obstacles**: a new static obstacle cell appears on the board every 5 points (`OBSTACLE_INTERVAL` in `core.py`), placed clear of the snake and food. Hitting one ends the game, same as a wall.

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
crashing — manual play never requires the ML deps at all. The model
generalizes across grid sizes (the observation is relative to the snake's
head, not the raw grid) so one trained model works at any setup-screen size.

## Training your own generations

`train.py` continues from `models/ppo_snake.zip` by default, so running it
again is a new generation built on the last one — no flag needed:

```bash
PYTHONPATH=src .venv/bin/python -m snake_rl.train --timesteps 300000
```

Pass `--reset` to throw away the current model and start fresh instead.

## Reward / punish the AI yourself

While AI mode is on, press `+` to reward or `-` to punish the move the AI
just made. Each press appends `{obs, action, reward}` to `feedback.jsonl` at
the repo root — a toast on the panel confirms it logged.

To fold that feedback into the next training run:

```bash
PYTHONPATH=src .venv/bin/python -m snake_rl.train --timesteps 300000 --feedback
```

`HumanFeedbackWrapper` (in `env.py`) looks up each `(observation, action)`
pair against what you logged and adds it as a reward bonus during training.
Observations are exact binary flag vectors, so this is an exact-match
lookup, not a fuzzy one — feedback only affects situations that recur
exactly. `--feedback path/to/file.jsonl` points at a different log if you
want to keep separate feedback sets.

## Structure

- `src/snake_rl/core.py` — `SnakeGame`: `reset(seed=None)`, `step(direction) -> alive`. Obstacles + progressive difficulty live here. No pygame import.
- `src/snake_rl/direction.py` — `Direction` enum with `opposite()` (blocks instant reversal).
- `src/snake_rl/env.py` — `build_observation(game)` (shared by training and the AI-play overlay), `SnakeEnv(gymnasium.Env)`, `HumanFeedbackWrapper`.
- `src/snake_rl/train.py` — PPO training script; continues from the last generation by default, saves `models/ppo_snake.zip`.
- `src/snake_rl/main.py` — pygame setup screen, window, input, rendering, AI toggle, confidence-bar panel, reward/punish logging. The only file that imports pygame.
- `tests/test_core.py` — assert-based self-check for the game rules (`python tests/test_core.py`).

## How the observation works

The policy doesn't see the raw grid — it sees 12 numbers relative to
the snake's head: 4 danger flags (wall/self/obstacle immediately
up/down/left/right), 4 food-direction flags, 4 one-hot current-heading
flags. This keeps training
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

---

<div align="center">

[![Author: cGradying](https://img.shields.io/badge/cGradying-AUTHOR-10B981?style=for-the-badge&labelColor=0B1120)](https://github.com/cGradying)

</div>
