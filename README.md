# snake-rl

Snake on a configurable grid (pygame), with `SnakeGame` wrapped as a
Gymnasium environment and trained via PPO (`stable-baselines3`). Game logic
is fully decoupled from rendering and from the RL stack: `core.py` has zero
pygame/gymnasium imports, so manual play never requires the ML dependencies.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m snake_rl.main
```

Setup screen: Left/Right or A/D sets `grid_size` in `[20, 150]`, step 10,
default 100. Enter starts.

In-game: arrow keys/WASD steer, `R` restarts after death, `M` toggles AI
control, `+`/`-` reward/punish the AI's last move (AI mode only), `Esc`
quits.

## Architecture

```
core.py   SnakeGame           pure game state machine, no dependencies
              |
env.py    build_observation() SnakeGame -> np.ndarray(12,)  [shared below]
          SnakeEnv             gymnasium.Env wrapping SnakeGame
          HumanFeedbackWrapper  gymnasium.Wrapper adding logged reward bonus
              |            \
train.py  PPO.learn()       main.py  Agent (loads PPO model), pygame loop
```

`build_observation()` is the single source both `SnakeEnv.step()` (training)
and `main.py`'s `Agent.predict()` (AI-play overlay) feed to the policy —
training and inference see identical inputs by construction, not by
convention.

## Module responsibilities

| File | Owns | Depends on |
|---|---|---|
| `core.py` | `SnakeGame`: grid, snake, food, obstacles, collision, scoring | stdlib only |
| `direction.py` | `Direction` enum, `opposite()` | stdlib only |
| `env.py` | `build_observation`, `SnakeEnv`, `HumanFeedbackWrapper` | `numpy`; `gymnasium` optional (guarded import — `env.py` still imports cleanly without it, `SnakeEnv`/`HumanFeedbackWrapper` just won't exist) |
| `train.py` | PPO training CLI, checkpointing | `stable-baselines3` |
| `main.py` | pygame window, input, rendering, `Agent` (lazy model loader) | `pygame`; `stable-baselines3`/`torch` lazily imported only when AI mode is toggled on |

## Observation spec

`build_observation(game) -> np.ndarray, shape (12,), dtype float32, values in {0.0, 1.0}`

Relative to the snake's head, not the raw grid — this is why one trained
model works at any grid size and why training cost doesn't scale with
`grid_size`.

| Index | Field | 1.0 when |
|---|---|---|
| 0 | `danger_up` | wall, own body (excluding tail), or obstacle 1 cell above the head |
| 1 | `danger_down` | same, below |
| 2 | `danger_left` | same, left |
| 3 | `danger_right` | same, right |
| 4 | `food_up` | food's y < head's y |
| 5 | `food_down` | food's y > head's y |
| 6 | `food_left` | food's x < head's x |
| 7 | `food_right` | food's x > head's x |
| 8-11 | `heading_{up,down,left,right}` | one-hot of `game.direction` |

The tail cell is excluded from the danger check because it vacates on a
non-growing move — same rule `SnakeGame.step()` uses for collision.
Consequence of the coarse 1-cell danger radius: the agent has no
multi-step lookahead and can still trap itself in its own body as it gets
long; this is the main ceiling on play quality, not training duration.

## Action / reward spec

`action: Discrete(4)`, mapped via `ACTIONS = [UP, DOWN, LEFT, RIGHT]` (`env.py`).

`SnakeEnv.step(action)` reward, in order of precedence:

| Condition | Reward |
|---|---|
| Move kills the snake | `-10.0` |
| Move eats food | `+10.0 - 0.01` |
| Move strictly decreases Manhattan distance to food | `+0.1 - 0.01` |
| Otherwise | `-0.1 - 0.01` |

The `-0.01` per-step cost discourages stalling. The distance-shaping term
(`±0.1`) is load-bearing: sparse `+10`/`-10` alone is too rare a signal to
learn from at grid sizes in the hundreds of cells.

`truncated = True` at `max_steps = grid_size**2 * 2` (default, overridable
via `SnakeEnv(max_steps=...)`).

## Difficulty scaling

Both formulas live in their respective owning module, not a shared config,
since one is render-loop state and the other is game state:

- **Speed** — `main.py::current_fps(score) = min(45, 12 + score * 0.6)`
- **Obstacles** — `core.py::OBSTACLE_INTERVAL = 5`; `SnakeGame._maybe_add_obstacle()` adds one obstacle whenever `score // 5` exceeds the current obstacle count, placed via rejection sampling (capped at 1000 attempts, then falls back to an exhaustive free-cell scan) excluding the snake, existing obstacles, and food.

## Feedback loop

`main.py::log_feedback(obs, action, reward, source)` appends one JSON line
to `feedback.jsonl` at the repo root:

```json
{"obs": [12 floats], "action": 0-3, "reward": 1.0 | -1.0, "source": "manual" | "auto"}
```

Two writers, both gated on AI mode being on:

- **Manual** (`main.py` event loop) — `+`/`-` keys log the AI's most recently taken `(obs, action)` pair.
- **Auto** (`main.py` main loop, post-step) — `+1` on any score increase, `-1` on the alive→dead transition, same `(obs, action)` pair that caused it.

`env.py::_load_feedback_table(path)` builds the lookup `HumanFeedbackWrapper`
uses at training time: groups entries by exact `(obs, action)` key (viable
because obs is a discrete 12-bit vector, not continuous — no nearest-
neighbor matching needed), buckets by `source`, and **manual entries win
outright over auto on a colliding key** rather than being averaged with
them — a deliberate human judgment call shouldn't get diluted by the
high-frequency, low-nuance auto signal. Entries with no `source` field
predate this schema and default to `manual` (all of `feedback.jsonl`'s
pre-existing entries were hand-logged before the auto-feedback feature
existed, so this default is accurate for them, not just a fallback).

`HumanFeedbackWrapper(env, bonus_scale=5.0)` adds `table.get(key, 0.0) *
bonus_scale` to `SnakeEnv`'s own reward at each step.

## Training

```bash
PYTHONPATH=src .venv/bin/pip install -r requirements-ml.txt
PYTHONPATH=src .venv/bin/python -m snake_rl.train [flags]
```

| Flag | Default | Effect |
|---|---|---|
| `--timesteps` | `300000` | PPO steps this run |
| `--grid-size` | `100` | `SnakeEnv` grid size for this run |
| `--reset` | off | discard `models/ppo_snake.zip` and start a fresh `PPO("MlpPolicy", ...)` instead of continuing it |
| `--feedback [path]` | off | wrap the env in `HumanFeedbackWrapper`; bare flag uses `feedback.jsonl`, or pass an explicit path |
| `--checkpoint-every` | `100000` | write `models/checkpoints/ppo_snake_<step>_steps.zip` at this interval via `CheckpointCallback`; `0` disables |

Without `--reset`, every invocation is a new generation stacked on
`models/ppo_snake.zip` — repeated runs accumulate, they don't restart.
`models/checkpoints/` is gitignored; only the final `models/ppo_snake.zip`
is committed.

`main.py`'s `Agent.load()` re-attempts loading `models/ppo_snake.zip` on
every `M` toggle until it succeeds — no need to restart the game process
after finishing a training run in another terminal.

## Testing

Assert-based, no framework, run directly:

```bash
.venv/bin/python tests/test_core.py       # SnakeGame rules; no ML deps needed
PYTHONPATH=src .venv/bin/python tests/test_env.py  # observation encoding, SnakeEnv reward, HumanFeedbackWrapper precedence; needs requirements-ml.txt
```

---

<div align="center">

[![Author: cGradying](https://img.shields.io/badge/cGradying-AUTHOR-10B981?style=for-the-badge&labelColor=0B1120)](https://github.com/cGradying)

</div>
