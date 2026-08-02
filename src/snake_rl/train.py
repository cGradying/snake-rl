import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import HumanFeedbackWrapper, SnakeEnv

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "ppo_snake.zip"
DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parents[2] / "feedback.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--reset", action="store_true", help="start a fresh model instead of continuing the last generation")
    parser.add_argument("--feedback", nargs="?", const=str(DEFAULT_FEEDBACK_PATH), default=None,
                         help="apply logged human +1/-1 feedback as reward shaping (default: feedback.jsonl)")
    parser.add_argument("--checkpoint-every", type=int, default=100_000,
                         help="save an intermediate checkpoint every N timesteps (0 disables)")
    args = parser.parse_args()

    env = SnakeEnv(grid_size=args.grid_size)
    if args.feedback:
        env = HumanFeedbackWrapper(env, args.feedback)

    if MODEL_PATH.exists() and not args.reset:
        print(f"continuing from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("starting a fresh model")
        model = PPO("MlpPolicy", env, verbose=1)

    callback = None
    if args.checkpoint_every > 0:
        checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_snake",
        )

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=args.reset, callback=callback)

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
