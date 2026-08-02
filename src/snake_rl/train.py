import argparse
from pathlib import Path

from stable_baselines3 import PPO

from .env import SnakeEnv

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--grid-size", type=int, default=100)
    args = parser.parse_args()

    env = SnakeEnv(grid_size=args.grid_size)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "ppo_snake.zip"
    model.save(out_path)
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
