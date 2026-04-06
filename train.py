from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import BipedEnv
from datetime import datetime
import pandas as pd
import subprocess
import os

def main():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def make_env(i):
        def _init():
            env = BipedEnv()
            env = Monitor(env, filename=f"logs/training_log_{timestamp}_env{i}")
            return env
        return _init

    env = DummyVecEnv([make_env(i) for i in range(8)])

    log_path = f"logs/training_log_{timestamp}"
    with open(f"{log_path}_notes.txt", "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write(f"Total timesteps: 30_000_000\n")
        f.write(f"=8 parallel envs, no ref cycle, removed both feet on ground reward\n")

    summary = pd.DataFrame([{
        "run": timestamp,
        "total_timesteps": 30_000_000,
        "notes": "8 parallel envs, no ref cycle, removed both feet on ground reward"
    }])
    summary.to_csv(f"{log_path}_summary.csv", index=False)

    model = PPO("MlpPolicy", env, verbose=1, clip_range=0.2, learning_rate=0.00003, tensorboard_log="logs/tensorboard/")
    model.learn(total_timesteps=30_000_000)
    model.save("models/louis")
    print("Training completed, Louis saved.")

    try:
        subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            r"C:\Users\shane\Documents\rlRobot\logs\louisNotes.ipynb",
            "--output",
            r"C:\Users\shane\Documents\rlRobot\logs\louisNotes.ipynb"
        ], check=True)
        print("Notebook updated.")
    except Exception as e:
        print(f"Notebook update failed: {e}")

main()