from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import BipedEnv
from datetime import datetime
import pandas as pd
import os

def main():
    os.makedirs("logs", exist_ok=True)
    env = BipedEnv()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/training_log_{timestamp}"
    env = Monitor(env, filename=log_path)

    with open(f"{log_path}_notes.txt", "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write(f"Total timesteps: 30_000_000\n")
        f.write(f"Notes: added HAA joint for more stability and added more lateral penalty, altered alive bonus and pose reward \n")

    summary = pd.DataFrame([{
        "run": timestamp,
        "total_timesteps":30_000_000 ,
        "notes": "added HAA joint for more stability and added more lateral penalty, altered alive bonus and pose reward"
    }])
    summary.to_csv(f"{log_path}_summary.csv", index=False)

    # PPO model, tboard for adv stats
    model = PPO("MlpPolicy", env, verbose=1, clip_range=0.2, learning_rate=0.00003, tensorboard_log="logs/tensorboard/")
    model.learn(total_timesteps=30_000_000)
    model.save("models/louis")
    print("Training completed, Louis saved.")

main()