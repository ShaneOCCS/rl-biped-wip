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

    # write training metadata to a notes file alongside the csv
    with open(f"{log_path}_notes.txt", "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write(f"Total timesteps: 10_000_000\n")
        f.write(f"Reward function: forward_velocity + alive_bonus + height_reward - energy\n")
        f.write(f"Notes: added height reward to fix crawling\n")
        f.write(f"Reward function: forward_velocity + alive_bonus + height_reward + foot_contact_reward - energy\n")
        f.write(f"Notes: added foot contact reward to fix hopping\n")

    # create a summary csv with run details
    summary = pd.DataFrame([{
        "run": timestamp,
        "total_timesteps": 10_000_000,
        "reward_function": "forward_velocity + alive_bonus + height_reward - energy",
        "notes": "added height reward to fix crawling"
    }])
    summary.to_csv(f"{log_path}_summary.csv", index=False)

    # create PPO model with a standard neural network, trained on louis's environment
    # verbose=1 prints training progress to the terminal
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000_000)
    model.save("models/model")  # saves louis's learning
    print("Training completed, Louis saved.")

main()