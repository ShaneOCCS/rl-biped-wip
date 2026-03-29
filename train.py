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

    # write training logs
    with open(f"{log_path}_notes.txt", "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write(f"Total timesteps: 5_000_000\n")
        f.write(f"Reward function: speed_reward + alive_bonus + height_reward + foot_contact_reward - lateral_penalty - energy\n")
        f.write(f"Notes: increased height reward to 4.0, raised termination height to 0.7 to fix crouching\n")

    summary = pd.DataFrame([{
        "run": timestamp,
        "total_timesteps": 5_000_000,
        "reward_function": "speed_reward + alive_bonus + height_reward + foot_contact_reward - lateral_penalty - energy",
        "notes": "increased height reward to 4.0, raised termination height to 0.7, episode limit 3000 steps"
    }])
    summary.to_csv(f"{log_path}_summary.csv", index=False)

    # create PPO model, clip_range=0.1 and learning_rate=0.0001 for stable learning
    model = PPO("MlpPolicy", env, verbose=1, clip_range=0.1, learning_rate=0.0001)
    model.learn(total_timesteps=5_000_000)
    model.save("models/model")
    print("Training completed, Louis saved.")

main()