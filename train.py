from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import BipedEnv
from datetime import datetime
import pandas as pd
import subprocess
import os

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    N_ENVS = 16
    TOTAL_STEPS = 50_000_000

    def make_env(i):
        def _init():
            env = BipedEnv()
            env = Monitor(env, filename=f"logs/training_log_{timestamp}_env{i}")
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # VecNormalize: critical for locomotion
    # normalizes obs (running mean/std) and rewards (running std only)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    log_path = f"logs/training_log_{timestamp}"
    notes = (
        "v2: position actuators (PD), periodic gait reward, clock obs (sin/cos), "
        "action smoothing 0.5, 50Hz control (10 substeps), VecNormalize, "
        "target_speed=0.8 m/s, airtime bonus, anti-hopping height-var penalty, "
        "upright tipover termination, reset noise."
    )
    with open(f"{log_path}_notes.txt", "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write(f"Total timesteps: {TOTAL_STEPS}\n")
        f.write(f"N envs: {N_ENVS}\n")
        f.write(notes + "\n")

    pd.DataFrame([{
        "run": timestamp,
        "total_timesteps": TOTAL_STEPS,
        "n_envs": N_ENVS,
        "notes": notes,
    }]).to_csv(f"{log_path}_summary.csv", index=False)

    # PPO hyperparameters tuned for biped locomotion
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,        # SB3 default; was way too low at 3e-5
        n_steps=2048,              # per env -> 2048*16 = 32768 per rollout
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        tensorboard_log="logs/tensorboard/",
    )

    # save every 1M steps in case training crashes or you want intermediate checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000_000 // N_ENVS, 1),
        save_path="models/checkpoints/",
        name_prefix=f"louis_{timestamp}",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb)
    model.save("models/louis")
    env.save("models/louis_vecnormalize.pkl")  # needed at inference time
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

if __name__ == "__main__":
    main()