from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import BipedEnv
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

MODEL_PATH = "models/louis"
VECNORM_PATH = "models/louis_vecnormalize.pkl"

# build a single-env VecNormalize wrapper using the saved stats
raw_env = BipedEnv()
venv = DummyVecEnv([lambda: BipedEnv()])
if os.path.exists(VECNORM_PATH):
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False     # freeze running stats
    venv.norm_reward = False  # we don't care about reward normalization at eval
    print("Loaded VecNormalize stats.")
else:
    print("WARNING: VecNormalize stats not found - policy may behave wrong.")

model = PPO.load(MODEL_PATH)

obs = venv.reset()

# render against the underlying raw env (VecNormalize doesn't expose model/data nicely)
# we step both the venv (for normalized obs) and mirror state to the raw env for viewing
underlying = venv.envs[0].unwrapped if hasattr(venv, 'envs') else raw_env

with mujoco.viewer.launch_passive(underlying.model, underlying.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

        if done[0]:
            obs = venv.reset()

        viewer.sync()
        time.sleep(underlying.control_dt)