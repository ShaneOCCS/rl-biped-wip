from environment import BipedEnv
from stable_baselines3 import PPO
import time
import mujoco.viewer
import mujoco

model = PPO.load("models/model")
env = BipedEnv()

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # model looks at observation and decides action
        action, _ = model.predict(obs)

        # apply action and get new observation
        obs, reward, terminated, truncated, info = env.step(action)

        # if louis fell or time ran out, reset for a new episode
        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(0.002)
        viewer.sync()