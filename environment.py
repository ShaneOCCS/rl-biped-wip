import gymnasium as gym
import numpy as np
import mujoco
import os

class BipedEnv(gym.Env):
    def __init__(self):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "robot/biped.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 1.0 would be fully extending a motor while -1.0 would be bending motor, 0.0 being zero force at all etc...
        # gear in xml handles actual physical force
        # louis will learn to walk by combining different number ratios over and over again
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


        self.action_space = gym.spaces.Box(low= np.inf, high=np.inf, shape=(15,), dtype=np.float32)