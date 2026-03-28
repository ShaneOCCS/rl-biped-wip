import gymnasium as gym
import numpy as np
import mujoco
import os

class BipedEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # load louis's physical model from xml
        xml_path = os.path.join(os.path.dirname(__file__), "robot/biped.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0

        # 6 motors, one per joint (hip, knee, ankle x2)
        # -1.0 = full bend, 0.0 = no force, 1.0 = full extend
        # gear in xml converts these values into actual physical force
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 25 sensor values louis can observe about himself (qpos=13, qvel=12)
        # unbounded (np.inf) because sensor readings can be any value
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # wipe simulation back to default, louis returns to spawn (0, 0, 1.5)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}  # empty dict required by gymnasium

    def _get_obs(self):
        # qpos: 3 torso position + 4 quaternion orientation + 6 joint angles = 13 values
        # qvel: 3 linear velocity + 3 angular velocity + 6 joint velocities = 12 values
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        # copy ai actions directly to the 6 motor controls
        self.data.ctrl[:] = action

        # advance physics by one timestep (0.002s), updates qpos and qvel
        mujoco.mj_step(self.model, self.data)

        # get louis's updated state after the action was applied
        obs = self._get_obs()

        # reward
        forward_velocity = self.data.qvel[0]        # x velocity, positive = moving forward
        torso_height = self.data.qpos[2]            # z position, drops when louis falls
        energy = np.sum(np.square(action))          # sum of squared motor forces
        alive_bonus = 1.0                           # flat reward each step for not falling
        height_reward = (torso_height - 0.5) * 2.0  # reward louis for standing tall, max reward at 1.3m

        # check if feet are touching the ground
        left_foot_contact = any(
            self.data.contact[i].geom2 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
            for i in range(self.data.ncon)
        )
        right_foot_contact = any(
            self.data.contact[i].geom2 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
            for i in range(self.data.ncon)
        )
        # reward if at least one foot is on the ground, penalize if both are off (hopping)
        foot_contact_reward = 1.0 if (left_foot_contact or right_foot_contact) else -1.0

        # reward forward movement and survival, penalize wasted energy
        # 0.001 weight keeps penalty small so louis still wants to move
        reward = forward_velocity + alive_bonus + height_reward + foot_contact_reward - (0.001 * energy)

        # --- episode end conditions ---
        terminated = torso_height < 0.5    # louis has fallen (torso too close to ground)
        self.step_count += 1
        truncated = self.step_count >= 1000  # episode time limit (2 seconds of simulation)
        return obs, reward, terminated, truncated, {}
