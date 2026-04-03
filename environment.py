import gymnasium as gym
import numpy as np
import mujoco
import os

class BipedEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Load Louis's physical model from xml
        xml_path = os.path.join(os.path.dirname(__file__), "robot/biped.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.cycle_index = 0

        # 12 position targets, 6 per leg (hip_rot, hip_abd, hip_flex, knee, ankle_flex, ankle_rot)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 37 sensor values (qpos=19: 7 torso + 12 joints, qvel=18: 6 torso + 12 joints)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        # reference walking cycle - 8 key poses (joint angles in radians)
        # order per leg: hip_rot, hip_abd, hip_flex, knee, ankle_flex, ankle_rot
        # left leg first, then right leg
        self.reference_cycle = np.array([
            # pose 1: left knee-high, right leg planted
            [0.0, 0.05, -0.5, 1.2, 0.3, 0.0, 0.0, -0.05, -0.2, 0.1, -0.1, 0.0],
            # pose 2: left leg driving down, right knee lifting
            [0.0, 0.02, -0.2, 0.5, 0.1, 0.0, 0.0, -0.02, -0.4, 0.8, 0.2, 0.0],
            # pose 3: left foot planted, right knee at peak
            [0.0, -0.02, 0.1, 0.1, -0.1, 0.0, 0.0, 0.05, -0.6, 1.2, 0.3, 0.0],
            # pose 4: left pushing back, right driving down
            [0.0, -0.05, 0.3, 0.2, -0.2, 0.0, 0.0, 0.02, -0.3, 0.5, 0.1, 0.0],
            # pose 5: right knee-high, left leg planted (mirror of pose 1)
            [0.0, -0.05, -0.2, 0.1, -0.1, 0.0, 0.0, 0.05, -0.5, 1.2, 0.3, 0.0],
            # pose 6: right leg driving down, left knee lifting
            [0.0, -0.02, -0.4, 0.8, 0.2, 0.0, 0.0, 0.02, -0.2, 0.5, 0.1, 0.0],
            # pose 7: right foot planted, left knee at peak
            [0.0, 0.05, -0.6, 1.2, 0.3, 0.0, 0.0, -0.02, 0.1, 0.1, -0.1, 0.0],
            # pose 8: right pushing back, left driving down
            [0.0, 0.02, -0.3, 0.5, 0.1, 0.0, 0.0, -0.05, 0.3, 0.2, -0.2, 0.0],
        ])

    def reset(self, seed=None, options=None):
        # wipe simulation back to default, louis returns to spawn (0, 0, 1.5)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:19] = 0.0  # start joints at neutral
        self.data.qvel[:] = 0.0  # no initial velocity
        self.step_count = 0
        self.cycle_index = 0  # reset walking cycle back to pose 1
        return self._get_obs(), {}  # empty dict required by gymnasium

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        # copy Louis's actions directly to the 12 motor controls
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # reward
        forward_velocity = self.data.qvel[0]  # x velocity, positive = moving forward
        torso_height = self.data.qpos[2]      # z position, drops when louis falls
        energy = np.sum(np.square(action))    # sum of squared motor forces
        alive_bonus = 0.1                     # flat reward each step for not falling
        height_reward = (torso_height - 0.5) * 4.0  # reward louis for standing tall

        # reward walking at human speed (~1.4 m/s), penalize too fast or too slow
        target_speed = 0.8
        speed_reward = -abs(forward_velocity - target_speed)

        # penalize sideways drift
        lateral_velocity = self.data.qvel[1]
        lateral_penalty = abs(lateral_velocity) * 3.0

        left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
        right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")

        # check if feet are touching the ground
        left_foot_contact = any(
            left_foot_id in (self.data.contact[i].geom1, self.data.contact[i].geom2)
            for i in range(self.data.ncon)
        )
        right_foot_contact = any(
            right_foot_id in (self.data.contact[i].geom1, self.data.contact[i].geom2)
            for i in range(self.data.ncon)
        )

        # reward alternating feet, penalize hopping
        if left_foot_contact and not right_foot_contact:
            foot_contact_reward = 1.0
        elif right_foot_contact and not left_foot_contact:
            foot_contact_reward = 1.0
        elif not left_foot_contact and not right_foot_contact:
            foot_contact_reward = -4.0  # hopping penalty
        else:
            foot_contact_reward = 0.0

        # get current target pose from the walking cycle
        target_pose = self.reference_cycle[self.cycle_index]

        # get louis's current joint angles (qpos[7:19] are the 12 joint angles)
        current_joints = self.data.qpos[7:19]

        # reward louis for matching the target pose, penalize deviation
        pose_error = np.sum(np.square(current_joints - target_pose))
        pose_reward = np.exp(-2.0 * pose_error) * 16.0

        # advance to next pose every 40 steps (~0.08 seconds per pose)
        if self.step_count % 40 == 0:
            self.cycle_index = (self.cycle_index + 1) % len(self.reference_cycle)

        # combine all rewards
        reward = speed_reward + alive_bonus + height_reward + foot_contact_reward + pose_reward - lateral_penalty - (0.001 * energy)

        terminated = torso_height < 0.5 or torso_height > 2.0 # terminate if louis falls too low or launches too high
        self.step_count += 1
        truncated = self.step_count >= 5000  # episode time limit (10 seconds of simulation)
        return obs, reward, terminated, truncated, {}
