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

        # 6 motors, one per joint (hip, knee, ankle x2)
        # -1.0 = full bend, 0.0 = no force, 1.0 = full extend
        # gear in xml converts these values into actual physical force
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 25 sensor values louis can observe about himself (qpos=13, qvel=12)
        # unbounded (np.inf) because sensor readings can be any value
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        # reference walking cycle - 8 key poses (joint angles in radians)
        # order: left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle
        self.reference_cycle = np.array([
            # pose 1: left knee-high, right leg planted
            [0.7, -1.0, 0.1, -0.3, -0.1, 0.2],
            # pose 2: left leg driving down, right knee starting to lift
            [0.4, -0.5, 0.2, -0.1, -0.4, 0.1],
            # pose 3: left foot planted, right knee pulling up
            [0.1, -0.1, 0.1, 0.2, -0.9, 0.0],
            # pose 4: left leg pushing back, right knee at peak
            [-0.2, -0.2, 0.2, 0.6, -1.0, 0.1],
            # pose 5: right knee-high, left leg planted (mirror of pose 1)
            [-0.3, -0.1, 0.2, 0.7, -0.5, 0.1],
            # pose 6: right leg driving down, left knee starting to lift
            [-0.1, -0.4, 0.1, 0.4, -0.1, 0.2],
            # pose 7: right foot planted, left knee pulling up
            [0.2, -0.9, 0.0, 0.1, -0.1, 0.1],
            # pose 8: right leg pushing back, left knee at peak (mirror of pose 4)
            [0.6, -1.0, 0.1, -0.2, -0.2, 0.2],
        ])

    def reset(self, seed=None, options=None):
        # wipe simulation back to default, louis returns to spawn (0, 0, 1.5)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.cycle_index = 0  # reset walking cycle back to pose 1
        return self._get_obs(), {}  # empty dict required by gymnasium

    def _get_obs(self):
        # qpos: 3 torso position + 4 quaternion orientation + 10 joint angles = 17 values
        # qvel: 3 linear velocity + 3 angular velocity + 10 joint velocities = 16 values
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        # copy Louis's actions directly to the 10 motor controls
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # reward
        forward_velocity = self.data.qvel[0]  # x velocity, positive = moving forward
        torso_height = self.data.qpos[2]      # z position, drops when louis falls
        energy = np.sum(np.square(action))    # sum of squared motor forces
        alive_bonus = 0.5                     # flat reward each step for not falling
        height_reward = (torso_height - 0.5) * 4.0  # reward louis for standing tall

        # reward walking at human speed (~1.4 m/s), penalize too fast or too slow
        target_speed = 0.8
        speed_reward = -abs(forward_velocity - target_speed)

        # penalize sideways drift
        lateral_velocity = self.data.qvel[1]
        lateral_penalty = abs(lateral_velocity)

        # check if feet are touching the ground
        left_foot_contact = any(
            self.data.contact[i].geom2 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
            for i in range(self.data.ncon)
        )
        right_foot_contact = any(
            self.data.contact[i].geom2 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
            for i in range(self.data.ncon)
        )

        # reward alternating feet, penalize hopping
        if left_foot_contact and not right_foot_contact:
            foot_contact_reward = 1.0
        elif right_foot_contact and not left_foot_contact:
            foot_contact_reward = 1.0
        elif not left_foot_contact and not right_foot_contact:
            foot_contact_reward = -2.0  # hopping penalty
        else:
            foot_contact_reward = 0.0

        # get current target pose from the walking cycle
        target_pose = self.reference_cycle[self.cycle_index]

        # get louis's current joint angles (qpos[7:13] are the 6 joint angles)
        current_joints = self.data.qpos[7:13]

        # reward louis for matching the target pose, penalize deviation
        pose_error = np.sum(np.square(current_joints - target_pose))
        pose_reward = np.exp(-2.0 * pose_error) * 8.0

        # advance to next pose every 40 steps (~0.08 seconds per pose)
        if self.step_count % 40 == 0:
            self.cycle_index = (self.cycle_index + 1) % len(self.reference_cycle)

        # combine all rewards
        reward = speed_reward + alive_bonus + height_reward + foot_contact_reward + pose_reward - lateral_penalty - (0.001 * energy)

        terminated = torso_height < 0.5 or torso_height > 2.0 # terminate if louis falls too low or launches too high
        self.step_count += 1
        truncated = self.step_count >= 5000  # episode time limit (10 seconds of simulation)
        return obs, reward, terminated, truncated, {}
