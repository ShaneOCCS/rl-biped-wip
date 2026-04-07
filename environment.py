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

        # 12 motors, 6 per leg (hip_rot, hip_abd, hip_flex, knee, ankle_flex, ankle_rot)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 37 sensor values (qpos=19: 7 torso + 12 joints, qvel=18: 6 torso + 12 joints)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:19] = 0.0  # start joints at neutral
        self.data.qvel[:] = 0.0     # no initial velocity
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        # apply actions to the 12 motors
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # base signals
        forward_velocity = self.data.qvel[0]   # x velocity, positive = moving forward
        torso_height     = self.data.qpos[2]   # z position, drops when louis falls
        energy           = np.sum(np.square(action))
        alive_bonus = 0.1

        # reward standing tall
        height_reward = (torso_height - 0.5) * 4.0

        # reward walking near target speed, penalize too fast or too slow
        target_speed = 0.3
        speed_reward = -abs(forward_velocity - target_speed)

        # penalize sideways drift
        lateral_velocity = self.data.qvel[1]
        lateral_penalty  = abs(lateral_velocity) * 1.5

        # foot contact
        left_foot_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
        right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")

        left_foot_contact = any(
            left_foot_id in (self.data.contact[i].geom1, self.data.contact[i].geom2)
            for i in range(self.data.ncon)
        )
        right_foot_contact = any(
            right_foot_id in (self.data.contact[i].geom1, self.data.contact[i].geom2)
            for i in range(self.data.ncon)
        )

        # reward alternating feet, penalize both feet up or standing still on both
        if left_foot_contact and not right_foot_contact:
            foot_contact_reward = 1.0
        elif right_foot_contact and not left_foot_contact:
            foot_contact_reward = 1.0
        elif not left_foot_contact and not right_foot_contact:
            foot_contact_reward = -8.0  # both feet off ground
        else:
            foot_contact_reward = 0.0

        # symmetry reward — opposite hip angles = natural alternating gait
        left_hip = self.data.qpos[9]  # LL_HF joint angle
        right_hip = self.data.qpos[15]  # LR_HF joint angle
        symmetry_reward = -abs(left_hip + right_hip) * 2.0

        # reward bent knees — straight legs get penalized
        left_knee = self.data.qpos[10]  # LL_KFE
        right_knee = self.data.qpos[16]  # LR_KFE
        knee_reward = (abs(left_knee) + abs(right_knee)) * 0.5

        # penalize hip rotation — stops spinning
        left_hip_rot = self.data.qpos[7]  # LL_HR
        right_hip_rot = self.data.qpos[13]  # LR_HR
        hip_rot_penalty = (abs(left_hip_rot) + abs(right_hip_rot)) * 3.0

        # combine all rewards
        reward = (
                speed_reward
                + alive_bonus
                + height_reward
                + foot_contact_reward
                + symmetry_reward
                + knee_reward
                - lateral_penalty
                - hip_rot_penalty
                - (0.0005 * energy)
        )

        # terminate if louis falls or launches too high
        terminated = torso_height < 0.5 or torso_height > 2.0
        self.step_count += 1
        truncated = self.step_count >= 5000
        return obs, reward, terminated, truncated, {}
