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

        # control / sim timing
        # physics runs at model.opt.timestep (0.002s = 500 Hz)
        # we step physics N_SUBSTEPS times per env.step() -> 50 Hz control
        self.n_substeps = 10
        self.control_dt = self.model.opt.timestep * self.n_substeps  # 0.02s

        # gait clock
        self.gait_period = 0.7      # seconds for full L+R cycle
        self.swing_ratio = 0.4      # fraction of cycle each foot is in swing
        self.phase = 0.0

        # action smoothing
        # ctrl_t = alpha * new_action + (1-alpha) * ctrl_{t-1}
        self.action_smoothing = 0.5
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)

        # joint range scaling
        # policy outputs in [-1, 1]; we map to each joint's actual ctrlrange
        self.ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid   = 0.5 * (self.ctrl_high + self.ctrl_low)
        self.ctrl_range = 0.5 * (self.ctrl_high - self.ctrl_low)

        # foot geom ids (cached)
        self.left_foot_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        self.left_foot_body  = self.model.geom_bodyid[self.left_foot_id]
        self.right_foot_body = self.model.geom_bodyid[self.right_foot_id]

        self.step_count = 0

        # 12 motors
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # qpos(19) + qvel(18) + clock(2) = 39
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # neutral pose with small random noise (helps robustness a lot)
        self.data.qpos[2] = 0.90  # torso start height
        self.data.qpos[7:19] = self.np_random.uniform(-0.05, 0.05, size=12)
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, size=self.model.nv)

        # randomize starting phase so episodes don't all start at left swing
        self.phase = float(self.np_random.uniform(0.0, 1.0))

        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        # initialize ctrl to neutral so PD doesn't yank
        self.data.ctrl[:] = self.ctrl_mid

        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        clock = np.array(
            [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
            dtype=np.float32,
        )
        return np.concatenate([self.data.qpos, self.data.qvel, clock]).astype(np.float32)

    def _foot_contact_force(self, geom_id):
        """Sum of normal contact force magnitudes on a foot geom."""
        total = 0.0
        f6 = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if geom_id == c.geom1 or geom_id == c.geom2:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += float(np.linalg.norm(f6[:3]))
        return total

    def _foot_lin_vel(self, body_id):
        """Linear velocity of the foot body in world frame."""
        # cvel layout is [angular(3), linear(3)] in world frame
        return float(np.linalg.norm(self.data.cvel[body_id][3:6]))

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # action smoothing (low-pass)
        smoothed = (
            self.action_smoothing * action
            + (1.0 - self.action_smoothing) * self.prev_action
        )
        self.prev_action = smoothed

        # ap [-1, 1] -> per-joint target angle
        ctrl_target = self.ctrl_mid + self.ctrl_range * smoothed
        self.data.ctrl[:] = ctrl_target

        # step physics N times at the same ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # advance gait clock
        self.phase = (self.phase + self.control_dt / self.gait_period) % 1.0
        left_phase  = self.phase
        right_phase = (self.phase + 0.5) % 1.0

        obs = self._get_obs()

        # reward
        forward_velocity = self.data.qvel[0]
        lateral_velocity = self.data.qvel[1]
        torso_height     = self.data.qpos[2]

        # quaternion: qpos[3:7] = [w, x, y, z]
        quat = self.data.qpos[3:7]
        # upright-ness: project body z-axis onto world z-axis
        # for quaternion [w,x,y,z], world-z component of body-z = 1 - 2(x^2 + y^2)
        upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)

        # velocity tracking (smooth, bounded [0, 1])
        target_speed = 0.8  # m/s — fast enough to need real swing
        speed_reward = np.exp(-((forward_velocity - target_speed) ** 2) / 0.25)

        # lateral drift penalty
        lateral_penalty = lateral_velocity ** 2

        # upright + height
        upright_reward = max(upright, 0.0)               # [0, 1]
        height_reward  = np.exp(-((torso_height - 0.85) ** 2) / 0.02)  # peaks at 0.85m

        #periodic gait reward
        left_frc  = self._foot_contact_force(self.left_foot_id)
        right_frc = self._foot_contact_force(self.right_foot_id)
        left_vel  = self._foot_lin_vel(self.left_foot_body)
        right_vel = self._foot_lin_vel(self.right_foot_body)

        # smooth swing/stance indicators in [0, 1]
        # E_swing(p)  ~ 1 when p in [0, swing_ratio], else ~0
        # E_stance(p) ~ 1 when p in [swing_ratio, 1], else ~0
        def swing_indicator(p):
            # 1 inside swing window, 0 outside, soft edges
            d_in  = p
            d_out = self.swing_ratio - p
            # cheap smooth box: sigmoid(k*d_in) * sigmoid(k*d_out)
            k = 25.0
            return _sigmoid(k * d_in) * _sigmoid(k * d_out)

        def stance_indicator(p):
            d_in  = p - self.swing_ratio
            d_out = 1.0 - p
            k = 25.0
            return _sigmoid(k * d_in) * _sigmoid(k * d_out)

        L_sw, L_st = swing_indicator(left_phase),  stance_indicator(left_phase)
        R_sw, R_st = swing_indicator(right_phase), stance_indicator(right_phase)

        # in swing: penalize force (foot should be in air)
        # in stance: penalize velocity (foot should be planted)
        # scale forces and vels to similar magnitudes
        frc_scale = 0.0005  # contact forces are ~3000-5000 N at this robot's weight
        vel_scale = 1.0
        # clip forces to prevent solver spikes from dominating
        left_frc_c = min(left_frc, 4000.0)
        right_frc_c = min(right_frc, 4000.0)
        periodic_reward = -(
                L_sw * frc_scale * left_frc_c + L_st * vel_scale * left_vel
                + R_sw * frc_scale * right_frc_c + R_st * vel_scale * right_vel
        )

        # foot air-time bonus (encourages clean stepping) tracked across env.step calls
        if not hasattr(self, "_left_air"):
            self._left_air = 0.0
            self._right_air = 0.0
        left_contact  = left_frc  > 1.0
        right_contact = right_frc > 1.0
        airtime_bonus = 0.0
        if left_contact and self._left_air > 0:
            airtime_bonus += min(self._left_air, 0.4)
        if right_contact and self._right_air > 0:
            airtime_bonus += min(self._right_air, 0.4)
        self._left_air  = 0.0 if left_contact  else self._left_air  + self.control_dt
        self._right_air = 0.0 if right_contact else self._right_air + self.control_dt

        # action smoothness penalty
        action_rate_penalty = np.sum(np.square(action - self.prev_action))

        # COM height variance penalty (anti-hopping)
        if not hasattr(self, "_prev_height"):
            self._prev_height = torso_height
        height_var_penalty = abs(torso_height - self._prev_height)
        self._prev_height = torso_height

        # energy
        energy = float(np.sum(np.square(self.data.actuator_force))) * 1e-5

        # alive bonus
        alive_bonus = 1.0

        # combine
        reward = (
              1.5 * speed_reward
            + 1.0 * upright_reward
            + 1.0 * height_reward
            + 1.0 * periodic_reward
            + 2.0 * airtime_bonus
            + 0.5 * alive_bonus
            - 0.5 * lateral_penalty
            - 0.1 * action_rate_penalty
            - 5.0 * height_var_penalty
            - energy
        )
        # fell, launched, or tipped over (upright dot product flipped)
        terminated = bool(
            torso_height < 0.55
            or torso_height > 1.5
            or upright < 0.3   # tipped past ~70 degrees
        )
        self.step_count += 1
        truncated = self.step_count >= 1000  # 20s episodes at 50Hz

        return obs, float(reward), terminated, truncated, {}


def _sigmoid(x):
    # numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))