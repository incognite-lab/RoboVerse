"""Run a five-action simple Chairman policy on the floating, full G1.

Actions retain the slider layout: [target_y, target_x, target_yaw, left
shoulder pitch, right shoulder pitch]. XY and yaw are absolute in the task
frame (the same world axes as the slider), not offsets from the reset pose.
The first five observations emulate the removed slider joints.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

try:
    from .SB3_chairman_env import StableBaseline3VecEnv as WalkingEnv
    from .SB3_chairman_simple_env import StableBaseline3VecEnv as SimpleEnv
except ImportError:
    from SB3_chairman_env import StableBaseline3VecEnv as WalkingEnv
    from SB3_chairman_simple_env import StableBaseline3VecEnv as SimpleEnv

from metasim.cfg.robots.g1_cfg_slider_simple import G1SliderSimpleCfg


class StableBaseline3VecEnv(WalkingEnv):
    MAIN_ROBOT_LINK_NAMES = (
        "pelvis", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    )

    def __init__(self, env):
        if env.scenario.robots[0].name != "g1_with_hands":
            raise ValueError("Simple walking requires robots: [g1_with_hands]")
        super().__init__(env)
        limits = G1SliderSimpleCfg().joint_limits
        self.action_names = tuple(limits)
        self.action_space = spaces.Box(
            np.array([v[0] for v in limits.values()], dtype=np.float32),
            np.array([v[1] for v in limits.values()], dtype=np.float32),
        )
        self.num_stages = 4
        self.left_shoulder_idx = self.sim_joint_names.index("left_shoulder_pitch_joint")
        self.right_shoulder_idx = self.sim_joint_names.index("right_shoulder_pitch_joint")
        self._shoulder_upper_indices = [
            self.upper_body_joint_names.index(name) for name in self.action_names[3:]
        ]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)

    def _planar_pose(self):
        robot = self.env.env.handler.get_states().robots[self.robot_name]
        pelvis = robot.body_state[:, self._pelvis_index].detach().cpu().numpy()
        w, x, y, z = pelvis[:, 3:7].T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return robot, pelvis[:, :2], yaw

    def _compose_robot_targets(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 5) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite simple walking actions of shape {(self.num_envs, 5)}")
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        _, xy, yaw = self._planar_pose()
        delta = actions[:, [1, 0]] - xy
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_error = np.arctan2(np.sin(actions[:, 2] - yaw), np.cos(actions[:, 2] - yaw))
        # Proportional position servo: 1 m error -> 1 m/s, 1 rad -> 1 rad/s.
        # The shared walking controller clips velocities to motion.pt limits.
        command = np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                                   -s * delta[:, 0] + c * delta[:, 1], yaw_error))
        command[np.linalg.norm(delta, axis=1) < 0.02, :2] = 0.0
        command[np.abs(yaw_error) < 0.02, 2] = 0.0
        upper = np.tile(self._upper_default_targets, (self.num_envs, 1))
        upper[:, self._shoulder_upper_indices] = actions[:, 3:5]
        return self._compose_upper_and_walk_targets(upper, command.astype(np.float32))

    def add_extra_to_obs(self, obs):
        robot, xy, yaw = self._planar_pose()
        shoulders = robot.joint_pos[:, [self.left_shoulder_idx, self.right_shoulder_idx]]
        virtual_joints = np.column_stack((xy[:, 1], xy[:, 0], yaw,
                                         shoulders.detach().cpu().numpy()))
        return SimpleEnv.add_extra_to_obs(self, virtual_joints)
