"""Chairman2: movable arm targets followed by body-frame [vx, vy, yaw_rate].

For g1_mygym_without_hand.urdf this is 10 arm angles (radians) and three
walking commands (m/s, m/s, rad/s). G1MotionPolicy owns all 12 leg targets;
waist joints hold their configured default angles. Fixed joints have no action.
The inherited multi-stage API supports both SB3 and Torch collectors.
The six inherited contact observation slots contain world-frame resultant
chair forces on the left/right hand [Lx, Ly, Lz, Rx, Ry, Rz], divided by 50 N
and clipped to [-1, 1]. Each hand includes its palm and fixed finger links.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from gymnasium import spaces

try:
    from .SB3_chairman_multi_env import StableBaseline3VecEnv as WalkingEnv
except ImportError:
    from SB3_chairman_multi_env import StableBaseline3VecEnv as WalkingEnv


class StableBaseline3VecEnv(WalkingEnv):
    HAND_CONTACT_FORCE_SCALE = 50.0
    MAIN_ROBOT_LINK_NAMES = (
        "pelvis", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
        "left_hand_palm_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
        "right_hand_palm_link",
    )

    def _policy_joint_limits(self, robot_cfg):
        # Configs may still list the fingers and locked wrist joints. Read the
        # selected URDF instead of assuming that every configured joint moves.
        path = Path(robot_cfg.urdf_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        joints = ET.parse(path).getroot().findall("joint")
        movable = {joint.attrib["name"]: joint for joint in joints
                   if joint.attrib["type"] in ("revolute", "prismatic", "continuous")}
        missing = set(movable) - set(robot_cfg.joint_limits)
        if missing:
            raise ValueError(f"Movable URDF joints missing configured limits: {sorted(missing)}")
        limits = {}
        for name, bounds in robot_cfg.joint_limits.items():
            if name not in movable:
                continue
            lower, upper = bounds
            urdf_limit = movable[name].find("limit")
            if urdf_limit is not None and movable[name].attrib["type"] != "continuous":
                lower = max(lower, float(urdf_limit.attrib["lower"]))
                upper = min(upper, float(urdf_limit.attrib["upper"]))
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"Invalid movable joint limits for {name}: {(lower, upper)}")
            limits[name] = (lower, upper)
        return limits

    def _policy_upper_joint_names(self):
        return tuple(name for name in self.robot_joint_names
                     if name.startswith(("left_shoulder_", "right_shoulder_",
                                         "left_elbow_", "right_elbow_",
                                         "left_wrist_", "right_wrist_",
                                         "left_hand_", "right_hand_")))

    def __init__(self, env):
        super().__init__(env)
        old_stages = self.num_stages
        self.num_stages = self.NUM_POLICY_STAGES
        # Base joint observations follow the simulator's actual DOF list.
        obs_dim = (self.observation_space.shape[0] - old_stages + self.num_stages
                   - len(self.robot_joint_names) + len(self.sim_joint_names))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        cfg = env.scenario.robots[0]
        controlled = set(self.leg_joint_names) | set(self.upper_body_joint_names)
        held_names = [name for name in self.robot_joint_names if name not in controlled]
        self._held_state_indices = [self.sim_joint_names.index(name) for name in held_names]
        self._held_targets = np.asarray(
            [cfg.default_joint_positions[name] for name in held_names], dtype=np.float32
        )
        self._held_state_indices_torch = torch.tensor(
            self._held_state_indices, dtype=torch.long, device=self.torch_device
        )
        self._held_targets_torch = torch.as_tensor(self._held_targets, device=self.torch_device)
        # Chairman2 checkers use actual palms rather than the old EE offsets.
        robot = env.env.handler.get_states().robots[self.robot_name]
        self.left_endffector = robot.body_names.index("left_hand_palm_link")
        self.right_endffector = robot.body_names.index("right_hand_palm_link")

    def _compose_robot_targets(self, actions):
        targets = super()._compose_robot_targets(actions)
        targets[:, self._held_state_indices] = self._held_targets
        return targets

    def _compose_robot_targets_torch(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.torch_device)
        if not torch.isfinite(actions).all():
            raise ValueError("Chairman2 policy produced NaN or infinite actions")
        targets = super()._compose_robot_targets_torch(actions)
        targets.index_copy_(1, self._held_state_indices_torch,
                            self._held_targets_torch.expand(self.num_envs, -1))
        return targets

    def _fingertip_chair_forces(self, states, robot):
        """Inherited observation hook; Chairman2 senses the whole fixed hand."""
        result = robot.joint_pos.new_zeros((self.num_envs, 6))
        contact = getattr(robot, "contact", None)
        if contact is None or contact["link_a"].shape[1] == 0:
            return result
        mapping = states.extras.get("global_link_map", {})
        chair_ids = [index for index, (name, _) in mapping.items() if name == "chair"]
        if not chair_ids:
            return result
        # Genesis link IDs are global solver indices, shared across batch rows.
        a, b = contact["link_a"], contact["link_b"]
        valid = contact["valid_mask"]
        chair_ids = torch.as_tensor(chair_ids, device=a.device)
        chair_a, chair_b = torch.isin(a, chair_ids), torch.isin(b, chair_ids)
        force_b = contact.get("force_b", contact.get("force"))
        if force_b is None:
            return result
        force_a = contact.get("force_a", -force_b)
        for side_index, side in enumerate(("left", "right")):
            hand_ids = [index for index, (name, link) in mapping.items()
                        if name == self.robot_name and link.startswith(f"{side}_hand_")]
            if not hand_ids:
                continue
            hand_ids = torch.as_tensor(hand_ids, device=a.device)
            hand_a = valid & chair_b & torch.isin(a, hand_ids)
            hand_b = valid & chair_a & torch.isin(b, hand_ids)
            forces = (torch.where(hand_a[..., None], force_a, 0.0)
                      + torch.where(hand_b[..., None], force_b, 0.0))
            result[:, side_index * 3:(side_index + 1) * 3] = forces.sum(dim=1)
        return (result / self.HAND_CONTACT_FORCE_SCALE).clamp(-1.0, 1.0)


Chairman2VecEnv = StableBaseline3VecEnv
