"""Chairman2: movable arm targets followed by body-frame [vx, vy, yaw_rate].

For g1_mygym_without_hand.urdf this is 10 arm angles (radians) and three
walking commands (m/s, m/s, rad/s). G1MotionPolicy owns all 12 leg targets;
waist joints hold their configured default angles. Fixed joints have no action.
The inherited multi-stage API supports both SB3 and Torch collectors.  Unlike
the generic Chairman wrapper, the policy observation is deliberately compact:
leg state is private to G1MotionPolicy and absolute poses of every arm link are
not exposed.  The six contact slots contain world-frame resultant chair forces
on the left/right hand [Lx, Ly, Lz, Rx, Ry, Rz], divided by 50 N and clipped to
[-1, 1]. Each hand includes its palm and fixed finger links.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from gymnasium import spaces

try:
    from .SB3_chairman_multi_env import (
        StableBaseline3VecEnv as WalkingEnv,
        _quaternion_error_vector,
    )
except ImportError:
    from SB3_chairman_multi_env import (
        StableBaseline3VecEnv as WalkingEnv,
        _quaternion_error_vector,
    )


class StableBaseline3VecEnv(WalkingEnv):
    NUM_POLICY_STAGES = 5
    from metasim.utils.chairman2_geometry import APPROACH_DISTANCE as FINAL_APPROACH_DISTANCE
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

    @property
    def stage_confirmation_steps(self) -> tuple[int, ...]:
        from metasim.cfg.checkers.stages_chairman2 import STAGE_TIMEOUTS, STAGE_TIMEOUT_REFERENCE_DT

        scenario = self.env.env.handler.scenario
        dt = (scenario.sim_params.dt or 0.002) * scenario.decimation
        return tuple(int(np.ceil(STAGE_TIMEOUTS[i] * STAGE_TIMEOUT_REFERENCE_DT / dt)) + 1
                     for i in range(self.NUM_POLICY_STAGES))

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
        self.num_stages = self.NUM_POLICY_STAGES
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
        initial_states = env.env.handler.get_states()
        robot = initial_states.robots[self.robot_name]
        chair = initial_states.objects["chair"]
        self.left_endffector = robot.body_names.index("left_hand_palm_link")
        self.right_endffector = robot.body_names.index("right_hand_palm_link")
        self._task_end_effector_indices = (
            robot.body_names.index("left_endeffector"),
            robot.body_names.index("endeffector"),
        )
        self._torso_index = robot.body_names.index("torso_link")
        self._chair_base_index = chair.body_names.index("base_link")
        self._chair_target_indices = (
            chair.body_names.index("target_hand_left"),
            chair.body_names.index("target_hand_right"),
        )

        # Keep this layout explicit.  Besides documenting the network input it
        # prevents a future parent-class observation from silently re-adding
        # leg joints or large world-frame link-state blocks.
        widths = (
            ("arm_joint_position", self.num_upper_body_actions),
            ("arm_joint_velocity", self.num_upper_body_actions),
            ("pelvis_linear_velocity_body", 3),
            ("pelvis_angular_velocity_body", 3),
            ("torso_up_world", 3),
            ("final_approach_body", 2),
            ("chair_back_direction_body", 2),
            ("chair_velocity_body", 3),
            ("chair_yaw_rate", 1),
            ("end_effector_target_delta_body", 6),
            ("palm_orientation_error", 6),
            ("hand_task_metrics", 6),
            ("end_effector_velocity_body", 6),
            ("hand_chair_force", 6),
            ("stage_context", 8),
            ("last_locomotion_command", 3),
            ("stage_one_hot", self.num_stages),
        )
        self.observation_feature_slices = {}
        offset = 0
        for name, width in widths:
            self.observation_feature_slices[name] = slice(offset, offset + width)
            offset += width
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(offset,), dtype=np.float32
        )

    @property
    def stage_observation_indices(self) -> tuple[int, ...]:
        stage_slice = self.observation_feature_slices["stage_one_hot"]
        return tuple(range(stage_slice.start, stage_slice.stop))

    def _task_context(self, states=None):
        from metasim.utils.chair_navigation import chair_back_direction_xy, world_vector_to_body_xy
        from metasim.utils.chairman2_geometry import PULL_DISTANCE
        handler = self.env.env.handler
        if states is None:
            states = handler.get_states()
        robot, chair = states.robots[self.robot_name], states.objects['chair']
        base = robot.body_state[:, self._pelvis_index]
        cb = chair.body_state[:, self._chair_base_index]
        task = handler.task
        stages = self.get_current_stages_torch()
        recorded = getattr(task, 'recorded_stage', torch.full_like(stages, -1))
        entering = recorded != stages
        ra = getattr(task, 'chairman_robot_anchor', base[:, :2])
        ca = getattr(task, 'chairman_chair_anchor', cb[:, :3])
        direction = getattr(task, 'chairman_pull_direction', chair_back_direction_xy(cb[:, 3:7]))
        # Before the first checker call of a new stage the anchor is its entry state.
        rd = torch.where(entering[:, None], 0., base[:, :2]-ra)
        cd = torch.where(entering[:, None], 0., cb[:, :2]-ca[:, :2])
        direction = torch.where(entering[:, None], chair_back_direction_xy(cb[:, 3:7]), direction)
        elapsed = getattr(task, 'stage_steps', torch.zeros_like(stages)).float()
        elapsed = torch.where(entering, 0., elapsed)
        scenario = self.env.scenario
        dt = (scenario.sim_params.dt or 0.002)*scenario.decimation
        return torch.cat((world_vector_to_body_xy(rd, base[:, 3:7]),
                          world_vector_to_body_xy(cd, base[:, 3:7]),
                          world_vector_to_body_xy(direction, base[:, 3:7]),
                          (PULL_DISTANCE-(cd*direction).sum(-1))[:, None],
                          (elapsed*dt/20.)[:, None]), dim=1)

    def add_extra_to_obs(self, obs):
        command = torch.as_tensor(
            self.last_locomotion_command,
            dtype=torch.float32,
            device=self.torch_device,
        )
        return self._compact_observation_torch(command).detach().cpu().numpy()

    def add_extra_to_obs_torch(self, obs):
        del obs  # The simulator's full joint vector also contains the legs.
        return self._compact_observation_torch(self.last_locomotion_command_torch)

    def _compact_observation_torch(self, locomotion_command):
        """Return only state observable and useful to the Chairman2 policy."""
        from metasim.utils.chair_navigation import (
            chair_back_direction_xy,
            world_vector_to_body_xy,
        )
        from metasim.utils.chairman2_geometry import (
            hand_task_space_metrics,
            rotate,
        )

        states = self.env.env.handler.get_states()
        robot = states.robots[self.robot_name]
        chair = states.objects["chair"]
        pelvis = robot.body_state[:, self._pelvis_index]
        torso = robot.body_state[:, self._torso_index]
        chair_body = chair.body_state[:, self._chair_base_index]
        pelvis_quat = pelvis[:, 3:7]

        arm_q = robot.joint_pos.index_select(1, self._upper_state_indices_torch)
        arm_qd = robot.joint_vel.index_select(1, self._upper_state_indices_torch)
        pelvis_linear_body = torch.cat(
            (world_vector_to_body_xy(pelvis[:, 7:9], pelvis_quat), pelvis[:, 9:10]),
            dim=1,
        )
        pelvis_angular_body = torch.cat(
            (world_vector_to_body_xy(pelvis[:, 10:12], pelvis_quat), pelvis[:, 12:13]),
            dim=1,
        )
        torso_up_world = rotate(
            torso[:, 3:7], torso.new_tensor([0.0, 0.0, 1.0])
        )

        chair_direction_world = chair_back_direction_xy(chair_body[:, 3:7])
        final_position = (
            chair_body[:, :2] + self.FINAL_APPROACH_DISTANCE * chair_direction_world
        )
        final_approach_body = world_vector_to_body_xy(
            final_position - pelvis[:, :2], pelvis_quat
        )
        chair_direction_body = world_vector_to_body_xy(
            chair_direction_world, pelvis_quat
        )
        chair_velocity_body = torch.cat(
            (
                world_vector_to_body_xy(chair_body[:, 7:9], pelvis_quat),
                chair_body[:, 9:10],
            ),
            dim=1,
        )

        end_effectors = robot.body_state[:, self._task_end_effector_indices]
        targets = chair.body_state[:, self._chair_target_indices]
        target_delta = targets[..., :3] - end_effectors[..., :3]
        target_delta_body = torch.cat(
            tuple(
                torch.cat(
                    (
                        world_vector_to_body_xy(target_delta[:, hand, :2], pelvis_quat),
                        target_delta[:, hand, 2:3],
                    ),
                    dim=1,
                )
                for hand in range(2)
            ),
            dim=1,
        )

        palms = robot.body_state[:, [self.left_endffector, self.right_endffector]]
        palm_orientation_error = torch.cat(
            tuple(
                _quaternion_error_vector(palms[:, hand, 3:7], targets[:, hand, 3:7])
                for hand in range(2)
            ),
            dim=1,
        )
        metrics = hand_task_space_metrics(
            torso, end_effectors, targets[..., :3], chair_direction_world
        )
        hand_task_metrics = torch.cat(
            (
                metrics["hand_forward_reach"],
                metrics["hand_height_above_backrest"],
                metrics["hand_behind_backrest"],
            ),
            dim=1,
        )
        end_effector_velocity_body = torch.cat(
            tuple(
                torch.cat(
                    (
                        world_vector_to_body_xy(
                            end_effectors[:, hand, 7:9], pelvis_quat
                        ),
                        end_effectors[:, hand, 9:10],
                    ),
                    dim=1,
                )
                for hand in range(2)
            ),
            dim=1,
        )
        hand_chair_force = self._fingertip_chair_forces(states, robot)
        stage_context = self._task_context(states)
        stages = self.get_current_stages_torch().clamp(0, self.num_stages - 1)
        stage_one_hot = torch.nn.functional.one_hot(
            stages, num_classes=self.num_stages
        ).to(dtype=torch.float32)

        result = torch.cat(
            (
                arm_q,
                arm_qd,
                pelvis_linear_body,
                pelvis_angular_body,
                torso_up_world,
                final_approach_body,
                chair_direction_body,
                chair_velocity_body,
                chair_body[:, 12:13],
                target_delta_body,
                palm_orientation_error,
                hand_task_metrics,
                end_effector_velocity_body,
                hand_chair_force,
                stage_context,
                locomotion_command.to(device=self.torch_device, dtype=torch.float32),
                stage_one_hot,
            ),
            dim=1,
        )
        expected = (self.num_envs, self.observation_space.shape[0])
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"Chairman2 observation has shape {tuple(result.shape)}, expected {expected}"
            )
        return result

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
