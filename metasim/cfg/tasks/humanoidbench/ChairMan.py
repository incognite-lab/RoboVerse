"""ChairMan task rewards for the full staged chair task."""

from __future__ import annotations

import math

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils import chairman2_geometry as chairman_geometry
from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS_BY_SIDE
from metasim.utils.chair_navigation import (
    CHAIR_FINAL_DISTANCE,
    CHAIR_STAGING_DISTANCE,
    chair_back_direction_xy,
    forward_direction_xy,
    smoothstep01,
)
from metasim.utils.humanoid_robot_util import neck_height_tensor

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


HEIGHT_THRESHOLD = 0.4


def _stage_mask(actual_stage: torch.Tensor, stages: int | tuple[int, ...] | list[int]) -> torch.BoolTensor:
    """Return a GPU boolean mask without allocating a stage tensor every step."""
    if isinstance(stages, int):
        return actual_stage == stages

    mask = actual_stage == stages[0]
    for stage in stages[1:]:
        mask = mask | (actual_stage == stage)
    return mask


# =============================================================================
# BASIC / AUXILIARY REWARDS
# =============================================================================

class TerminationCfg(HumanoidBaseReward):
    """Penalty signal for every unsuccessful termination from the checker.

    The checker can terminate an environment for more reasons than a fall
    (for example when the chair is displaced too far).  ``termination_events``
    is populated by ``_ChairManChecker`` before rewards are evaluated.
    """
    def __init__(self):
        super().__init__()
        self.termination_events = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.termination_events is not None:
            self.termination_events[env_ids] = False

    def __call__(self, states: EnvState, robot_name) -> torch.FloatTensor:
        if self.termination_events is not None:
            robot = states.robots[robot_name]
            return self.termination_events.to(
                device=robot.joint_pos.device,
                dtype=robot.joint_pos.dtype,
            )

        # Backward-compatible fallback for direct use outside the staged
        # Chairman checker.
        neck_heights = neck_height_tensor(states, robot_name)
        return torch.where(
            neck_heights < HEIGHT_THRESHOLD,
            torch.ones_like(neck_heights),
            torch.zeros_like(neck_heights),
        )


class DeltaActionRateCfg(HumanoidBaseReward):
    """
    Penalty magnitude for abrupt target changes in actions.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_actions = None
        self.controlled_indices = None
        self.scale = 0.35

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        actions = robot.joint_pos_target
        if (
            self.prev_actions is None
            or self.prev_actions.shape != actions.shape
            or self.prev_actions.device != actions.device
        ):
            # Genesis may expose six floating-base DOFs in the reset state
            # (49 values) and only the 43 actuated joints after the first
            # simulation step.  Such layouts must never be subtracted.
            self.prev_actions = actions.detach().clone()
        else:
            self.prev_actions[env_ids] = actions[env_ids].detach().clone()

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        actions = states.robots[robot_name].joint_pos_target

        if (
            self.prev_actions is None
            or self.prev_actions.shape != actions.shape
            or self.prev_actions.device != actions.device
        ):
            self.prev_actions = actions.detach().clone()
            return torch.zeros(
                actions.shape[0], device=actions.device, dtype=actions.dtype
            )

        # motion.pt intentionally produces a periodic leg trajectory. Fingers
        # also need to travel by as much as 1.7 rad when stage 2 starts. Neither
        # belongs in the upper-body target-rate penalty.
        if self.controlled_indices is None:
            leg_names = {
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            }
            indices = [
                i for i, name in enumerate(states.robots[robot_name].joint_names)
                if name not in leg_names and "_hand_" not in name
            ]
            self.controlled_indices = torch.tensor(indices, dtype=torch.long, device=actions.device)
        elif self.controlled_indices.device != actions.device:
            self.controlled_indices = self.controlled_indices.to(actions.device)

        delta_actions = (actions - self.prev_actions).index_select(1, self.controlled_indices)
        self.prev_actions = actions.detach().clone()

        mean_abs_delta = torch.mean(torch.abs(delta_actions), dim=1)
        penalty = torch.clamp(mean_abs_delta / self.scale, min=0.0, max=1.0)
        return penalty


class DoFVelocityAccelerationCfg(HumanoidBaseReward):
    """
    Penalty magnitude for high joint velocities and accelerations (excluding fingers).

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_joint_vel = None
        self.controlled_indices = None

        self.vel_scale = 6.0
        self.acc_scale = 8.0

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        joint_vel = robot.joint_vel
        if (
            self.prev_joint_vel is None
            or self.prev_joint_vel.shape != joint_vel.shape
            or self.prev_joint_vel.device != joint_vel.device
        ):
            self.prev_joint_vel = joint_vel.detach().clone()
        else:
            self.prev_joint_vel[env_ids] = joint_vel[env_ids].detach().clone()

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_vel = robot.joint_vel
        device = joint_vel.device

        if self.controlled_indices is None:
            # Regularize only waist and arms. Leg motion comes from the frozen
            # gait controller and fingers must remain free to close in stage 2.
            indices = [
                idx for idx, joint in enumerate(robot.joint_names)
                if not any(token in joint for token in ("hip", "knee", "ankle", "hand"))
            ]
            self.controlled_indices = torch.tensor(indices, dtype=torch.long, device=device)
        elif self.controlled_indices.device != device:
            self.controlled_indices = self.controlled_indices.to(device)

        target_vel = joint_vel.index_select(1, self.controlled_indices)

        mean_abs_vel = torch.mean(torch.abs(target_vel), dim=-1)

        if (
            self.prev_joint_vel is None
            or self.prev_joint_vel.shape != joint_vel.shape
            or self.prev_joint_vel.device != joint_vel.device
        ):
            mean_abs_acc = torch.zeros_like(mean_abs_vel)
            self.prev_joint_vel = joint_vel.detach().clone()
        else:
            prev_target_vel = self.prev_joint_vel.index_select(1, self.controlled_indices)

            delta_vel = target_vel - prev_target_vel
            mean_abs_acc = torch.mean(torch.abs(delta_vel), dim=-1)
            self.prev_joint_vel = joint_vel.detach().clone()

        vel_penalty = torch.clamp(mean_abs_vel / self.vel_scale, min=0.0, max=1.0)
        acc_penalty = torch.clamp(mean_abs_acc / self.acc_scale, min=0.0, max=1.0)

        return 0.7 * vel_penalty + 0.3 * acc_penalty


class LocomotionCommandPenalty(HumanoidBaseReward):
    """Penalize command jumps and nonzero walking commands during manipulation."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.command = None
        self.previous_command = None
        self.delta_scale = torch.tensor([0.08, 0.06, 0.15])
        self.command_scale = torch.tensor([0.50, 0.30, 0.80])
        self.delta_scale_device = None
        self.command_scale_device = None
        self.stop_radius = 0.45

    def set_control_context(self, command, previous_command, device=None):
        """Receive physical [vx, vy, yaw_rate] commands from the SB3 wrapper."""
        self.command = torch.as_tensor(command, dtype=torch.float32, device=device)
        self.previous_command = torch.as_tensor(
            previous_command, dtype=torch.float32, device=device
        )

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.command is not None:
            self.command[env_ids] = 0.0
            self.previous_command[env_ids] = 0.0
        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]
        if self.actual_stage is None or self.command is None:
            return torch.zeros(num_envs, device=device)

        command = self.command.to(device)
        previous = self.previous_command.to(device)
        if self.delta_scale_device is None or self.delta_scale_device.device != device:
            self.delta_scale_device = self.delta_scale.to(device)
            self.command_scale_device = self.command_scale.to(device)
        smoothness = torch.mean(
            torch.clamp(
                torch.abs(command - previous) / self.delta_scale_device, 0.0, 1.0
            ),
            dim=-1,
        )
        stop_command = torch.mean(
            torch.clamp(torch.abs(command) / self.command_scale_device, 0.0, 1.0),
            dim=-1,
        )

        manipulation = (self.actual_stage == 1) | (self.actual_stage == 2)
        chair = states.objects["chair"]
        pelvis_idx = robot.body_names.index("pelvis")
        chair_idx = chair.body_names.index("base_link")
        pelvis_xy = robot.body_state[:, pelvis_idx, :2]
        chair_state = chair.body_state[:, chair_idx]
        final_xy = (
            chair_state[:, :2]
            + CHAIR_FINAL_DISTANCE * chair_back_direction_xy(chair_state[:, 3:7])
        )
        near_final = torch.norm(pelvis_xy - final_xy, dim=-1) < self.stop_radius
        pull_target = torch.tensor(
            [-0.25, 0.0, 0.1], dtype=chair_state.dtype, device=device
        )
        near_pull_target = torch.norm(chair_state[:, :3] - pull_target, dim=-1) < 0.20
        post_pull = (self.actual_stage == 4) | (self.actual_stage == 5)
        stop_gate = (
            manipulation
            | post_pull
            | ((self.actual_stage == 0) & near_final)
            | ((self.actual_stage == 3) & near_pull_target)
        )

        penalty = torch.where(
            stop_gate, 0.25 * smoothness + 0.75 * stop_command, smoothness
        )
        # During stage 3, smooth commands are important for a single
        # uninterrupted pull.  Stages 4 and 5 require a zero walking command
        # while the fingers are released and the arms are lowered.
        active = (self.actual_stage >= 0) & (self.actual_stage <= 5)
        return torch.clamp(penalty, 0.0, 1.0) * active.float()


class DofPositionLimitsCfg(HumanoidBaseReward):
    """
    Soft penalty magnitude for approaching joint limits.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.limit_buffer = 0.05
        self.joint_limits: dict[str, tuple[float, float]] = {
            "waist_yaw_joint": (-2.618, 2.618),
            "waist_roll_joint": (-0.52, 0.52),
            "waist_pitch_joint": (-0.52, 0.52),
            "left_shoulder_pitch_joint": (-3.0892, 2.6704),
            "left_shoulder_roll_joint": (-1.5882, 2.2515),
            "left_shoulder_yaw_joint": (-2.618, 2.618),
            "left_elbow_joint": (-1.0472, 2.0944),
            "left_wrist_roll_joint": (-1.972222054, 1.972222054),
            "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
            "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
            "right_shoulder_pitch_joint": (-3.0892, 2.6704),
            "right_shoulder_roll_joint": (-2.2515, 1.5882),
            "right_shoulder_yaw_joint": (-2.618, 2.618),
            "right_elbow_joint": (-1.0472, 2.0944),
            "right_wrist_roll_joint": (-1.972222054, 1.972222054),
            "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
            "right_wrist_yaw_joint": (-1.614429558, 1.614429558),
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        joint_names = robot.joint_names
        device = joint_pos.device

        total_violation = torch.zeros(joint_pos.shape[0], device=device)
        counted = 0

        for i, name in enumerate(joint_names):
            if name not in self.joint_limits:
                continue

            counted += 1
            low, high = self.joint_limits[name]
            low_tensor = torch.tensor(low, device=device)
            high_tensor = torch.tensor(high, device=device)

            below_low = torch.relu((low_tensor + self.limit_buffer) - joint_pos[:, i])
            above_high = torch.relu(joint_pos[:, i] - (high_tensor - self.limit_buffer))
            total_violation = total_violation + below_low + above_high

        if counted == 0:
            return torch.zeros(joint_pos.shape[0], device=device)

        mean_violation = total_violation / counted
        penalty = torch.clamp(mean_violation / 0.25, min=0.0, max=1.0)
        return penalty


class HumanlyDofLimitCfg(HumanoidBaseReward):
    """
    Penalty magnitude for exceeding more human-like upper-body limits.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.dof_indices = None
        self.q_lower_tensor = None
        self.q_upper_tensor = None
        self.limit_buffer = 0.05

        self.human_limits: dict[str, tuple[float, float]] = {
            "waist_yaw_joint": (-1.0, 1.0),
            "waist_roll_joint": (-0.3, 0.3),
            "waist_pitch_joint": (-0.3, 0.5),

            "left_shoulder_pitch_joint": (-2.8, 2.5),
            "right_shoulder_pitch_joint": (-2.8, 2.5),

            "left_shoulder_roll_joint": (-0.5, 2.0),
            "right_shoulder_roll_joint": (-2.0, 0.5),

            "left_shoulder_yaw_joint": (-1.6, 1.6),
            "right_shoulder_yaw_joint": (-1.6, 1.6),

            "left_elbow_joint": (0.0, 2.1),
            "right_elbow_joint": (0.0, 2.1),

            "left_wrist_roll_joint": (-1.5, 1.5),
            "left_wrist_pitch_joint": (-1.0, 1.0),
            "left_wrist_yaw_joint": (-1.0, 1.0),
            "right_wrist_roll_joint": (-1.5, 1.5),
            "right_wrist_pitch_joint": (-1.0, 1.0),
            "right_wrist_yaw_joint": (-1.0, 1.0),
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device

        if self.dof_indices is None:
            self.dof_indices = []
            lower_vals = []
            upper_vals = []

            for i, name in enumerate(robot.joint_names):
                if name in self.human_limits:
                    self.dof_indices.append(i)
                    low, high = self.human_limits[name]
                    lower_vals.append(low + self.limit_buffer)
                    upper_vals.append(high - self.limit_buffer)

            if not self.dof_indices:
                return torch.zeros(joint_pos.shape[0], device=device)

            self.dof_indices = torch.tensor(self.dof_indices, device=device, dtype=torch.long)
            self.q_lower_tensor = torch.tensor(lower_vals, device=device).unsqueeze(0)
            self.q_upper_tensor = torch.tensor(upper_vals, device=device).unsqueeze(0)

        q_active = joint_pos[:, self.dof_indices]
        violation_lower = torch.clamp(self.q_lower_tensor - q_active, min=0.0)
        violation_upper = torch.clamp(q_active - self.q_upper_tensor, min=0.0)
        total_violation = violation_lower + violation_upper

        mean_violation = torch.mean(total_violation, dim=-1)
        penalty = torch.clamp(mean_violation / 0.20, min=0.0, max=1.0)
        return penalty


class UprightPenaltyCfg(HumanoidBaseReward):
    """
    Penalty magnitude for torso not being upright.

    Assumes quaternion order [w, x, y, z] in body_state[:, :, 3:7].

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.target_z = torch.tensor([0.0, 0.0, 1.0])

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        torso_link_idx = robot.body_names.index("torso_link")
        root_quat = robot.body_state[:, torso_link_idx, 3:7]
        device = root_quat.device

        if self.target_z.device != device:
            self.target_z = self.target_z.to(device)

        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]

        current_z_x = 2 * (x * z + y * w)
        current_z_y = 2 * (y * z - x * w)
        current_z_z = 1 - 2 * (x * x + y * y)

        current_z_axis = torch.stack([current_z_x, current_z_y, current_z_z], dim=1)
        alignment = torch.sum(current_z_axis * self.target_z, dim=-1)

        penalty = torch.clamp((1.0 - alignment) / 2.0, min=0.0, max=1.0)
        return penalty


# =============================================================================
# STAGE 0
# =============================================================================
class Stage0ArmPos(HumanoidBaseReward):
    """
    Reward for keeping arms in position suitable for stage 0 (approaching chair).

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stage = 0
        self.dof_indices = None
        self.required_pos_tensor = None
        self.required_pos_list = None
        self.arm_sigma = 0.18
        self.waist_sigma = 0.08
        self.finger_sigma = 0.25

        self.required_pos: dict[str, float] = {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            "left_shoulder_pitch_joint": 0.28,
            "right_shoulder_pitch_joint": 0.28,

            "left_shoulder_roll_joint": 0.35,
            "right_shoulder_roll_joint": -0.35,

            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,

            "left_elbow_joint": 0.77,
            "right_elbow_joint": 0.77,

            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,

            # Left hand fingers
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            # Right hand fingers
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device
        num_envs = joint_pos.shape[0]

        # ``actual_stage`` is shared with this reward by the Chairman checker.
        # Until it is initialized, and in every stage other than stage 0, this
        # reward contributes exactly zero for the corresponding environment.
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = self.actual_stage.to(device=device) == self.active_stage
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.dof_indices is None:
            self.dof_indices = []
            self.required_pos_list = []


            for i, name in enumerate(robot.joint_names):
                if name in self.required_pos:
                    self.dof_indices.append(i)
                    pos = self.required_pos[name]
                    self.required_pos_list.append(pos)

            if not self.dof_indices:
                return torch.zeros(num_envs, device=device)

            self.dof_indices = torch.tensor(self.dof_indices, device=device, dtype=torch.long)
            self.required_pos_tensor = torch.tensor(self.required_pos_list, device=device).unsqueeze(0)

        if self.dof_indices.device != device:
            self.dof_indices = self.dof_indices.to(device)
        if self.required_pos_tensor.device != device:
            self.required_pos_tensor = self.required_pos_tensor.to(device)

        q_active = joint_pos[:, self.dof_indices]
        error = torch.abs(q_active - self.required_pos_tensor)
        selected_names = [robot.joint_names[int(i)] for i in self.dof_indices.tolist()]
        waist_mask = torch.tensor(
            [name.startswith("waist_") for name in selected_names], device=device
        )
        finger_mask = torch.tensor(
            ["_hand_" in name for name in selected_names], device=device
        )
        arm_mask = ~(waist_mask | finger_mask)

        def group_reward(mask, sigma):
            if not mask.any():
                return torch.ones(num_envs, device=device)
            return torch.exp(-torch.mean(torch.square(error[:, mask] / sigma), dim=-1))

        # Do not dilute three waist joints by averaging them with 28 arm/finger
        # joints. This makes torso motion during walking visibly expensive.
        reward = (
            0.45 * group_reward(waist_mask, self.waist_sigma)
            + 0.45 * group_reward(arm_mask, self.arm_sigma)
            + 0.10 * group_reward(finger_mask, self.finger_sigma)
        )
        return reward * stage_mask.float()




class WalkToChairProgressReward(HumanoidBaseReward):
    """
    Stage 0 navigation through a staging point behind the chair.

    The robot first walks to a point 1.5 m behind the backrest.  Around that
    point it is rewarded for stopping and facing the chair.  Once aligned, the
    desired velocity opens smoothly toward the final point 0.75 m behind the
    backrest.  There is no discontinuous phase switch inside this reward.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands", target_speed=0.5):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.staging_distance = CHAIR_STAGING_DISTANCE
        self.final_distance = CHAIR_FINAL_DISTANCE
        self.target_speed = target_speed
        self.vel_sigma = 0.18
        self.staging_braking_distance = 0.55
        self.final_braking_distance = 0.55
        self.transition_half_width = 0.25
        self.corridor_sigma = 0.45
        self.turn_reward_radius = 0.55
        self.stop_position_sigma = 0.15
        self.stop_speed_sigma = 0.12
        self.direction_speed_scale = target_speed
        self.heading_gate_start = 0.50   # 60 degrees from the chair
        self.heading_gate_full = 0.94    # about 20 degrees from the chair
        self.overshoot_margin = 0.08
        self.saved_chair_pos = None
        self.saved_chair_quat = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        chair = states.objects["chair"]
        chair_base_idx = chair.body_names.index("base_link")
        chair_state = chair.body_state[:, chair_base_idx]
        chair_pos = chair_state[:, :3]
        chair_quat = chair_state[:, 3:7]

        if self.saved_chair_pos is None:
            self.saved_chair_pos = chair_pos.clone()
            self.saved_chair_quat = chair_quat.clone()
        else:
            self.saved_chair_pos[env_ids] = chair_pos[env_ids].clone()
            self.saved_chair_quat[env_ids] = chair_quat[env_ids].clone()

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(
            self.actual_stage,
            torch.tensor(self.active_stages, device=device)
        )
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        base_idx = robot.body_names.index("pelvis")
        root_pos_xy = robot.body_state[:, base_idx, :2]
        root_quat = robot.body_state[:, base_idx, 3:7]
        root_vel_xy = robot.body_state[:, base_idx, 7:9]

        chair_base_idx = chair.body_names.index("base_link")
        if self.saved_chair_pos is None:
            chair_state = chair.body_state[:, chair_base_idx]
            self.saved_chair_pos = chair_state[:, :3].clone()
            self.saved_chair_quat = chair_state[:, 3:7].clone()

        chair_pos_xy = self.saved_chair_pos[:, :2]
        back_dir = chair_back_direction_xy(self.saved_chair_quat)
        staging_pos = chair_pos_xy + self.staging_distance * back_dir
        final_pos = chair_pos_xy + self.final_distance * back_dir

        to_staging = staging_pos - root_pos_xy
        to_final = final_pos - root_pos_xy
        to_chair = chair_pos_xy - root_pos_xy
        staging_dist = torch.norm(to_staging, dim=-1)
        final_dist = torch.norm(to_final, dim=-1)
        chair_dist = torch.norm(to_chair, dim=-1)
        staging_dir = to_staging / (staging_dist.unsqueeze(-1) + 1.0e-6)
        final_dir = to_final / (final_dist.unsqueeze(-1) + 1.0e-6)
        face_chair_dir = to_chair / (chair_dist.unsqueeze(-1) + 1.0e-6)

        # Chair-frame coordinates make the path independent of world rotation.
        chair_to_robot = root_pos_xy - chair_pos_xy
        along_back = torch.sum(chair_to_robot * back_dir, dim=-1)
        side_dir = torch.stack((-back_dir[:, 1], back_dir[:, 0]), dim=-1)
        lateral_error = torch.abs(torch.sum(chair_to_robot * side_dir, dim=-1))
        corridor_gate = torch.exp(
            -torch.square(lateral_error) / (2.0 * self.corridor_sigma ** 2)
        )

        # The blend grows while crossing the staging plane toward the chair.
        # Multiplication by corridor_gate prevents cutting the corner when the
        # robot is still far to one side of the desired approach line.
        transition_input = (
            self.staging_distance + self.transition_half_width - along_back
        ) / (2.0 * self.transition_half_width)
        approach_blend = smoothstep01(transition_input) * corridor_gate

        forward_dir = forward_direction_xy(root_quat)
        heading_cos = torch.sum(forward_dir * face_chair_dir, dim=-1)
        heading_input = (
            heading_cos - self.heading_gate_start
        ) / (self.heading_gate_full - self.heading_gate_start)
        heading_gate = smoothstep01(heading_input)
        heading_reward = torch.clamp((heading_cos + 1.0) * 0.5, min=0.0, max=1.0)

        staging_speed = self.target_speed * torch.clamp(
            staging_dist / self.staging_braking_distance, min=0.0, max=1.0
        )
        final_speed = self.target_speed * torch.clamp(
            final_dist / self.final_braking_distance, min=0.0, max=1.0
        ) * heading_gate
        staging_vel = staging_speed.unsqueeze(-1) * staging_dir
        final_vel = final_speed.unsqueeze(-1) * final_dir
        target_vel_vec = (
            (1.0 - approach_blend).unsqueeze(-1) * staging_vel
            + approach_blend.unsqueeze(-1) * final_vel
        )

        vel_error_sq = torch.sum(torch.square(root_vel_xy - target_vel_vec), dim=-1)
        velocity_reward = torch.exp(-vel_error_sq / (2.0 * self.vel_sigma ** 2))

        desired_speed = torch.norm(target_vel_vec, dim=-1)
        desired_dir = target_vel_vec / (desired_speed.unsqueeze(-1) + 1.0e-6)
        velocity_projection = torch.sum(root_vel_xy * desired_dir, dim=-1)
        direction_reward = torch.clamp(
            velocity_projection / self.direction_speed_scale,
            min=0.0,
            max=1.0,
        )
        backward_amount = torch.clamp(-velocity_projection, min=0.0)
        backward_penalty = torch.clamp(backward_amount / 0.4, min=0.0, max=1.0)
        backward_factor = 1.0 - backward_penalty

        # Going past the final chair-frame plane must never be attractive.
        overshoot_amount = torch.clamp(
            (self.final_distance - self.overshoot_margin) - along_back, min=0.0
        )
        overshoot_penalty = torch.clamp(overshoot_amount / 0.10, min=0.0, max=1.0)
        overshoot_factor = 1.0 - overshoot_penalty

        speed_xy = torch.norm(root_vel_xy, dim=-1)
        stop_position_reward = torch.exp(
            -torch.square(final_dist) / (2.0 * self.stop_position_sigma ** 2)
        )
        stop_speed_reward = torch.exp(
            -torch.square(speed_xy) / (2.0 * self.stop_speed_sigma ** 2)
        )
        arrival_stop_reward = stop_position_reward * stop_speed_reward * heading_reward

        staging_proximity = torch.exp(
            -torch.square(staging_dist) / (2.0 * self.turn_reward_radius ** 2)
        )
        turn_context = torch.maximum(staging_proximity, approach_blend)
        turn_reward = turn_context * heading_reward

        base_reward = (
            0.15 * velocity_reward
            + 0.35 * direction_reward
            + 0.25 * turn_reward
            + 0.25 * arrival_stop_reward
        )
        total_reward = base_reward * backward_factor * overshoot_factor
        total_reward = torch.clamp(total_reward, min=0.0, max=1.0)
        return total_reward * stage_mask.float()


class KeepChairStillPenalty(HumanoidBaseReward):
    """
    Stage 0 and 1:
    Penalty magnitude for moving the chair before grasp.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0, 1]
        self.lin_scale_stage0 = 0.08
        self.ang_scale_stage0 = 0.30
        self.lin_scale_stage1 = 0.20
        self.ang_scale_stage1 = 0.70

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        active_tensor = torch.tensor(self.active_stages, device=device)
        stage_mask = torch.isin(self.actual_stage, active_tensor)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        chair_base_idx = chair.body_names.index("base_link")
        chair_lin_vel = chair.body_state[:, chair_base_idx, 7:10]
        chair_ang_vel = chair.body_state[:, chair_base_idx, 10:13]

        lin_norm = torch.norm(chair_lin_vel, dim=-1)
        ang_norm = torch.norm(chair_ang_vel, dim=-1)

        lin_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.lin_scale_stage0, device=device),
            torch.full((num_envs,), self.lin_scale_stage1, device=device),
        )
        ang_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.ang_scale_stage0, device=device),
            torch.full((num_envs,), self.ang_scale_stage1, device=device),
        )

        lin_penalty = torch.clamp(lin_norm / lin_scale, min=0.0, max=1.0)
        ang_penalty = torch.clamp(ang_norm / ang_scale, min=0.0, max=1.0)

        penalty = 0.7 * lin_penalty + 0.3 * ang_penalty
        return penalty * stage_mask.float()


class OpenGraspReward(HumanoidBaseReward):
    """
    Stage 0 and 1:
    Keep fingers open and calm.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0, 1]

        self.target_angle = 0.0
        self.pos_scale = 0.7
        self.vel_scale = 2.0

        self.finger_indices = None
        self.target_tensor = None
        self.finger_keywords = ["thumb", "index", "middle"]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        active_tensor = torch.tensor(self.active_stages, device=device)
        stage_mask = torch.isin(self.actual_stage, active_tensor)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.finger_indices is None:
            indices = []
            for idx, joint_name in enumerate(robot.joint_names):
                if any(k in joint_name for k in self.finger_keywords):
                    indices.append(idx)

            if not indices:
                return torch.zeros(num_envs, device=device)

            self.finger_indices = torch.tensor(indices, device=device, dtype=torch.long)
            self.target_tensor = torch.full((1, len(indices)), self.target_angle, device=device)

        q_finger = robot.joint_pos[:, self.finger_indices]
        dq_finger = robot.joint_vel[:, self.finger_indices]

        pos_error = torch.mean(torch.abs(q_finger - self.target_tensor), dim=-1)
        vel_norm = torch.mean(torch.abs(dq_finger), dim=-1)

        pos_reward = torch.clamp(1.0 - pos_error / self.pos_scale, min=0.0, max=1.0)
        vel_reward = torch.clamp(1.0 - vel_norm / self.vel_scale, min=0.0, max=1.0)

        total_reward = 0.75 * pos_reward + 0.25 * vel_reward
        return total_reward * stage_mask.float()


class FaceChairReward(HumanoidBaseReward):
    """
    Stage 0 reward for keeping the robot facing the chair.

    Reward torso forward-axis alignment with the chair in the XY plane.
    Every increase in angular error reduces the reward: +1 at zero error,
    zero at 20 degrees, and negative beyond that. No progress bonus or
    angular tolerance can mask a misaligned torso.

    Output: <-1, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.zero_reward_angle = math.radians(20.0)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        try:
            base_idx = robot.body_names.index("torso_link")
            chair_base_idx = chair.body_names.index("base_link")
        except ValueError:
            return torch.zeros(num_envs, device=device)

        base_pos_xy = robot.body_state[:, base_idx, :2]
        base_quat = robot.body_state[:, base_idx, 3:7]
        chair_pos_xy = chair.body_state[:, chair_base_idx, :2]

        to_chair = chair_pos_xy - base_pos_xy
        chair_dist = torch.norm(to_chair, dim=-1)
        chair_dir = to_chair / torch.clamp(chair_dist.unsqueeze(-1), min=1.0e-6)
        forward_dir = forward_direction_xy(base_quat)

        alignment = torch.sum(forward_dir * chair_dir, dim=-1)
        cross = forward_dir[:, 0] * chair_dir[:, 1] - forward_dir[:, 1] * chair_dir[:, 0]
        heading_error = torch.atan2(torch.abs(cross), alignment)
        alignment_reward = 2.0 * torch.exp(
            -math.log(2.0) * heading_error / self.zero_reward_angle
        ) - 1.0
        # A coincident target or vertical forward axis has no planar heading.
        valid_heading = (chair_dist > 1.0e-6) & (torch.norm(forward_dir, dim=-1) > 1.0e-6)
        alignment_reward = torch.where(valid_heading, alignment_reward, -torch.ones_like(alignment_reward))
        return alignment_reward * stage_mask.float()


# =============================================================================
# STAGE 1
# =============================================================================

class ReachChairProgressReward(HumanoidBaseReward):
    """
    Stage 1 dense reward for bringing *both* hands to their targets.

    A reciprocal state reward keeps a useful gradient even when the hands are
    initially far away.  The minimum of the two hand scores prevents one hand
    from compensating for the other one, matching the stage checker which
    requires both hands to succeed.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [1]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.progress_scale = 0.01
        self.distance_scale = 0.12
        self.prev_distances = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_distances is not None:
            self.prev_distances[env_ids] = torch.nan

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        try:
            l_hand_idx = robot.body_names.index(self.robot_left_hand)
            r_hand_idx = robot.body_names.index(self.robot_right_hand)
            l_target_idx = chair.body_names.index(self.chair_target_left)
            r_target_idx = chair.body_names.index(self.chair_target_right)
        except ValueError:
            return torch.zeros(num_envs, device=device)

        p_hand_left = robot.body_state[:, l_hand_idx, :3]
        p_hand_right = robot.body_state[:, r_hand_idx, :3]
        p_target_left = chair.body_state[:, l_target_idx, :3]
        p_target_right = chair.body_state[:, r_target_idx, :3]

        dist_left = torch.norm(p_hand_left - p_target_left, dim=-1)
        dist_right = torch.norm(p_hand_right - p_target_right, dim=-1)
        distances = torch.stack((dist_left, dist_right), dim=-1)

        if self.prev_distances is None or self.prev_distances.shape != distances.shape:
            self.prev_distances = distances.detach().clone()
            progress_per_hand = torch.zeros_like(distances)
        else:
            previous = torch.where(
                torch.isnan(self.prev_distances), distances, self.prev_distances
            )
            progress_per_hand = torch.clamp(
                (previous - distances) / self.progress_scale, min=0.0, max=1.0
            )
            self.prev_distances = torch.where(
                stage_mask.unsqueeze(-1), distances.detach(), self.prev_distances
            )

        state_per_hand = 1.0 / (
            1.0 + torch.square(distances / self.distance_scale)
        )
        state_reward = (
            0.35 * torch.mean(state_per_hand, dim=-1)
            + 0.65 * torch.min(state_per_hand, dim=-1).values
        )
        progress_reward = (
            0.35 * torch.mean(progress_per_hand, dim=-1)
            + 0.65 * torch.min(progress_per_hand, dim=-1).values
        )

        total_reward = 0.80 * state_reward + 0.20 * progress_reward
        return total_reward * stage_mask.float()


class HandOrientationProgressReward(HumanoidBaseReward):
    """
    Stage 1 dense orientation reward using the checker's quaternion metric.

    The checker uses ``1 - abs(q_target dot q_hand)``.  Using that exact error
    here avoids a reward/checker mismatch and avoids the unstable derivative
    of ``acos`` close to perfect alignment.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [1]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.progress_scale = 0.015
        self.error_scale = 0.08
        self.position_gate_scale = 0.25
        self.prev_errors = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_errors is not None:
            self.prev_errors[env_ids] = torch.nan

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        try:
            l_hand_idx = robot.body_names.index(self.robot_left_hand)
            r_hand_idx = robot.body_names.index(self.robot_right_hand)
            l_target_idx = chair.body_names.index(self.chair_target_left)
            r_target_idx = chair.body_names.index(self.chair_target_right)
        except ValueError:
            return torch.zeros(num_envs, device=device)

        q_hand_left = robot.body_state[:, l_hand_idx, 3:7]
        q_hand_right = robot.body_state[:, r_hand_idx, 3:7]
        q_target_left = chair.body_state[:, l_target_idx, 3:7]
        q_target_right = chair.body_state[:, r_target_idx, 3:7]

        q_hand_left = torch.nn.functional.normalize(q_hand_left, dim=-1)
        q_hand_right = torch.nn.functional.normalize(q_hand_right, dim=-1)
        q_target_left = torch.nn.functional.normalize(q_target_left, dim=-1)
        q_target_right = torch.nn.functional.normalize(q_target_right, dim=-1)
        errors = torch.stack(
            (
                1.0 - torch.abs(torch.sum(q_hand_left * q_target_left, dim=-1)),
                1.0 - torch.abs(torch.sum(q_hand_right * q_target_right, dim=-1)),
            ),
            dim=-1,
        )

        if self.prev_errors is None or self.prev_errors.shape != errors.shape:
            self.prev_errors = errors.detach().clone()
            progress_per_hand = torch.zeros_like(errors)
        else:
            previous = torch.where(torch.isnan(self.prev_errors), errors, self.prev_errors)
            progress_per_hand = torch.clamp(
                (previous - errors) / self.progress_scale, min=0.0, max=1.0
            )
            self.prev_errors = torch.where(
                stage_mask.unsqueeze(-1), errors.detach(), self.prev_errors
            )

        state_per_hand = 1.0 / (1.0 + torch.square(errors / self.error_scale))
        state_reward = (
            0.35 * torch.mean(state_per_hand, dim=-1)
            + 0.65 * torch.min(state_per_hand, dim=-1).values
        )
        progress_reward = (
            0.35 * torch.mean(progress_per_hand, dim=-1)
            + 0.65 * torch.min(progress_per_hand, dim=-1).values
        )

        # Orientation remains learnable far away, but becomes most important
        # once the palms approach the actual handle targets.
        left_dist = torch.norm(
            robot.body_state[:, l_hand_idx, :3] - chair.body_state[:, l_target_idx, :3], dim=-1
        )
        right_dist = torch.norm(
            robot.body_state[:, r_hand_idx, :3] - chair.body_state[:, r_target_idx, :3], dim=-1
        )
        max_dist = torch.maximum(left_dist, right_dist)
        position_gate = 1.0 / (
            1.0 + torch.square(max_dist / self.position_gate_scale)
        )
        total_reward = (0.80 * state_reward + 0.20 * progress_reward) * (
            0.30 + 0.70 * position_gate
        )
        return total_reward * stage_mask.float()


class HandTargetStillnessReward(HumanoidBaseReward):
    """
    Stage 1 and 2:
    Reward for having both hands:
    1) close to targets,
    2) with low linear velocity.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [1,2]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.dist_scale = 0.07
        self.vel_scale = 0.15

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        try:
            l_hand_idx = robot.body_names.index(self.robot_left_hand)
            r_hand_idx = robot.body_names.index(self.robot_right_hand)
            l_target_idx = chair.body_names.index(self.chair_target_left)
            r_target_idx = chair.body_names.index(self.chair_target_right)
        except ValueError:
            return torch.zeros(num_envs, device=device)

        p_hand_left = robot.body_state[:, l_hand_idx, :3]
        p_hand_right = robot.body_state[:, r_hand_idx, :3]
        v_hand_left = robot.body_state[:, l_hand_idx, 7:10]
        v_hand_right = robot.body_state[:, r_hand_idx, 7:10]

        p_target_left = chair.body_state[:, l_target_idx, :3]
        p_target_right = chair.body_state[:, r_target_idx, :3]

        dist_left = torch.norm(p_hand_left - p_target_left, dim=-1)
        dist_right = torch.norm(p_hand_right - p_target_right, dim=-1)
        vel_left = torch.norm(v_hand_left, dim=-1)
        vel_right = torch.norm(v_hand_right, dim=-1)

        distances = torch.stack((dist_left, dist_right), dim=-1)
        velocities = torch.stack((vel_left, vel_right), dim=-1)
        dist_per_hand = 1.0 / (
            1.0 + torch.pow(distances / self.dist_scale, 4)
        )
        vel_per_hand = torch.exp(
            -torch.square(velocities) / (2.0 * self.vel_scale ** 2)
        )
        per_hand = dist_per_hand * (0.30 + 0.70 * vel_per_hand)
        total_reward = (
            0.25 * torch.mean(per_hand, dim=-1)
            + 0.75 * torch.min(per_hand, dim=-1).values
        )
        return total_reward * stage_mask.float()


class PreciseHandTargetReward(HumanoidBaseReward):
    """Stage 1-2 reward for holding both end effectors on their targets.

    The reward is zero unless both hands are within the checker's 7 cm
    threshold. Inside that region, position shaping smoothly reaches the full
    precision bonus at 3 cm. The bonus includes quaternion alignment and hand
    stillness, so position alone is not sufficient.

    Output: <0, 1>
    """

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        # A persistent positive pose reward in stage 3 made standing still with
        # the chair more profitable than completing the pull. Stage 3 uses a
        # zero-at-goal drift penalty instead.
        self.active_stages = (1, 2)
        self.precise_distance = 0.03
        self.shaping_distance = 0.07
        self.precise_orientation_error = 0.03
        self.shaping_orientation_error = 0.08
        self.precise_speed = 0.15
        self.shaping_speed = 0.30
        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(
            self.actual_stage.to(device=device),
            torch.tensor(self.active_stages, dtype=torch.long, device=device),
        )
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        try:
            left_hand_idx = robot.body_names.index(self.robot_left_hand)
            right_hand_idx = robot.body_names.index(self.robot_right_hand)
            left_target_idx = chair.body_names.index(self.chair_target_left)
            right_target_idx = chair.body_names.index(self.chair_target_right)
        except ValueError:
            return torch.zeros(num_envs, device=device)

        left_state = robot.body_state[:, left_hand_idx]
        right_state = robot.body_state[:, right_hand_idx]
        left_target = chair.body_state[:, left_target_idx]
        right_target = chair.body_state[:, right_target_idx]

        distances = torch.stack(
            (
                torch.norm(left_state[:, :3] - left_target[:, :3], dim=-1),
                torch.norm(right_state[:, :3] - right_target[:, :3], dim=-1),
            ),
            dim=-1,
        )
        per_hand_proximity = 1.0 / (
            1.0 + torch.pow(distances / self.precise_distance, 4)
        )
        bilateral_proximity = (
            0.10 * torch.mean(per_hand_proximity, dim=-1)
            + 0.90 * torch.min(per_hand_proximity, dim=-1).values
        )
        max_distance = torch.max(distances, dim=-1).values
        both_hands_near = max_distance <= self.shaping_distance
        precision_bonus = smoothstep01(
            (self.shaping_distance - max_distance)
            / (self.shaping_distance - self.precise_distance)
        )
        position_score = 0.40 * bilateral_proximity + 0.60 * precision_bonus

        left_quat = torch.nn.functional.normalize(left_state[:, 3:7], dim=-1)
        right_quat = torch.nn.functional.normalize(right_state[:, 3:7], dim=-1)
        left_target_quat = torch.nn.functional.normalize(left_target[:, 3:7], dim=-1)
        right_target_quat = torch.nn.functional.normalize(right_target[:, 3:7], dim=-1)
        orientation_errors = torch.stack(
            (
                1.0 - torch.abs(torch.sum(left_quat * left_target_quat, dim=-1)),
                1.0 - torch.abs(torch.sum(right_quat * right_target_quat, dim=-1)),
            ),
            dim=-1,
        )
        max_orientation_error = torch.max(orientation_errors, dim=-1).values
        orientation_score = smoothstep01(
            (self.shaping_orientation_error - max_orientation_error)
            / (self.shaping_orientation_error - self.precise_orientation_error)
        )

        # Relative rather than world speed is essential in stage 3: when the
        # chair is pulled, a hand moving together with its target is still a
        # stable hold and must not be penalized.
        max_speed = torch.maximum(
            torch.norm(left_state[:, 7:10] - left_target[:, 7:10], dim=-1),
            torch.norm(right_state[:, 7:10] - right_target[:, 7:10], dim=-1),
        )
        stillness_score = smoothstep01(
            (self.shaping_speed - max_speed)
            / (self.shaping_speed - self.precise_speed)
        )

        checker_alignment = (
            (0.10 + 0.90 * orientation_score)
            * (0.20 + 0.80 * stillness_score)
        )
        return (
            position_score
            * checker_alignment
            * both_hands_near.float()
            * stage_mask.float()
        )


class StayNearAnchorReward(HumanoidBaseReward):
    """
    Stages 1, 2 and 4:
    Reward for keeping the pelvis near its anchor position.

    Output: <0, 1>, where 1 means no drift and 0 means the robot has
    drifted by ``max_xy_drift`` or more. Use with a positive weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        # Stage 3 deliberately remains excluded because the robot must move
        # while pulling. On entry to stage 4 a new pelvis anchor is captured.
        self.active_stages = [1, 2, 4]

        self.saved_positions_xy = None
        self.prev_stages = None
        self.robot_name_for_reset = robot_name
        self.max_xy_drift = 0.12

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name_for_reset]
        base_idx = robot.body_names.index("pelvis")
        current_xy = robot.body_state[:, base_idx, :2]

        if self.saved_positions_xy is None:
            self.saved_positions_xy = current_xy.clone()
        else:
            self.saved_positions_xy[env_ids] = current_xy[env_ids].clone()

        if self.actual_stage is not None:
            if self.prev_stages is None:
                self.prev_stages = self.actual_stage.clone()
            else:
                self.prev_stages[env_ids] = self.actual_stage[env_ids].clone()

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        base_idx = robot.body_names.index("pelvis")
        current_xy = robot.body_state[:, base_idx, :2]

        if self.saved_positions_xy is None:
            self.saved_positions_xy = current_xy.clone()
            self.prev_stages = self.actual_stage.clone()

        stage_changed = (self.actual_stage != self.prev_stages)
        update_mask = stage_changed & stage_mask

        if update_mask.any():
            self.saved_positions_xy[update_mask] = current_xy[update_mask].clone()

        self.prev_stages = self.actual_stage.clone()

        drift = torch.norm(current_xy - self.saved_positions_xy, dim=-1)
        reward = torch.clamp(
            1.0 - drift / self.max_xy_drift,
            min=0.0,
            max=1.0,
        )

        return reward * stage_mask.float()


# =============================================================================
# STAGE 2
# =============================================================================

class CloseGraspReward(HumanoidBaseReward):
    """
    Stage 2 dense reward for closing both hands around the chair.

    Targets use the deeper, collision-tested grasp pose from ``debug2``.  The
    old targets only partially bent the index and middle fingers, so they did
    not reliably reach the chair. Weakest-joint terms prevent a few closed
    fingers from hiding fingers that remain open. Signed progress and velocity
    terms distinguish movement toward closure from reopening before contact.

    Output: <-0.2, 1>. Negative values mean that fingers reopened in this step.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [2]

        self.progress_scale = 0.03
        self.velocity_scale = 1.5

        self.finger_targets_dict = {
            "left_hand_thumb_0_joint": 0.396,
            "left_hand_thumb_1_joint": 0.700,
            "left_hand_thumb_2_joint": 1.000,
            "left_hand_middle_0_joint": -1.500,
            "left_hand_middle_1_joint": -1.700,
            "left_hand_index_0_joint": -1.500,
            "left_hand_index_1_joint": -1.700,

            "right_hand_thumb_0_joint": -0.396,
            "right_hand_thumb_1_joint": -0.700,
            "right_hand_thumb_2_joint": -1.000,
            "right_hand_middle_0_joint": 1.500,
            "right_hand_middle_1_joint": 1.700,
            "right_hand_index_0_joint": 1.500,
            "right_hand_index_1_joint": 1.700,
        }

        self.finger_indices = None
        self.target_tensor = None
        self.prev_closure_per_joint = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_closure_per_joint is not None:
            self.prev_closure_per_joint[env_ids] = torch.nan

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        active_tensor = torch.tensor(self.active_stages, device=device)
        stage_mask = torch.isin(self.actual_stage, active_tensor)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.finger_indices is None:
            indices = []
            targets = []
            joint_names = list(robot.joint_names)

            for name, target_val in self.finger_targets_dict.items():
                if name in joint_names:
                    indices.append(joint_names.index(name))
                    targets.append(target_val)

            if len(indices) != len(self.finger_targets_dict):
                missing = [name for name in self.finger_targets_dict if name not in joint_names]
                raise ValueError(f"CloseGraspReward is missing finger joints: {missing}")

            self.finger_indices = torch.tensor(indices, device=device, dtype=torch.long)
            self.target_tensor = torch.tensor(
                targets, device=device, dtype=robot.joint_pos.dtype
            ).unsqueeze(0)

        q_finger = robot.joint_pos[:, self.finger_indices]
        target_magnitude = torch.clamp(torch.abs(self.target_tensor), min=0.1)
        closure_per_joint = torch.clamp(
            1.0 - torch.abs(q_finger - self.target_tensor) / target_magnitude,
            min=0.0,
            max=1.0,
        )
        left_closure = torch.mean(closure_per_joint[:, :7], dim=-1)
        right_closure = torch.mean(closure_per_joint[:, 7:], dim=-1)
        mean_closure = torch.mean(closure_per_joint, dim=-1)
        balanced_hands = torch.minimum(left_closure, right_closure)
        left_worst_three = torch.topk(
            closure_per_joint[:, :7], k=3, dim=-1, largest=False
        ).values.mean(dim=-1)
        right_worst_three = torch.topk(
            closure_per_joint[:, 7:], k=3, dim=-1, largest=False
        ).values.mean(dim=-1)
        balanced_worst_three = torch.minimum(left_worst_three, right_worst_three)
        weakest_joint = torch.min(closure_per_joint, dim=-1).values
        closure_state = (
            0.15 * mean_closure
            + 0.20 * balanced_hands
            + 0.30 * balanced_worst_three
            + 0.35 * weakest_joint
        )

        if (
            self.prev_closure_per_joint is None
            or self.prev_closure_per_joint.shape != closure_per_joint.shape
            or self.prev_closure_per_joint.device != device
        ):
            self.prev_closure_per_joint = closure_per_joint.detach().clone()
            signed_progress_per_joint = torch.zeros_like(closure_per_joint)
        else:
            previous = torch.where(
                torch.isnan(self.prev_closure_per_joint),
                closure_per_joint,
                self.prev_closure_per_joint,
            )
            signed_progress_per_joint = torch.clamp(
                (closure_per_joint - previous) / self.progress_scale,
                min=-1.0,
                max=1.0,
            )
            self.prev_closure_per_joint = torch.where(
                stage_mask.unsqueeze(-1),
                closure_per_joint.detach(),
                self.prev_closure_per_joint,
            )

        progress_mean = torch.mean(signed_progress_per_joint, dim=-1)
        progress_worst_three = torch.topk(
            signed_progress_per_joint, k=3, dim=-1, largest=False
        ).values.mean(dim=-1)
        progress_weakest = torch.min(signed_progress_per_joint, dim=-1).values
        signed_progress = (
            0.50 * progress_mean
            + 0.30 * progress_worst_three
            + 0.20 * progress_weakest
        )

        # Give an immediate signed signal for physically moving each joint in
        # the correct direction. The remaining-error gate removes the incentive
        # to oscillate once a joint has reached its target.
        dq_finger = robot.joint_vel[:, self.finger_indices]
        toward_target = torch.sign(self.target_tensor - q_finger) * dq_finger
        remaining_error = 1.0 - closure_per_joint
        signed_motion_per_joint = torch.clamp(
            toward_target / self.velocity_scale, min=-1.0, max=1.0
        ) * remaining_error
        signed_motion = torch.mean(signed_motion_per_joint, dim=-1)

        # Stage 2 starts only after both palms pass the reach checker. A sharp
        # palm-proximity multiplier previously erased the finger gradient after
        # even small whole-body sway; hand-retention rewards remain active.
        auxiliary_gate = 1.0 - closure_state
        total_reward = (
            closure_state
            + auxiliary_gate * (
                0.12 * signed_progress
                + 0.08 * signed_motion
            )
        )
        return torch.clamp(total_reward, min=-0.2, max=1.0) * stage_mask.float()


class GraspForceReward(HumanoidBaseReward):
    """
    Stage 2:
    Dense fingertip contact reward aligned with the checker force threshold.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

        self.active_stages = [2]
        self.force_threshold = 0.5

        self.tip_map = {
            "left_hand_thumb_2": 0,
            "left_hand_index_1": 1,
            "left_hand_middle_1": 2,
            "right_hand_thumb_2": 3,
            "right_hand_index_1": 4,
            "right_hand_middle_1": 5,
        }

        self.base_idx_to_tip = None
        self.chair_ids = None
        self.num_bodies = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        contact_data = robot.contact
        if contact_data is None:
            return torch.zeros(num_envs, device=device)

        if self.base_idx_to_tip is None:
            global_map = states.extras.get("global_link_map", {})
            num_bodies = states.extras.get("num_bodies_per_env", 1000)

            idx_to_tip = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            chair_ids = []

            for idx, (o_name, l_name) in global_map.items():
                if o_name == robot_name:
                    for tip_name, tip_id in self.tip_map.items():
                        if tip_name in l_name:
                            idx_to_tip[idx] = tip_id
                elif o_name == "chair":
                    chair_ids.append(idx)

            self.base_idx_to_tip = idx_to_tip
            self.chair_ids = torch.tensor(chair_ids, device=device, dtype=torch.long)
            self.num_bodies = num_bodies

        link_a = contact_data["link_a"]
        if link_a.shape[1] == 0:
            return torch.zeros(num_envs, device=device)

        link_b = contact_data["link_b"]
        valid_mask = contact_data["valid_mask"]

        forces = contact_data.get("force_b", contact_data.get("force", None))
        if forces is None:
            forces = torch.zeros((*link_a.shape, 3), device=device)

        force_mags = torch.norm(forces, dim=-1)

        base_a = link_a % self.num_bodies
        base_b = link_b % self.num_bodies

        a_is_chair = torch.isin(base_a, self.chair_ids)
        b_is_chair = torch.isin(base_b, self.chair_ids)

        tip_a = self.base_idx_to_tip[base_a]
        tip_b = self.base_idx_to_tip[base_b]

        contact_tip = torch.where(
            b_is_chair,
            tip_a,
            torch.where(a_is_chair, tip_b, torch.tensor(-1, device=device))
        )

        valid_interaction = (contact_tip >= 0) & valid_mask

        tip_forces = torch.zeros((num_envs, 6), device=device)

        for tip_id in range(6):
            tip_mask = valid_interaction & (contact_tip == tip_id)
            tip_force_vals = force_mags * tip_mask.float()
            max_f, _ = torch.max(tip_force_vals, dim=1)
            tip_forces[:, tip_id] = max_f

        # Weak first contacts already provide a signal, but the largest part
        # of the reward is available only when both hands and every fingertip
        # satisfy the same condition as the stage checker.
        tip_rewards = torch.sqrt(
            torch.clamp(tip_forces / self.force_threshold, min=0.0, max=1.0)
        )
        left_reward = torch.mean(tip_rewards[:, 0:3], dim=1)
        right_reward = torch.mean(tip_rewards[:, 3:6], dim=1)
        all_tips_reward = torch.min(tip_rewards, dim=1).values
        total_reward = (
            0.25 * torch.mean(tip_rewards, dim=1)
            + 0.35 * torch.minimum(left_reward, right_reward)
            + 0.40 * all_tips_reward
        )

        return total_reward * stage_mask.float()


# =============================================================================
# STAGE 3
# =============================================================================

class MaintainAnyGraspReward(HumanoidBaseReward):
    """
    Stage 3:
    Signed constraint reward for retaining a robust bilateral grasp.

    A fully loaded grasp returns 0. Weak or missing contacts return values down
    to -1, so this term cannot be farmed by standing still with the chair.

    Output: <-1, 0>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.force_threshold = 0.5
        self.robust_force = 1.0

        self.tip_map = {
            "left_hand_thumb_2": (0, 0),
            "left_hand_index_1": (0, 1),
            "left_hand_middle_1": (0, 2),
            "right_hand_thumb_2": (1, 0),
            "right_hand_index_1": (1, 1),
            "right_hand_middle_1": (1, 2),
        }

        self.base_idx_to_hand = None
        self.base_idx_to_tip = None
        self.chair_ids = None
        self.num_bodies = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        contact_data = robot.contact
        if contact_data is None:
            return -stage_mask.float()

        if self.base_idx_to_hand is None:
            global_map = states.extras.get("global_link_map", {})
            num_bodies = states.extras.get("num_bodies_per_env", 1000)

            idx_to_hand = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            idx_to_tip = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            chair_ids = []

            for idx, (o_name, l_name) in global_map.items():
                if o_name == robot_name:
                    for tip_name, (hand_id, tip_id) in self.tip_map.items():
                        if tip_name in l_name:
                            idx_to_hand[idx] = hand_id
                            idx_to_tip[idx] = tip_id
                elif o_name == "chair":
                    chair_ids.append(idx)

            self.base_idx_to_hand = idx_to_hand
            self.base_idx_to_tip = idx_to_tip
            self.chair_ids = torch.tensor(chair_ids, device=device, dtype=torch.long)
            self.num_bodies = num_bodies

        link_a = contact_data["link_a"]
        if link_a.shape[1] == 0 or self.chair_ids.numel() == 0:
            return -stage_mask.float()

        link_b = contact_data["link_b"]
        valid_mask = contact_data["valid_mask"]

        forces = contact_data.get("force_b", contact_data.get("force", None))
        if forces is None:
            forces = torch.zeros((*link_a.shape, 3), device=device)

        force_mags = torch.norm(forces, dim=-1)
        base_a = link_a % self.num_bodies
        base_b = link_b % self.num_bodies

        a_is_chair = torch.isin(base_a, self.chair_ids)
        b_is_chair = torch.isin(base_b, self.chair_ids)

        hand_a = self.base_idx_to_hand[base_a]
        hand_b = self.base_idx_to_hand[base_b]
        tip_a = self.base_idx_to_tip[base_a]
        tip_b = self.base_idx_to_tip[base_b]

        contact_hand = torch.where(
            b_is_chair,
            hand_a,
            torch.where(a_is_chair, hand_b, torch.tensor(-1, device=device)),
        )
        contact_tip = torch.where(
            b_is_chair,
            tip_a,
            torch.where(a_is_chair, tip_b, torch.tensor(-1, device=device)),
        )

        valid_interaction = (contact_hand >= 0) & (contact_tip >= 0) & valid_mask
        contact_rewards = torch.sqrt(
            torch.clamp(force_mags / self.robust_force, min=0.0, max=1.0)
        )

        best_tip_rewards = torch.zeros((num_envs, 2, 3), device=device)
        for hand_id in range(2):
            for tip_id in range(3):
                tip_mask = valid_interaction & (contact_hand == hand_id) & (contact_tip == tip_id)
                max_reward, _ = torch.max(contact_rewards * tip_mask.float(), dim=1)
                best_tip_rewards[:, hand_id, tip_id] = max_reward

        left_any = torch.max(best_tip_rewards[:, 0, :], dim=1)[0]
        right_any = torch.max(best_tip_rewards[:, 1, :], dim=1)[0]
        bilateral_any = torch.minimum(left_any, right_any)
        bilateral_coverage = torch.minimum(
            torch.mean(best_tip_rewards[:, 0, :], dim=1),
            torch.mean(best_tip_rewards[:, 1, :], dim=1),
        )
        grasp_quality = 0.75 * bilateral_any + 0.25 * bilateral_coverage
        penalty_reward = torch.clamp(grasp_quality, 0.0, 1.0) - 1.0
        return penalty_reward * stage_mask.float()


class Stage3HandDriftPenalty(HumanoidBaseReward):
    """Stage 3 penalty for sliding either palm away from its moving target.

    The checker terminates at 10 cm drift.  The penalty starts at 4 cm and is
    zero inside that safe region, avoiding another positive reward for waiting.

    Output: <0, 1>, use with a negative weight.
    """

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.safe_distance = 0.04
        self.failure_distance = 0.10
        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = self.actual_stage.to(device=device) == 3
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        left_hand = robot.body_state[:, robot.body_names.index(self.robot_left_hand), :3]
        right_hand = robot.body_state[:, robot.body_names.index(self.robot_right_hand), :3]
        left_target = chair.body_state[:, chair.body_names.index(self.chair_target_left), :3]
        right_target = chair.body_state[:, chair.body_names.index(self.chair_target_right), :3]
        max_distance = torch.maximum(
            torch.norm(left_hand - left_target, dim=-1),
            torch.norm(right_hand - right_target, dim=-1),
        )
        penalty = smoothstep01(
            (max_distance - self.safe_distance)
            / (self.failure_distance - self.safe_distance)
        )
        return penalty * stage_mask.float()


class PullChairReward(HumanoidBaseReward):
    """
    Stage 3:
    Signed reward for one uninterrupted pull from x=0.75 to x=-0.25.

    Forward progress and target-speed tracking are positive. Pausing on the
    path earns almost zero; moving in the wrong direction is negative. At the target the
    objective switches to stopping both the chair and robot.

    Output: <-1, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.initial_chair_pos = torch.tensor([0.75, 0.0, 0.1])
        self.target_chair_pos = torch.tensor([-0.25, 0.0, 0.1])
        self.pull_distance = 1.0
        self.target_pull_speed = 0.35
        self.vel_sigma = 0.12
        self.brake_fraction = 0.20
        self.progress_step_scale = 0.004
        self.path_tolerance = 0.30
        self.lateral_speed_scale = 0.15
        self.prev_progress = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_progress is not None:
            self.prev_progress[env_ids] = torch.nan

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        initial = self.initial_chair_pos.to(device)
        target = self.target_chair_pos.to(device)
        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_vel = chair.body_state[:, chair_idx, 7:10]

        pulled_x = initial[0] - chair_pos[:, 0]
        progress = torch.clamp(pulled_x / self.pull_distance, min=0.0, max=1.0)

        if (
            self.prev_progress is None
            or self.prev_progress.shape != progress.shape
            or self.prev_progress.device != device
        ):
            self.prev_progress = progress.detach().clone()
            signed_progress = torch.zeros(num_envs, device=device)
        else:
            previous = torch.where(
                torch.isnan(self.prev_progress), progress, self.prev_progress
            )
            signed_progress = torch.clamp(
                (progress - previous) / self.progress_step_scale,
                min=-1.0,
                max=1.0,
            )
            self.prev_progress = torch.where(
                stage_mask, progress.detach(), self.prev_progress
            )

        remaining = torch.clamp(1.0 - progress, min=0.0, max=1.0)
        speed_factor = smoothstep01(remaining / self.brake_fraction)
        desired_speed = self.target_pull_speed * speed_factor
        pull_speed = -chair_vel[:, 0]
        speed_tracking = torch.exp(
            -torch.square(pull_speed - desired_speed)
            / (2.0 * self.vel_sigma ** 2)
        )
        required_motion_speed = torch.clamp(desired_speed, min=0.05)
        continuity = smoothstep01(
            torch.clamp(pull_speed, min=0.0) / required_motion_speed
        )

        lateral_error = torch.norm(chair_pos[:, 1:3] - target[1:3], dim=-1)
        path_quality = torch.clamp(
            1.0 - lateral_error / self.path_tolerance, min=0.0, max=1.0
        )
        lateral_speed_penalty = torch.clamp(
            torch.norm(chair_vel[:, 1:3], dim=-1) / self.lateral_speed_scale,
            min=0.0,
            max=1.0,
        )
        moving_reward = (
            0.65 * signed_progress
            + 0.20 * speed_tracking
            + 0.15 * continuity
        )
        moving_reward = (
            moving_reward * (0.50 + 0.50 * path_quality)
            - 0.20 * (1.0 - path_quality)
            - 0.15 * lateral_speed_penalty
        )

        chair_speed = torch.norm(chair_vel, dim=-1)
        chair_stop = torch.clamp(1.0 - chair_speed / 0.20, min=0.0, max=1.0)

        base_idx = robot.body_names.index("pelvis")
        robot_speed = torch.norm(robot.body_state[:, base_idx, 7:9], dim=-1)
        robot_stop = torch.clamp(1.0 - robot_speed / 0.20, min=0.0, max=1.0)
        target_error = torch.norm(chair_pos - target, dim=-1)
        target_quality = torch.clamp(
            1.0 - target_error / self.path_tolerance, min=0.0, max=1.0
        )
        stopping_reward = (
            2.0
            * target_quality
            * (0.60 * chair_stop + 0.40 * robot_stop)
            - 1.0
        )

        # Start braking only in the final centimetres. Before that, a pause is
        # explicitly worse than continuing the pull.
        ready_to_stop = progress >= 1.0
        reward = torch.where(ready_to_stop, stopping_reward, moving_reward)
        return torch.clamp(reward, min=-1.0, max=1.0) * stage_mask.float()


# =============================================================================
# STAGE 4 AND 5
# =============================================================================

class PulledChairStillnessReward(HumanoidBaseReward):
    """
    Stage 4 and 5:
    Penalty for moving the pulled chair or robot away from the final pose.

    This is zero at the desired stable state and approaches one at the same
    position/velocity limits used by the checker. Use with a negative weight.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [4, 5]
        self.target_chair_pos = torch.tensor([-0.25, 0.0, 0.1])

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        target = self.target_chair_pos.to(device)
        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_vel = chair.body_state[:, chair_idx, 7:10]

        base_idx = robot.body_names.index("pelvis")
        robot_vel = robot.body_state[:, base_idx, 7:10]

        position_violation = torch.clamp(
            torch.norm(chair_pos - target, dim=-1) / 0.40, min=0.0, max=1.0
        )
        chair_motion = torch.clamp(
            torch.norm(chair_vel, dim=-1) / 0.20, min=0.0, max=1.0
        )
        robot_motion = torch.clamp(
            torch.norm(robot_vel[:, :2], dim=-1) / 0.20, min=0.0, max=1.0
        )
        penalty = torch.maximum(
            position_violation, torch.maximum(chair_motion, robot_motion)
        )
        return penalty * stage_mask.float()


class ReleaseFingersReward(HumanoidBaseReward):
    """
    Stage 4:
    Signed progress reward for opening every finger after the chair is stopped.

    Output: <-1, 1>. Re-closing fingers is negative.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [4]
        self.finger_targets_dict = {
            "left_hand_thumb_0_joint": 0.396,
            "left_hand_thumb_1_joint": 0.700,
            "left_hand_thumb_2_joint": 1.000,
            "left_hand_middle_0_joint": -1.500,
            "left_hand_middle_1_joint": -1.700,
            "left_hand_index_0_joint": -1.500,
            "left_hand_index_1_joint": -1.700,
            "right_hand_thumb_0_joint": -0.396,
            "right_hand_thumb_1_joint": -0.700,
            "right_hand_thumb_2_joint": -1.000,
            "right_hand_middle_0_joint": 1.500,
            "right_hand_middle_1_joint": 1.700,
            "right_hand_index_0_joint": 1.500,
            "right_hand_index_1_joint": 1.700,
        }
        self.finger_indices = None
        self.closed_scale = None
        self.prev_openness = None
        self.open_threshold = 0.15
        self.goal_shaping_angle = 0.30
        self.progress_scale = 0.04
        self.velocity_scale = 1.5

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_openness is not None:
            self.prev_openness[env_ids] = torch.nan
        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.finger_indices is None:
            joint_names = list(robot.joint_names)
            missing = [name for name in self.finger_targets_dict if name not in joint_names]
            if missing:
                raise ValueError(f"ReleaseFingersReward is missing finger joints: {missing}")
            self.finger_indices = torch.tensor(
                [joint_names.index(name) for name in self.finger_targets_dict],
                device=device,
                dtype=torch.long,
            )
            self.closed_scale = torch.tensor(
                [abs(value) for value in self.finger_targets_dict.values()],
                device=device,
                dtype=robot.joint_pos.dtype,
            ).unsqueeze(0)

        q_fingers = robot.joint_pos[:, self.finger_indices]
        openness = torch.clamp(
            1.0 - torch.abs(q_fingers) / self.closed_scale, min=0.0, max=1.0
        )
        if (
            self.prev_openness is None
            or self.prev_openness.shape != openness.shape
            or self.prev_openness.device != device
        ):
            self.prev_openness = openness.detach().clone()
            signed_progress_per_joint = torch.zeros_like(openness)
        else:
            previous = torch.where(
                torch.isnan(self.prev_openness), openness, self.prev_openness
            )
            signed_progress_per_joint = torch.clamp(
                (openness - previous) / self.progress_scale, min=-1.0, max=1.0
            )
            self.prev_openness = torch.where(
                stage_mask.unsqueeze(-1), openness.detach(), self.prev_openness
            )

        progress_mean = torch.mean(signed_progress_per_joint, dim=-1)
        progress_worst_four = torch.topk(
            signed_progress_per_joint, k=4, dim=-1, largest=False
        ).values.mean(dim=-1)
        signed_progress = 0.60 * progress_mean + 0.40 * progress_worst_four

        dq_fingers = robot.joint_vel[:, self.finger_indices]
        toward_open = -torch.sign(q_fingers) * dq_fingers
        signed_motion = torch.mean(
            torch.clamp(toward_open / self.velocity_scale, min=-1.0, max=1.0)
            * (1.0 - openness),
            dim=-1,
        )
        max_finger_angle = torch.max(torch.abs(q_fingers), dim=-1)[0]
        goal_score = smoothstep01(
            (self.goal_shaping_angle - max_finger_angle)
            / (self.goal_shaping_angle - self.open_threshold)
        )
        reward = 0.55 * signed_progress + 0.25 * signed_motion + 0.20 * goal_score
        return torch.clamp(reward, min=-1.0, max=1.0) * stage_mask.float()


class ArmDownReward(HumanoidBaseReward):
    """
    Stage 5:
    Signed progress reward for placing both arms in the final resting pose.

    Output: <-1, 1>. Moving away from the resting pose is negative.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [5]
        self.arm_joint_scales = {
            "left_shoulder_pitch_joint": 2.5,
            "right_shoulder_pitch_joint": 2.5,
            "left_shoulder_roll_joint": 1.5,
            "right_shoulder_roll_joint": 1.5,
            "left_shoulder_yaw_joint": 1.5,
            "right_shoulder_yaw_joint": 1.5,
            "left_elbow_joint": 2.0,
            "right_elbow_joint": 2.0,
        }
        self.arm_indices = None
        self.scale_tensor = None
        self.prev_rest_scores = None
        self.rest_threshold = 0.35
        self.goal_shaping_angle = 0.55
        self.progress_scale = 0.035
        self.velocity_scale = 1.2

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_rest_scores is not None:
            self.prev_rest_scores[env_ids] = torch.nan
        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.arm_indices is None:
            joint_names = list(robot.joint_names)
            missing = [name for name in self.arm_joint_scales if name not in joint_names]
            if missing:
                raise ValueError(f"ArmDownReward is missing arm joints: {missing}")
            self.arm_indices = torch.tensor(
                [joint_names.index(name) for name in self.arm_joint_scales],
                device=device,
                dtype=torch.long,
            )
            self.scale_tensor = torch.tensor(
                list(self.arm_joint_scales.values()),
                device=device,
                dtype=robot.joint_pos.dtype,
            ).unsqueeze(0)

        q_arms = robot.joint_pos[:, self.arm_indices]
        rest_scores = torch.clamp(
            1.0 - torch.abs(q_arms) / self.scale_tensor, min=0.0, max=1.0
        )
        if (
            self.prev_rest_scores is None
            or self.prev_rest_scores.shape != rest_scores.shape
            or self.prev_rest_scores.device != device
        ):
            self.prev_rest_scores = rest_scores.detach().clone()
            signed_progress_per_joint = torch.zeros_like(rest_scores)
        else:
            previous = torch.where(
                torch.isnan(self.prev_rest_scores), rest_scores, self.prev_rest_scores
            )
            signed_progress_per_joint = torch.clamp(
                (rest_scores - previous) / self.progress_scale, min=-1.0, max=1.0
            )
            self.prev_rest_scores = torch.where(
                stage_mask.unsqueeze(-1), rest_scores.detach(), self.prev_rest_scores
            )

        progress_mean = torch.mean(signed_progress_per_joint, dim=-1)
        progress_worst_four = torch.topk(
            signed_progress_per_joint, k=4, dim=-1, largest=False
        ).values.mean(dim=-1)
        signed_progress = 0.60 * progress_mean + 0.40 * progress_worst_four

        dq_arms = robot.joint_vel[:, self.arm_indices]
        toward_rest = -torch.sign(q_arms) * dq_arms
        signed_motion = torch.mean(
            torch.clamp(toward_rest / self.velocity_scale, min=-1.0, max=1.0)
            * (1.0 - rest_scores),
            dim=-1,
        )
        max_arm_angle = torch.max(torch.abs(q_arms), dim=-1)[0]
        goal_score = smoothstep01(
            (self.goal_shaping_angle - max_arm_angle)
            / (self.goal_shaping_angle - self.rest_threshold)
        )
        reward = 0.55 * signed_progress + 0.25 * signed_motion + 0.20 * goal_score
        return torch.clamp(reward, min=-1.0, max=1.0) * stage_mask.float()


class KeepFingersOpenPenalty(HumanoidBaseReward):
    """Stage 5 penalty for closing fingers again while lowering the arms."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [5]
        self.finger_indices = None
        self.open_threshold = 0.15
        self.closed_angle = 0.50

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)
        stage_mask = self.actual_stage.to(device=device) == 5
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.finger_indices is None:
            indices = [
                i for i, name in enumerate(robot.joint_names)
                if any(token in name for token in ("thumb", "index", "middle"))
            ]
            if not indices:
                return torch.zeros(num_envs, device=device)
            self.finger_indices = torch.tensor(indices, dtype=torch.long, device=device)

        max_angle = torch.max(
            torch.abs(robot.joint_pos[:, self.finger_indices]), dim=-1
        ).values
        penalty = smoothstep01(
            (max_angle - self.open_threshold)
            / (self.closed_angle - self.open_threshold)
        )
        return penalty * stage_mask.float()


# =============================================================================
# OPTIONAL / DISABLED REWARDS
# =============================================================================

class ArmRestingPosePenaltyCfg(HumanoidBaseReward):
    """
    Optional penalty magnitude for arm resting pose in stage 0.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.dof_indices = None
        self.q_lower_tensor = None
        self.q_upper_tensor = None

        self.resting_limits: dict[str, tuple[float, float]] = {
            "left_shoulder_pitch_joint": (-0.3, 0.3),
            "right_shoulder_pitch_joint": (-0.3, 0.3),
            "left_shoulder_roll_joint": (-0.1, 0.1),
            "right_shoulder_roll_joint": (-0.1, 0.1),
            "left_shoulder_yaw_joint": (-0.1, 0.1),
            "right_shoulder_yaw_joint": (-0.1, 0.1),
            "left_elbow_joint": (-0.1, 0.3),
            "right_elbow_joint": (-0.1, 0.3),
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device
        num_envs = joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.dof_indices is None:
            self.dof_indices = []
            lower_vals = []
            upper_vals = []

            for i, name in enumerate(robot.joint_names):
                if name in self.resting_limits:
                    self.dof_indices.append(i)
                    low, high = self.resting_limits[name]
                    lower_vals.append(low)
                    upper_vals.append(high)

            if not self.dof_indices:
                return torch.zeros(num_envs, device=device)

            self.dof_indices = torch.tensor(self.dof_indices, device=device, dtype=torch.long)
            self.q_lower_tensor = torch.tensor(lower_vals, device=device).unsqueeze(0)
            self.q_upper_tensor = torch.tensor(upper_vals, device=device).unsqueeze(0)

        q_active = joint_pos[:, self.dof_indices]
        violation_lower = torch.clamp(self.q_lower_tensor - q_active, min=0.0)
        violation_upper = torch.clamp(q_active - self.q_upper_tensor, min=0.0)
        total_violation = violation_lower + violation_upper

        mean_violation = torch.mean(total_violation, dim=-1)
        penalty = torch.clamp(mean_violation / 0.25, min=0.0, max=1.0)

        return penalty * stage_mask.float()
class StageProgressCfg(HumanoidBaseReward):
    """
    One-shot reward for completing a stage.

    The checker increments ``actual_stage`` first, therefore transition k->k+1
    returns k+1 once and zero on all following steps.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # Pokud není actual_stage inicializováno, vrátíme 0
        if self.actual_stage is None or self.completed_stages is None:
            robot = states.robots[robot_name]
            return torch.zeros(robot.joint_pos.shape[0], device=robot.joint_pos.device)

        if self.completed_stages.any():
            ret = self.completed_stages.float() * self.actual_stage.float()
            self.completed_stages.zero_()
            return ret
        else:
            return torch.zeros_like(self.completed_stages)
class ContinuousStageReward(HumanoidBaseReward):
    """
    Continuous Stage Reward: Dává permanentní odměnu za to, ve kterém Stage se robot nachází.
    Stage 0 = 0 bodů
    Stage 1 = 1 * váha
    Stage 2 = 2 * váha
    ... atd.

    Tímto robotovi jasně říkáme, že udržet se v pozdějších fázích je matematicky
    nejvýhodnější věc v celé hře.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # 1. Ochrana pro úplně první krok, kdy stage ještě nemusí být zinicializován
        if self.actual_stage is None:
            robot = states.robots[robot_name]
            num_envs = robot.joint_pos.shape[0]
            device = robot.joint_pos.device
            return torch.zeros(num_envs, device=device)

        # 2. Jednoduše vrátíme aktuální číslo stage (0, 1, 2, 3...)
        # Váš framework (Metasim/Gym wrapper) tuto hodnotu následně
        # automaticky vynásobí váhou, kterou máte definovanou v configu.
        return self.actual_stage.float()
# =============================================================================
# BODY REWARDS FOR ONE POLICY THROUGH THE COMPLETE TASK
# Independent copies of the separate/multi calculations. Stage masks select
# the current physical objective; they do not split the policy or its return.
# =============================================================================

_ARM_TARGETS = {
    "left": {
        "left_shoulder_pitch_joint": 0.28,
        "left_shoulder_roll_joint": 0.35,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.77,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
    },
    "right": {
        "right_shoulder_pitch_joint": 0.28,
        "right_shoulder_roll_joint": -0.35,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.77,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
}

_FINGER_TARGETS = {
    "left": {
        "left_hand_thumb_0_joint": 0.396,
        "left_hand_thumb_1_joint": 0.700,
        "left_hand_thumb_2_joint": 1.000,
        "left_hand_middle_0_joint": -1.500,
        "left_hand_middle_1_joint": -1.700,
        "left_hand_index_0_joint": -1.500,
        "left_hand_index_1_joint": -1.700,
    },
    "right": {
        "right_hand_thumb_0_joint": -0.396,
        "right_hand_thumb_1_joint": -0.700,
        "right_hand_thumb_2_joint": -1.000,
        "right_hand_middle_0_joint": 1.500,
        "right_hand_middle_1_joint": 1.700,
        "right_hand_index_0_joint": 1.500,
        "right_hand_index_1_joint": 1.700,
    },
}

_STAGE2_FINGER_TARGETS = STAGE2_FINGER_JOINT_TARGETS_BY_SIDE


class Stage0ReferenceVelocityReward(HumanoidBaseReward):
    """Reward motion toward the final chair pose and stopping at the target.

    Farther than ``slowdown_distance`` from the target, the reference speed is
    constant.  During the last ten centimetres it decreases linearly to zero,
    so matching the same vector teaches both approach direction and braking.

    The Gaussian tracking term alone is nearly zero for both standing still and
    walking in the wrong direction.  A signed progress term therefore rewards
    velocity toward the target and explicitly penalizes velocity away from it.
    Its influence fades inside the slowdown zone, where velocity tracking must
    instead reward the robot for coming to a complete stop.
    """

    def __init__(
        self,
        robot_name="g1_with_hands",
        target_speed=0.5,
        slowdown_distance=0.10,
        velocity_sigma=0.25,
        progress_weight=0.5,
    ):
        super().__init__(robot_name)
        if not math.isfinite(target_speed) or target_speed < 0:
            raise ValueError("target_speed must be finite and non-negative")
        if not math.isfinite(slowdown_distance) or slowdown_distance <= 0:
            raise ValueError("slowdown_distance must be finite and positive")
        if not math.isfinite(velocity_sigma) or velocity_sigma <= 0:
            raise ValueError("velocity_sigma must be finite and positive")
        if (
            not math.isfinite(progress_weight)
            or not 0.0 <= progress_weight <= 1.0
        ):
            raise ValueError("progress_weight must be in [0, 1]")
        self.active_stage = 0
        self.final_distance = CHAIR_FINAL_DISTANCE
        self.target_speed = float(target_speed)
        self.slowdown_distance = float(slowdown_distance)
        self.velocity_sigma = float(velocity_sigma)
        self.progress_weight = float(progress_weight)

    def __call__(
        self, states: list["EnvState"], robot_name: str = None
    ) -> torch.FloatTensor:
        robot = states.robots[robot_name or self.robot_name]
        device = robot.body_state.device
        dtype = robot.body_state.dtype
        num_envs = robot.body_state.shape[0]
        if self.actual_stage is None:
            return torch.zeros(num_envs, dtype=dtype, device=device)

        pelvis_idx = robot.body_names.index("pelvis")
        pelvis_state = robot.body_state[:, pelvis_idx]
        chair = states.objects["chair"]
        chair_idx = chair.body_names.index("base_link")
        chair_state = chair.body_state[:, chair_idx]

        target_xy = (
            chair_state[:, :2]
            + self.final_distance
            * chair_back_direction_xy(chair_state[:, 3:7])
        )
        target_delta = target_xy - pelvis_state[:, :2]
        target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
        target_direction = target_delta / torch.clamp(
            target_distance.unsqueeze(-1), min=1.0e-6
        )

        speed_fraction = torch.clamp(
            target_distance / self.slowdown_distance, min=0.0, max=1.0
        )
        reference_velocity = (
            target_direction
            * (self.target_speed * speed_fraction).unsqueeze(-1)
        )
        velocity_error = torch.linalg.vector_norm(
            pelvis_state[:, 7:9] - reference_velocity, dim=-1
        )
        tracking_reward = torch.exp(
            -0.5 * torch.square(velocity_error / self.velocity_sigma)
        )

        # Signed speed along the target direction.  Unlike the Gaussian term,
        # this distinguishes moving toward the target from moving away from it.
        toward_speed = torch.sum(
            pelvis_state[:, 7:9] * target_direction, dim=-1
        )
        if self.target_speed > 0.0:
            signed_progress = torch.clamp(
                toward_speed / self.target_speed, min=-1.0, max=1.0
            )
        else:
            signed_progress = torch.zeros_like(toward_speed)

        # Far from the target, combine velocity tracking with signed progress.
        # At the target speed_fraction is zero, leaving pure tracking reward so
        # that standing still at the final pose still receives reward 1.
        progress_mix = self.progress_weight * speed_fraction
        reward = (
            (1.0 - progress_mix) * tracking_reward
            + progress_mix * signed_progress
        )
        reward = torch.nan_to_num(reward, nan=-1.0, posinf=1.0, neginf=-1.0)
        stage_mask = self.actual_stage.to(device=device) == self.active_stage
        return reward * stage_mask.to(dtype=dtype)

class PelvisFacingChairReward(HumanoidBaseReward):
    """Reward the pelvis forward axis for pointing directly at the chair.

    The reward is continuous with no angular tolerance: it is +1 at zero
    heading error, zero at ``zero_reward_angle_degrees`` and approaches -1
    when the robot turns sideways or backwards relative to the chair.
    """

    def __init__(self, robot_name="g1_with_hands",
                 zero_reward_angle_degrees=30.0):
        super().__init__(robot_name)
        if not 0.0 < zero_reward_angle_degrees < 180.0:
            raise ValueError("zero_reward_angle_degrees must be in (0, 180)")
        self.zero_reward_angle = math.radians(zero_reward_angle_degrees)

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        chair = states.objects["chair"]
        device = robot.body_state.device
        num_envs = robot.body_state.shape[0]
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        pelvis_idx = robot.body_names.index("pelvis")
        chair_idx = chair.body_names.index("base_link")
        pelvis_xy = robot.body_state[:, pelvis_idx, :2]
        pelvis_quat = robot.body_state[:, pelvis_idx, 3:7]
        to_chair = chair.body_state[:, chair_idx, :2] - pelvis_xy
        distance = torch.linalg.vector_norm(to_chair, dim=-1)
        chair_direction = to_chair / distance.clamp_min(1.0e-6).unsqueeze(-1)
        pelvis_forward = forward_direction_xy(pelvis_quat)

        alignment = torch.sum(pelvis_forward * chair_direction, dim=-1)
        cross = (pelvis_forward[:, 0] * chair_direction[:, 1]
                 - pelvis_forward[:, 1] * chair_direction[:, 0])
        heading_error = torch.atan2(torch.abs(cross), alignment)
        reward = 2.0 * torch.exp(
            -math.log(2.0) * torch.square(
                heading_error / self.zero_reward_angle)) - 1.0
        reward = torch.where(distance > 1.0e-6, reward, torch.zeros_like(reward))
        reward = torch.nan_to_num(reward, nan=-1.0, posinf=1.0, neginf=-1.0)
        active = (self.actual_stage >= 0) & (self.actual_stage <= 5)
        return reward * active.to(device=device, dtype=reward.dtype)

class KeepChairStillPenaltyWalk(HumanoidBaseReward):
    """
    Stage 0 and 1:
    Penalty magnitude for moving the chair before grasp.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.lin_scale_stage0 = 0.08
        self.ang_scale_stage0 = 0.30
        self.lin_scale_stage1 = 0.20
        self.ang_scale_stage1 = 0.70

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        chair_base_idx = chair.body_names.index("base_link")
        chair_lin_vel = chair.body_state[:, chair_base_idx, 7:10]
        chair_ang_vel = chair.body_state[:, chair_base_idx, 10:13]

        lin_norm = torch.norm(chair_lin_vel, dim=-1)
        ang_norm = torch.norm(chair_ang_vel, dim=-1)

        lin_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.lin_scale_stage0, device=device),
            torch.full((num_envs,), self.lin_scale_stage1, device=device),
        )
        ang_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.ang_scale_stage0, device=device),
            torch.full((num_envs,), self.ang_scale_stage1, device=device),
        )

        lin_penalty = torch.clamp(lin_norm / lin_scale, min=0.0, max=1.0)
        ang_penalty = torch.clamp(ang_norm / ang_scale, min=0.0, max=1.0)

        penalty = 0.7 * lin_penalty + 0.3 * ang_penalty
        return penalty * stage_mask.float()

class KeepChairStillPenaltyArms(HumanoidBaseReward):
    """
    Stage 0 and 1:
    Penalty magnitude for moving the chair before grasp.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [1]
        self.lin_scale_stage0 = 0.08
        self.ang_scale_stage0 = 0.30
        self.lin_scale_stage1 = 0.20
        self.ang_scale_stage1 = 0.70

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        chair_base_idx = chair.body_names.index("base_link")
        chair_lin_vel = chair.body_state[:, chair_base_idx, 7:10]
        chair_ang_vel = chair.body_state[:, chair_base_idx, 10:13]

        lin_norm = torch.norm(chair_lin_vel, dim=-1)
        ang_norm = torch.norm(chair_ang_vel, dim=-1)

        lin_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.lin_scale_stage0, device=device),
            torch.full((num_envs,), self.lin_scale_stage1, device=device),
        )
        ang_scale = torch.where(
            self.actual_stage == 0,
            torch.full((num_envs,), self.ang_scale_stage0, device=device),
            torch.full((num_envs,), self.ang_scale_stage1, device=device),
        )

        lin_penalty = torch.clamp(lin_norm / lin_scale, min=0.0, max=1.0)
        ang_penalty = torch.clamp(ang_norm / ang_scale, min=0.0, max=1.0)

        penalty = 0.7 * lin_penalty + 0.3 * ang_penalty
        return penalty * stage_mask.float()

class _SideReward(HumanoidBaseReward):
    side = None

    @property
    def hand_name(self):
        return "left_endeffector" if self.side == "left" else "endeffector"

    @property
    def target_name(self):
        return "target_hand_left" if self.side == "left" else "target_hand_right"

    def active_mask(self, device, stages):
        if self.actual_stage is None:
            return None
        return _stage_mask(self.actual_stage.to(device=device), stages)

class _SideStage0ArmPoseReward(_SideReward):
    """Reward only this arm for matching its seven-joint stage-0 pose."""

    def __init__(self, robot_name="g1_with_hands", sigma=0.18):
        super().__init__(robot_name)
        self.sigma = float(sigma)
        self.indices = None
        self.targets = None

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_pos.device
        mask = self.active_mask(device, 0)
        if mask is None:
            return torch.zeros(robot.joint_pos.shape[0], device=device)
        if self.indices is None:
            names = list(robot.joint_names)
            target_map = _ARM_TARGETS[self.side]
            missing = [name for name in target_map if name not in names]
            if missing:
                raise ValueError(f"{type(self).__name__} missing joints: {missing}")
            ordered = list(target_map)
            self.indices = torch.tensor([names.index(name) for name in ordered],
                                        dtype=torch.long, device=device)
            self.targets = torch.tensor([target_map[name] for name in ordered],
                                        dtype=robot.joint_pos.dtype,
                                        device=device).unsqueeze(0)
        error = robot.joint_pos.index_select(1, self.indices) - self.targets
        reward = torch.exp(-torch.mean(torch.square(error / self.sigma), dim=-1))
        return reward * mask.float()

class LeftStage0ArmPoseReward(_SideStage0ArmPoseReward):
    side = "left"

class RightStage0ArmPoseReward(_SideStage0ArmPoseReward):
    side = "right"

class _SideHandDistanceReward(_SideReward):
    """Independent hand-to-target distance reward in stages 1--4."""

    def __init__(self, robot_name="g1_with_hands", distance_scale=0.20):
        super().__init__(robot_name)
        self.distance_scale = float(distance_scale)

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        chair = states.objects["chair"]
        device = robot.body_state.device
        mask = self.active_mask(device, (1, 2, 3, 4))
        if mask is None:
            return torch.zeros(robot.body_state.shape[0], device=device)
        hand = robot.body_state[:, robot.body_names.index(self.hand_name), :3]
        target = chair.body_state[:, chair.body_names.index(self.target_name), :3]
        reward = torch.exp(
            -torch.linalg.vector_norm(hand - target, dim=-1) / self.distance_scale)
        return reward * mask.float()

class LeftHandDistanceReward(_SideHandDistanceReward):
    side = "left"

class RightHandDistanceReward(_SideHandDistanceReward):
    side = "right"

class _SideHandOrientationReward(_SideReward):
    """Independent target-orientation reward in stages 1--4."""

    def __init__(self, robot_name="g1_with_hands", error_scale=0.05):
        super().__init__(robot_name)
        self.error_scale = float(error_scale)

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        chair = states.objects["chair"]
        device = robot.body_state.device
        mask = self.active_mask(device, (1, 2, 3, 4))
        if mask is None:
            return torch.zeros(robot.body_state.shape[0], device=device)
        hand_q = torch.nn.functional.normalize(
            robot.body_state[:, robot.body_names.index(self.hand_name), 3:7], dim=-1)
        target_q = torch.nn.functional.normalize(
            chair.body_state[:, chair.body_names.index(self.target_name), 3:7], dim=-1)
        error = 1.0 - torch.abs(torch.sum(hand_q * target_q, dim=-1))
        return torch.exp(-error / self.error_scale) * mask.float()

class LeftHandOrientationReward(_SideHandOrientationReward):
    side = "left"

class RightHandOrientationReward(_SideHandOrientationReward):
    side = "right"

class _SideArmVelocityPenalty(_SideReward):
    """Penalize excessive speed of this arm only in stages 1, 2 and 3."""

    def __init__(self, robot_name="g1_with_hands", speed_limit=1.5,
                 full_penalty_speed=3.0):
        super().__init__(robot_name)
        self.speed_limit = float(speed_limit)
        self.full_penalty_speed = float(full_penalty_speed)
        self.indices = None

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_vel.device
        mask = self.active_mask(device, (1, 2, 3))
        if mask is None:
            return torch.zeros(robot.joint_vel.shape[0], device=device)
        if self.indices is None:
            wanted = _ARM_TARGETS[self.side]
            self.indices = torch.tensor(
                [i for i, name in enumerate(robot.joint_names) if name in wanted],
                dtype=torch.long, device=device)
        speed = robot.joint_vel.index_select(1, self.indices).abs().amax(dim=-1)
        penalty = torch.clamp(
            (speed - self.speed_limit)
            / (self.full_penalty_speed - self.speed_limit), 0.0, 1.0)
        return penalty * mask.float()

class LeftArmVelocityPenalty(_SideArmVelocityPenalty):
    side = "left"

class RightArmVelocityPenalty(_SideArmVelocityPenalty):
    side = "right"

class _SideStage3HandDriftPenalty(_SideReward):
    """Penalize one palm sliding away from its moving target in stage 3."""

    def __init__(self, robot_name="g1_with_hands", safe_distance=0.04,
                 failure_distance=0.10):
        super().__init__(robot_name)
        self.safe_distance = float(safe_distance)
        self.failure_distance = float(failure_distance)
        if not 0.0 <= self.safe_distance < self.failure_distance:
            raise ValueError(
                "safe_distance must be non-negative and below failure_distance"
            )

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        chair = states.objects["chair"]
        device = robot.body_state.device
        mask = self.active_mask(device, 3)
        if mask is None:
            return torch.zeros(robot.body_state.shape[0], device=device)
        hand = robot.body_state[:, robot.body_names.index(self.hand_name), :3]
        target = chair.body_state[:, chair.body_names.index(self.target_name), :3]
        distance = torch.linalg.vector_norm(hand - target, dim=-1)
        penalty = smoothstep01(
            (distance - self.safe_distance)
            / (self.failure_distance - self.safe_distance)
        )
        return penalty * mask.to(dtype=penalty.dtype)

class LeftStage3HandDriftPenalty(_SideStage3HandDriftPenalty):
    side = "left"

class RightStage3HandDriftPenalty(_SideStage3HandDriftPenalty):
    side = "right"

class CenteredUpperBodyCOMReward(HumanoidBaseReward):
    """Exponentially reward the upper-body COM directly above the pelvis.

    The horizontal error has no dead zone: every non-zero displacement lowers
    the reward. ``sigma`` controls the width of the Gaussian reward curve.
    """

    def __init__(self, robot_name="g1_with_hands", sigma=0.05):
        super().__init__(robot_name)
        self.sigma = float(sigma)
        if self.sigma <= 0.0:
            raise ValueError("CenteredUpperBodyCOMReward sigma must be positive")

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        pelvis_idx = robot.body_names.index("pelvis")
        upper_com = chairman_geometry.upper_body_center_of_mass(
            robot, str(chairman_geometry.G1_WITH_HANDS_URDF))
        horizontal_error = upper_com[:, :2] - robot.body_state[:, pelvis_idx, :2]
        squared_distance = horizontal_error.square().sum(dim=-1)
        reward = torch.exp(-0.5 * squared_distance / (self.sigma * self.sigma))
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        if self.actual_stage is None:
            return torch.zeros_like(reward)
        active = (self.actual_stage >= 0) & (self.actual_stage <= 5)
        return reward * active.to(reward.dtype)

class FaceChairAllStagesReward(FaceChairReward):
    """Face the chair in every currently trained stage."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [0, 1, 2, 3, 4, 5]

class WaistVelocityPenalty(HumanoidBaseReward):
    """Penalize excessive physical velocity of the three waist joints."""

    def __init__(self, robot_name="g1_with_hands", speed_limit=0.8,
                 full_penalty_speed=2.0):
        super().__init__(robot_name)
        self.speed_limit = float(speed_limit)
        self.full_penalty_speed = float(full_penalty_speed)
        self.indices = None

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_vel.device
        if self.actual_stage is None:
            return torch.zeros(robot.joint_vel.shape[0], device=device)
        if self.indices is None:
            indices = [i for i, name in enumerate(robot.joint_names)
                       if name.startswith("waist_")]
            self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        speed = robot.joint_vel.index_select(1, self.indices).abs().amax(dim=-1)
        return torch.clamp(
            (speed - self.speed_limit)
            / (self.full_penalty_speed - self.speed_limit), 0.0, 1.0)

class ApproachAndStandReward(Stage0ReferenceVelocityReward):
    """Use the approach/stop velocity target throughout stages 0, 1 and 2."""

    def __call__(self, states, robot_name=None):
        original_stage = self.actual_stage
        reward = super().__call__(states, robot_name)
        if original_stage is None:
            return reward
        active = _stage_mask(original_stage.to(device=reward.device), (0, 1, 2))
        # The parent masks stage 0. Recompute only its final mask by dividing
        # out that mask is unsafe, so temporarily evaluate active rows as stage 0.
        if not ((original_stage == 1) | (original_stage == 2)).any():
            return reward
        try:
            self.actual_stage = torch.where(
                active, torch.zeros_like(original_stage), original_stage)
            return super().__call__(states, robot_name) * active.float()
        finally:
            self.actual_stage = original_stage

class Stage3ReverseVelocityReward(HumanoidBaseReward):
    """Track the backward velocity that carries the chair to its stage-3 target."""

    def __init__(self, robot_name="g1_with_hands", target_speed=0.35,
                 slowdown_distance=0.20, velocity_sigma=0.12):
        super().__init__(robot_name)
        self.target_speed = float(target_speed)
        self.slowdown_distance = float(slowdown_distance)
        self.velocity_sigma = float(velocity_sigma)
        self.target_chair_xy = torch.tensor([-0.25, 0.0])

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        chair = states.objects["chair"]
        device = robot.body_state.device
        if self.actual_stage is None:
            return torch.zeros(robot.body_state.shape[0], device=device)
        mask = self.actual_stage.to(device=device) == 3
        pelvis = robot.body_state[:, robot.body_names.index("pelvis")]
        chair_state = chair.body_state[:, chair.body_names.index("base_link")]
        delta = self.target_chair_xy.to(device) - chair_state[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / torch.clamp(distance.unsqueeze(-1), min=1.0e-6)
        speed = self.target_speed * torch.clamp(
            distance / self.slowdown_distance, 0.0, 1.0)
        reference_velocity = direction * speed.unsqueeze(-1)
        error = torch.linalg.vector_norm(
            pelvis[:, 7:9] - reference_velocity, dim=-1)
        reward = torch.exp(-0.5 * torch.square(error / self.velocity_sigma))
        return reward * mask.float()

class _SideOpenFingersReward(_SideReward):
    """Reward this hand for open, calm fingers in stages 0 and 1."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.indices = None

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_pos.device
        mask = self.active_mask(device, (0, 1))
        if mask is None:
            return torch.zeros(robot.joint_pos.shape[0], device=device)
        if self.indices is None:
            prefix = f"{self.side}_hand_"
            self.indices = torch.tensor(
                [i for i, name in enumerate(robot.joint_names)
                 if name.startswith(prefix)], dtype=torch.long, device=device)
        position = robot.joint_pos.index_select(1, self.indices)
        velocity = robot.joint_vel.index_select(1, self.indices)
        reward = (0.8 * torch.exp(-position.abs().mean(dim=-1) / 0.35)
                  + 0.2 * torch.exp(-velocity.abs().mean(dim=-1) / 1.0))
        return reward * mask.float()

class LeftOpenFingersReward(_SideOpenFingersReward):
    side = "left"

class RightOpenFingersReward(_SideOpenFingersReward):
    side = "right"

class _SideStage2FingerJointPositionReward(_SideReward):
    """Keep one hand in its complete grasp pose during stages 2 and 3."""

    def __init__(self, robot_name="g1_with_hands", error_scale=0.25):
        super().__init__(robot_name)
        if error_scale <= 0.0:
            raise ValueError("error_scale must be positive")
        self.error_scale = float(error_scale)
        self.indices = None
        self.targets = None

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_pos.device
        mask = self.active_mask(device, (2, 3))
        if mask is None:
            return torch.zeros(robot.joint_pos.shape[0], device=device)

        if self.indices is None:
            joint_names = list(robot.joint_names)
            target_map = _STAGE2_FINGER_TARGETS[self.side]
            missing = [name for name in target_map if name not in joint_names]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} is missing finger joints: {missing}"
                )
            ordered_names = list(target_map)
            self.indices = torch.tensor(
                [joint_names.index(name) for name in ordered_names],
                dtype=torch.long,
                device=device,
            )
            self.targets = torch.tensor(
                [target_map[name] for name in ordered_names],
                dtype=robot.joint_pos.dtype,
                device=device,
            ).unsqueeze(0)

        positions = robot.joint_pos.index_select(1, self.indices)
        mean_joint_error = torch.mean(torch.abs(positions - self.targets), dim=-1)
        reward = torch.exp(-mean_joint_error / self.error_scale)
        return reward * mask.to(dtype=reward.dtype)

class LeftStage2FingerJointPositionReward(_SideStage2FingerJointPositionReward):
    side = "left"

class RightStage2FingerJointPositionReward(_SideStage2FingerJointPositionReward):
    side = "right"

class _SideReleaseFingersReward(_SideReward):
    """Signed stage-4 opening reward for the fingers of one hand only."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.indices = None
        self.closed_scale = None
        self.prev_openness = None
        self.open_threshold = 0.15
        self.goal_shaping_angle = 0.30
        self.progress_scale = 0.04
        self.velocity_scale = 1.5

    def reset(self, env_ids, states):
        if self.prev_openness is not None:
            self.prev_openness[env_ids] = torch.nan
        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        device = robot.joint_pos.device
        mask = self.active_mask(device, 4)
        if mask is None:
            return torch.zeros(robot.joint_pos.shape[0], device=device)

        if self.indices is None:
            names = list(robot.joint_names)
            target_map = _FINGER_TARGETS[self.side]
            ordered = list(target_map)
            missing = [name for name in ordered if name not in names]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} is missing finger joints: {missing}"
                )
            self.indices = torch.tensor(
                [names.index(name) for name in ordered],
                dtype=torch.long,
                device=device,
            )
            self.closed_scale = torch.tensor(
                [abs(target_map[name]) for name in ordered],
                dtype=robot.joint_pos.dtype,
                device=device,
            ).unsqueeze(0)

        positions = robot.joint_pos.index_select(1, self.indices)
        openness = torch.clamp(
            1.0 - positions.abs() / self.closed_scale, min=0.0, max=1.0
        )
        if (
            self.prev_openness is None
            or self.prev_openness.shape != openness.shape
            or self.prev_openness.device != device
        ):
            self.prev_openness = openness.detach().clone()
            progress_per_joint = torch.zeros_like(openness)
        else:
            previous = torch.where(
                torch.isnan(self.prev_openness), openness, self.prev_openness
            )
            progress_per_joint = torch.clamp(
                (openness - previous) / self.progress_scale,
                min=-1.0,
                max=1.0,
            )
            self.prev_openness = torch.where(
                mask.unsqueeze(-1), openness.detach(), self.prev_openness
            )

        progress_mean = progress_per_joint.mean(dim=-1)
        progress_worst = torch.topk(
            progress_per_joint,
            k=min(4, progress_per_joint.shape[1]),
            dim=-1,
            largest=False,
        ).values.mean(dim=-1)
        signed_progress = 0.60 * progress_mean + 0.40 * progress_worst

        velocity = robot.joint_vel.index_select(1, self.indices)
        toward_open = -torch.sign(positions) * velocity
        signed_motion = torch.mean(
            torch.clamp(
                toward_open / self.velocity_scale, min=-1.0, max=1.0
            )
            * (1.0 - openness),
            dim=-1,
        )
        max_angle = positions.abs().amax(dim=-1)
        goal_score = smoothstep01(
            (self.goal_shaping_angle - max_angle)
            / (self.goal_shaping_angle - self.open_threshold)
        )
        reward = (
            0.55 * signed_progress
            + 0.25 * signed_motion
            + 0.20 * goal_score
        )
        return torch.clamp(reward, min=-1.0, max=1.0) * mask.float()

class LeftReleaseFingersReward(_SideReleaseFingersReward):
    side = "left"

class RightReleaseFingersReward(_SideReleaseFingersReward):
    side = "right"


class StageCompletionReward(HumanoidBaseReward):
    """One unit for each successful stage exit, consumed exactly once.

    The shared reward wrapper evaluates dense rewards with the pre-transition
    stage. This bonus therefore reads the completion flag, not actual_stage;
    completing stage 0 is rewarded too. Checker events remain available for
    logging. A single policy continues learning across intermediate exits.
    """

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        if self.completed_stages is None:
            return torch.zeros_like(robot.joint_pos[:, 0])
        reward = self.completed_stages.to(dtype=robot.joint_pos.dtype).clone()
        self.completed_stages.zero_()
        return reward


# =============================================================================
# WEIGHTS FOR ONE POLICY
# Initial dense scale matches the adapted multi task: 0.25 of the separate
# YAML body weights, with each side contributing half of a paired objective.
# Stage-specific weights change the objective, not the policy. There is no
# persistent reward just for occupying a later stage. Intermediate completion
# bonuses and the final bonus are each +500, within one continuing return.
# =============================================================================

_COMMON_REWARD_WEIGHTS = {
    "CenteredUpperBodyCOMReward": 0.25,
    "FaceChairAllStagesReward": 0.075,
    "PelvisFacingChairReward": 0.0375,
    "WaistVelocityPenalty": -0.05,
    # Physical upper-body target changes and walking-command changes replace
    # SeparatePPOTrainer's per-body PolicyDeltaAction (unavailable here).
    "DeltaActionRateCfg": -0.05,
    "LocomotionCommandPenalty": -0.025,
    "StageCompletionReward": 500.0,
}

STAGE0_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -10.0,
    "ApproachAndStandReward": 0.5,
    "LeftStage0ArmPoseReward": 0.125,
    "RightStage0ArmPoseReward": 0.125,
    "LeftOpenFingersReward": 0.0625,
    "RightOpenFingersReward": 0.0625,
    "KeepChairStillPenaltyWalk": -0.25,
}

STAGE1_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -10.0,
    "ApproachAndStandReward": 0.5,
    "LeftHandDistanceReward": 0.15,
    "RightHandDistanceReward": 0.15,
    "LeftHandOrientationReward": 0.1,
    "RightHandOrientationReward": 0.1,
    "LeftArmVelocityPenalty": -0.0125,
    "RightArmVelocityPenalty": -0.0125,
    "LeftOpenFingersReward": 0.0625,
    "RightOpenFingersReward": 0.0625,
    "KeepChairStillPenaltyArms": -0.25,
}

STAGE2_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -10.0,
    "ApproachAndStandReward": 0.5,
    "StayNearAnchorReward": 0.0375,
    "LeftHandDistanceReward": 0.15,
    "RightHandDistanceReward": 0.15,
    "LeftHandOrientationReward": 0.1,
    "RightHandOrientationReward": 0.1,
    "LeftArmVelocityPenalty": -0.0125,
    "RightArmVelocityPenalty": -0.0125,
    "LeftStage2FingerJointPositionReward": 0.125,
    "RightStage2FingerJointPositionReward": 0.125,
}

STAGE3_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -250.0,
    "Stage3ReverseVelocityReward": 0.5,
    "PullChairReward": 0.3125,
    "LeftHandDistanceReward": 0.15,
    "RightHandDistanceReward": 0.15,
    "LeftHandOrientationReward": 0.1,
    "RightHandOrientationReward": 0.1,
    "LeftArmVelocityPenalty": -0.0125,
    "RightArmVelocityPenalty": -0.0125,
    "LeftStage3HandDriftPenalty": -0.0125,
    "RightStage3HandDriftPenalty": -0.0125,
    "LeftStage2FingerJointPositionReward": 0.125,
    "RightStage2FingerJointPositionReward": 0.125,
}

STAGE4_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -250.0,
    "StayNearAnchorReward": 0.0375,
    "LeftHandDistanceReward": 0.15,
    "RightHandDistanceReward": 0.15,
    "LeftHandOrientationReward": 0.1,
    "RightHandOrientationReward": 0.1,
    "LeftReleaseFingersReward": 0.125,
    "RightReleaseFingersReward": 0.125,
    "PulledChairStillnessReward": -0.025,
}

STAGE5_REWARD_WEIGHTS = {
    **_COMMON_REWARD_WEIGHTS,
    "TerminationCfg": -250.0,
    "PulledChairStillnessReward": -0.025,
    # Final objective for the same policy that performs stages 0..4.
    "ArmDownReward": 0.25,
    "KeepFingersOpenPenalty": -0.0125,
}


# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class ChairmanCfg(HumanoidTaskCfg):
    """Chair task for humanoid robots - full staged reward shaping."""

    success_bar = 0.9
    episode_length = 2500
    log_reward_components: bool = True
    reset_rewards_on_stage_change: bool = True
    stage_reward_weights: dict = {
        0: STAGE0_REWARD_WEIGHTS,
        1: STAGE1_REWARD_WEIGHTS,
        2: STAGE2_REWARD_WEIGHTS,
        3: STAGE3_REWARD_WEIGHTS,
        4: STAGE4_REWARD_WEIGHTS,
        5: STAGE5_REWARD_WEIGHTS,
    }

    objects = [
        ArticulationObjCfg(
            name="chair",
            urdf_path="roboverse_data/assets/humanoidbench/chairs/chair3/foldable_chair_debug.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False,
            batch_fixed_verts=True,
        )
    ]

    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"
    checker = _ChairManChecker()

    reward_functions = [
        TerminationCfg(),
        DeltaActionRateCfg(),
        LocomotionCommandPenalty(),
        CenteredUpperBodyCOMReward(),
        FaceChairAllStagesReward(),
        PelvisFacingChairReward(),
        WaistVelocityPenalty(),
        ApproachAndStandReward(),
        Stage3ReverseVelocityReward(),
        StayNearAnchorReward(),
        LeftStage0ArmPoseReward(),
        RightStage0ArmPoseReward(),
        LeftHandDistanceReward(),
        RightHandDistanceReward(),
        LeftHandOrientationReward(),
        RightHandOrientationReward(),
        LeftArmVelocityPenalty(),
        RightArmVelocityPenalty(),
        LeftStage3HandDriftPenalty(),
        RightStage3HandDriftPenalty(),
        LeftOpenFingersReward(),
        RightOpenFingersReward(),
        LeftStage2FingerJointPositionReward(),
        RightStage2FingerJointPositionReward(),
        LeftReleaseFingersReward(),
        RightReleaseFingersReward(),
        KeepChairStillPenaltyWalk(),
        KeepChairStillPenaltyArms(),
        PullChairReward(),
        PulledChairStillnessReward(),
        ArmDownReward(),
        KeepFingersOpenPenalty(),
        StageCompletionReward(),
    ]
    # Missing stage entries contribute zero; all active values are above.
    reward_weights = [0.0] * len(reward_functions)

    def extra_spec(self):
        return {}
