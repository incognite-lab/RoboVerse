"""ChairMan task rewards for the full staged chair task."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.types import EnvState
from metasim.utils import configclass
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


# =============================================================================
# BASIC / AUXILIARY REWARDS
# =============================================================================

class TerminationCfg(HumanoidBaseReward):
    """Termination condition based on humanoid neck height."""
    def __init__(self):
        super().__init__()

    def __call__(self, states: EnvState, robot_name) -> torch.FloatTensor:
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

        delta_actions = actions - self.prev_actions
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
        self.fingers = None

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

        if self.fingers is None:
            self.fingers = []
            for idx, joint in enumerate(robot.joint_names):
                if "hand" in joint:
                    self.fingers.append(idx)

        if self.fingers:
            num_dof = joint_vel.shape[1]
            all_indices = torch.arange(num_dof, device=device)
            finger_tensor = torch.tensor(self.fingers, device=device)
            non_finger_mask = ~torch.isin(all_indices, finger_tensor)
            target_vel = joint_vel[:, non_finger_mask]
        else:
            target_vel = joint_vel

        mean_abs_vel = torch.mean(torch.abs(target_vel), dim=-1)

        if (
            self.prev_joint_vel is None
            or self.prev_joint_vel.shape != joint_vel.shape
            or self.prev_joint_vel.device != joint_vel.device
        ):
            mean_abs_acc = torch.zeros_like(mean_abs_vel)
            self.prev_joint_vel = joint_vel.detach().clone()
        else:
            if self.fingers:
                prev_target_vel = self.prev_joint_vel[:, non_finger_mask]
            else:
                prev_target_vel = self.prev_joint_vel

            delta_vel = target_vel - prev_target_vel
            mean_abs_acc = torch.mean(torch.abs(delta_vel), dim=-1)
            self.prev_joint_vel = joint_vel.detach().clone()

        vel_penalty = torch.clamp(mean_abs_vel / self.vel_scale, min=0.0, max=1.0)
        acc_penalty = torch.clamp(mean_abs_acc / self.acc_scale, min=0.0, max=1.0)

        return 0.7 * vel_penalty + 0.3 * acc_penalty


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
        self.constraint_buffer = 0.20

        self.required_pos: dict[str, float] = {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            "left_shoulder_pitch_joint": 0.28,
            "right_shoulder_pitch_joint": 0.28,

            "left_shoulder_roll_joint": -0.35,
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
        excess_error = torch.clamp(error - self.constraint_buffer, min=0.0)
        joint_reward = torch.clamp(1.0 - excess_error / self.constraint_buffer, min=0.0, max=1.0)
        return joint_reward.mean(dim=-1) * stage_mask.float()




class WalkToChairProgressReward(HumanoidBaseReward):
    """
    Stage 0 navigation through a staging point behind the chair.

    The robot first walks to a point 1.5 m behind the backrest.  Around that
    point it is rewarded for stopping and facing the chair.  Once aligned, the
    desired velocity opens smoothly toward the final point 0.75 m behind the
    backrest.  There is no discontinuous phase switch inside this reward.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands", target_speed=0.8):
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
            0.45 * velocity_reward
            + 0.15 * direction_reward
            + 0.25 * turn_reward
            + 0.15 * arrival_stop_reward
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
    Face Chair: Udržuje pohled robota na židli (Trychtýřová odměna & Trest za odvracení)
    Odměňuje robota za to, že osa X jeho hlavy směřuje k židli.
    Tvrdě penalizuje, pokud úhlová rychlost hlavy směřuje pohled pryč od židle.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        # Může být aktivní ve všech fázích, kdy chceme, aby robot sledoval cíl
        self.active_stages = [1, 2, 3, 4, 5]

        # O kolik metrů výše nad base_link židle se má robot dívat (na sedák)
        self.chair_look_z_offset = 0.4

        # Váha trestu za odvracení zraku (úhlová rychlost pryč od cíle)
        self.look_away_penalty_weight = 2.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        try:
            head_link_idx = robot.body_names.index("head_link")
            chair_base_idx = chair.body_names.index("base_link")

            # 1. Pozice hlavy a židle
            head_pos = robot.body_state[:, head_link_idx, :3]
            chair_pos = chair.body_state[:, chair_base_idx, :3]

            # 2. Úhlová rychlost hlavy [N, 3] (indexy 10:13)
            head_ang_vel = robot.body_state[:, head_link_idx, 10:13]

            # 3. Orientace hlavy (Quaternion)
            q = robot.body_state[:, head_link_idx, 3:7]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        except ValueError:
            return torch.zeros(num_envs, device=device)

        # --- A. VÝPOČET SMĚRŮ ---
        # Zvedneme cíl pohledu na úroveň sedáku
        target_pos = chair_pos.clone()
        target_pos[:, 2] += self.chair_look_z_offset

        # Vektor od hlavy k židli (Normalizovaný)
        vec_to_target = target_pos - head_pos
        dir_to_target = vec_to_target / (torch.norm(vec_to_target, dim=-1, keepdim=True) + 1e-6)

        # Vektor, kam reálně hlava KOUKÁ (Osa X z quaternionu)
        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x*y + w*z)
        forward_z = 2 * (x*z - w*y)
        head_forward_vec = torch.stack([forward_x, forward_y, forward_z], dim=-1)

        # --- B. ODMĚNA ZA POHLED (Trychtýřová odměna) ---
        # Dot product: 1.0 = kouká přesně tam, -1.0 = kouká dozadu
        alignment = torch.sum(head_forward_vec * dir_to_target, dim=-1)

        # Uděláme z toho chybu: 0.0 = perfektní, 2.0 = nejhorší
        look_error = 1.0 - alignment

        # Trychtýř (Inverse Distance): Čím menší chyba, tím strměji roste odměna k 1.0
        rew_look = 1.0 / (1.0 + 5.0 * look_error)

        # --- C. PENALIZACE ZA ODVRACENÍ ZRAKU (Angular Velocity Penalty) ---
        # Křížový součin (Cross Product) nám dá OSU, kolem které se musí hlava
        # otočit, aby se forward_vec srovnal s dir_to_target.
        correction_axis = torch.cross(head_forward_vec, dir_to_target, dim=-1)

        # Promítneme reálnou úhlovou rychlost hlavy na tuto ideální korekční osu.
        # - Kladné číslo = hlava se otáčí K židli (Správně)
        # - Záporné číslo = hlava se otáčí PRYČ od židle (Špatně!)
        turn_progress = torch.sum(head_ang_vel * correction_axis, dim=-1)

        # Ořízneme kladné hodnoty (neodměňujeme za rychlost otáčení, chceme jen klidný pohled)
        # a ponecháme jen záporné hodnoty (odvracení zraku)
        turning_away = torch.clamp(turn_progress, max=0.0)

        # Aplikace trestu
        penalty_turn = turning_away * self.look_away_penalty_weight

        # --- D. CELKOVÉ SKÓRE ---
        # Robot dostává body za to, že kouká na židli (rew_look),
        # ale pokud cukne hlavou jinam, dostane facku (penalty_turn).
        total_reward = rew_look + penalty_turn

        return total_reward * stage_mask.float()


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


class StayNearAnchorReward(HumanoidBaseReward):
    """
    Stage 1 and 2:
    Reward for keeping pelvis near anchor position.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [1, 2]

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

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
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
        reward = torch.clamp(1.0 - drift / self.max_xy_drift, min=0.0, max=1.0)

        return reward * stage_mask.float()


# =============================================================================
# STAGE 2
# =============================================================================

class CloseGraspReward(HumanoidBaseReward):
    """
    Stage 2 dense reward for closing both hands around the chair.

    Targets use the deeper, collision-tested grasp pose from ``debug2``.  The
    old targets only partially bent the index and middle fingers, so they did
    not reliably reach the chair.  The score starts at zero for an open hand
    and rises independently for every joint, providing a learning signal long
    before a contact-force reward becomes available.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [2]

        self.progress_scale = 0.03
        self.proximity_scale = 0.10

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
        self.prev_closure = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_closure is not None:
            self.prev_closure[env_ids] = torch.nan

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
        closure = 0.40 * torch.mean(closure_per_joint, dim=-1) + 0.60 * torch.minimum(
            left_closure, right_closure
        )

        if self.prev_closure is None or self.prev_closure.shape != closure.shape:
            self.prev_closure = closure.detach().clone()
            progress_reward = torch.zeros_like(closure)
        else:
            previous = torch.where(torch.isnan(self.prev_closure), closure, self.prev_closure)
            progress_reward = torch.clamp(
                (closure - previous) / self.progress_scale, min=0.0, max=1.0
            )
            self.prev_closure = torch.where(
                stage_mask, closure.detach(), self.prev_closure
            )

        # Do not make closing far away from the handle the optimal solution.
        chair = states.objects["chair"]
        left_hand_idx = robot.body_names.index("left_endeffector")
        right_hand_idx = robot.body_names.index("endeffector")
        left_target_idx = chair.body_names.index("target_hand_left")
        right_target_idx = chair.body_names.index("target_hand_right")
        left_dist = torch.norm(
            robot.body_state[:, left_hand_idx, :3]
            - chair.body_state[:, left_target_idx, :3],
            dim=-1,
        )
        right_dist = torch.norm(
            robot.body_state[:, right_hand_idx, :3]
            - chair.body_state[:, right_target_idx, :3],
            dim=-1,
        )
        max_dist = torch.maximum(left_dist, right_dist)
        proximity_gate = 1.0 / (
            1.0 + torch.pow(max_dist / self.proximity_scale, 4)
        )
        total_reward = (0.80 * closure + 0.20 * progress_reward) * (
            0.20 + 0.80 * proximity_gate
        )
        return total_reward * stage_mask.float()


class GraspForceReward(HumanoidBaseReward):
    """
    Stage 2:
    Exact fingertip grasp reward aligned with checker logic.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

        self.active_stages = [2]
        self.force_threshold = 2.0

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
    Reward for keeping at least one fingertip on the chair with each hand.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.force_threshold = 2.0

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
            return torch.zeros(num_envs, device=device)

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
        contact_rewards = torch.clamp(force_mags / self.force_threshold, min=0.0, max=1.0)

        best_tip_rewards = torch.zeros((num_envs, 2, 3), device=device)
        for hand_id in range(2):
            for tip_id in range(3):
                tip_mask = valid_interaction & (contact_hand == hand_id) & (contact_tip == tip_id)
                max_reward, _ = torch.max(contact_rewards * tip_mask.float(), dim=1)
                best_tip_rewards[:, hand_id, tip_id] = max_reward

        left_any = torch.max(best_tip_rewards[:, 0, :], dim=1)[0]
        right_any = torch.max(best_tip_rewards[:, 1, :], dim=1)[0]
        reward = torch.sqrt(left_any * right_any + 1e-8)
        return torch.clamp(reward, 0.0, 1.0) * stage_mask.float()


class PullChairReward(HumanoidBaseReward):
    """
    Stage 3:
    Reward for pulling the chair from x=0.75 to x=-0.25, then stopping.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.initial_chair_pos = torch.tensor([0.75, 0.0, 0.1])
        self.target_chair_pos = torch.tensor([-0.25, 0.0, 0.1])
        self.pull_distance = 1.0
        self.target_pull_speed = 0.45
        self.vel_sigma = 0.25
        self.stop_zone = 0.15
        self.prev_progress = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_progress is not None:
            self.prev_progress[env_ids] = 0.0

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        initial = self.initial_chair_pos.to(device)
        target = self.target_chair_pos.to(device)
        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_vel = chair.body_state[:, chair_idx, 7:10]

        pulled_x = initial[0] - chair_pos[:, 0]
        progress = torch.clamp(pulled_x / self.pull_distance, min=0.0, max=1.0)

        if self.prev_progress is None:
            self.prev_progress = progress.clone()
            progress_delta_reward = torch.zeros(num_envs, device=device)
        else:
            delta = progress - self.prev_progress
            progress_delta_reward = torch.clamp(delta / 0.015, min=0.0, max=1.0)
            self.prev_progress = progress.clone()

        lateral_error = torch.norm(chair_pos[:, 1:3] - target[1:3], dim=-1)
        lateral_reward = torch.clamp(1.0 - lateral_error / 0.35, min=0.0, max=1.0)

        pull_speed = torch.clamp(-chair_vel[:, 0], min=0.0)
        speed_factor = torch.clamp((1.0 - progress) / 0.35, min=0.0, max=1.0)
        desired_speed = self.target_pull_speed * speed_factor
        speed_reward = torch.exp(-torch.square(pull_speed - desired_speed) / (2.0 * self.vel_sigma ** 2))

        target_error = torch.norm(chair_pos - target, dim=-1)
        near_target = torch.clamp(1.0 - target_error / self.stop_zone, min=0.0, max=1.0)

        chair_speed = torch.norm(chair_vel, dim=-1)
        stop_reward = torch.clamp(1.0 - chair_speed / 0.20, min=0.0, max=1.0)

        base_idx = robot.body_names.index("pelvis")
        robot_speed = torch.norm(robot.body_state[:, base_idx, 7:10], dim=-1)
        robot_stop_reward = torch.clamp(1.0 - robot_speed / 0.25, min=0.0, max=1.0)

        moving_part = (
            0.45 * progress +
            0.25 * progress_delta_reward +
            0.20 * speed_reward +
            0.10 * lateral_reward
        )
        finish_part = near_target * (0.60 * stop_reward + 0.40 * robot_stop_reward)
        reward = torch.where(near_target > 0.0, 0.55 * moving_part + 0.45 * finish_part, moving_part)
        return torch.clamp(reward, 0.0, 1.0) * stage_mask.float()


# =============================================================================
# STAGE 4 AND 5
# =============================================================================

class PulledChairStillnessReward(HumanoidBaseReward):
    """
    Stage 4 and 5:
    Reward for keeping the pulled chair and robot still.

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

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        target = self.target_chair_pos.to(device)
        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_vel = chair.body_state[:, chair_idx, 7:10]

        base_idx = robot.body_names.index("pelvis")
        robot_vel = robot.body_state[:, base_idx, 7:10]

        pos_reward = torch.clamp(1.0 - torch.norm(chair_pos - target, dim=-1) / 0.40, min=0.0, max=1.0)
        chair_still = torch.clamp(1.0 - torch.norm(chair_vel, dim=-1) / 0.20, min=0.0, max=1.0)
        robot_still = torch.clamp(1.0 - torch.norm(robot_vel, dim=-1) / 0.20, min=0.0, max=1.0)

        reward = pos_reward * chair_still * robot_still
        return reward * stage_mask.float()


class ReleaseFingersReward(HumanoidBaseReward):
    """
    Stage 4:
    Reward for opening fingers after the chair is pulled and stopped.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [4]
        self.finger_keywords = ["thumb", "index", "middle"]
        self.finger_indices = None
        self.open_threshold = 0.15

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
            indices = [
                i for i, name in enumerate(robot.joint_names)
                if any(keyword in name for keyword in self.finger_keywords)
            ]
            if not indices:
                return torch.zeros(num_envs, device=device)
            self.finger_indices = torch.tensor(indices, device=device, dtype=torch.long)

        q_fingers = robot.joint_pos[:, self.finger_indices]
        max_finger_angle = torch.max(torch.abs(q_fingers), dim=-1)[0]
        reward = torch.clamp(1.0 - max_finger_angle / self.open_threshold, min=0.0, max=1.0)
        return reward * stage_mask.float()


class ArmDownReward(HumanoidBaseReward):
    """
    Stage 5:
    Reward for placing both arms in the resting pose checked by the final stage.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.active_stages = [5]
        self.arm_joints_to_check = [
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_elbow_joint", "right_elbow_joint",
        ]
        self.arm_indices = None
        self.rest_threshold = 0.35

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        if self.arm_indices is None:
            indices = [
                i for i, name in enumerate(robot.joint_names)
                if name in self.arm_joints_to_check
            ]
            if not indices:
                return torch.zeros(num_envs, device=device)
            self.arm_indices = torch.tensor(indices, device=device, dtype=torch.long)

        q_arms = robot.joint_pos[:, self.arm_indices]
        max_arm_angle = torch.max(torch.abs(q_arms), dim=-1)[0]
        reward = torch.clamp(1.0 - max_arm_angle / self.rest_threshold, min=0.0, max=1.0)
        return reward * stage_mask.float()


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
    Stage progress: Odměna za aktuální dosažený stage.
    Podle DoorMan paperu (Table 2) je váha 1.0.

    Formula: stage_current
    Funguje jako dense reward, který motivuje robota zůstat ve vyšších fázích.
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
# WEIGHTS
# reward functions output either:
# - reward in <0,1>  -> use positive weight
# - penalty in <0,1> -> use negative weight
# =============================================================================

# A fall must be clearly worse than any single successful task step, without
# creating the critic spikes caused by the previous -1000 value.
TERMINATION_WEIGHT = -100.0

# General optional penalties / rewards
DELTA_ACTION_RATE_WEIGHT = -0.02
DOF_VELOCITY_ACCELERATION_WEIGHT = -0.02
DOF_POSITION_LIMITS_WEIGHT = -0.25
HUMANLY_DOF_LIMIT_WEIGHT = -0.25
UPRIGHT_PENALTY_WEIGHT = -1.00
FACE_CHAIR_REWARD_WEIGHT = 0.25
ARM_RESTING_POSE_PENALTY_WEIGHT = -0.05
# Sparse transition bonus: stage 0->1 gives 20, 1->2 gives 40 and 2->3 gives 60.
STAGE_PROGRESS_WEIGHT = 20.0
# Small stage baseline; it must not dominate the action-dependent rewards.
CONTINUOUS_STAGE_REWARD_WEIGHT = 0.25

# Stage 0
STAGE0_ARM_POS_REWARD_WEIGHT = 0.75
WALK_TO_CHAIR_REWARD_WEIGHT = 6.0
OPEN_GRASP_REWARD_WEIGHT = 0.50
KEEP_CHAIR_STILL_PENALTY_WEIGHT = -1.0

# Stage 1
REACH_CHAIR_REWARD_WEIGHT = 6.0
REACH_ORIENTATION_REWARD_WEIGHT = 3.0
HAND_TARGET_STILLNESS_REWARD_WEIGHT = 3.0
STAY_NEAR_ANCHOR_REWARD_WEIGHT = 1.0

# Stage 2
CLOSE_GRASP_REWARD_WEIGHT = 7.0
FORCE_GRASP_REWARD_WEIGHT = 8.0

# Stage 3
MAINTAIN_ANY_GRASP_REWARD_WEIGHT = 3.0
PULL_CHAIR_REWARD_WEIGHT = 7.0

# Stage 4
PULLED_CHAIR_STILLNESS_REWARD_WEIGHT = 4.0
RELEASE_FINGERS_REWARD_WEIGHT = 5.0

# Stage 5
ARM_DOWN_REWARD_WEIGHT = 6.0


# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class ChairmanCfg(HumanoidTaskCfg):
    """Chair task for humanoid robots - full staged reward shaping."""

    success_bar = 0.9
    episode_length = 1000

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

    reward_weights = [
        TERMINATION_WEIGHT,
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,
        # DOF_POSITION_LIMITS_WEIGHT,
        # HUMANLY_DOF_LIMIT_WEIGHT,
        UPRIGHT_PENALTY_WEIGHT,

        STAGE0_ARM_POS_REWARD_WEIGHT,
        WALK_TO_CHAIR_REWARD_WEIGHT,
        KEEP_CHAIR_STILL_PENALTY_WEIGHT,
        OPEN_GRASP_REWARD_WEIGHT,

        REACH_CHAIR_REWARD_WEIGHT,
        REACH_ORIENTATION_REWARD_WEIGHT,
        HAND_TARGET_STILLNESS_REWARD_WEIGHT,
        STAY_NEAR_ANCHOR_REWARD_WEIGHT,

        CLOSE_GRASP_REWARD_WEIGHT,
        FORCE_GRASP_REWARD_WEIGHT,

        # MAINTAIN_ANY_GRASP_REWARD_WEIGHT,
        # PULL_CHAIR_REWARD_WEIGHT,

        # PULLED_CHAIR_STILLNESS_REWARD_WEIGHT,
        # RELEASE_FINGERS_REWARD_WEIGHT,
        # ARM_DOWN_REWARD_WEIGHT,

        FACE_CHAIR_REWARD_WEIGHT,
        # ARM_RESTING_POSE_PENALTY_WEIGHT,
        STAGE_PROGRESS_WEIGHT,
        CONTINUOUS_STAGE_REWARD_WEIGHT,
    ]

    reward_functions = [
        TerminationCfg(),
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),
        # DofPositionLimitsCfg(),
        # HumanlyDofLimitCfg(),
        UprightPenaltyCfg(),

        Stage0ArmPos(),
        WalkToChairProgressReward(),
        KeepChairStillPenalty(),
        OpenGraspReward(),

        ReachChairProgressReward(),
        HandOrientationProgressReward(),
        HandTargetStillnessReward(),
        StayNearAnchorReward(),

        CloseGraspReward(),
        GraspForceReward(),

        # MaintainAnyGraspReward(),
        # PullChairReward(),

        # PulledChairStillnessReward(),
        # ReleaseFingersReward(),
        # ArmDownReward(),

        FaceChairReward(),
        # ArmRestingPosePenaltyCfg(),
        StageProgressCfg(),
        ContinuousStageReward(),
    ]

    def extra_spec(self):
        return {}
