"""ChairMan task rewards - version focused only up to Stage 2."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairManCheckerSimple
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import neck_height_tensor

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg





class DeltaActionRateCfg(HumanoidBaseReward):
    """
    Penalty magnitude for abrupt target changes in actions.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)
        self.prev_actions = None
        self.scale = 0.35

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        actions = robot.joint_pos_target
        if self.prev_actions is None:
            self.prev_actions = actions.detach().clone()
        else:
            self.prev_actions[env_ids] = actions[env_ids].detach().clone()

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        actions = states.robots[robot_name].joint_pos_target

        if self.prev_actions is None:
            self.prev_actions = actions.detach().clone()
            return torch.zeros(actions.shape[0], device=actions.device)

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
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)
        self.prev_joint_vel = None
        self.fingers = None

        self.vel_scale = 6.0
        self.acc_scale = 8.0

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        joint_vel = robot.joint_vel
        if self.prev_joint_vel is None:
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

        if self.prev_joint_vel is None:
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

class WalkToChairProgressReward(HumanoidBaseReward):
    """
    Stage 0:
    Reward for approaching the chair with proper braking.

    Combines:
    1) progress per step,
    2) state reward for being near desired stop distance,
    3) velocity tracking toward distance-dependent target speed,
    4) reward for moving in the correct direction,
    5) suppression of reward when moving backward,
    6) suppression of reward when overshooting the stop zone.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider_simple", target_speed=0.8):
        super().__init__(robot_name)
        self.active_stages = [0]

        # cílová vzdálenost od židle
        self.stop_distance = 0.75

        # od jaké vzdálenosti už začínáme výrazně brzdit
        self.braking_distance = 0.70

        # progress shaping
        self.progress_scale = 0.03     # 3 cm zlepšení za krok -> progress ~1

        # state shaping
        self.state_scale = 1.5

        # velocity tracking
        self.target_speed = target_speed
        self.vel_sigma = 0.18

        # direction shaping
        self.direction_speed_scale = 0.8

        # overshoot tolerance
        self.overshoot_margin = 0.05   # tolerance 5 cm

        self.saved_chair_pos = None
        self.prev_dist = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        chair = states.objects["chair"]
        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]

        if self.saved_chair_pos is None:
            self.saved_chair_pos = chair_pos.clone()
        else:
            self.saved_chair_pos[env_ids] = chair_pos[env_ids].clone()

        if self.prev_dist is not None:
            self.prev_dist[env_ids] = 0.0

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
        root_pos = robot.body_state[:, base_idx, :3]
        root_vel = robot.body_state[:, base_idx, 7:10]

        chair_base_idx = chair.body_names.index("base_link")
        if self.saved_chair_pos is None:
            self.saved_chair_pos = chair.body_state[:, chair_base_idx, :3].clone()

        target_pos = self.saved_chair_pos

        # směr k židli v XY
        vec_to_chair = target_pos - root_pos
        vec_to_chair[:, 2] = 0.0
        dist = torch.norm(vec_to_chair, dim=-1)
        dir_to_chair = vec_to_chair / (dist.unsqueeze(-1) + 1e-6)

        # -------------------------------------------------
        # 1) progress reward
        # -------------------------------------------------
        if self.prev_dist is None:
            self.prev_dist = dist.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            delta = self.prev_dist - dist
            progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
            self.prev_dist = dist.clone()

        # -------------------------------------------------
        # 2) state reward: blízkost ke správné stop distance
        # -------------------------------------------------
        dist_error = torch.abs(dist - self.stop_distance)
        state_reward = torch.clamp(1.0 - dist_error / self.state_scale, min=0.0, max=1.0)

        # -------------------------------------------------
        # 3) velocity tracking with braking
        # -------------------------------------------------
        dist_to_stop = torch.clamp(dist - self.stop_distance, min=0.0)

        # čím dál od cíle, tím víc se blíží target_speed
        speed_factor = torch.clamp(dist_to_stop / self.braking_distance, min=0.0, max=1.0)
        dynamic_speed = self.target_speed * speed_factor

        target_vel_vec = dynamic_speed.unsqueeze(-1) * dir_to_chair
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)
        velocity_reward = torch.exp(-vel_error_sq / (2.0 * self.vel_sigma ** 2))

        # -------------------------------------------------
        # 4) reward for moving in correct direction
        # -------------------------------------------------
        velocity_projection = torch.sum(root_vel * dir_to_chair, dim=-1)
        direction_reward = torch.clamp(
            velocity_projection / self.direction_speed_scale,
            min=0.0,
            max=1.0
        )

        # -------------------------------------------------
        # 5) backward suppression
        # když robot couvá, reward výrazně potlačíme
        # -------------------------------------------------
        backward_amount = torch.clamp(-velocity_projection, min=0.0)
        backward_penalty = torch.clamp(backward_amount / 0.4, min=0.0, max=1.0)
        backward_factor = 1.0 - backward_penalty

        # -------------------------------------------------
        # 6) overshoot suppression
        # pokud zajede moc blízko k židli, reward se potlačí
        # -------------------------------------------------
        overshoot_amount = torch.clamp((self.stop_distance - self.overshoot_margin) - dist, min=0.0)
        overshoot_penalty = torch.clamp(overshoot_amount / 0.10, min=0.0, max=1.0)
        overshoot_factor = 1.0 - overshoot_penalty

        # -------------------------------------------------
        # final reward
        #
        # progress + state + velocity tracking + direction
        # a pak potlačení při couvání a přejetí
        # -------------------------------------------------
        base_reward = (
            0.10 * progress_reward +
            0.05 * state_reward +
            0.80 * velocity_reward +
            0.05 * direction_reward
        )

        total_reward = base_reward * backward_factor * overshoot_factor
        total_reward = torch.clamp(total_reward, min=0.0, max=1.0)

        return total_reward * stage_mask.float()

class BackToTargetProgressReward(HumanoidBaseReward):
    """
    Stage 2:
    Robot has hands down, moves backward to a target position, and stops there.

    Combines:
    1) progress per step toward target position
    2) state reward for being near target position
    3) velocity tracking toward distance-dependent target speed
    4) reward for moving in correct direction toward target
    5) reward for moving backward in robot body frame
    6) suppression when overshooting target
    7) optional small reward for keeping arms down

    Output: <0, 1>
    """

    def __init__(
        self,
        robot_name="g1_slider_simple",
        target_speed=0.6,
        target_xy=(-0.8, 0.0)

    ):
        super().__init__(robot_name)
        self.active_stages = [2]

        # world target position for pelvis in XY
        self.target_xy = target_xy

        # braking / target reaching
        self.stop_distance = 0.08
        self.braking_distance = 0.80

        # shaping
        self.progress_scale = 0.03
        self.state_scale = 1.0

        # velocity tracking
        self.target_speed = target_speed
        self.vel_sigma = 0.18

        # direction shaping
        self.direction_speed_scale = 0.6

        # overshoot tolerance
        self.overshoot_margin = 0.03

        self.arm_sigma = 0.35

        self.prev_dist = None
        self.left_shoulder_idx = None
        self.right_shoulder_idx = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.prev_dist is None:
            self.prev_dist = torch.full((num_envs,), float("nan"), device=device)
        else:
            self.prev_dist[env_ids] = float("nan")

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
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
        root_pos = robot.body_state[:, base_idx, :3]
        root_vel = robot.body_state[:, base_idx, 7:10]
        root_quat = robot.body_state[:, base_idx, 3:7]  # [w, x, y, z]

        # -------------------------------------------------
        # target position in world
        # -------------------------------------------------
        target_pos = torch.zeros_like(root_pos)
        target_pos[:, 0] = self.target_xy[0]
        target_pos[:, 1] = self.target_xy[1]
        target_pos[:, 2] = root_pos[:, 2]

        vec_to_target = target_pos - root_pos
        vec_to_target[:, 2] = 0.0
        dist = torch.norm(vec_to_target, dim=-1)
        dir_to_target = vec_to_target / (dist.unsqueeze(-1) + 1e-6)

        # -------------------------------------------------
        # 1) progress reward
        # -------------------------------------------------
        if self.prev_dist is None:
            self.prev_dist = dist.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            invalid_mask = torch.isnan(self.prev_dist)
            delta = self.prev_dist - dist
            progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
            progress_reward[invalid_mask] = 0.0
            self.prev_dist[invalid_mask] = dist[invalid_mask]
            self.prev_dist[~invalid_mask] = dist[~invalid_mask]

        # -------------------------------------------------
        # 2) state reward: closeness to target stop zone
        # -------------------------------------------------
        dist_error = torch.abs(dist - self.stop_distance)
        state_reward = torch.clamp(
            1.0 - dist_error / self.state_scale,
            min=0.0,
            max=1.0
        )

        # -------------------------------------------------
        # 3) velocity tracking with braking
        # -------------------------------------------------
        dist_to_stop = torch.clamp(dist - self.stop_distance, min=0.0)
        speed_factor = torch.clamp(dist_to_stop / self.braking_distance, min=0.0, max=1.0)
        dynamic_speed = self.target_speed * speed_factor

        target_vel_vec = dynamic_speed.unsqueeze(-1) * dir_to_target
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)
        velocity_reward = torch.exp(-vel_error_sq / (2.0 * self.vel_sigma ** 2))

        # -------------------------------------------------
        # 4) reward for moving in correct direction toward target
        # -------------------------------------------------
        velocity_projection = torch.sum(root_vel * dir_to_target, dim=-1)
        direction_reward = torch.clamp(
            velocity_projection / self.direction_speed_scale,
            min=0.0,
            max=1.0
        )

        # -------------------------------------------------
        # 5) reward for moving backward in body frame
        # -------------------------------------------------
        # pelvis forward axis from quaternion
        w = root_quat[:, 0]
        x = root_quat[:, 1]
        y = root_quat[:, 2]
        z = root_quat[:, 3]

        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x * y + w * z)

        body_forward = torch.stack([forward_x, forward_y], dim=-1)
        body_forward = body_forward / (torch.norm(body_forward, dim=-1, keepdim=True) + 1e-6)

        root_vel_xy = root_vel[:, :2]

        # kladné = dopředu, záporné = dozadu
        body_forward_speed = torch.sum(root_vel_xy * body_forward, dim=-1)

        backward_amount = torch.clamp(-body_forward_speed, min=0.0)
        backward_reward = torch.clamp(backward_amount / 0.35, min=0.0, max=1.0)

        forward_amount = torch.clamp(body_forward_speed, min=0.0)
        forward_penalty = torch.clamp(forward_amount / 0.35, min=0.0, max=1.0)
        backward_factor = 1.0 - forward_penalty

        # -------------------------------------------------
        # 6) overshoot suppression
        # pokud už je robot moc za cílem / moc blízko, reward se potlačí
        # -------------------------------------------------
        overshoot_amount = torch.clamp((self.stop_distance - self.overshoot_margin) - dist, min=0.0)
        overshoot_penalty = torch.clamp(overshoot_amount / 0.08, min=0.0, max=1.0)
        overshoot_factor = 1.0 - overshoot_penalty


        # -------------------------------------------------
        # final reward
        # -------------------------------------------------
        base_reward = (
            0.10 * progress_reward +
            0.15 * state_reward +
            0.45 * velocity_reward +
            0.10 * direction_reward +
            0.15 * backward_reward
        )

        total_reward = base_reward * backward_factor * overshoot_factor
        total_reward = torch.clamp(total_reward, min=0.0, max=1.0)

        return total_reward * stage_mask.float()

def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class ArmPostureReward(HumanoidBaseReward):
    """
    Dense reward for desired shoulder posture by stage.
    Sharper near target than the original version.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        up_target_left=-1.86,
        up_target_right=-1.86,
        down_target_left=-1.0,
        down_target_right=-1.0,
        sigma=0.20,
    ):
        super().__init__(robot_name)

        self.up_stages = [0, 3]
        self.down_stages = [1, 2]

        self.up_target_left = up_target_left
        self.up_target_right = up_target_right
        self.down_target_left = down_target_left
        self.down_target_right = down_target_right
        self.sigma = sigma

        self.left_idx = None
        self.right_idx = None

    def _lazy_init(self, robot):
        if self.left_idx is None:
            joint_names = robot.joint_names.tolist()
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        self._lazy_init(robot)

        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]

        target_left = torch.zeros(num_envs, device=device)
        target_right = torch.zeros(num_envs, device=device)

        up_mask = torch.isin(self.actual_stage, torch.tensor(self.up_stages, device=device))
        down_mask = torch.isin(self.actual_stage, torch.tensor(self.down_stages, device=device))
        active_mask = up_mask | down_mask

        target_left[up_mask] = self.up_target_left
        target_right[up_mask] = self.up_target_right
        target_left[down_mask] = self.down_target_left
        target_right[down_mask] = self.down_target_right

        if not active_mask.any():
            return torch.zeros(num_envs, device=device)

        err_left = q_left - target_left
        err_right = q_right - target_right

        # quadratic mean error místo abs, ostřejší kolem cíle
        err = torch.sqrt(0.5 * (err_left**2 + err_right**2))

        reward = torch.exp(-(err ** 2) / (2.0 * self.sigma ** 2))
        return torch.clamp(reward, 0.0, 1.0) * active_mask.float()

class FaceForwardReward(HumanoidBaseReward):
    """
    Reward za to, že pelvis/torso míří dopředu v globálním +X směru.

    Aktivní ve stage 0 a 2.

    Output: <0, 1>
    Use with POSITIVE weight.
    """
    def __init__(self, robot_name="g1_slider_simple", forward_axis="x", sharpness=2.0):
        super().__init__(robot_name)
        self.active_stages = [0, 2]
        self.forward_axis = forward_axis
        self.sharpness = sharpness
        self.body_idx = None

    def _lazy_init(self, robot):
        if self.body_idx is None:
            # můžeš změnit na torso_link, pokud je stabilnější než pelvis
            self.body_idx = robot.body_names.index("pelvis")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
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

        self._lazy_init(robot)

        q = robot.body_state[:, self.body_idx, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # forward vector těla = lokální osa X převedená do world
        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x*y + w*z)
        forward_vec = torch.stack([forward_x, forward_y], dim=-1)

        forward_vec = forward_vec / (torch.norm(forward_vec, dim=-1, keepdim=True) + 1e-6)

        if self.forward_axis == "x":
            desired = torch.tensor([1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
        else:
            desired = torch.tensor([0.0, 1.0], device=device).unsqueeze(0).repeat(num_envs, 1)

        alignment = torch.sum(forward_vec * desired, dim=-1)  # [-1, 1]
        reward = torch.clamp((alignment + 1.0) / 2.0, 0.0, 1.0) ** self.sharpness
        return reward * stage_mask.float()


class StayAtAnchorPenalty(HumanoidBaseReward):
    """
    Penalizace za pohyb od místa, kde robot vstoupil do stage 1 nebo 3.

    Stage 1: má stát a dávat ruce dolů
    Stage 3: má stát a dávat ruce nahoru

    Penalizuje:
    - posun v XY
    - rotaci kolem Z
    - lineární rychlost
    - úhlovou rychlost kolem Z

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        pos_scale=0.06,
        yaw_scale=0.20,
        lin_vel_scale=0.12,
        yaw_vel_scale=0.50,
    ):
        super().__init__(robot_name)
        self.active_stages = [1, 3]

        self.pos_scale = pos_scale
        self.yaw_scale = yaw_scale
        self.lin_vel_scale = lin_vel_scale
        self.yaw_vel_scale = yaw_vel_scale

        self.base_idx = None
        self.prev_stage = None
        self.anchor_xy = None
        self.anchor_yaw = None

    def _lazy_init(self, robot):
        if self.base_idx is None:
            self.base_idx = robot.body_names.index("pelvis")

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        self._lazy_init(robot)

        num_envs = robot.joint_pos.shape[0]
        device = robot.joint_pos.device

        if self.anchor_xy is None:
            self.anchor_xy = torch.zeros(num_envs, 2, device=device)
            self.anchor_yaw = torch.zeros(num_envs, device=device)
            self.prev_stage = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        self.anchor_xy[env_ids] = 0.0
        self.anchor_yaw[env_ids] = 0.0
        self.prev_stage[env_ids] = -1

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        self._lazy_init(robot)

        pos = robot.body_state[:, self.base_idx, :3]
        lin_vel = robot.body_state[:, self.base_idx, 7:10]
        ang_vel = robot.body_state[:, self.base_idx, 10:13]
        q = robot.body_state[:, self.base_idx, 3:7]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        active_mask = torch.isin(
            self.actual_stage,
            torch.tensor(self.active_stages, device=device)
        )

        if self.anchor_xy is None:
            self.anchor_xy = pos[:, :2].clone()
            self.anchor_yaw = yaw.clone()
            self.prev_stage = self.actual_stage.clone()

        # uložit anchor při vstupu do stage 1 nebo 3
        entered_active = active_mask & (~torch.isin(
            self.prev_stage,
            torch.tensor(self.active_stages, device=device)
        ))
        if entered_active.any():
            self.anchor_xy[entered_active] = pos[entered_active, :2]
            self.anchor_yaw[entered_active] = yaw[entered_active]

        self.prev_stage = self.actual_stage.clone()

        if not active_mask.any():
            return torch.zeros(num_envs, device=device)

        pos_err = torch.norm(pos[:, :2] - self.anchor_xy, dim=-1)
        yaw_err = torch.abs(_wrap_to_pi(yaw - self.anchor_yaw))
        lin_speed = torch.norm(lin_vel[:, :2], dim=-1)
        yaw_speed = torch.abs(ang_vel[:, 2])

        pos_pen = torch.clamp(pos_err / self.pos_scale, 0.0, 1.0)
        yaw_pen = torch.clamp(yaw_err / self.yaw_scale, 0.0, 1.0)
        lin_pen = torch.clamp(lin_speed / self.lin_vel_scale, 0.0, 1.0)
        yaw_vel_pen = torch.clamp(yaw_speed / self.yaw_vel_scale, 0.0, 1.0)

        penalty = 0.40 * pos_pen + 0.20 * yaw_pen + 0.25 * lin_pen + 0.15 * yaw_vel_pen
        return penalty * active_mask.float()

class ArmPoseErrorPenalty(HumanoidBaseReward):
    """
    Penalizace za odchylku ramen od požadované polohy.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        up_target_left=-1.86,
        up_target_right=-1.86,
        down_target_left=-1.0,
        down_target_right=-1.0,
        scale=1.0,
    ):
        super().__init__(robot_name)

        self.up_stages = [0, 3]
        self.down_stages = [1, 2]

        self.up_target_left = up_target_left
        self.up_target_right = up_target_right
        self.down_target_left = down_target_left
        self.down_target_right = down_target_right
        self.scale = scale

        self.left_idx = None
        self.right_idx = None

    def _lazy_init(self, robot):
        if self.left_idx is None:
            joint_names = robot.joint_names.tolist()
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        self._lazy_init(robot)

        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]

        target_left = torch.zeros(num_envs, device=device)
        target_right = torch.zeros(num_envs, device=device)

        up_mask = torch.isin(self.actual_stage, torch.tensor(self.up_stages, device=device))
        down_mask = torch.isin(self.actual_stage, torch.tensor(self.down_stages, device=device))
        active_mask = up_mask | down_mask

        target_left[up_mask] = self.up_target_left
        target_right[up_mask] = self.up_target_right
        target_left[down_mask] = self.down_target_left
        target_right[down_mask] = self.down_target_right

        err = 0.5 * (torch.abs(q_left - target_left) + torch.abs(q_right - target_right))
        penalty = torch.clamp(err / self.scale, 0.0, 1.0)
        return penalty * active_mask.float()


class StageProgressCfg(HumanoidBaseReward):
    """
    Stage progress: Odměna za aktuální dosažený stage.
    Podle DoorMan paperu (Table 2) je váha 1.0.

    Formula: stage_current
    Funguje jako dense reward, který motivuje robota zůstat ve vyšších fázích.
    """
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # Pokud není actual_stage inicializováno, vrátíme 0
        if self.completed_stages.any():
            ret = self.completed_stages * self.actual_stage.float()
            self.completed_stages = torch.zeros_like(self.completed_stages) # Reset pro další výpočet
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
    def __init__(self, robot_name="g1_slider_simple"):
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





TERMINATION_WEIGHT = -1000.0

# General optional penalties / rewards
DELTA_ACTION_RATE_WEIGHT = -0.05
DOF_VELOCITY_ACCELERATION_WEIGHT = -0.05
DOF_POSITION_LIMITS_WEIGHT = -0.20
HUMANLY_DOF_LIMIT_WEIGHT = -0.10
UPRIGHT_PENALTY_WEIGHT = -0.20
FACE_CHAIR_REWARD_WEIGHT = 0.20
ARM_RESTING_POSE_PENALTY_WEIGHT = -0.02
STAGE_PROGRESS_WEIGHT = 50.0
CONTINUOUS_STAGE_REWARD_WEIGHT = 4.0


WALK_TO_CHAIR_REWARD_WEIGHT = 2.0


ARM_POSTURE_REWARD_WEIGHT = 4.0
FACE_FORWARD_REWARD_WEIGHT = 1.0
STAY_AT_ANCHOR_PENALTY_WEIGHT = -1.0
ARM_POSE_ERROR_PENALTY_WEIGHT = -1.0   # volitelné
BACK_TO_TARGET_PROGRESS_WEIGHT = 2.0

# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class ChairmansimpleCfg(HumanoidTaskCfg):
    """Chair task for humanoid robots - tuned only up to Stage 2."""

    success_bar = 0.9
    episode_length = 1000

    objects = [
        ArticulationObjCfg(
            name="chair",
            urdf_path="roboverse_data/assets/humanoidbench/chairs/chair1/foldable_chair_debug.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False,
            batch_fixed_verts=True,
        ),
        # RigidObjCfg(
        #     name="room",
        #     urdf_path="/home/roboversepc/Documents/rooms/room5/room.urdf",
        #     default_position= [0.0, 0.0, 0.0],
        #     fix_base_link=True
        # )
    ]

    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"
    checker = _ChairManCheckerSimple()

    reward_weights = [
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,

        WALK_TO_CHAIR_REWARD_WEIGHT,
        BACK_TO_TARGET_PROGRESS_WEIGHT,
        ARM_POSTURE_REWARD_WEIGHT,
        FACE_FORWARD_REWARD_WEIGHT,
        STAY_AT_ANCHOR_PENALTY_WEIGHT,
        STAGE_PROGRESS_WEIGHT,
        #ARM_POSE_ERROR_PENALTY_WEIGHT,


    ]

    reward_functions = [
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),


        WalkToChairProgressReward(),
        BackToTargetProgressReward(),
        ArmPostureReward(),
        FaceForwardReward(),
        StayAtAnchorPenalty(),
        StageProgressCfg(),
        #ArmPoseErrorPenalty(),



    ]

    def extra_spec(self):
        return {}
