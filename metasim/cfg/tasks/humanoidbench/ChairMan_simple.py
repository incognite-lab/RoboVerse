"""ChairMan simple task rewards."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairManCheckerSimple, _ChairManCheckerSimpleGRPO
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.types import EnvState
from metasim.utils import configclass

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
        robot_name = robot_name or self.robot_name
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
        robot_name = robot_name or self.robot_name
        robot = states.robots[robot_name]
        joint_vel = robot.joint_vel
        device = joint_vel.device

        if self.fingers is None:
            joint_names = robot.joint_names.tolist() if hasattr(robot.joint_names, "tolist") else list(robot.joint_names)
            self.fingers = []
            for idx, joint in enumerate(joint_names):
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


# class WalkToChairProgressReward(HumanoidBaseReward):
#     """
#     Stage 0:
#     Reward for approaching the chair with proper braking.

#     Combines:
#     1) progress per step,
#     2) state reward for being near desired stop distance,
#     3) velocity tracking toward distance-dependent target speed,
#     4) reward for moving in the correct direction,
#     5) suppression of reward when moving backward,
#     6) suppression of reward when overshooting the stop zone.

#     Output: <0, 1>
#     """
#     def __init__(self, robot_name="g1_slider", target_speed=0.8):
#         super().__init__(robot_name)
#         self.active_stages = [0]

#         # cílová vzdálenost od židle
#         self.stop_distance = 0.75

#         # od jaké vzdálenosti už začínáme výrazně brzdit
#         self.braking_distance = 0.70

#         # progress shaping
#         self.progress_scale = 0.03     # 3 cm zlepšení za krok -> progress ~1

#         # state shaping
#         self.state_scale = 1.5

#         # velocity tracking
#         self.target_speed = target_speed
#         self.vel_sigma = 0.18

#         # direction shaping
#         self.direction_speed_scale = 0.8

#         # overshoot tolerance
#         self.overshoot_margin = 0.05   # tolerance 5 cm

#         self.saved_chair_pos = None
#         self.prev_dist = None

#     def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
#         chair = states.objects["chair"]
#         chair_base_idx = chair.body_names.index("base_link")
#         chair_pos = chair.body_state[:, chair_base_idx, :3]

#         if self.saved_chair_pos is None:
#             self.saved_chair_pos = chair_pos.clone()
#         else:
#             self.saved_chair_pos[env_ids] = chair_pos[env_ids].clone()

#         if self.prev_dist is not None:
#             self.prev_dist[env_ids] = 0.0

#         if hasattr(super(), "reset"):
#             super().reset(env_ids, states)

#     def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
#         robot = states.robots[robot_name]
#         chair = states.objects["chair"]
#         device = robot.joint_pos.device
#         num_envs = robot.joint_pos.shape[0]

#         if self.actual_stage is None:
#             return torch.zeros(num_envs, device=device)

#         stage_mask = torch.isin(
#             self.actual_stage,
#             torch.tensor(self.active_stages, device=device)
#         )
#         if not stage_mask.any():
#             return torch.zeros(num_envs, device=device)

#         base_idx = robot.body_names.index("pelvis")
#         root_pos = robot.body_state[:, base_idx, :3]
#         root_vel = robot.body_state[:, base_idx, 7:10]

#         chair_base_idx = chair.body_names.index("base_link")
#         if self.saved_chair_pos is None:
#             self.saved_chair_pos = chair.body_state[:, chair_base_idx, :3].clone()

#         target_pos = self.saved_chair_pos

#         # směr k židli v XY
#         vec_to_chair = target_pos - root_pos
#         vec_to_chair[:, 2] = 0.0
#         dist = torch.norm(vec_to_chair, dim=-1)
#         dir_to_chair = vec_to_chair / (dist.unsqueeze(-1) + 1e-6)

#         # -------------------------------------------------
#         # 1) progress reward
#         # -------------------------------------------------
#         if self.prev_dist is None:
#             self.prev_dist = dist.clone()
#             progress_reward = torch.zeros(num_envs, device=device)
#         else:
#             delta = self.prev_dist - dist
#             progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
#             self.prev_dist = dist.clone()

#         # -------------------------------------------------
#         # 2) state reward: blízkost ke správné stop distance
#         # -------------------------------------------------
#         dist_error = torch.abs(dist - self.stop_distance)
#         state_reward = torch.clamp(1.0 - dist_error / self.state_scale, min=0.0, max=1.0)

#         # -------------------------------------------------
#         # 3) velocity tracking with braking
#         # -------------------------------------------------
#         # dist_to_stop = torch.clamp(dist - self.stop_distance, min=0.0)

#         # # čím dál od cíle, tím víc se blíží target_speed
#         # speed_factor = torch.clamp(dist_to_stop / self.braking_distance, min=0.0, max=1.0)
#         # dynamic_speed = self.target_speed * speed_factor

#         # target_vel_vec = dynamic_speed.unsqueeze(-1) * dir_to_chair
#         # vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)
#         # velocity_reward = torch.exp(-vel_error_sq / (2.0 * self.vel_sigma ** 2))

#         # -------------------------------------------------
#         # 4) reward for moving in correct direction
#         # -------------------------------------------------
#         velocity_projection = torch.sum(root_vel * dir_to_chair, dim=-1)
#         direction_reward = torch.clamp(
#             velocity_projection / self.direction_speed_scale,
#             min=0.0,
#             max=1.0
#         )

#         # -------------------------------------------------
#         # 5) backward suppression
#         # když robot couvá, reward výrazně potlačíme
#         # -------------------------------------------------
#         backward_amount = torch.clamp(-velocity_projection, min=0.0)
#         backward_penalty = torch.clamp(backward_amount / 0.4, min=0.0, max=1.0)
#         backward_factor = 1.0 - backward_penalty

#         # -------------------------------------------------
#         # 6) overshoot suppression
#         # pokud zajede moc blízko k židli, reward se potlačí
#         # -------------------------------------------------
#         overshoot_amount = torch.clamp((self.stop_distance - self.overshoot_margin) - dist, min=0.0)
#         overshoot_penalty = torch.clamp(overshoot_amount / 0.10, min=0.0, max=1.0)
#         overshoot_factor = 1.0 - overshoot_penalty

#         # -------------------------------------------------
#         # final reward
#         #
#         # progress + state + velocity tracking + direction
#         # a pak potlačení při couvání a přejetí
#         # -------------------------------------------------
#         base_reward = (
#             0.10 * progress_reward +
#             0.10 * state_reward +
#             #0.80 * velocity_reward +
#             0.80 * direction_reward
#         )

#         total_reward = base_reward * backward_factor * overshoot_factor
#         total_reward = torch.clamp(total_reward, min=0.0, max=1.0)

#         return total_reward * stage_mask.float()

class WalkToChairDirectReward(HumanoidBaseReward):
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.stop_distance = 0.75
        self.goal_activation_distance = 0.30
        self.goal_sigma = 0.08
        self.forward_speed_scale = 0.35
        self.direction_sharpness = 6.0
        self.overshoot_margin = 0.05
        self.left_idx = None
        self.right_idx = None

    def _lazy_init(self, robot):
        if self.left_idx is None:
            joint_names = robot.joint_names.tolist() if hasattr(robot.joint_names, "tolist") else list(robot.joint_names)
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
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

        self._lazy_init(robot)

        base_idx = robot.body_names.index("pelvis")
        root_pos = robot.body_state[:, base_idx, :3]
        root_vel = robot.body_state[:, base_idx, 7:10]

        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]

        vec_to_chair = chair_pos - root_pos
        vec_to_chair[:, 2] = 0.0

        dist = torch.norm(vec_to_chair, dim=-1)
        dir_to_chair = vec_to_chair / (dist.unsqueeze(-1) + 1e-6)

        root_vel_xy = root_vel[:, :2]
        dir_to_chair_xy = dir_to_chair[:, :2]

        speed = torch.norm(root_vel_xy, dim=-1)
        forward_speed = torch.sum(root_vel_xy * dir_to_chair_xy, dim=-1)

        direction_cos = forward_speed / (speed + 1e-6)
        direction_cos = torch.clamp(direction_cos, 0.0, 1.0)

        sharp_direction = direction_cos ** self.direction_sharpness
        forward_amount = torch.clamp(forward_speed / self.forward_speed_scale, 0.0, 1.0)
        direction_reward = sharp_direction * forward_amount

        dist_error = torch.abs(dist - self.stop_distance)
        goal_gate = torch.clamp(1.0 - dist_error / self.goal_activation_distance, 0.0, 1.0)
        goal_peak = torch.exp(-(dist_error ** 2) / (2.0 * self.goal_sigma ** 2))
        goal_reward = torch.clamp(goal_gate * goal_peak, 0.0, 1.0)

        overshoot_amount = torch.clamp((self.stop_distance - self.overshoot_margin) - dist, min=0.0)
        overshoot_penalty = torch.clamp(overshoot_amount / 0.10, 0.0, 1.0)
        overshoot_factor = 1.0 - overshoot_penalty

        total_reward = 0.80 * direction_reward + 0.20 * goal_reward
        total_reward = total_reward * overshoot_factor

        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]
        arm_err = 0.5 * (torch.abs(q_left + 1.86) + torch.abs(q_right + 1.86))
        arm_factor = torch.exp(-(arm_err ** 2) / (2.0 * 0.12 ** 2))

        total_reward = total_reward * (0.2 + 0.8 * arm_factor)
        total_reward = torch.clamp(total_reward, 0.0, 1.0)

        return total_reward * stage_mask.float()
class ArmPosturePenalty(HumanoidBaseReward):
    """
    Tvrdá penalizace za velkou odchylku ramen od targetu.
    Output: <0,1>
    Use with NEGATIVE weight.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        up_target_left=-1.86,
        up_target_right=-1.86,
        down_target_left=-1.32,
        down_target_right=-1.32,
        bad_error_start=0.20,
        bad_error_full=0.50,
    ):
        super().__init__(robot_name)

        self.up_stages = [0, 3]
        self.down_stages = [1, 2]

        self.up_target_left = up_target_left
        self.up_target_right = up_target_right
        self.down_target_left = down_target_left
        self.down_target_right = down_target_right

        self.bad_error_start = bad_error_start
        self.bad_error_full = bad_error_full

        self.left_idx = None
        self.right_idx = None

    def _lazy_init(self, robot):
        if self.left_idx is None:
            joint_names = robot.joint_names.tolist() if hasattr(robot.joint_names, "tolist") else list(robot.joint_names)
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
        robot = states.robots[robot_name]

        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        self._lazy_init(robot)

        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]

        up_mask = torch.isin(self.actual_stage, torch.tensor(self.up_stages, device=device))
        down_mask = torch.isin(self.actual_stage, torch.tensor(self.down_stages, device=device))
        active_mask = up_mask | down_mask

        if not active_mask.any():
            return torch.zeros(num_envs, device=device)

        target_left = torch.zeros(num_envs, device=device)
        target_right = torch.zeros(num_envs, device=device)

        target_left[up_mask] = self.up_target_left
        target_right[up_mask] = self.up_target_right
        target_left[down_mask] = self.down_target_left
        target_right[down_mask] = self.down_target_right

        err = 0.5 * (
            torch.abs(q_left - target_left) +
            torch.abs(q_right - target_right)
        )

        penalty = (err - self.bad_error_start) / (self.bad_error_full - self.bad_error_start + 1e-6)
        penalty = torch.clamp(penalty, 0.0, 1.0)

        return penalty * active_mask.float()
class ArmPostureReward(HumanoidBaseReward):
    """
    Reward za správnou polohu ramen, progres k targetu a jejich udržení.

    Skládá se z:
    - position reward: ramena jsou blízko targetu
    - progress reward: chyba ramen se oproti minulému kroku zmenšila
    - hold reward: ramena se u targetu moc nehýbou
    - target bonus: extra bonus když jsou opravdu dobře trefená

    Output: <0, 1>
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        up_target_left=-1.86,
        up_target_right=-1.86,
        down_target_left=-1.32,
        down_target_right=-1.32,
        pos_sigma_up=0.2,
        pos_sigma_down=0.2,
        vel_sigma=0.20,
        target_tol=0.08,
        hold_gate_threshold=0.8,
        progress_scale=0.03,
    ):
        super().__init__(robot_name)

        self.up_stages = [0, 3]
        self.down_stages = [1, 2]

        self.up_target_left = up_target_left
        self.up_target_right = up_target_right
        self.down_target_left = down_target_left
        self.down_target_right = down_target_right

        self.pos_sigma_up = pos_sigma_up
        self.pos_sigma_down = pos_sigma_down
        self.vel_sigma = vel_sigma
        self.target_tol = target_tol
        self.hold_gate_threshold = hold_gate_threshold
        self.progress_scale = progress_scale

        self.left_idx = None
        self.right_idx = None
        self.prev_err = None

    def _lazy_init(self, robot):
        if self.left_idx is None:
            joint_names = robot.joint_names.tolist() if hasattr(robot.joint_names, "tolist") else list(robot.joint_names)
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.prev_err is None:
            self.prev_err = torch.full((num_envs,), float("nan"), device=device)
        else:
            self.prev_err[env_ids] = float("nan")

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
        robot = states.robots[robot_name]

        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        self._lazy_init(robot)

        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]

        qd_left = robot.joint_vel[:, self.left_idx]
        qd_right = robot.joint_vel[:, self.right_idx]

        up_mask = torch.isin(self.actual_stage, torch.tensor(self.up_stages, device=device))
        down_mask = torch.isin(self.actual_stage, torch.tensor(self.down_stages, device=device))
        active_mask = up_mask | down_mask

        if not active_mask.any():
            return torch.zeros(num_envs, device=device)

        target_left = torch.zeros(num_envs, device=device)
        target_right = torch.zeros(num_envs, device=device)

        target_left[up_mask] = self.up_target_left
        target_right[up_mask] = self.up_target_right
        target_left[down_mask] = self.down_target_left
        target_right[down_mask] = self.down_target_right

        err_left = torch.abs(q_left - target_left)
        err_right = torch.abs(q_right - target_right)
        err = 0.5 * (err_left + err_right)

        shoulder_speed = 0.5 * (torch.abs(qd_left) + torch.abs(qd_right))

        pos_sigma = torch.full((num_envs,), self.pos_sigma_down, device=device)
        pos_sigma[up_mask] = self.pos_sigma_up

        # 1) reward za správnou pozici
        pos_reward = torch.exp(-(err ** 2) / (2.0 * pos_sigma ** 2))

        # 2) progress reward
        if self.prev_err is None:
            self.prev_err = err.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            invalid_mask = torch.isnan(self.prev_err)
            delta_err = self.prev_err - err
            progress_reward = torch.clamp(delta_err / self.progress_scale, min=0.0, max=1.0)
            progress_reward[invalid_mask] = 0.0
            self.prev_err = err.clone()

        # 3) hold reward se počítá jen pokud je pozice dost dobrá
        raw_hold_reward = torch.exp(-(shoulder_speed ** 2) / (2.0 * self.vel_sigma ** 2))
        hold_mask = pos_reward >= self.hold_gate_threshold
        hold_reward = raw_hold_reward * hold_mask.float()

        # 4) bonus za opravdu přesné trefení
        target_bonus = (err < self.target_tol).float()

        reward = (
            0.45 * pos_reward +
            0.25 * progress_reward +
            0.20 * hold_reward +
            0.10 * target_bonus
        )

        reward = torch.clamp(reward, 0.0, 1.0)
        return reward * active_mask.float()
class BackToTargetReward(HumanoidBaseReward):
    """
    Stage 2:
    Robot má:
    - ruce dole
    - couvat se židlí
    - dostat se na target pozici
    - u targetu zastavit

    Reward je navržený přímo podle stage2 checkeru.

    Složky:
    1) progress k robot targetu
    2) state reward za blízkost robot targetu
    3) progress odtažení židle
    4) state reward za dostatečně odtaženou židli
    5) reward za pohyb směrem k targetu
    6) reward za couvání v body frame
    7) still reward u cíle
    8) gate přes ruce dole

    Output: <0, 1>
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        target_xy=(-0.80, 0.0),
        chair_start_x=0.75,
        chair_move_goal=0.30,
        down_target_left=-1.32,
        down_target_right=-1.32,
        robot_progress_scale=0.03,
        chair_progress_scale=0.015,
        target_sigma=0.12,
        target_speed_scale=0.25,
        backward_speed_scale=0.20,
        still_lin_sigma=0.08,
        still_yaw_sigma=0.25,
        still_activation_dist=0.20,
        arm_sigma=0.12,
    ):
        super().__init__(robot_name)
        self.active_stages = [2]

        self.target_xy = target_xy
        self.chair_start_x = chair_start_x
        self.chair_move_goal = chair_move_goal

        self.down_target_left = down_target_left
        self.down_target_right = down_target_right

        self.robot_progress_scale = robot_progress_scale
        self.chair_progress_scale = chair_progress_scale
        self.target_sigma = target_sigma
        self.target_speed_scale = target_speed_scale
        self.backward_speed_scale = backward_speed_scale
        self.still_lin_sigma = still_lin_sigma
        self.still_yaw_sigma = still_yaw_sigma
        self.still_activation_dist = still_activation_dist
        self.arm_sigma = arm_sigma

        self.base_idx = None
        self.left_idx = None
        self.right_idx = None

        self.prev_robot_dist = None
        self.prev_chair_move = None

    def _lazy_init(self, robot):
        if self.base_idx is None:
            self.base_idx = robot.body_names.index("pelvis")

        if self.left_idx is None:
            joint_names = robot.joint_names.tolist() if hasattr(robot.joint_names, "tolist") else list(robot.joint_names)
            self.left_idx = joint_names.index("left_shoulder_pitch_joint")
            self.right_idx = joint_names.index("right_shoulder_pitch_joint")

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        self._lazy_init(robot)

        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.prev_robot_dist is None:
            self.prev_robot_dist = torch.full((num_envs,), float("nan"), device=device)
        else:
            self.prev_robot_dist[env_ids] = float("nan")

        if self.prev_chair_move is None:
            self.prev_chair_move = torch.full((num_envs,), float("nan"), device=device)
        else:
            self.prev_chair_move[env_ids] = float("nan")

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
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

        self._lazy_init(robot)

        # -------------------------------------------------
        # robot state
        # -------------------------------------------------
        root_pos = robot.body_state[:, self.base_idx, :3]
        root_vel = robot.body_state[:, self.base_idx, 7:10]
        root_ang_vel = robot.body_state[:, self.base_idx, 10:13]
        root_quat = robot.body_state[:, self.base_idx, 3:7]

        root_xy = root_pos[:, :2]
        root_vel_xy = root_vel[:, :2]

        # -------------------------------------------------
        # chair state
        # -------------------------------------------------
        chair_base_idx = chair.body_names.index("base_link")
        chair_x = chair.body_state[:, chair_base_idx, 0]

        # -------------------------------------------------
        # target position for robot
        # -------------------------------------------------
        target_xy = torch.tensor(self.target_xy, device=device).unsqueeze(0).repeat(num_envs, 1)
        vec_to_target = target_xy - root_xy
        dist_to_target = torch.norm(vec_to_target, dim=-1)
        dir_to_target = vec_to_target / (dist_to_target.unsqueeze(-1) + 1e-6)

        # -------------------------------------------------
        # 1) robot progress reward
        # -------------------------------------------------
        invalid_robot = torch.isnan(self.prev_robot_dist)
        delta_robot = self.prev_robot_dist - dist_to_target
        robot_progress_reward = torch.clamp(
            delta_robot / self.robot_progress_scale,
            min=0.0,
            max=1.0,
        )
        robot_progress_reward[invalid_robot] = 0.0
        self.prev_robot_dist = dist_to_target.clone()

        # -------------------------------------------------
        # 2) robot target state reward
        # -------------------------------------------------
        robot_state_reward = torch.exp(
            -(dist_to_target ** 2) / (2.0 * self.target_sigma ** 2)
        )
        robot_state_reward = torch.clamp(robot_state_reward, 0.0, 1.0)

        # -------------------------------------------------
        # 3) chair move progress reward
        # -------------------------------------------------
        chair_move = torch.clamp(self.chair_start_x - chair_x, min=0.0)

        invalid_chair = torch.isnan(self.prev_chair_move)
        delta_chair = chair_move - self.prev_chair_move
        chair_progress_reward = torch.clamp(
            delta_chair / self.chair_progress_scale,
            min=0.0,
            max=1.0,
        )
        chair_progress_reward[invalid_chair] = 0.0
        self.prev_chair_move = chair_move.clone()

        # -------------------------------------------------
        # 4) chair moved enough state reward
        # checker chce jen "alespoň dost"
        # takže reward saturuje na 1 po dosažení goal
        # -------------------------------------------------
        chair_state_reward = torch.clamp(
            chair_move / self.chair_move_goal,
            min=0.0,
            max=1.0,
        )

        # -------------------------------------------------
        # 5) motion toward robot target
        # -------------------------------------------------
        target_speed = torch.sum(root_vel_xy * dir_to_target, dim=-1)
        target_motion_reward = torch.clamp(
            target_speed / self.target_speed_scale,
            min=0.0,
            max=1.0,
        )

        # -------------------------------------------------
        # 6) backward motion reward in body frame
        # -------------------------------------------------
        w = root_quat[:, 0]
        x = root_quat[:, 1]
        y = root_quat[:, 2]
        z = root_quat[:, 3]

        forward_x = 1.0 - 2.0 * (y**2 + z**2)
        forward_y = 2.0 * (x * y + w * z)

        body_forward = torch.stack([forward_x, forward_y], dim=-1)
        body_forward = body_forward / (torch.norm(body_forward, dim=-1, keepdim=True) + 1e-6)

        body_forward_speed = torch.sum(root_vel_xy * body_forward, dim=-1)
        backward_reward = torch.clamp(
            -body_forward_speed / self.backward_speed_scale,
            min=0.0,
            max=1.0,
        )

        # -------------------------------------------------
        # 7) still reward only near target
        # -------------------------------------------------
        lin_speed = torch.norm(root_vel_xy, dim=-1)
        yaw_speed = torch.abs(root_ang_vel[:, 2])

        raw_still_reward = torch.exp(
            -(lin_speed ** 2) / (2.0 * self.still_lin_sigma ** 2)
        ) * torch.exp(
            -(yaw_speed ** 2) / (2.0 * self.still_yaw_sigma ** 2)
        )

        still_gate = (dist_to_target < self.still_activation_dist).float()
        still_reward = raw_still_reward * still_gate

        # -------------------------------------------------
        # 8) arms down factor
        # -------------------------------------------------
        q_left = robot.joint_pos[:, self.left_idx]
        q_right = robot.joint_pos[:, self.right_idx]

        arm_err = 0.5 * (
            torch.abs(q_left - self.down_target_left) +
            torch.abs(q_right - self.down_target_right)
        )

        arms_down_reward = torch.exp(
            -(arm_err ** 2) / (2.0 * self.arm_sigma ** 2)
        )
        arms_down_reward = torch.clamp(arms_down_reward, 0.0, 1.0)

        # -------------------------------------------------
        # final
        # -------------------------------------------------
        base_reward = (
            0.22 * robot_progress_reward +
            0.20 * robot_state_reward +
            0.12 * chair_progress_reward +
            0.16 * chair_state_reward +
            0.12 * target_motion_reward +
            0.10 * backward_reward +
            0.08 * still_reward
        )

        # ruce dole jsou pro stage 2 nutné -> použijeme silný gate
        total_reward = base_reward * (0.25 + 0.75 * arms_down_reward)

        total_reward = torch.clamp(total_reward, 0.0, 1.0)
        return total_reward * stage_mask.float()


class StillnessReward(HumanoidBaseReward):
    """
    Reward for standing still.

    Active in stage 1 and stage 3.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        lin_sigma=0.08,
        yaw_sigma=0.25,
    ):
        super().__init__(robot_name)
        self.active_stages = [1, 3]
        self.lin_sigma = lin_sigma
        self.yaw_sigma = yaw_sigma
        self.base_idx = None

    def _lazy_init(self, robot):
        if self.base_idx is None:
            self.base_idx = robot.body_names.index("pelvis")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
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

        lin_vel = robot.body_state[:, self.base_idx, 7:10]
        ang_vel = robot.body_state[:, self.base_idx, 10:13]

        lin_speed = torch.norm(lin_vel[:, :2], dim=-1)
        yaw_speed = torch.abs(ang_vel[:, 2])

        lin_reward = torch.exp(-(lin_speed ** 2) / (2.0 * self.lin_sigma ** 2))
        yaw_reward = torch.exp(-(yaw_speed ** 2) / (2.0 * self.yaw_sigma ** 2))

        reward = torch.clamp(lin_reward * yaw_reward, 0.0, 1.0)
        return reward * stage_mask.float()


class StageProgressCfg(HumanoidBaseReward):
    """
    Small bonus when a new stage is completed.
    """
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None or self.completed_stages is None:
            return torch.zeros(num_envs, device=device)

        if self.completed_stages.any():
            ret = self.completed_stages.float()
            self.completed_stages = torch.zeros_like(self.completed_stages)
            return ret

        return torch.zeros(num_envs, device=device)
class ChairStillPenalty(HumanoidBaseReward):
    """
    Stage 0:
    Penalizace za pohyb židle.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        speed_scale=0.08,
    ):
        super().__init__(robot_name)
        self.active_stages = [0]
        self.speed_scale = speed_scale

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
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

        chair_idx = states.objects["chair"].body_names.index("base_link")
        chair_lin_vel = states.objects["chair"].body_state[:, chair_idx, 7:10]

        chair_speed = torch.norm(chair_lin_vel[:, :2], dim=-1)

        penalty = torch.clamp(chair_speed / self.speed_scale, min=0.0, max=1.0)
        return penalty * stage_mask.float()
class ContinuousStageReward(HumanoidBaseReward):
    """
    Permanentní reward za aktuální stage.
    """
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name

        if self.actual_stage is None:
            robot = states.robots[robot_name]
            num_envs = robot.joint_pos.shape[0]
            device = robot.joint_pos.device
            return torch.zeros(num_envs, device=device)

        return self.actual_stage.float()
class FinalReward(HumanoidBaseReward):
    """
    Permanentní reward za aktuální stage.
    """
    def __init__(self, robot_name="g1_slider_simple"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
        mask_success = self.actual_stage == 4
        return mask_success.float()
class FaceChairReward(HumanoidBaseReward):
    """
    Reward za to, že pelvis míří čelem k židli.

    Output: <0, 1>
    Use with POSITIVE weight.
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        active_stages=(0, 1, 2, 3),
        sharpness=6.0,
        only_forward=True,
    ):
        super().__init__(robot_name)
        self.active_stages = list(active_stages)
        self.sharpness = sharpness
        self.only_forward = only_forward
        self.base_idx = None

    def _lazy_init(self, robot):
        if self.base_idx is None:
            self.base_idx = robot.body_names.index("pelvis")

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot_name = robot_name or self.robot_name
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

        self._lazy_init(robot)

        root_pos = robot.body_state[:, self.base_idx, :3]
        root_quat = robot.body_state[:, self.base_idx, 3:7]  # [w, x, y, z]

        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]

        # směr k židli v XY
        vec_to_chair = chair_pos - root_pos
        vec_to_chair[:, 2] = 0.0
        dir_to_chair = vec_to_chair[:, :2] / (
            torch.norm(vec_to_chair[:, :2], dim=-1, keepdim=True) + 1e-6
        )

        # forward osa pelvisu v XY
        w = root_quat[:, 0]
        x = root_quat[:, 1]
        y = root_quat[:, 2]
        z = root_quat[:, 3]

        forward_x = 1.0 - 2.0 * (y**2 + z**2)
        forward_y = 2.0 * (x * y + w * z)

        body_forward = torch.stack([forward_x, forward_y], dim=-1)
        body_forward = body_forward / (
            torch.norm(body_forward, dim=-1, keepdim=True) + 1e-6
        )

        # cos úhlu mezi forward směrem pelvisu a směrem k židli
        alignment = torch.sum(body_forward * dir_to_chair, dim=-1)  # <- v intervalu [-1, 1]

        if self.only_forward:
            # bokem nebo dozadu = 0 reward
            reward = torch.clamp(alignment, min=0.0, max=1.0) ** self.sharpness
        else:
            # dozadu stále něco málo dostane
            reward = torch.clamp((alignment + 1.0) / 2.0, 0.0, 1.0) ** self.sharpness

        return reward * stage_mask.float()
# =============================================================================
# WEIGHTS
# reward functions output either:
# - reward in <0,1>  -> use positive weight
# - penalty in <0,1> -> use negative weight
# =============================================================================

TERMINATION_WEIGHT = -1000.0

# General optional penalties / rewards
DELTA_ACTION_RATE_WEIGHT = -0.002
DOF_VELOCITY_ACCELERATION_WEIGHT = -0.002

CHAIR_APPROACH_WEIGHT = 2.0
ARM_POSTURE_WEIGHT = 2.0
CHAIR_STILL_PENALTY_WEIGHT = -0.5
CHAIR_PULL_WEIGHT = 5.0
STILLNESS_WEIGHT = 1.5
STAGE_PROGRESS_WEIGHT = 4.0
CONTINUOUS_STAGE_WEIGHT = 4.0
ARM_PENALTY_WEIGHT = -5.0
FACE_CHAIR_WEIGHT = 2.0
FINAL_WEIGHT = 3000.0

# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class ChairmansimpleCfg(HumanoidTaskCfg):
    """Simple chair task for humanoid robots."""

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
        RigidObjCfg(
            name="room",
            urdf_path="/home/roboversepc/Documents/rooms/room5/room.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            batch_fixed_verts=True
        )
    ]

    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"
    checker = _ChairManCheckerSimple()

    reward_weights = [
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,
        CHAIR_APPROACH_WEIGHT,
        ARM_POSTURE_WEIGHT,
        CHAIR_PULL_WEIGHT,
        STILLNESS_WEIGHT,
        STAGE_PROGRESS_WEIGHT,
        CHAIR_STILL_PENALTY_WEIGHT,
        CONTINUOUS_STAGE_WEIGHT,
        ARM_PENALTY_WEIGHT,
        FACE_CHAIR_WEIGHT,
        FINAL_WEIGHT,
    ]

    reward_functions = [
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),
        WalkToChairDirectReward(),
        ArmPostureReward(),
        BackToTargetReward(),
        StillnessReward(),
        StageProgressCfg(),
        ChairStillPenalty(),
        ContinuousStageReward(),
        ArmPosturePenalty(),
        FaceChairReward(),
        FinalReward(),
    ]

    def extra_spec(self):
        return {}
@configclass
class ChairmansimplegrpoCfg(HumanoidTaskCfg):
    """ChairMan task config for GRPO fine-tuning."""

    success_bar = 0.0
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
        RigidObjCfg(
            name="room",
            urdf_path="/home/roboversepc/Documents/rooms/room5/room.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            batch_fixed_verts=True
        ),
    ]

    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"

    # necháme původní checker – ten už ví, kdy je task solved / failed
    checker = _ChairManCheckerSimpleGRPO()

    # jen jednoduché regularizační penalty
    reward_weights = [
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,
    ]

    reward_functions = [
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),
    ]

    def extra_spec(self):
        return {}


@configclass
class ChairmansimplegaussianCfg(ChairmansimpleCfg):
    """ChairMan simple task with the room supplied by a Gaussian splat camera."""

    objects = [
        ArticulationObjCfg(
            name="chair",
            urdf_path="roboverse_data/assets/humanoidbench/chairs/chair1/foldable_chair_debug.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False,
            batch_fixed_verts=True,
        ),
    ]


@configclass
class ChairmansimplegaussiangrpoCfg(ChairmansimplegrpoCfg):
    """ChairMan GRPO task with the room supplied by a Gaussian splat camera."""

    objects = [
        ArticulationObjCfg(
            name="chair",
            urdf_path="roboverse_data/assets/humanoidbench/chairs/chair1/foldable_chair_debug.urdf",
            default_position=[0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False,
            batch_fixed_verts=True,
        ),
    ]
