"""ChairMan simple task rewards."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairManCheckerSimple
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


class ChairApproachReward(HumanoidBaseReward):
    """
    Stage 0:
    Reward only for direct motion toward the pre-grasp point in front of the chair.

    Robot is rewarded mainly when:
    - distance to target decreases
    - velocity points toward target
    - lateral motion is small

    Output: <0, 1>
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        target_distance_x=0.75,
        progress_scale=0.02,
        target_sigma=0.12,
        direction_speed_scale=0.25,
        lateral_speed_scale=0.08,
    ):
        super().__init__(robot_name)
        self.active_stages = [0]

        self.target_distance_x = target_distance_x
        self.progress_scale = progress_scale
        self.target_sigma = target_sigma
        self.direction_speed_scale = direction_speed_scale
        self.lateral_speed_scale = lateral_speed_scale

        self.chair_pos = None
        self.prev_dist = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        chair = states.objects["chair"]
        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]

        if self.chair_pos is None:
            self.chair_pos = chair_pos.clone()
        else:
            self.chair_pos[env_ids] = chair_pos[env_ids].clone()

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

        base_idx = robot.body_names.index("pelvis")
        root_pos = robot.body_state[:, base_idx, :3]
        root_vel = robot.body_state[:, base_idx, 7:10]

        chair_base_idx = chair.body_names.index("base_link")
        if self.chair_pos is None:
            self.chair_pos = chair.body_state[:, chair_base_idx, :3].clone()

        chair_pos = self.chair_pos

        # -------------------------------------------------
        # target point = point in front of the chair
        # robot should approach this point directly
        # -------------------------------------------------
        target_pos = torch.zeros_like(root_pos)
        target_pos[:, 0] = chair_pos[:, 0] - self.target_distance_x
        target_pos[:, 1] = chair_pos[:, 1]
        target_pos[:, 2] = root_pos[:, 2]

        vec_to_target = target_pos - root_pos
        vec_to_target[:, 2] = 0.0
        dist = torch.norm(vec_to_target, dim=-1)
        dir_to_target = vec_to_target / (dist.unsqueeze(-1) + 1e-6)

        root_vel_xy = root_vel[:, :2]
        dir_xy = dir_to_target[:, :2]

        # -------------------------------------------------
        # 1) progress reward
        # only if distance truly decreases
        # -------------------------------------------------
        invalid_mask = torch.isnan(self.prev_dist)
        delta_dist = self.prev_dist - dist
        progress_reward = torch.clamp(delta_dist / self.progress_scale, min=0.0, max=1.0)
        progress_reward[invalid_mask] = 0.0
        self.prev_dist = dist.clone()

        # -------------------------------------------------
        # 2) state reward
        # reward for being close to the exact target point
        # -------------------------------------------------
        state_reward = torch.exp(-(dist ** 2) / (2.0 * self.target_sigma ** 2))
        state_reward = torch.clamp(state_reward, 0.0, 1.0)

        # -------------------------------------------------
        # 3) direction reward
        # reward only forward motion toward target
        # -------------------------------------------------
        forward_speed = torch.sum(root_vel_xy * dir_xy, dim=-1)
        direction_reward = torch.clamp(
            forward_speed / self.direction_speed_scale,
            min=0.0,
            max=1.0
        )

        # -------------------------------------------------
        # 4) lateral motion suppression
        # if robot moves sideways, reward is strongly reduced
        # -------------------------------------------------
        lateral_vec = root_vel_xy - forward_speed.unsqueeze(-1) * dir_xy
        lateral_speed = torch.norm(lateral_vec, dim=-1)

        lateral_penalty = torch.clamp(
            lateral_speed / self.lateral_speed_scale,
            min=0.0,
            max=1.0
        )
        lateral_factor = 1.0 - lateral_penalty

        # -------------------------------------------------
        # 5) backward motion suppression
        # if robot moves away from target, kill reward
        # -------------------------------------------------
        backward_amount = torch.clamp(-forward_speed, min=0.0)
        backward_penalty = torch.clamp(
            backward_amount / self.direction_speed_scale,
            min=0.0,
            max=1.0
        )
        backward_factor = 1.0 - backward_penalty

        # -------------------------------------------------
        # final reward
        # only good if robot goes directly toward the target
        # -------------------------------------------------
        base_reward = (
            0.35 * progress_reward +
            0.25 * state_reward +
            0.40 * direction_reward
        )

        reward = base_reward * lateral_factor * backward_factor
        reward = torch.clamp(reward, 0.0, 1.0)

        return reward * stage_mask.float()


class ChairPullReward(HumanoidBaseReward):
    """
    Stage 2:
    Reward for moving the chair backward.

    Simple version:
    - 30 % chair progress reward
    - 50 % chair state reward
    - 20 % backward motion reward
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        target_chair_x=0.45,
        progress_scale=0.02,
        chair_sigma=0.10,
        backward_speed_scale=0.25,
    ):
        super().__init__(robot_name)
        self.active_stages = [2]

        self.target_chair_x = target_chair_x
        self.progress_scale = progress_scale
        self.chair_sigma = chair_sigma
        self.backward_speed_scale = backward_speed_scale

        self.prev_chair_x = None
        self.base_idx = None

    def _lazy_init(self, robot):
        if self.base_idx is None:
            self.base_idx = robot.body_names.index("pelvis")

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        robot = states.robots[self.robot_name]
        self._lazy_init(robot)

        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.prev_chair_x is None:
            self.prev_chair_x = torch.full((num_envs,), float("nan"), device=device)
        else:
            self.prev_chair_x[env_ids] = float("nan")

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

        root_vel = robot.body_state[:, self.base_idx, 7:10]
        root_quat = robot.body_state[:, self.base_idx, 3:7]

        chair_base_idx = chair.body_names.index("base_link")
        chair_x = chair.body_state[:, chair_base_idx, 0]

        # -------------------------------------------------
        # 1) chair progress reward
        # -------------------------------------------------
        invalid_mask = torch.isnan(self.prev_chair_x)
        delta_x = self.prev_chair_x - chair_x
        progress_reward = torch.clamp(delta_x / self.progress_scale, min=0.0, max=1.0)
        progress_reward[invalid_mask] = 0.0
        self.prev_chair_x = chair_x.clone()

        # -------------------------------------------------
        # 2) chair state reward
        # -------------------------------------------------
        chair_err = torch.abs(chair_x - self.target_chair_x)
        state_reward = torch.exp(-(chair_err ** 2) / (2.0 * self.chair_sigma ** 2))
        state_reward = torch.clamp(state_reward, 0.0, 1.0)

        # -------------------------------------------------
        # 3) backward motion reward in body frame
        # -------------------------------------------------
        w = root_quat[:, 0]
        x = root_quat[:, 1]
        y = root_quat[:, 2]
        z = root_quat[:, 3]

        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x * y + w * z)

        body_forward = torch.stack([forward_x, forward_y], dim=-1)
        body_forward = body_forward / (torch.norm(body_forward, dim=-1, keepdim=True) + 1e-6)

        root_vel_xy = root_vel[:, :2]
        body_forward_speed = torch.sum(root_vel_xy * body_forward, dim=-1)

        backward_amount = torch.clamp(-body_forward_speed, min=0.0)
        backward_reward = torch.clamp(backward_amount / self.backward_speed_scale, min=0.0, max=1.0)

        reward = 0.3 * progress_reward + 0.5 * state_reward + 0.2 * backward_reward
        reward = torch.clamp(reward, 0.0, 1.0)

        return reward * stage_mask.float()


class ArmPostureReward(HumanoidBaseReward):
    """
    Reward for desired shoulder posture by stage.

    Simple version:
    - 30 % progress reward
    - 70 % state reward
    """
    def __init__(
        self,
        robot_name="g1_slider_simple",
        up_target_left=-1.86,
        up_target_right=-1.86,
        down_target_left=-1.32,
        down_target_right=-1.32,
        sigma=0.1,
        progress_scale=0.04,
    ):
        super().__init__(robot_name)

        self.up_stages = [0, 3]
        self.down_stages = [1, 2]

        self.up_target_left = up_target_left
        self.up_target_right = up_target_right
        self.down_target_left = down_target_left
        self.down_target_right = down_target_right

        self.sigma = sigma
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
        err = torch.sqrt(0.5 * (err_left**2 + err_right**2))

        # -------------------------------------------------
        # 1) state reward
        # -------------------------------------------------
        state_reward = torch.exp(-(err ** 2) / (2.0 * self.sigma ** 2))
        state_reward = torch.clamp(state_reward, 0.0, 1.0)

        # -------------------------------------------------
        # 2) progress reward
        # -------------------------------------------------
        invalid_mask = torch.isnan(self.prev_err)
        delta_err = self.prev_err - err
        progress_reward = torch.clamp(delta_err / self.progress_scale, min=0.0, max=1.0)
        progress_reward[invalid_mask] = 0.0
        self.prev_err = err.clone()

        reward = 0.01 * progress_reward + 0.99 * state_reward
        reward = torch.clamp(reward, 0.0, 1.0)

        return reward * active_mask.float()


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
    Continuous Stage Reward: Dává permanentní odměnu za to, ve kterém Stage se robot nachází.
    Stage 0 = 0 bodů
    Stage 1 = 1 * váha
    Stage 2 = 2 * váha
    ... atd.

    Tímto robotovi jasně říkáme, že udržet se v pozdějších fázích je matematicky
    nejvýhodnější věc v celé hře.
    """
    def __init__(self, robot_name="g1_slider"):
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


class ArmPoseErrorPenalty(HumanoidBaseReward):
    """
    Penalty for shoulder posture error by stage.

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
        scale=0.5,
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

        err_left = torch.abs(q_left - target_left)
        err_right = torch.abs(q_right - target_right)
        err = 0.5 * (err_left + err_right)

        penalty = torch.clamp(err / self.scale, min=0.0, max=1.0)
        return penalty * active_mask.float()
# =============================================================================
# WEIGHTS
# reward functions output either:
# - reward in <0,1>  -> use positive weight
# - penalty in <0,1> -> use negative weight
# =============================================================================

TERMINATION_WEIGHT = -1000.0

# General optional penalties / rewards
DELTA_ACTION_RATE_WEIGHT = -0.01
DOF_VELOCITY_ACCELERATION_WEIGHT = -0.01

CHAIR_APPROACH_WEIGHT = 2.0
ARM_POSTURE_WEIGHT = 3.0
CHAIR_PULL_WEIGHT = 3.0
STILLNESS_WEIGHT = 1.5
STAGE_PROGRESS_WEIGHT = 50.0
CHAIR_STILL_PENALTY_WEIGHT = -1.0
CONTINUOUS_STAGE_WEIGHT = 4.0
ARM_PENALITY_WEIGHT = -1.5

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
        # RigidObjCfg(
        #     name="room",
        #     urdf_path="/home/roboversepc/Documents/rooms/room5/room.urdf",
        #     default_position=[0.0, 0.0, 0.0],
        #     fix_base_link=True
        # )
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
        ARM_PENALITY_WEIGHT,
    ]

    reward_functions = [
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),
        ChairApproachReward(),
        ArmPostureReward(),
        ChairPullReward(),
        StillnessReward(),
        StageProgressCfg(),
        ChairStillPenalty(),
        ContinuousStageReward(),
        ArmPoseErrorPenalty(),
    ]

    def extra_spec(self):
        return {}
