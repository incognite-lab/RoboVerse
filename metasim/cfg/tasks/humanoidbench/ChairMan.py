"""ChairMan task rewards - version focused only up to Stage 2."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.types import EnvState
from metasim.utils import configclass
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
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
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


class DofPositionLimitsCfg(HumanoidBaseReward):
    """
    Soft penalty magnitude for approaching joint limits.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider", target_speed=0.8):
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


class KeepChairStillPenalty(HumanoidBaseReward):
    """
    Stage 0 and 1:
    Penalty magnitude for moving the chair before grasp.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        # Může být aktivní ve všech fázích, kdy chceme, aby robot sledoval cíl
        self.active_stages = [0, 1, 2, 3, 4, 5]

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
    Stage 1:
    Reward for bringing both hands closer to chair handles.

    Combines:
    1) progress per step,
    2) current state reward.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.progress_scale = 0.015
        self.state_scale = 0.20

        self.prev_mean_dist = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_mean_dist is not None:
            self.prev_mean_dist[env_ids] = 0.0

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
        mean_dist = 0.5 * (dist_left + dist_right)

        if self.prev_mean_dist is None:
            self.prev_mean_dist = mean_dist.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            delta = self.prev_mean_dist - mean_dist
            progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
            self.prev_mean_dist = mean_dist.clone()

        state_reward = torch.clamp(1.0 - mean_dist / self.state_scale, min=0.0, max=1.0)

        total_reward = 0.30 * progress_reward + 0.70 * state_reward
        return total_reward * stage_mask.float()


class HandOrientationProgressReward(HumanoidBaseReward):
    """
    Stage 1:
    Reward for aligning both hand orientations with handle target orientations.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.progress_scale = 0.08
        self.state_scale = 0.50

        self.prev_mean_angle = None

    def _quat_diff_angle(self, q1, q2):
        dot = torch.sum(q1 * q2, dim=-1)
        dot = torch.clamp(torch.abs(dot), max=1.0)
        return 2.0 * torch.acos(dot)

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_mean_angle is not None:
            self.prev_mean_angle[env_ids] = 0.0

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

        angle_left = self._quat_diff_angle(q_hand_left, q_target_left)
        angle_right = self._quat_diff_angle(q_hand_right, q_target_right)
        mean_angle = 0.5 * (angle_left + angle_right)

        if self.prev_mean_angle is None:
            self.prev_mean_angle = mean_angle.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            delta = self.prev_mean_angle - mean_angle
            progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
            self.prev_mean_angle = mean_angle.clone()

        state_reward = torch.clamp(1.0 - mean_angle / self.state_scale, min=0.0, max=1.0)

        total_reward = 0.40 * progress_reward + 0.60 * state_reward
        return total_reward * stage_mask.float()


class HandTargetStillnessReward(HumanoidBaseReward):
    """
    Stage 1 and 2:
    Reward for having both hands:
    1) close to targets,
    2) with low linear velocity.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1,2]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        self.dist_scale = 0.08
        self.vel_scale = 0.12

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
        mean_dist = 0.5 * (dist_left + dist_right)

        vel_left = torch.norm(v_hand_left, dim=-1)
        vel_right = torch.norm(v_hand_right, dim=-1)
        mean_vel = 0.5 * (vel_left + vel_right)

        dist_reward = torch.clamp(1.0 - mean_dist / self.dist_scale, min=0.0, max=1.0)
        vel_reward = torch.clamp(1.0 - mean_vel / self.vel_scale, min=0.0, max=1.0)

        total_reward = dist_reward * vel_reward
        return total_reward * stage_mask.float()


class StayNearAnchorReward(HumanoidBaseReward):
    """
    Stage 1 and 2:
    Reward for keeping pelvis near anchor position.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
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
    Stage 2:
    Progressive finger-closing reward.

    Kombinuje:
    1) progress per step - zda se robot mezi kroky přiblížil cílovému sevření
    2) state reward - jak blízko je aktuálně cílovému sevření

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [2]

        self.threshold = 0.10

        # škálování
        self.progress_scale = 0.01   # cca 0.02 rad průměrného zlepšení -> reward 1
        self.state_scale = 1.0     # průměrná chyba 0.35 rad -> reward 0

        self.finger_targets_dict = {
            "left_hand_thumb_0_joint": 0.396 + self.threshold,
            "left_hand_thumb_1_joint": 0.214 + self.threshold,
            "left_hand_thumb_2_joint": 0.357 + self.threshold,
            "left_hand_middle_0_joint": -0.523 - self.threshold,
            "left_hand_middle_1_joint": -0.527 - self.threshold,
            "left_hand_index_0_joint": -0.485 - self.threshold,
            "left_hand_index_1_joint": -0.542 - self.threshold,

            "right_hand_thumb_0_joint": -0.389 - self.threshold,
            "right_hand_thumb_1_joint": -0.208 - self.threshold,
            "right_hand_thumb_2_joint": -0.358 - self.threshold,
            "right_hand_middle_0_joint": 0.505 + self.threshold,
            "right_hand_middle_1_joint": 0.518 + self.threshold,
            "right_hand_index_0_joint": 0.485 + self.threshold,
            "right_hand_index_1_joint": 0.541 + self.threshold,
        }

        self.finger_indices = None
        self.target_tensor = None
        self.prev_mean_error = None

    def reset(self, env_ids: torch.Tensor, states: list["EnvState"]):
        if self.prev_mean_error is not None:
            self.prev_mean_error[env_ids] = 0.0

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

            if not indices:
                return torch.zeros(num_envs, device=device)

            self.finger_indices = torch.tensor(indices, device=device, dtype=torch.long)
            self.target_tensor = torch.tensor(targets, device=device).unsqueeze(0)

        q_finger = robot.joint_pos[:, self.finger_indices]

        # průměrná absolutní chyba vůči cílovému sevření
        mean_error = torch.mean(torch.abs(q_finger - self.target_tensor), dim=-1)

        # 1) progress reward
        if self.prev_mean_error is None:
            self.prev_mean_error = mean_error.clone()
            progress_reward = torch.zeros(num_envs, device=device)
        else:
            delta = self.prev_mean_error - mean_error
            progress_reward = torch.clamp(delta / self.progress_scale, min=0.0, max=1.0)
            self.prev_mean_error = mean_error.clone()

        # 2) state reward
        state_reward = torch.clamp(1.0 - mean_error / self.state_scale, min=0.0, max=1.0)

        total_reward = (
            0.30 * progress_reward +
            0.70 * state_reward
        )

        return total_reward * stage_mask.float()
class GraspForceReward(HumanoidBaseReward):
    """
    Stage 2:
    Exact fingertip grasp reward aligned with checker logic.

    Output: <0, 1>
    """
    def __init__(self, robot_name="g1_slider"):
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

        tip_rewards = torch.clamp(tip_forces / self.force_threshold, min=0.0, max=1.0)

        left_reward = torch.mean(tip_rewards[:, 0:3], dim=1)
        right_reward = torch.mean(tip_rewards[:, 3:6], dim=1)

        total_reward = torch.sqrt(left_reward * right_reward + 1e-8)
        total_reward = torch.clamp(total_reward, min=0.0, max=1.0)

        return total_reward * stage_mask.float()


# =============================================================================
# OPTIONAL / DISABLED REWARDS
# =============================================================================

class ArmRestingPosePenaltyCfg(HumanoidBaseReward):
    """
    Optional penalty magnitude for arm resting pose in stage 0.

    Output: <0, 1>
    Use with NEGATIVE weight.
    """
    def __init__(self, robot_name="g1_slider"):
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
    def __init__(self, robot_name="g1_slider"):
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
STAGE_PROGRESS_WEIGHT = 500.0
CONTINUOUS_STAGE_REWARD_WEIGHT = 4.0

# Stage 0
WALK_TO_CHAIR_REWARD_WEIGHT = 4.0
OPEN_GRASP_REWARD_WEIGHT = 0.8
KEEP_CHAIR_STILL_PENALTY_WEIGHT = -1.0

# Stage 1
REACH_CHAIR_REWARD_WEIGHT = 5.0
REACH_ORIENTATION_REWARD_WEIGHT = 2.5
HAND_TARGET_STILLNESS_REWARD_WEIGHT = 4.0
STAY_NEAR_ANCHOR_REWARD_WEIGHT = 1.5

# Stage 2
CLOSE_GRASP_REWARD_WEIGHT = 5.5
FORCE_GRASP_REWARD_WEIGHT = 4.0


# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class ChairmanCfg(HumanoidTaskCfg):
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
        RigidObjCfg(
            name="room",
            urdf_path="/home/roboversepc/Documents/rooms/room5/room.urdf",
            default_position= [0.0, 0.0, 0.0],
            fix_base_link=True
        )
    ]

    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"
    checker = _ChairManChecker()

    reward_weights = [
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,
        DOF_POSITION_LIMITS_WEIGHT,
        HUMANLY_DOF_LIMIT_WEIGHT,
        # UPRIGHT_PENALTY_WEIGHT,

        WALK_TO_CHAIR_REWARD_WEIGHT,
        OPEN_GRASP_REWARD_WEIGHT,
        KEEP_CHAIR_STILL_PENALTY_WEIGHT,

        REACH_CHAIR_REWARD_WEIGHT,
        REACH_ORIENTATION_REWARD_WEIGHT,
        HAND_TARGET_STILLNESS_REWARD_WEIGHT,
        STAY_NEAR_ANCHOR_REWARD_WEIGHT,

        CLOSE_GRASP_REWARD_WEIGHT,
        FORCE_GRASP_REWARD_WEIGHT,

        FACE_CHAIR_REWARD_WEIGHT,
        ARM_RESTING_POSE_PENALTY_WEIGHT,
        STAGE_PROGRESS_WEIGHT,
        #CONTINUOUS_STAGE_REWARD_WEIGHT,
    ]

    reward_functions = [
        DeltaActionRateCfg(),
        DoFVelocityAccelerationCfg(),
        DofPositionLimitsCfg(),
        HumanlyDofLimitCfg(),
        # UprightPenaltyCfg(),

        WalkToChairProgressReward(),
        OpenGraspReward(),
        KeepChairStillPenalty(),

        ReachChairProgressReward(),
        HandOrientationProgressReward(),
        HandTargetStillnessReward(),
        StayNearAnchorReward(),

        CloseGraspReward(),
        GraspForceReward(),

        FaceChairReward(),
        ArmRestingPosePenaltyCfg(),
        StageProgressCfg(),
        #ContinuousStageReward(),
    ]

    def extra_spec(self):
        return {}
