"""Chairman2 rewards, grouped by stage like ChairMan_multi.

Stages: 0 walk with arms tucked, 1 extend above backrest, 2 cross toward
the seat, 3 pull, 4 release.
Each reward owns its mask, state and reset. Targets remain shared with checkers
in chairman2_geometry; weights and task registration are at the end of this file.
"""
from __future__ import annotations

import math
import torch
from metasim.types import EnvState
from metasim.cfg.checkers import _ChairMan2Checker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass
from metasim.utils import chairman2_geometry as g
from metasim.utils.chair_navigation import chair_back_direction_xy, forward_direction_xy, smoothstep01
from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


def _stage_mask(actual_stage, stages):
    mask = torch.zeros_like(actual_stage, dtype=torch.bool)
    for stage in stages:
        mask |= actual_stage == stage
    return mask


def bounded(error, scale):
    x = error / scale
    return x / (1 + x)


def bilateral(error, scale):
    # The worst hand/joint matters more than the average.
    return 0.75 * bounded(error.amax(-1), scale) + 0.25 * bounded(error.mean(-1), scale)


def heading_cost(m):
    # Strict decrease of reward for every increase of torso heading error.
    return 1 - torch.exp(-math.log(2) * m['heading'] / math.radians(20))


def still_cost(m):
    return bounded(m['robot_speed'], g.STILL_SPEED) + bounded(m['robot_yaw_speed'], g.STILL_YAW_SPEED)


def anchor_cost(m):
    return bounded(m['robot_drift'], g.ROBOT_DRIFT_TOLERANCE) + bounded(m['chair_drift'], g.CHAIR_DRIFT_TOLERANCE)


def chair_still_cost(m):
    return bounded(m['chair_speed'], g.STILL_SPEED) + bounded(m['chair_yaw_speed'], g.STILL_YAW_SPEED)


def hand_shape_cost(m):
    return bilateral(m['palm_angle'], g.PALM_ANGLE_TOLERANCE) + bilateral(m['elbow_angle'], g.ELBOW_ANGLE_TOLERANCE)




# =============================================================================
# BASIC / AUXILIARY REWARDS
# =============================================================================

class MotionRegularization(HumanoidBaseReward):
    """Ve všech stages penalizuje prudké povely, rychlé klouby a náklon trupu."""

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.previous_targets = None
        self.command = self.previous_command = None

    def set_control_context(self, command, previous_command, device=None):
        self.command = torch.as_tensor(command, device=device).clone()
        self.previous_command = torch.as_tensor(previous_command, device=device).clone()

    def reset(self, env_ids, states):
        if self.previous_targets is not None:
            self.previous_targets[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        ids = [list(robot.joint_names).index(name) for name in g.STAGE0_JOINT_TARGETS]
        targets = getattr(robot, 'joint_pos_target', robot.joint_pos)[:, ids]
        if self.previous_targets is None:
            self.previous_targets = targets.detach().clone()
        previous = torch.where(torch.isnan(self.previous_targets), targets, self.previous_targets)
        rate = ((targets-previous).abs()/max(self.control_dt, 1e-6)).mean(-1)
        speed = torch.relu(robot.joint_vel[:, ids].abs()-1.5).mean(-1)
        self.previous_targets = targets.detach().clone()
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name or self.robot_name, self)
        cost = (
            0.03 * bounded(rate, 3.0)
            + 0.05 * bounded(speed, 1.5)
            + 0.01 * (1 - m['upright']).clamp(min=0)
        )
        if self.command is not None:
            cost += 0.02*bounded((self.command-self.previous_command).abs().mean(-1)/self.control_dt, 3.0)
        return -torch.nan_to_num(cost, nan=1.0, posinf=1.0)*self.control_dt/0.02


class UpperBodyCenterOfMassReward(HumanoidBaseReward):
    """Ve všech stages penalizuje vodorovné vychýlení COM vršku těla od pelvisu."""

    def __init__(self, robot_name="g1_without_hands"):
        super().__init__('g1_with_hands')
        self.robot_name = robot_name
        self.metrics = None
        self.control_dt = 0.01

    def __call__(self, states, robot_name=None):
        name = robot_name or self.robot_name
        robot = states.robots[name]
        metrics = self.metrics if self.metrics is not None else g.measure(states, name, self)
        outside = torch.relu(
            metrics['upper_body_com_horizontal_error'] - g.UPPER_BODY_COM_DEADZONE
        )
        normalized = outside / g.UPPER_BODY_COM_SCALE
        cost = normalized.square() / (1 + normalized.square())
        # Zero is the optimum. A non-positive running reward cannot be farmed
        # by delaying stage completion, and dt scaling keeps its strength
        # stable when simulation decimation changes.
        #print("UpperBodyCenterOfMassReward cost:", (-torch.nan_to_num(cost, nan=1.0, posinf=1.0) * self.control_dt / 0.02))
        return -torch.nan_to_num(cost, nan=1.0, posinf=1.0) * self.control_dt / 0.02


# =============================================================================
# STAGE COMPLETION / FAILURE
# =============================================================================

class StageOutcomeReward(HumanoidBaseReward):
    """Dává bonus za dokončení stage a penalizaci za neúspěšné ukončení."""

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.termination_events = None

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        value = q.new_zeros(q.shape[0])
        if self.completed_stages is not None:
            # Do not consume checker events; all collectors need the same event.
            value = 10.0*self.completed_stages.float()
            if self.actual_stage is not None:
                value += 10.0*self.completed_stages.float()*(self.actual_stage == 4)
        if self.termination_events is not None:
            value = torch.where(self.termination_events, -10.0, value)
        return value


# =============================================================================
# STAGE 0: WALK TO CHAIR WITH ARMS READY
# =============================================================================

class WalkToChairProgressReward(HumanoidBaseReward):
    """Stage 0: odměňuje chůzi k cíli za židlí a následné zastavení."""
    def __init__(self, robot_name="g1_without_hands", target_speed=0.5):
        super().__init__('g1_with_hands')
        self.robot_name = robot_name
        self.active_stages = [0]
        self.final_distance = g.APPROACH_DISTANCE
        self.target_speed = target_speed
        self.min_walk_speed = 0.28
        self.slow_radius = 0.35
        self.stop_radius = g.POSITION_TOLERANCE
        self.distance_progress_scale = 0.02
        self.velocity_sigma = 0.18
        self.saved_chair_pos = None
        self.saved_chair_quat = None
        self.prev_final_distance = None

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

        if self.prev_final_distance is not None:
            self.prev_final_distance[env_ids] = torch.nan

        if hasattr(super(), "reset"):
            super().reset(env_ids, states)

    def __call__(self, states: list["EnvState"], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = _stage_mask(self.actual_stage, self.active_stages)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        base_idx = robot.body_names.index("pelvis")
        root_pos_xy = robot.body_state[:, base_idx, :2]
        root_vel_xy = robot.body_state[:, base_idx, 7:9]

        chair_base_idx = chair.body_names.index("base_link")
        if self.saved_chair_pos is None:
            chair_state = chair.body_state[:, chair_base_idx]
            self.saved_chair_pos = chair_state[:, :3].clone()
            self.saved_chair_quat = chair_state[:, 3:7].clone()

        chair_pos_xy = self.saved_chair_pos[:, :2]
        back_dir = chair_back_direction_xy(self.saved_chair_quat)
        final_pos = chair_pos_xy + self.final_distance * back_dir

        to_final = final_pos - root_pos_xy
        final_dist = torch.norm(to_final, dim=-1)
        final_dir = to_final / torch.clamp(final_dist.unsqueeze(-1), min=1.0e-6)

        if (
            self.prev_final_distance is None
            or self.prev_final_distance.shape != final_dist.shape
            or self.prev_final_distance.device != device
        ):
            self.prev_final_distance = final_dist.detach().clone()
            distance_progress = torch.zeros_like(final_dist)
        else:
            previous_distance = torch.where(
                torch.isnan(self.prev_final_distance), final_dist, self.prev_final_distance
            )
            distance_progress = torch.clamp(
                (previous_distance - final_dist) / self.distance_progress_scale,
                min=-1.0,
                max=1.0,
            )
            self.prev_final_distance = torch.where(
                stage_mask, final_dist.detach(), self.prev_final_distance
            )

        desired_speed = torch.where(
            final_dist > self.slow_radius,
            torch.full_like(final_dist, self.target_speed),
            torch.where(
                final_dist > self.stop_radius,
                torch.full_like(final_dist, self.min_walk_speed),
                torch.zeros_like(final_dist),
            ),
        )
        desired_velocity = final_dir * desired_speed.unsqueeze(-1)
        velocity_error = torch.norm(root_vel_xy - desired_velocity, dim=-1)
        tracking_reward = 2.0 * torch.exp(
            -torch.square(velocity_error) / (2.0 * self.velocity_sigma ** 2)
        ) - 1.0

        velocity_projection = torch.sum(root_vel_xy * final_dir, dim=-1)
        signed_direction = torch.clamp(
            velocity_projection / torch.clamp(desired_speed, min=self.min_walk_speed),
            min=-1.0,
            max=1.0,
        )

        speed_xy = torch.norm(root_vel_xy, dim=-1)
        stop_position_reward = smoothstep01(
            (self.stop_radius - final_dist) / self.stop_radius
        )
        stop_speed_reward = torch.clamp(1.0 - speed_xy / g.STILL_SPEED, min=0.0, max=1.0)
        arrival_stop_reward = stop_position_reward * stop_speed_reward

        moving_reward = (
            0.45 * signed_direction
            + 0.35 * distance_progress
            + 0.20 * tracking_reward
        )
        total_reward = torch.where(
            final_dist <= self.stop_radius,
            0.75 * arrival_stop_reward + 0.25 * distance_progress,
            moving_reward,
        )
        total_reward = torch.clamp(total_reward, min=-1.0, max=1.0)
        return total_reward * stage_mask.float()

class FaceChairReward(HumanoidBaseReward):
    """Stage 0: odměňuje natočení trupu směrem k židli."""
    def __init__(self, robot_name="g1_without_hands"):
        super().__init__('g1_with_hands')
        self.robot_name = robot_name
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


class Stage0ArmPos(HumanoidBaseReward):
    """Stage 0: odměňuje přiblížení všech zadaných kloubů k chodecké póze."""
    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.active_stage = 0
        self.previous_score = None
        self.progress_weight = 10.0
        self.remaining_error_weight = 0.01
        self.hold_bonus = 0.25

    def reset(self, env_ids, states):
        """Forget progress history only for environments that were reset."""
        if self.previous_score is not None:
            self.previous_score[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        robot = states.robots[robot_name or self.robot_name]
        if self.actual_stage is None:
            return robot.joint_pos.new_zeros(robot.joint_pos.shape[0])

        active = self.actual_stage == self.active_stage
        if not active.any():
            return robot.joint_pos.new_zeros(robot.joint_pos.shape[0])

        names = list(robot.joint_names)
        indices = [names.index(name) for name in g.STAGE0_JOINT_TARGETS]
        error = (robot.joint_pos[:, indices] - robot.joint_pos.new_tensor(
            list(g.STAGE0_JOINT_TARGETS.values()))).abs()
        normalized_error = error / g.JOINT_TOLERANCE

        # Unlike a narrow Gaussian (or a squared rational kernel), this score
        # keeps a strong slope when a joint starts many tolerances away.
        joint_score = 1.0 / (1.0 + normalized_error)
        mean_score = joint_score.mean(-1)
        worst_score = joint_score.amin(-1)
        inside_fraction = (error <= g.JOINT_TOLERANCE).float().mean(-1)
        current_score = (
            0.25 * mean_score
            + 0.65 * worst_score
            + 0.10 * inside_fraction
        )

        if (
            self.previous_score is None
            or self.previous_score.shape != current_score.shape
            or self.previous_score.device != current_score.device
        ):
            self.previous_score = torch.full_like(current_score, torch.nan)

        previous = torch.where(
            torch.isnan(self.previous_score), current_score, self.previous_score
        )
        progress = current_score - previous
        all_inside = (error <= g.JOINT_TOLERANCE).all(-1)

        reward = (
            self.progress_weight * progress
            - self.remaining_error_weight * (1.0 - current_score)
            + self.hold_bonus * all_inside.float()
        )
        self.previous_score = torch.where(
            active, current_score.detach(), self.previous_score
        )
        return torch.where(
            active,
            torch.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0),
            torch.zeros_like(reward),
        )


class KeepChairStillPenalty(HumanoidBaseReward):
    """Stage 0: penalizuje posun, rotaci a rychlost židle před manipulací."""
    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.active_stages = [0]

    def __call__(self, states, robot_name=None):
        robot_name = robot_name or self.robot_name
        if self.actual_stage is None:
            return states.robots[robot_name].joint_pos.new_zeros(states.robots[robot_name].joint_pos.shape[0])
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name, self)
        penalty = (bounded(m['chair_drift'], g.CHAIR_DRIFT_TOLERANCE)
                   + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
                   + chair_still_cost(m)) / 4.0
        return penalty * _stage_mask(self.actual_stage, self.active_stages)


# =============================================================================
# STAGE 1: EXTEND ARMS
# =============================================================================

class ExtendArmsReward(HumanoidBaseReward):
    """Stage 1: odměňuje natažení obou end-effectorů před tělo nad opěradlo."""
    stage = 1

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.previous_cost = None

    def reset(self, env_ids, states):
        if self.previous_cost is not None:
            self.previous_cost[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name or self.robot_name, self)
        reach_shortfall = torch.relu(g.HAND_FORWARD_REACH_MIN - m['hand_forward_reach'])
        height_shortfall = torch.relu(
            g.HAND_ABOVE_BACKREST_MIN - m['hand_height_above_backrest']
        )
        task_space_cost = (
            4.0 * bilateral(reach_shortfall, 0.15)
            + 3.0 * bilateral(height_shortfall, 0.15)
        )
        near_goal = (
            (reach_shortfall.amax(-1) <= 0.05)
            & (height_shortfall.amax(-1) <= 0.05)
        )
        cost = (
            task_space_cost
            + anchor_cost(m)
            + heading_cost(m)
            + chair_still_cost(m)
            + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
            + near_goal.float() * bounded(m['arm_speed'], 0.2)
        )
        active = self.actual_stage == self.stage
        if self.previous_cost is None or self.previous_cost.shape != cost.shape:
            self.previous_cost = torch.full_like(cost, torch.nan)
        previous = torch.where(torch.isnan(self.previous_cost), cost, self.previous_cost)
        progress = previous - cost
        self.previous_cost = torch.where(active, cost.detach(), self.previous_cost)
        # Time-normalized running costs; no positive reward for waiting.
        result = 4.0 * progress - (0.02 + 0.15 * cost) * self.control_dt / 0.02
        return torch.where(active, torch.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0), 0.0)


# =============================================================================
# STAGE 2: MOVE HANDS BEHIND THE BACKREST
# =============================================================================

class MoveHandsBehindBackrestReward(HumanoidBaseReward):
    """Stage 2: odměňuje přesun obou end-effectorů za opěradlo směrem k sedáku."""
    stage = 2

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.previous_cost = None

    def reset(self, env_ids, states):
        if self.previous_cost is not None:
            self.previous_cost[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name or self.robot_name, self)
        depth_shortfall = torch.relu(
            g.HAND_BEHIND_BACKREST_MIN - m['hand_behind_backrest']
        )
        height_excess = torch.relu(-m['hand_below_target'])
        task_space_cost = (
            5.0 * bilateral(depth_shortfall, 0.10)
            + 3.0 * bilateral(height_excess, 0.15)
        )
        near_goal = (
            (depth_shortfall.amax(-1) <= 0.03)
            & (height_excess.amax(-1) <= 0.03)
        )
        cost = (
            task_space_cost
            + anchor_cost(m)
            + chair_still_cost(m)
            + heading_cost(m)
            + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
            + near_goal.float() * bounded(m['arm_speed'], 0.2)
        )
        active = self.actual_stage == self.stage
        if self.previous_cost is None or self.previous_cost.shape != cost.shape:
            self.previous_cost = torch.full_like(cost, torch.nan)
        previous = torch.where(torch.isnan(self.previous_cost), cost, self.previous_cost)
        progress = previous - cost
        self.previous_cost = torch.where(active, cost.detach(), self.previous_cost)
        # Time-normalized running costs; no positive reward for waiting.
        result = 4.0 * progress - (0.02 + 0.15 * cost) * self.control_dt / 0.02
        return torch.where(active, torch.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0), 0.0)


# =============================================================================
# STAGE 3: PULL AND STOP
# =============================================================================

class PullChairReward(HumanoidBaseReward):
    """Stage 3: odměňuje stabilní úchop, rovný tah židle a zastavení v cíli."""
    stage = 3

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.previous_cost = None

    def reset(self, env_ids, states):
        if self.previous_cost is not None:
            self.previous_cost[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name or self.robot_name, self)
        near = torch.exp(-m['pull_error']/0.20)
        cost = (
            4 * bounded(m['pull_error'], 0.5)
            + 2 * (1 - m['contact'].float().mean(-1))
            + hand_shape_cost(m)
            + bilateral(m['palm_error'], g.CONTACT_RADIUS)
            + bilateral(m['hand_slip'], 0.08)
            + bounded(m['lateral'], g.PULL_TOLERANCE)
            + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
            + heading_cost(m)
            + bounded(m['sideways_speed'], 0.08)
            + bounded(torch.relu(-m['backward_speed']), 0.08)
            + near * (still_cost(m) + chair_still_cost(m))
            + bilateral(torch.relu(m['contact_force'] - g.CONTACT_FORCE_MAX), g.CONTACT_FORCE_MAX)
        )
        active = self.actual_stage == self.stage
        if self.previous_cost is None or self.previous_cost.shape != cost.shape:
            self.previous_cost = torch.full_like(cost, torch.nan)
        previous = torch.where(torch.isnan(self.previous_cost), cost, self.previous_cost)
        progress = previous - cost
        eligible = m['contact'].all(-1) & (m['palm_angle'] <= g.PALM_ANGLE_TOLERANCE).all(-1) \
            & (m['elbow_angle'] <= g.ELBOW_ANGLE_TOLERANCE).all(-1)
        progress = torch.where(eligible | (progress <= 0), progress, torch.zeros_like(progress))
        self.previous_cost = torch.where(active, cost.detach(), self.previous_cost)
        # Time-normalized running costs; no positive reward for waiting.
        result = 4.0 * progress - (0.02 + 0.15 * cost) * self.control_dt / 0.02
        return torch.where(active, torch.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0), 0.0)


# =============================================================================
# STAGE 4: RELEASE AND LIFT
# =============================================================================

class LiftHandsReward(HumanoidBaseReward):
    """Stage 4: odměňuje puštění židle a zvednutí obou rukou nad targety."""
    stage = 4

    def __init__(self):
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01
        self.previous_cost = None

    def reset(self, env_ids, states):
        if self.previous_cost is not None:
            self.previous_cost[env_ids] = torch.nan

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        m = self.metrics if self.metrics is not None else g.measure(states, robot_name or self.robot_name, self)
        cost = (
            3 * bilateral(m['lift_error'], 0.1)
            + m['any_contact'].float().mean(-1)
            + anchor_cost(m)
            + still_cost(m)
            + chair_still_cost(m)
            + heading_cost(m)
            + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
            + torch.exp(-m['lift_error'].amax(-1) / 0.05) * bounded(m['arm_speed'], 0.2)
        )
        active = self.actual_stage == self.stage
        if self.previous_cost is None or self.previous_cost.shape != cost.shape:
            self.previous_cost = torch.full_like(cost, torch.nan)
        previous = torch.where(torch.isnan(self.previous_cost), cost, self.previous_cost)
        progress = previous - cost
        self.previous_cost = torch.where(active, cost.detach(), self.previous_cost)
        # Time-normalized running costs; no positive reward for waiting.
        result = 4.0 * progress - (0.02 + 0.15 * cost) * self.control_dt / 0.02
        return torch.where(active, torch.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0), 0.0)


# =============================================================================
# WEIGHTS
# =============================================================================

MOTION_REGULARIZATION_WEIGHT = 0.1
UPPER_BODY_COM_REWARD_WEIGHT = 1.0
STAGE_OUTCOME_WEIGHT = 50.0  # 500 per completed stage, -500 on failure.
STAGE0_ARM_POS_REWARD_WEIGHT = 0.4
WALK_TO_CHAIR_REWARD_WEIGHT = 0.4
FACE_CHAIR_REWARD_WEIGHT = 0.1
KEEP_CHAIR_STILL_PENALTY_WEIGHT = -1.0
EXTEND_ARMS_REWARD_WEIGHT = 0.1
MOVE_HANDS_BEHIND_BACKREST_REWARD_WEIGHT = 0.1
PULL_CHAIR_REWARD_WEIGHT = 0.1
LIFT_HANDS_REWARD_WEIGHT = 0.1


# =============================================================================
# TASK CONFIG
# =============================================================================

@configclass
class Chairman2Cfg(HumanoidTaskCfg):
    success_bar = 0.9
    episode_length = 6000
    num_policy_stages: int = g.NUM_STAGES
    task_version: str = g.TASK_VERSION
    reset_to_stage0: bool = False
    use_snapshot_curriculum: bool = True
    eval_start_stage: int | None = None
    train_stage: int | None = None
    curriculum_max_stage: int | None = None
    snapshot_save_probability: float = 0.1
    verbose_motion_diagnostics: bool = False
    visualize_center_of_mass: bool = False
    log_reward_components: bool = True
    reset_rewards_on_stage_change: bool = True
    objects = [ArticulationObjCfg(
        name='chair', urdf_path='roboverse_data/assets/humanoidbench/chairs/chair3/foldable_chair_debug.urdf',
        default_position=[0., 0., 0.], fix_base_link=True, colapse_fixed_joints=False, batch_fixed_verts=True)]
    traj_filepath = 'roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json'
    checker = _ChairMan2Checker()
    reward_weights = [
        STAGE_OUTCOME_WEIGHT,
        MOTION_REGULARIZATION_WEIGHT,
        UPPER_BODY_COM_REWARD_WEIGHT,
        STAGE0_ARM_POS_REWARD_WEIGHT,
        WALK_TO_CHAIR_REWARD_WEIGHT,
        FACE_CHAIR_REWARD_WEIGHT,
        KEEP_CHAIR_STILL_PENALTY_WEIGHT,
        EXTEND_ARMS_REWARD_WEIGHT,
        MOVE_HANDS_BEHIND_BACKREST_REWARD_WEIGHT,
        PULL_CHAIR_REWARD_WEIGHT,
        LIFT_HANDS_REWARD_WEIGHT,
    ]
    reward_functions = [
        StageOutcomeReward(),
        MotionRegularization(),
        UpperBodyCenterOfMassReward(),
        Stage0ArmPos(),
        WalkToChairProgressReward(),
        FaceChairReward(),
        KeepChairStillPenalty(),
        ExtendArmsReward(),
        MoveHandsBehindBackrestReward(),
        PullChairReward(),
        LiftHandsReward(),
    ]

    def extra_spec(self):
        return {}
