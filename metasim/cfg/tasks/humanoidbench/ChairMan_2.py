"""Rewards for Chairman2 stages 0..5, using the checker targets and tolerances.

0: walking arm pose; 1: walk; 2: forward arm pose; 3: lower palms;
4: pull and stop; 5: lift palms. Joint angles are radians, distances metres.
Walking retains ChairMan_multi's reward formulas and weights. Other stages
use a negative remaining-error cost plus signed progress; waiting short of
the goal never earns a positive pose reward. Only the checker awards completion.
"""
from __future__ import annotations

import torch

from metasim.cfg.checkers import _ChairMan2Checker
from metasim.cfg.checkers import stages_chairman2 as checks
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass
from metasim.utils.chair_navigation import chair_back_direction_xy
from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg
from . import ChairMan_multi as proven


def _robot_body(states, robot_name):
    robot = states.robots[robot_name]
    return robot.body_state[:, robot.body_names.index("pelvis")]


def _chair_body(states):
    chair = states.objects["chair"]
    return chair.body_state[:, chair.body_names.index("base_link")]


def _palms_and_targets(states, robot_name):
    robot, chair = states.robots[robot_name], states.objects["chair"]
    palms = torch.stack([robot.body_state[:, robot.body_names.index(f"{s}_hand_palm_link"), :3]
                         for s in ("left", "right")], dim=1)
    targets = torch.stack([chair.body_state[:, chair.body_names.index(f"target_hand_{s}"), :3]
                           for s in ("left", "right")], dim=1)
    return palms, targets


def _score(normalized_error):
    """Maximum is reached only when EVERY condition lies inside tolerance.

    The mean gives all joints/hands a learning signal and the minimum prevents
    a good hand or joint from hiding the worst one. Rational decay stays dense
    even when the initial joint pose is far from its target.
    """
    component = 1.0 / (1.0 + (normalized_error - 1.0).clamp(min=0.0))
    return 0.5 * component.mean(dim=-1) + 0.5 * component.amin(dim=-1)


class StageReward(HumanoidBaseReward):
    def __init__(self, stages):
        super().__init__()
        self.robot_name = "g1_without_hands"
        self.active_stages = tuple(stages)
        # Shared, per-environment anchors published by the checker.
        self.robot_anchor = self.chair_anchor = self.pull_direction = None

    def mask(self, states, robot_name):
        q = states.robots[robot_name].joint_pos
        mask = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
        if self.actual_stage is not None:
            for stage in self.active_stages:
                mask |= self.actual_stage == stage
        return mask


class ProgressReward(StageReward):
    """Signed progress and a nonpositive cost; reset memory per environment."""
    def __init__(self, stages):
        super().__init__(stages)
        self.previous_score = self.previous_stage = None
        self.progress_gain = 5.0
        # Keep the per-step cost small relative to the -50 failure penalty:
        # a large negative pose cost can teach the agent to end episodes early.
        self.remaining_cost_scale = 0.02

    def reset(self, env_ids, states):
        if self.previous_score is not None:
            self.previous_score[env_ids] = torch.nan
            self.previous_stage[env_ids] = -1

    def __call__(self, states, robot_name):
        active = self.mask(states, robot_name)
        q = states.robots[robot_name].joint_pos
        if not active.any():
            return q.new_zeros(q.shape[0])
        score = self.score(states, robot_name)
        if self.previous_score is None or self.previous_score.shape != score.shape:
            self.previous_score = torch.full_like(score, torch.nan)
            self.previous_stage = torch.full_like(self.actual_stage, -1)
        valid = torch.isfinite(self.previous_score) & (self.previous_stage == self.actual_stage)
        delta = torch.where(valid, score - self.previous_score, torch.zeros_like(score))
        self.previous_score = torch.where(active, score.detach(), self.previous_score)
        self.previous_stage = torch.where(active, self.actual_stage, self.previous_stage).clone()
        value = self.progress_gain * delta + self.remaining_cost_scale * (score - 1.0)
        return torch.where(active, value, torch.zeros_like(score))


class JointPoseReward(ProgressReward):
    def __init__(self, stage):
        if stage not in (0, 2):
            raise ValueError("Joint pose stage must be 0 or 2")
        super().__init__((stage,))
        self.stage = stage

    def score(self, states, robot_name):
        targets = checks.STAGE0_JOINT_TARGETS if self.stage == 0 else checks.STAGE2_JOINT_TARGETS
        tolerances = checks.STAGE0_JOINT_TOLERANCES if self.stage == 0 else checks.STAGE2_JOINT_TOLERANCES
        robot = states.robots[robot_name]
        names = list(robot.joint_names)
        missing = set(targets) - set(names)
        if missing:
            raise ValueError(f"Checker target joints missing from robot: {sorted(missing)}")
        q = robot.joint_pos[:, [names.index(n) for n in targets]]
        target = q.new_tensor(list(targets.values()))
        tolerance = q.new_tensor([tolerances.get(n, checks.JOINT_POSITION_TOLERANCE) for n in targets])
        return _score((q - target).abs() / tolerance.clamp(min=1e-6))


class LowerPalmsReward(ProgressReward):
    def __init__(self):
        super().__init__((3,))

    def score(self, states, robot_name):
        palms, targets = _palms_and_targets(states, robot_name)
        direction = chair_back_direction_xy(_chair_body(states)[:, 3:7])
        targets[:, :, :2] += checks.PALM_FRONT_OFFSET * direction[:, None, :]
        return _score(torch.linalg.vector_norm(palms - targets, dim=-1) / checks.PALM_POSITION_TOLERANCE)


class PullAndStopReward(ProgressReward):
    def __init__(self):
        super().__init__((4,))

    def score(self, states, robot_name):
        if self.chair_anchor is None or self.pull_direction is None:
            raise RuntimeError("PullAndStopReward requires checker stage-entry anchors")
        robot, chair = _robot_body(states, robot_name), _chair_body(states)
        displacement = chair[:, :2] - self.chair_anchor[:, :2]
        distance = (displacement * self.pull_direction).sum(dim=-1)
        lateral = torch.linalg.vector_norm(displacement - distance[:, None] * self.pull_direction, dim=-1)
        position = torch.stack(((distance - checks.CHAIR_PULL_DISTANCE_THRESHOLD).abs(), lateral), dim=-1)
        position_score = _score(position / checks.CHAIR_PULL_TOLERANCE)
        # Braking is valuable near the goal; far away it must not discourage pulling.
        speed = torch.stack((torch.linalg.vector_norm(robot[:, 7:10], dim=-1),
                             torch.linalg.vector_norm(chair[:, 7:10], dim=-1)), dim=-1)
        # Aim below the strict checker threshold, not exactly on its boundary.
        stop_score = _score(speed / (0.5 * checks.VELOCITY_THRESHOLD))
        return position_score * (0.75 + 0.25 * stop_score)


class LiftPalmsReward(ProgressReward):
    def __init__(self):
        super().__init__((5,))

    def score(self, states, robot_name):
        palms, targets = _palms_and_targets(states, robot_name)
        xy = torch.linalg.vector_norm(palms[:, :, :2] - targets[:, :, :2], dim=-1)
        height_deficit = (targets[:, :, 2] + checks.PALM_LIFT_HEIGHT - palms[:, :, 2]).clamp(min=0.0)
        # Zero deficit yields normalized error 1, hence no artificial upper height limit.
        return _score(torch.cat((xy / checks.PALM_LIFT_XY_TOLERANCE,
                                 1.0 + height_deficit / checks.PALM_LIFT_HEIGHT), dim=-1))


class RobotAnchorPenalty(StageReward):
    def __init__(self):
        super().__init__((0, 2, 3))

    def __call__(self, states, robot_name):
        active = self.mask(states, robot_name)
        robot = _robot_body(states, robot_name)
        if not active.any():
            return robot.new_zeros(robot.shape[0])
        if self.robot_anchor is None:
            raise RuntimeError("RobotAnchorPenalty requires checker stage-entry anchors")
        drift = torch.linalg.vector_norm(robot[:, :2] - self.robot_anchor, dim=-1)
        return (drift / checks.ROBOT_DRIFT_THRESHOLD).square().clamp(max=2.0) * active


class ChairAnchorPenalty(StageReward):
    def __init__(self):
        super().__init__((1, 2))

    def __call__(self, states, robot_name):
        active = self.mask(states, robot_name)
        chair = _chair_body(states)
        if not active.any():
            return chair.new_zeros(chair.shape[0])
        if self.chair_anchor is None:
            raise RuntimeError("ChairAnchorPenalty requires checker stage-entry anchors")
        drift = torch.linalg.vector_norm(chair[:, :3] - self.chair_anchor, dim=-1)
        return (drift / checks.CHAIR_DRIFT_THRESHOLD).square().clamp(max=2.0) * active


class WalkingStageReward(HumanoidBaseReward):
    """Evaluate an unchanged old stage-0 reward on Chairman2 stage 1 only."""
    def __init__(self, reward):
        super().__init__()
        self.robot_name = "g1_without_hands"
        self.reward = reward

    def reset(self, env_ids, states):
        self.reward.robot_name = self.robot_name
        reset = getattr(self.reward, "reset", None)
        if reset is not None:
            reset(env_ids, states)

    def set_control_context(self, command, previous_command, device=None):
        setter = getattr(self.reward, "set_control_context", None)
        if setter is not None:
            setter(command, previous_command, device=device)

    def __call__(self, states, robot_name):
        q = states.robots[robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        active = self.actual_stage == 1
        self.reward.actual_stage = torch.where(active, 0, -1)
        # Preserve the proven pose formula but use the user's current walking
        # arm configuration, so stage 1 does not undo stage 0.
        if isinstance(self.reward, proven.Stage0ArmPos):
            self.reward.required_pos.update(checks.STAGE0_JOINT_TARGETS)
        return self.reward(states, robot_name) * active


class ManipulationCommandPenalty(StageReward):
    def __init__(self):
        super().__init__((0, 2, 3, 4, 5))
        self.command = self.previous_command = None

    def set_control_context(self, command, previous_command, device=None):
        # Own copies: reward resets must not mutate wrapper command buffers.
        self.command = torch.as_tensor(command, device=device).clone()
        self.previous_command = torch.as_tensor(previous_command, device=device).clone()

    def reset(self, env_ids, states):
        if self.command is not None:
            self.command[env_ids] = 0
            self.previous_command[env_ids] = 0

    def __call__(self, states, robot_name):
        q = states.robots[robot_name].joint_pos
        active = self.mask(states, robot_name)
        if self.command is None or not active.any():
            return q.new_zeros(q.shape[0])
        smooth = ((self.command - self.previous_command).abs() / q.new_tensor([0.08, 0.06, 0.15])).clamp(max=1).mean(-1)
        stop = (self.command.abs() / q.new_tensor([0.5, 0.3, 0.8])).clamp(max=1).mean(-1)
        # Crouching/balancing corrections remain possible; suppress travel in
        # stationary stages. Allow walking throughout the pull until near goal.
        stop_gate = self.actual_stage != 4
        if self.chair_anchor is not None:
            delta = _chair_body(states)[:, :2] - self.chair_anchor[:, :2]
            target_delta = delta - checks.CHAIR_PULL_DISTANCE_THRESHOLD * self.pull_direction
            near_goal = torch.linalg.vector_norm(target_delta, dim=-1) < 2 * checks.CHAIR_PULL_TOLERANCE
            stop_gate |= near_goal
        return torch.where(stop_gate, 0.25 * smooth + 0.75 * stop, smooth) * active


def _for_robot(reward):
    # The legacy base reward constructor does not recognize g1_without_hands;
    # its formulas already work with named joints of this G1 variant.
    reward.robot_name = "g1_without_hands"
    return reward


# Function/weight pairs keep task registration aligned. Walking-specific terms
# and global regularizers retain the values from ChairmanmultiCfg.
REWARD_TERMS = [
    (_for_robot(proven.TerminationCfg()), proven.TERMINATION_WEIGHT),
    (_for_robot(proven.DeltaActionRateCfg()), proven.DELTA_ACTION_RATE_WEIGHT),
    (_for_robot(proven.DoFVelocityAccelerationCfg()), proven.DOF_VELOCITY_ACCELERATION_WEIGHT),
    (_for_robot(proven.UprightPenaltyCfg()), proven.UPRIGHT_PENALTY_WEIGHT),
    (WalkingStageReward(proven.Stage0ArmPos()), proven.STAGE0_ARM_POS_REWARD_WEIGHT),
    (WalkingStageReward(proven.WalkToChairProgressReward()), proven.WALK_TO_CHAIR_REWARD_WEIGHT),
    (WalkingStageReward(proven.FaceChairReward()), proven.FACE_CHAIR_REWARD_WEIGHT),
    (WalkingStageReward(proven.KeepChairStillPenalty()), proven.KEEP_CHAIR_STILL_PENALTY_WEIGHT),
    (WalkingStageReward(proven.OpenGraspReward()), proven.OPEN_GRASP_REWARD_WEIGHT),
    (WalkingStageReward(proven.LocomotionCommandPenalty()), proven.LOCOMOTION_COMMAND_PENALTY_WEIGHT),
    (JointPoseReward(0), 10.0),
    (JointPoseReward(2), 10.0),
    (LowerPalmsReward(), 12.0),
    (PullAndStopReward(), 16.0),
    (LiftPalmsReward(), 12.0),
    (RobotAnchorPenalty(), -4.0),
    (ChairAnchorPenalty(), -4.0),
    (ManipulationCommandPenalty(), proven.LOCOMOTION_COMMAND_PENALTY_WEIGHT),
    (_for_robot(proven.MultiPolicyStageCompletionReward()), proven.MULTI_POLICY_STAGE_COMPLETION_WEIGHT),
]


@configclass
class Chairman2Cfg(HumanoidTaskCfg):
    """Six-stage Chairman2 task for G1 with fixed finger/wrist joints."""

    success_bar = 0.9
    episode_length = 2500
    num_policy_stages: int = 6
    reset_to_stage0: bool = False
    use_snapshot_curriculum: bool = True
    eval_start_stage: int | None = None
    snapshot_save_probability: float = 1.0
    verbose_motion_diagnostics: bool = False

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
    checker = _ChairMan2Checker()

    reward_weights = [weight for _, weight in REWARD_TERMS]
    reward_functions = [reward for reward, _ in REWARD_TERMS]

    def extra_spec(self):
        return {}
