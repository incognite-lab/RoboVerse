"""Five-policy Chairman2: approach, extend, contact, pull, release.

Dense shaping is signed progress plus non-positive state costs. Staying in an
unfinished stage cannot earn a stream of positive pose/contact rewards.
"""
import math
import torch
from metasim.cfg.checkers import _ChairMan2Checker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass
from metasim.utils import chairman2_geometry as g
from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


def bounded(error, scale):
    x = error / scale
    return x / (1 + x)


def bilateral(error, scale):
    # The worst hand/joint matters more than the average.
    return 0.75 * bounded(error.amax(-1), scale) + 0.25 * bounded(error.mean(-1), scale)


class Chairman2Reward(HumanoidBaseReward):
    def __init__(self):
        # Initialize known G1 anthropometrics, then use the actual robot key.
        super().__init__('g1_with_hands')
        self.robot_name = 'g1_without_hands'
        self.metrics = None
        self.control_dt = 0.01

    def measure(self, states, robot_name):
        if self.metrics is not None:
            return self.metrics
        return g.measure(states, robot_name or self.robot_name, self)


class StagePotentialReward(Chairman2Reward):
    stage = -1

    def __init__(self):
        super().__init__()
        self.previous_cost = None

    def reset(self, env_ids, states):
        if self.previous_cost is not None:
            self.previous_cost[env_ids] = torch.nan

    def cost(self, m):
        raise NotImplementedError

    def __call__(self, states, robot_name=None):
        q = states.robots[robot_name or self.robot_name].joint_pos
        if self.actual_stage is None:
            return q.new_zeros(q.shape[0])
        m = self.measure(states, robot_name)
        cost = self.cost(m)
        active = self.actual_stage == self.stage
        if self.previous_cost is None or self.previous_cost.shape != cost.shape:
            self.previous_cost = torch.full_like(cost, torch.nan)
        previous = torch.where(torch.isnan(self.previous_cost), cost, self.previous_cost)
        progress = previous - cost
        if self.stage == 3:
            eligible = m['contact'].all(-1) & (m['palm_angle'] <= g.PALM_ANGLE_TOLERANCE).all(-1) \
                & (m['elbow_angle'] <= g.ELBOW_ANGLE_TOLERANCE).all(-1)
            progress = torch.where(eligible | (progress <= 0), progress, torch.zeros_like(progress))
        self.previous_cost = torch.where(active, cost.detach(), self.previous_cost)
        # Time-normalized running costs; no positive reward for waiting.
        result = 4.0 * progress - (0.02 + 0.15 * cost) * self.control_dt / 0.02
        return torch.where(active, torch.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0), 0.0)


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


class WalkToChairReward(StagePotentialReward):
    stage = 0

    def cost(self, m):
        near = torch.exp(-m['approach_error'] / 0.35)
        return (3 * bounded(m['approach_error'], 1.0) + heading_cost(m)
                + (0.3 + near) * bilateral(m['pose0'], g.JOINT_TOLERANCE)
                + near * still_cost(m) + bounded(m['chair_drift'], g.CHAIR_DRIFT_TOLERANCE)
                + bounded(m['chair_yaw'], g.HEADING_TOLERANCE))


class ExtendArmsReward(StagePotentialReward):
    stage = 1

    def cost(self, m):
        pose = bilateral(m['pose1'], g.JOINT_TOLERANCE)
        return (3 * pose + anchor_cost(m) + still_cost(m) + heading_cost(m)
                + chair_still_cost(m) + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
                + (1-pose)*bounded(m['arm_speed'], 0.20))


class PlaceHandsReward(StagePotentialReward):
    stage = 2

    def cost(self, m):
        contact_missing = 1 - m['contact'].float().mean(-1)
        excess_force = torch.relu(m['contact_force'] - g.CONTACT_FORCE_MAX)
        return (3 * bilateral(m['palm_error'], g.CONTACT_RADIUS) + hand_shape_cost(m)
                + contact_missing + anchor_cost(m) + still_cost(m) + chair_still_cost(m)
                + heading_cost(m) + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
                + bilateral(excess_force, g.CONTACT_FORCE_MAX)
                + bilateral(m['hand_slip']*m['any_contact'], 0.08))


class PullChairReward(StagePotentialReward):
    stage = 3

    def cost(self, m):
        near = torch.exp(-m['pull_error']/0.20)
        return (4 * bounded(m['pull_error'], 0.5) + 2*(1-m['contact'].float().mean(-1))
                + hand_shape_cost(m) + bilateral(m['palm_error'], g.CONTACT_RADIUS)
                + bilateral(m['hand_slip'], 0.08) + bounded(m['lateral'], g.PULL_TOLERANCE)
                + bounded(m['chair_yaw'], g.HEADING_TOLERANCE) + heading_cost(m)
                + bounded(m['sideways_speed'], 0.08) + bounded(torch.relu(-m['backward_speed']), 0.08)
                + near*(still_cost(m)+chair_still_cost(m))
                + bilateral(torch.relu(m['contact_force']-g.CONTACT_FORCE_MAX), g.CONTACT_FORCE_MAX))


class LiftHandsReward(StagePotentialReward):
    stage = 4

    def cost(self, m):
        return (3*bilateral(m['lift_error'], 0.10) + m['any_contact'].float().mean(-1)
                + anchor_cost(m) + still_cost(m) + chair_still_cost(m) + heading_cost(m)
                + bounded(m['chair_yaw'], g.HEADING_TOLERANCE)
                + torch.exp(-m['lift_error'].amax(-1)/0.05)*bounded(m['arm_speed'], 0.20))


class MotionRegularization(Chairman2Reward):
    def __init__(self):
        super().__init__()
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
        m = self.measure(states, robot_name)
        cost = 0.03*bounded(rate, 3.0) + 0.05*bounded(speed, 1.5) + 0.1*(1-m['upright']).clamp(min=0)
        if self.command is not None:
            cost += 0.02*bounded((self.command-self.previous_command).abs().mean(-1)/self.control_dt, 3.0)
        return -torch.nan_to_num(cost, nan=1.0, posinf=1.0)*self.control_dt/0.02


class StageOutcomeReward(Chairman2Reward):
    def __init__(self):
        super().__init__()
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
    log_reward_components: bool = True
    reset_rewards_on_stage_change: bool = True
    objects = [ArticulationObjCfg(
        name='chair', urdf_path='roboverse_data/assets/humanoidbench/chairs/chair3/foldable_chair_debug.urdf',
        default_position=[0., 0., 0.], fix_base_link=True, colapse_fixed_joints=False, batch_fixed_verts=True)]
    traj_filepath = 'roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json'
    checker = _ChairMan2Checker()
    reward_functions = [StageOutcomeReward(), WalkToChairReward(), ExtendArmsReward(), PlaceHandsReward(),
                        PullChairReward(), LiftHandsReward(), MotionRegularization()]
    reward_weights = [1.0] * len(reward_functions)

    def extra_spec(self):
        return {}
