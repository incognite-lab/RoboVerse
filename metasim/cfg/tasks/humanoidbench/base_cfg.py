"""Base class for humanoid tasks."""

from __future__ import annotations

import logging

import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.control import ControlCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    neck_height_tensor,
    robot_local_velocity_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from humanoid_bench/tasks/basic_locomotion_envs.py
########################################################

# Height of head above which stand reward is 1.
H1_STAND_HEAD_HEIGHT = 1.65
H1_STAND_NECK_HEIGHT = 1.41
H1_CRAWL_HEAD_HEIGHT = 0.8
G1_STAND_HEAD_HEIGHT = 1.28
G1_STAND_NECK_HEIGHT = 1.08
G1_CRAWL_HEAD_HEIGHT = 0.6


class HumanoidBaseReward:
    """Base class for humanoid rewards."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the humanoid reward."""
        self.robot_name = robot_name
        if (
            robot_name == "h1"
            or robot_name == "h1_simple_hand"
            or robot_name == "h1_hand"
            or robot_name == "h1_body_collision"
        ):
            self._stand_height = H1_STAND_HEAD_HEIGHT
            self._stand_neck_height = H1_STAND_NECK_HEIGHT
            self._crawl_height = H1_CRAWL_HEAD_HEIGHT
        elif robot_name == "g1_no_hands" or robot_name == "g1":
            self._stand_height = G1_STAND_HEAD_HEIGHT
            self._stand_neck_height = G1_STAND_NECK_HEIGHT
            self._crawl_height = G1_CRAWL_HEAD_HEIGHT
        elif robot_name == "g1_with_hands_simple" or robot_name == "g1":
            self._stand_height = G1_STAND_HEAD_HEIGHT
            self._stand_neck_height = G1_STAND_NECK_HEIGHT
            self._crawl_height = G1_CRAWL_HEAD_HEIGHT
        elif robot_name == "g1_with_hands":
            self._stand_height = G1_STAND_HEAD_HEIGHT
            self._stand_neck_height = G1_STAND_NECK_HEIGHT
            self._crawl_height = G1_CRAWL_HEAD_HEIGHT
            self.initial_pos = {  # = target angles [rad] when action = 0.0
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
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
        else:
            raise ValueError(f"Unknown robot {robot_name}")


class StableReward(HumanoidBaseReward):
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    _stand_shoulder_height = 1.2  # nastav podle svého robota
    _fall_height = 0.5             # pokud hlava spadne pod tuto výšku, hlava je moc nízko a přichází trest
    _torso_margin = 0.2            # tolerance naklonění trupu

    def __init__(self, robot_name="h1"):
        """Initialize the locomotion reward."""
        super().__init__(robot_name)
        self._timestep = 0  # lokální counter
    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        ret_rewards = []
        head_height = neck_height_tensor(states, self.robot_name)
        standing = humanoid_reward_util.tolerance_tensor(
            head_height,  # Adjust for neck height
            bounds=(self._stand_neck_height, float("inf")),
            margin=self._stand_neck_height / 4,
        )
        #print("standing", standing)
        upright = humanoid_reward_util.tolerance_tensor(
            torso_upright_tensor(states, self.robot_name),
            bounds=(0.9, float("inf")),
            margin=1.9,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * upright
        small_control = humanoid_reward_util.tolerance_tensor(
            actuator_forces_tensor(states, self.robot_name),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean(dim=-1)


        almost_fallen = head_height < self._fall_height
        fall_penalty = torch.where(almost_fallen, torch.tensor(-1.0, device=head_height.device), torch.tensor(0.0, device=head_height.device))

        small_control = small_control#(4 + small_control) / 5
        stable_reward = stand_reward
        full_reward = stable_reward * small_control + fall_penalty
        #print("full reward", full_reward )
        return full_reward


class BaseLocomotionReward(HumanoidBaseReward):
    """Base class for locomotion rewards."""

    _move_speed = None
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = None

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the locomotion reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the locomotion reward."""
        if robot_name is None:
            robot_name = self.robot_name  # fallback to default

        stable_rewards = StableReward(robot_name)(states)
        if self._move_speed == None:
            return stable_rewards
        if self._move_speed == 0:
            horizontal_velocity = robot_velocity_tensor(states, robot_name)[:, [0, 1]]
            dont_move = humanoid_reward_util.tolerance_tensor(horizontal_velocity, margin=2).mean(dim=-1)
            moving_reward = dont_move
        else:
            com_x_velocity = robot_local_velocity_tensor(states, robot_name)[:, 0]
            move = humanoid_reward_util.tolerance_tensor(
                com_x_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            moving_reward = move
        #print("full reward", stable_rewards )
        return stable_rewards * moving_reward


@configclass
class HumanoidTaskCfg(BaseRLTaskCfg):
    """Base class for humanoid tasks."""

    decimation: int = 10
    source_benchmark = BenchmarkType.HUMANOIDBENCH
    task_type = TaskType.LOCOMOTION
    episode_length = 500  # TODO: may change
    objects = []
    reward_weights = [1.0]
    sim_params = SimParamCfg(
        dt=0.002,
        contact_offset=0.01,
        num_position_iterations=8,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.5,
        replace_cylinder_with_capsule=True,
    )
    control = ControlCfg(action_scale=0.5, action_offset=True, torque_limit_scale=0.85)

    # @staticmethod
    def humanoid_obs_flatten_func(self, envstates: list[EnvState]) -> torch.Tensor:
        """Observation function for humanoid tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.

        Returns:
            torch.Tensor: Flattened observations for all environments.
        """
        env_obs = []
        results_state = []
        for _, object_state in sorted(envstates.objects.items()):
            results_state.append(object_state.root_state)
        for _, robot_state in sorted(envstates.robots.items()):
            results_state.append(robot_state.root_state)
            results_state.append(robot_state.joint_pos)
            results_state.append(robot_state.joint_vel)
        for _, sensor_state in sorted(envstates.sensors.items()):
            # vyflattenovat vše kromě batch dim
            sensor_state = sensor_state.reshape(sensor_state.shape[0], -1)
            results_state.append(sensor_state)
        return torch.cat(results_state, dim=1)

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
