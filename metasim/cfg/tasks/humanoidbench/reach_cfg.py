"""Rotate Cube in hand task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ReachChecker
from metasim.cfg.objects import RigidObjCfg, ArticulationObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward
from metasim.utils.humanoid_robot_util import right_palm_position


class StandingReward(HumanoidBaseReward):
    """Reward function for maintaining standing posture."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the standing reward."""
        super().__init__(robot_name)
        self._stand_height = 0.6

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the standing reward."""
        results_still = []
        for state in states:
            com_vel = humanoid_robot_util.center_of_mass_velocity(state, self._robot_name)
            still_x = humanoid_reward_util.tolerance(com_vel[0], bounds=(0.0, 0.0), margin=2)
            still_y = humanoid_reward_util.tolerance(com_vel[1], bounds=(0.0, 0.0), margin=2)
            still_reward = (still_x + still_y) / 2
            results_still.append(still_reward)

        stable_rewards = StableReward(robot_name=self._robot_name)(states)
        return torch.tensor(results_still) * stable_rewards
class ReachReward(HumanoidBaseReward):
    """Reward function for reaching a target position."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the reach reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the reach reward."""

        cube1_state = states.objects["cube_1"].root_state[:, :3]  # [num_envs, 3]
        right_hand_pos = right_palm_position(states, self.robot_name)  # [num_envs, 3]

        # vzdálenost
        distance = torch.norm(right_hand_pos - cube1_state, dim=1)  # [num_envs]

        # tolerance - použij torch variantu, ne numpy
        distance_np = distance.detach().cpu().numpy()
        reward_np = humanoid_reward_util.tolerance(
            distance_np,
            bounds=(0.0, 0.05),   # <-- dej malé tolerance okno místo (0,0)
            margin=1.0
        )
        # vrať to zpátky na stejný device a dtype
        reach_reward = torch.as_tensor(reward_np, device=distance.device, dtype=distance.dtype)

        return reach_reward


class OrientationReward(HumanoidBaseReward):
    """Reward function for cube orientation alignment."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the orientation reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the orientation reward."""
        results = []
        for state in states:
            left_cube_rot = state["metasim_body_cube_1/cube_1"]["rot"]
            right_cube_rot = state["metasim_body_cube_2/cube_2"]["rot"]
            target_cube_rot = state["metasim_body_cube_destination/cube_destination"]["rot"]

            left_alignment = torch.norm(left_cube_rot - target_cube_rot)
            right_alignment = torch.norm(right_cube_rot - target_cube_rot)

            results.append(left_alignment + right_alignment)
        return torch.tensor(results)


class HandProximityReward(HumanoidBaseReward):
    """Reward function for hand-cube proximity."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the hand proximity reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the hand proximity reward."""
        results = []
        for state in states:
            left_hand_pos = humanoid_robot_util.left_hand_position(state, self._robot_name)
            right_hand_pos = humanoid_robot_util.right_hand_position(state, self._robot_name)
            cube1_pos = state["metasim_body_cube_1/cube_1"]["pos"]
            cube2_pos = state["metasim_body_cube_2/cube_2"]["pos"]

            left_dist = torch.norm(left_hand_pos - cube1_pos)
            right_dist = torch.norm(right_hand_pos - cube2_pos)

            left_proximity = humanoid_reward_util.tolerance(left_dist, bounds=(0.0, 0.0), margin=0.5)
            right_proximity = humanoid_reward_util.tolerance(right_dist, bounds=(0.0, 0.0), margin=0.5)

            results.append((left_proximity + right_proximity) / 2)
        return torch.tensor(results)


@configclass
class ReachCfg(HumanoidTaskCfg):
    """Cube task for humanoid robots."""

    episode_length = 1000
    objects = [
        ArticulationObjCfg(
            name="cube_1",
            mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_2/mjcf/cube_2.xml",
            urdf_path="roboverse_data/assets/humanoidbench/cube/cube_2/cube_2.urdf",
            default_position= [0.3, 0.2, 0.9],

        ),
        # RigidObjCfg(
        #     name="cube_2",
        #     mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_2/mjcf/cube_2.xml",
        #     physics=PhysicStateType.GEOM,
        # ),
        # RigidObjCfg(
        #     name="cube_destination",
        #     mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_destination/mjcf/cube_destination.xml",
        #     physics=PhysicStateType.GEOM,
        #     fix_base_link=True,
        # ),
        ArticulationObjCfg(
            name="table",
            urdf_path="roboverse_data/assets/humanoidbench/table/urdf/table.urdf",
            mjcf_path="roboverse_data/assets/humanoidbench/table/mjcf/table.mjcf",
            default_position= [0.0, 0.0, -0.2],
            fix_base_link=True,
        ),

    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/cube/v2/g1/initial_state_v2.json"
    #traj_filepath = "my_env/initial_state_g1_v2.json"
    checker = _ReachChecker()
    reward_weights = [0.2, 0.5, 0.3]
    reward_functions = [ReachReward()]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
