"""Exit Door task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _DoorChecker
from metasim.cfg.objects import RigidObjCfg, ArticulationObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward


class DoorReward(HumanoidBaseReward):
    """Reward function for the door task."""

    def __init__(self, robot_name="h1"):
        """Initialize the door reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState],robot_name:str = None) -> torch.FloatTensor:
        """Compute the door reward."""
        results = []
        for state in states:
            door_angle = humanoid_robot_util.door_angle_tensor(state, "door")
            door_opened = humanoid_reward_util.tolerance(
                door_angle,
                bounds=(1.0, 1.57),
                margin=1.57
                )
            results.append(door_opened)
        return torch.tensor(results)


@configclass
class DoorCfg(HumanoidTaskCfg):
    """Door task for humanoid robots."""
    success_bar = 0.9
    episode_length = 200
    objects = [
        ArticulationObjCfg(
            name="door",
            urdf_path="roboverse_data/assets/humanoidbench/door/door.urdf",
            default_position= [1.0, 0.0, 0.0],
            fix_base_link=True,
        )
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/door/initial_state_v2.json"
    checker = _DoorChecker()
    reward_weights = [1.0]
    reward_functions = [DoorReward()]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
