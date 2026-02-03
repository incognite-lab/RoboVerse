"""Stand task for humanoid robots."""

from metasim.cfg.checkers import _StandChecker
from metasim.utils import configclass

from .base_cfg import BaseLocomotionReward, HumanoidTaskCfg, HumanoidBaseReward
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util
import torch
from metasim.utils.humanoid_robot_util import feet_position


class AnkleDownReward(HumanoidBaseReward):
    """Reward function for legs down position."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the legs down reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the legs down reward."""
        left_ankle, right_ankle = feet_position(states, self.robot_name)
        sum_height = left_ankle[:, 2] + right_ankle[:, 2]
        results = humanoid_reward_util.tolerance_tensor(
            sum_height,
            bounds=(0.0, 0.4),
            margin=0.6,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return results
class HandDownReward(HumanoidBaseReward):
    """Reward function for hand down position."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the hand down reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState],robot_name: str = None) -> torch.FloatTensor:
        """Compute the hand down reward."""
        left_hand_pos = humanoid_robot_util.left_hand_position(states, self.robot_name)
        right_hand_pos = humanoid_robot_util.right_hand_position(states, self.robot_name)
        sum_height = left_hand_pos[:, 2] + right_hand_pos[:, 2]
        results = humanoid_reward_util.tolerance_tensor(
            sum_height,
            bounds=(0.0, 1.0),
            margin=1.0,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return results
class DistanceFromInitPos(HumanoidBaseReward):

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the distance from initial position reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        diff = None
        for name, state in self.initial_pos.items():
            joint_names = states.robots[self.robot_name].joint_names
            state_idx = list(joint_names).index(name)
            joint_pos = states.robots[self.robot_name].joint_pos[:,state_idx]
            state = torch.tensor(state, device=joint_pos.device)
            if diff is None:
                diff = torch.abs(joint_pos - state)
            else:
                diff += torch.abs(joint_pos - state)
        results = humanoid_reward_util.tolerance_tensor(
            diff,
            bounds=(0.0, 1.0),
            margin=5.0,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return results

class StandReward(BaseLocomotionReward):
    """Reward function for the stand task."""

    _move_speed = 0
    success_bar = 500


@configclass
class StandCfg(HumanoidTaskCfg):
    """Stand task for humanoid robots."""

    #episode_length = 100
    # traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/h1_v2.pkl"
    #traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"
    #traj_filepath = "my_env/initial_state_g1_v2.json"
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"

    checker = _StandChecker()
    reward_weights = [0.7,0.3]
    reward_functions = [StandReward(),DistanceFromInitPos()]
    # reward_weights = [0.8,0.1,0.1]
    # reward_functions = [StandReward(), AnkleDownReward(), HandDownReward()]
    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
