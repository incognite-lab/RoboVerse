"""Exit Door task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _DoorChecker
from metasim.cfg.objects import RigidObjCfg, ArticulationObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward




class ReachHandleDoorPosReward(HumanoidBaseReward):
    """Reward function for reaching the door handle."""
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_dist = None
    def __call__(self, states:list[EnvState], robot_name:str = None) -> torch.FloatTensor:
        """Compute the reach pos door handle reward."""
        eefektor = "endeffector"
        handle_idx = states.objects["door"].body_names.index("door_handle")
        handle_pos = states.objects["door"].body_state[:, handle_idx, :3]  # [num_envs, 3]
        #print("Handle position:", handle_pos)
        right_hand_pos = humanoid_robot_util.right_palm_position(states, self.robot_name,ee_name=eefektor)  # [num_envs, 3]
        distance = torch.norm(right_hand_pos - handle_pos, dim=1)  # [num_envs]
        #print("Right hand position:", right_hand_pos)
        #print("Distance to handle:", distance)
        reach_reward = humanoid_reward_util.tolerance(
            distance,
            bounds=(0.0, 0.03),
            margin=0.2,
            sigmoid="gaussian"
        )
        if self.prev_dist is None:
            self.prev_dist = distance.clone()
            approach_bonus = torch.zeros_like(distance)
        else:
            delta_dist = self.prev_dist - distance
            approach_bonus = torch.clamp(delta_dist * 10.0, min=0.0)* 0.1
            self.prev_dist = distance.clone()

        total_reward = reach_reward + approach_bonus
        total_reward = torch.clamp(total_reward, max=1.0)




        #print("Reach reward:", reach_reward)
        return total_reward
class ReachHandleDoorOriReward(HumanoidBaseReward):
    """Reward function for reaching the door handle orientation."""
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
    def __call__(self, states:list[EnvState], robot_name:str = None) -> torch.FloatTensor:
        """Compute the reach ori door handle reward."""
        eefector = "endeffector"
        handle_idx = states.objects["door"].body_names.index("door_handle")
        handle_ori = states.objects["door"].body_state[:, handle_idx, 3:7]  # [num_envs, 4]
        handle_pos = states.objects["door"].body_state[:, handle_idx, :3]  # [num_envs, 3]
        right_hand_ori = humanoid_robot_util.right_palm_orientation(states, self.robot_name,ee_name=eefector)  # [num_envs, 4]
        right_hand_pos = humanoid_robot_util.right_palm_position(states, self.robot_name,ee_name=eefector)  # [num_envs, 3]
        #if torch.norm(right_hand_pos - handle_pos, dim=1)
        # Compute quaternion distance
        dot_product = torch.abs(torch.sum(handle_ori * right_hand_ori, dim=1))  # [num_envs]
        ori_reward = humanoid_reward_util.tolerance(
            dot_product,
            bounds=(0.9, 1.0),
            margin=0.2,
            sigmoid="gaussian"
        )

        distance_weight = torch.exp(-10.0 * torch.clamp(torch.norm(right_hand_pos - handle_pos, dim=1) - 0.05, min=0.0, max=1.0))
        ori_reward = ori_reward * distance_weight
        #print("Orientation reward:", ori_reward)
        return ori_reward
class DoorReward(HumanoidBaseReward):
    """Reward function for the door task."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the door reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the door reward (add door-op reward only for envs that pass the reach threshold)."""
        reach_reward_pos = ReachHandleDoorPosReward(self.robot_name)(states, robot_name)
        reach_reward_ori = ReachHandleDoorOriReward(self.robot_name)(states, robot_name)
        reach_reward = 0.5 * reach_reward_pos + 0.5 * reach_reward_ori  # [num_envs]

        # boolean mask per-env where reach is above threshold
        thresh = 0.98
        #print("Reach reward:", reach_reward)
        #print("Reach reward total:", reach_reward)
        mask = reach_reward > thresh  # tensor of shape [num_envs], dtype=bool

        # compute door open reward for all envs (must be tensor)
        obj_name = "door"
        door_angle = humanoid_robot_util.door_angle_tensor(states, obj_name)
        #print("Door angle:", door_angle)
        door_op_reward = humanoid_reward_util.tolerance(
            -door_angle,
            bounds=(0.785, 1.57),
            sigmoid="linear",
            margin=1.0
        )
        #print("Door open reward (before mask):", door_op_reward)
        # final: reach component scaled by 0.5 for all envs, add door_op_reward*0.5 only where mask is True
        final_reward = reach_reward * 0.5 + torch.where(mask, door_op_reward * 0.5, torch.zeros_like(door_op_reward))
        #print("Door open reward:", door_op_reward)
        #print("Final reward:", final_reward)
        return final_reward


@configclass
class DoorCfg(HumanoidTaskCfg):
    """Door task for humanoid robots."""
    success_bar = 0.9
    episode_length = 400
    objects = [
        ArticulationObjCfg(
            name="door",
            urdf_path="roboverse_data/assets/humanoidbench/door/door.urdf",
            usd_path = "urdf2usd_convert/g1/usd/door.usd",
            default_position= [0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False
        )
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/door/initial_state_v2.json"
    checker = _DoorChecker()
    reward_weights = [1.0]
    reward_functions = [DoorReward(robot_name="g1_with_hands_simple")]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
