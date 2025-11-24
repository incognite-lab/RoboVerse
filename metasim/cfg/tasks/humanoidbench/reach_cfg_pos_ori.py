"""Rotate Cube in hand task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _ReachCheckerPosOri
from metasim.cfg.objects import RigidObjCfg, ArticulationObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util
from scipy.spatial.transform import Rotation as R



from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward
from metasim.utils.humanoid_robot_util import right_palm_position,right_palm_orientation
class ReachReward(HumanoidBaseReward):
    """Reward function for reaching a target position."""
    success_bar = 0.9
    def __init__(self, robot_name="g1_with_hands_simple"):
        """Initialize the reach reward."""
        super().__init__(robot_name)
        self.prev_distance = None  # pro výpočet změny vzdálenosti

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the reach reward."""
        ee_name = "endeffector"

        cube1_state = states.objects["cube_1"].root_state[:, :3]  # [num_envs, 3]
        #right_hand_pos = right_palm_position(states, self.robot_name)
        right_hand_pos = right_palm_position(states, self.robot_name,ee_name)  # [num_envs, 3]

        # aktuální vzdálenost ruky od cíle
        distance = torch.norm(right_hand_pos - cube1_state, dim=1)  # [num_envs]

        # základní odměna – blízkost k cíli
        #distance_np = distance.detach().cpu().numpy()
        base_reward = humanoid_reward_util.tolerance(
            distance,
            bounds=(0.0, 0.03),   # cílové okno
            margin=0.2,
            sigmoid="gaussian"
        )
        #base_reward = torch.as_tensor(base_reward_np, device=distance.device, dtype=distance.dtype)

        # --- BONUS: odměna za pohyb směrem k cíli ---
        if self.prev_distance is None:
            self.prev_distance = distance.clone()
            approach_bonus = torch.zeros_like(distance)
        else:
            # delta_distance < 0 → přiblížil se
            delta_distance = self.prev_distance - distance
            approach_bonus = torch.clamp(delta_distance, min=0.0) * 1.0  # zesil efekt
            self.prev_distance = distance.clone()

        # --- Penalizace za vzdálení ---
        away_penalty = torch.clamp(distance - 1.0, min=0.0) * 0.1  # pokud >1m, mírná penalizace

        # --- Celková odměna ---
        total_reward = base_reward + approach_bonus - away_penalty

        # omez hodnoty (pro stabilitu)
        total_reward = torch.clamp(total_reward, 0.0, 5.0)
        #print("dist Reward:", total_reward)
        return total_reward


class OrientationReward(HumanoidBaseReward):
    """Reward for aligning hand orientation with cube orientation (Euler-based)
       + bonus for moving toward the target orientation.
    """

    def __init__(self, robot_name="g1_with_hands_simple"):
        super().__init__(robot_name)
        self.epsilon = 1e-6
        self.prev_angle_error = None  # for movement-towards-target bonus

    def quat_to_euler(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to Euler XYZ (roll,pitch,yaw)."""

        # torch quaternion format might be [x,y,z,w], convert if necessary
        # zde je orientace [x,y,z,w], takže ji přehodíme na [w,x,y,z]
        q = torch.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], dim=1)

        # rot matrix (B,3,3)
        qw, qx, qy, qz = q[:,0], q[:,1], q[:,2], q[:,3]

        # Rotation matrix from quaternion (torch friendly)
        Rm = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
        Rm[:,0,0] = 1 - 2*(qy*qy + qz*qz)
        Rm[:,0,1] = 2*(qx*qy - qz*qw)
        Rm[:,0,2] = 2*(qx*qz + qy*qw)

        Rm[:,1,0] = 2*(qx*qy + qz*qw)
        Rm[:,1,1] = 1 - 2*(qx*qx + qz*qz)
        Rm[:,1,2] = 2*(qy*qz - qx*qw)

        Rm[:,2,0] = 2*(qx*qz - qy*qw)
        Rm[:,2,1] = 2*(qy*qz + qx*qw)
        Rm[:,2,2] = 1 - 2*(qx*qx + qy*qy)

        # Euler extraction (XYZ) roll, pitch, yaw
        roll = torch.atan2(Rm[:,2,1], Rm[:,2,2])
        pitch = torch.asin(torch.clamp(-Rm[:,2,0], -1.0 + self.epsilon, 1.0 - self.epsilon))
        yaw = torch.atan2(Rm[:,1,0], Rm[:,0,0])

        return torch.stack([roll, pitch, yaw], dim=1)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        ee_name = "endeffector"

        # --- get quaternions ---
        right_hand_ori = right_palm_orientation(states, self.robot_name, ee_name)  # [B, 4]
        cube1_ori     = states.objects["cube_1"].root_state[:, 3:7]             # [B, 4]

        # --- convert to Euler ---
        hand_eul = self.quat_to_euler(right_hand_ori)  # [B,3]
        cube_eul = self.quat_to_euler(cube1_ori)       # [B,3]

        # --- angular difference ---
        # normalize angle difference to [-pi, pi]
        diff = hand_eul - cube_eul
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi

        # L1 angular error (good approximation)
        angle_error = torch.norm(diff, dim=1)  # [B]

        # --- orientation reward ---
        orient_reward = humanoid_reward_util.tolerance(
            angle_error,
            bounds=(0.0, 0.1),
            margin=0.6,
            sigmoid="gaussian"
        )

        # --- weight based on distance (same as before) ---
        right_hand_pos = right_palm_position(states, self.robot_name, ee_name)
        cube1_pos = states.objects["cube_1"].root_state[:, :3]
        distance = torch.norm(right_hand_pos - cube1_pos, dim=1)

        distance_weight = torch.exp(-10.0 * torch.clamp(distance - 0.05, min=0.0))
        orient_reward = orient_reward * distance_weight

        # ===============================
        #   BONUS: moving toward target
        # ===============================

        if self.prev_angle_error is None:
            self.prev_angle_error = angle_error.clone()
            directional_bonus = torch.zeros_like(angle_error)
        else:
            improvement = self.prev_angle_error - angle_error
            directional_bonus = torch.clamp(improvement, min=0.0) * 0.5
            self.prev_angle_error = angle_error.clone()

        # add directional bonus
        total_reward = orient_reward + directional_bonus

        # bonus for being very close
        close_mask = (distance < 0.05) & (angle_error < 0.1)
        total_reward = total_reward + 0.5 * close_mask.float()

        total_reward = torch.clamp(total_reward, 0.0, 3.0)
        return total_reward



@configclass
class ReachposoriCfg(HumanoidTaskCfg):
    """Cube task for humanoid robots."""
    success_bar = 0.9
    episode_length = 200
    objects = [
        # ArticulationObjCfg(
        #     name="cube_1",
        #     mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_2/mjcf/cube_2.xml",
        #     urdf_path="roboverse_data/assets/humanoidbench/cube/cube_2/cube_2.urdf",
        #     usd_path="urdf2usd_convert/g1/usd/cube_2.usd",
        #     default_position= [0.3, 0.2, 0.9],
        #     fix_base_link=True,

        # ),
        RigidObjCfg(
            name="cube_1",
            mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_2/mjcf/cube_2.xml",
            urdf_path="roboverse_data/assets/humanoidbench/cube/cube_2/cube_2.urdf",
            usd_path="urdf2usd_convert/g1/usd/cube_2.usd",
            default_position= [0.3, 0.2, 0.9],
            physics=PhysicStateType.XFORM,

        ),
        RigidObjCfg(
            name="table",
            urdf_path="roboverse_data/assets/humanoidbench/table/urdf/table.urdf",
            usd_path="urdf2usd_convert/g1/usd/table.usd",
            default_orientation= [0.0, 0.0, 0.0, 1.0],
            default_position= [0.0, 0.0, 5.0],
            physics=PhysicStateType.GEOM,
        )
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
        # ArticulationObjCfg(
        #     name="table",
        #     urdf_path="roboverse_data/assets/humanoidbench/table/urdf/table.urdf",
        #     mjcf_path="roboverse_data/assets/humanoidbench/table/mjcf/table.mjcf",
        #     usd_path="urdf2usd_convert/g1/usd/table.usd",
        #     default_position= [0.0, 0.0, -0.2],
        #     fix_base_link=True,
        # ),

    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/cube/v2/g1/initial_state_v2.json"
    #traj_filepath = "my_env/initial_state_g1_v2.json"
    checker = _ReachCheckerPosOri()
    reward_weights = [0.5, 0.5]
    reward_functions = [ReachReward(), OrientationReward()]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
