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
class BalancedOrientPosReward(HumanoidBaseReward):
    """
    Reward = 0.5 * orientation_reward + 0.5 * distance_reward
    Vždy se hodnotí obě složky. Bez gate, bez vypínání distance rewardu.
    Stabilní, hladká kombinace.
    """

    def __init__(self, robot_name="g1_with_hands_simple"):
        super().__init__(robot_name)

        self.prev_distance = None
        self.prev_angle_error = None

        self.max_reward = 1.0

    def quat_diff_angle(self, q1, q2):
        """Quaternion angular difference in radians."""
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)

        dot = torch.sum(q1 * q2, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)

        return 2 * torch.acos(torch.abs(dot))

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        ee = "endeffector"

        # pozice a orientace
        hand_pos = right_palm_position(states, self.robot_name, ee)
        hand_ori = right_palm_orientation(states, self.robot_name, ee)

        cube_pos = states.objects["cube_1"].root_state[:, :3]
        cube_ori = states.objects["cube_1"].root_state[:, 3:7]

        # ---------------------------------------------------------
        # ORIENTATION REWARD
        # ---------------------------------------------------------
        angle_error = self.quat_diff_angle(hand_ori, cube_ori)

        orient_reward = humanoid_reward_util.tolerance(
            angle_error,
            bounds=(0.0, 0.05),   # perfektní orientace
            margin=0.7,           # kritický úhel kde reward klesá k 0
            sigmoid="gaussian",
        )

        # Bonus za zlepšení orientace
        if self.prev_angle_error is None:
            orient_improve_bonus = torch.zeros_like(angle_error)
        else:
            delta = torch.clamp(self.prev_angle_error - angle_error, min=0.0)
            orient_improve_bonus = 0.3 * delta

        self.prev_angle_error = angle_error.clone()

        orientation_total = orient_reward + orient_improve_bonus

        # ---------------------------------------------------------
        # DISTANCE REWARD
        # ---------------------------------------------------------
        distance = torch.norm(hand_pos - cube_pos, dim=1)

        dist_reward = humanoid_reward_util.tolerance(
            distance,
            bounds=(0.0, 0.03),   # velmi blízko objektu = max
            margin=0.35,
            sigmoid="gaussian",
        )

        # bonus za přiblížení
        if self.prev_distance is None:
            approach_bonus = torch.zeros_like(distance)
        else:
            d_improve = torch.clamp(self.prev_distance - distance, min=0.0)
            approach_bonus = 0.5 * d_improve

        self.prev_distance = distance.clone()

        distance_total = dist_reward + approach_bonus

        # ---------------------------------------------------------
        # 50:50 kombinace
        # ---------------------------------------------------------
        total = 0.5 * orientation_total + 0.5 * distance_total

        return torch.clamp(total, 0.0, self.max_reward)

class OrientThenReachReward(HumanoidBaseReward):
    """
    First reward robot ONLY for correct hand orientation.
    Once orientation is good enough, distance reward activates.
    """
    def __init__(self, robot_name="g1_with_hands_simple"):
        super().__init__(robot_name)
        self.prev_distance = None
        self.prev_angle_error = None

        # thresholds
        self.orientation_threshold = 0.25  # ~15 degrees (in radians)
        self.max_reward = 1.0              # safety clamp

    def quat_diff_angle(self, q1, q2):
        """Quaternion angular difference in radians."""
        # ensure normalized
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)

        # quaternion inner product → angle
        dot = torch.sum(q1 * q2, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)

        return 2 * torch.acos(torch.abs(dot))

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        ee = "endeffector"

        # --- Get positions and orientations ---
        hand_pos = right_palm_position(states, self.robot_name, ee)
        hand_ori = right_palm_orientation(states, self.robot_name, ee)

        cube_pos = states.objects["cube_1"].root_state[:, :3]
        cube_ori = states.objects["cube_1"].root_state[:, 3:7]

        # ==========================================================
        #   1) ORIENTATION REWARD
        # ==========================================================
        angle_error = self.quat_diff_angle(hand_ori, cube_ori)

        # orientation reward is 1 when perfect, 0 when >0.6 rad
        orient_reward = humanoid_reward_util.tolerance(
            angle_error,
            bounds=(0.0, 0.05),
            margin=0.6,
            sigmoid="gaussian"
        )

        # directional bonus (move toward correct orientation)
        if self.prev_angle_error is None:
            direction_bonus = torch.zeros_like(angle_error)
        else:
            improvement = torch.clamp(self.prev_angle_error - angle_error, min=0.0)
            direction_bonus = improvement * 0.5

        self.prev_angle_error = angle_error.clone()

        # total orientation component
        orientation_total = orient_reward + direction_bonus

        # ==========================================================
        #   Important logic:
        #   If orientation is bad → RETURN ONLY ORIENTATION REWARD
        # ==========================================================
        orientation_good = angle_error < self.orientation_threshold

        if not torch.any(orientation_good):
            # none of the envs has good orientation, distance reward disabled
            return torch.clamp(orientation_total, 0.0, self.max_reward)

        # ==========================================================
        #   2) DISTANCE REWARD (activated only after orientation OK)
        # ==========================================================
        distance = torch.norm(hand_pos - cube_pos, dim=1)

        dist_reward = humanoid_reward_util.tolerance(
            distance,
            bounds=(0.0, 0.03),
            margin=0.3,
            sigmoid="gaussian"
        )

        # approach bonus
        if self.prev_distance is None:
            approach_bonus = torch.zeros_like(distance)
        else:
            d_improve = torch.clamp(self.prev_distance - distance, min=0.0)
            approach_bonus = d_improve * 1.0

        self.prev_distance = distance.clone()

        # combine only for envs that passed orientation gate
        total_reward = 0.5 * orientation_total + 0.5*(dist_reward + approach_bonus) * orientation_good.float()


        return torch.clamp(total_reward, 0.0, self.max_reward)
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
    reward_weights = [1.0]
    reward_functions = [BalancedOrientPosReward()]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
