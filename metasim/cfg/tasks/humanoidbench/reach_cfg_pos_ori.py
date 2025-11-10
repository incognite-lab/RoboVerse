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
    """Reward function for aligning hand orientation with the cube orientation."""

    def __init__(self, robot_name="g1_with_hands_simple"):
        super().__init__(robot_name)
        self.epsilon = 1e-6

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        ee_name = "endeffector"

        # --- 1. Orientace ruky a objektu ---
        right_hand_ori = right_palm_orientation(states, self.robot_name, ee_name)  # [B, 4]
        cube1_orient = states.objects["cube_1"].body_state[:, 0, 3:7]              # [B, 4]

        # --- 2. (volitelné) pozice – kvůli vážení ---
        right_hand_pos = right_palm_position(states, self.robot_name, ee_name)
        cube1_pos = states.objects["cube_1"].root_state[:, :3]
        distance = torch.norm(right_hand_pos - cube1_pos, dim=1)                   # [B]

        # --- 3. Úhlová chyba mezi kvaterniony ---
        # Quaternion inner product → cos(theta/2)
        dot = torch.sum(right_hand_ori * cube1_orient, dim=1)
        dot = torch.clamp(torch.abs(dot), 0.0, 1.0 - self.epsilon)
        # Úhel chyby (radians): malý úhel → lepší orientace
        angle_error = 2.0 * torch.acos(dot)

        # --- 4. Přepočet na hladkou odměnu (1 = perfektní orientace, 0 = opačná) ---
        orient_reward = humanoid_reward_util.tolerance(
            angle_error,
            bounds=(0.0, 0.1),   # 0–0.1 rad ≈ 0–6°
            margin=0.6,          # plynulý přechod až do cca 35°
            sigmoid="gaussian"
        )

        # --- 5. Váhování podle vzdálenosti (orientace se počítá hlavně blízko cíle) ---
        # distance < 0.05 → váha ~1, distance = 0.1 → ~0.6, distance = 0.2 → ~0.14
        distance_weight = torch.exp(-10.0 * torch.clamp(distance - 0.05, min=0.0))
        orient_reward = orient_reward * distance_weight

        # --- 6. Dodatečný „bonus“ za velmi přesnou orientaci, pokud jsme blízko ---
        close_mask = (distance < 0.05) & (angle_error < 0.1)
        orient_reward = orient_reward + 0.5 * close_mask.float()

        # --- 7. Ořez a návrat ---
        orient_reward = torch.clamp(orient_reward, 0.0, 2.0)
        return orient_reward



@configclass
class ReachposoriCfg(HumanoidTaskCfg):
    """Cube task for humanoid robots."""
    success_bar = 0.9
    episode_length = 200
    objects = [
        ArticulationObjCfg(
            name="cube_1",
            mjcf_path="roboverse_data/assets/humanoidbench/cube/cube_2/mjcf/cube_2.xml",
            urdf_path="roboverse_data/assets/humanoidbench/cube/cube_2/cube_2.urdf",
            default_position= [0.3, 0.2, 0.9],
            fix_base_link=True,

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
    checker = _ReachCheckerPosOri()
    reward_weights = [0.5, 0.5]
    reward_functions = [ReachReward(), OrientationReward()]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
