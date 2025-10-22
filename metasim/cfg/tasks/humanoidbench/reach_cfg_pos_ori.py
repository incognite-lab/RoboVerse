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
    success_bar = 0.9
    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the reach reward."""
        super().__init__(robot_name)
        self.prev_distance = None  # pro výpočet změny vzdálenosti

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the reach reward."""
        cube1_state = states.objects["cube_1"].root_state[:, :3]  # [num_envs, 3]
        right_hand_pos = right_palm_position(states, self.robot_name)
        #right_hand_pos = right_palm_position(states, self.robot_name,"endeffector")  # [num_envs, 3]

        # aktuální vzdálenost ruky od cíle
        distance = torch.norm(right_hand_pos - cube1_state, dim=1)  # [num_envs]

        # základní odměna – blízkost k cíli
        #distance_np = distance.detach().cpu().numpy()
        base_reward = humanoid_reward_util.tolerance(
            distance,
            bounds=(0.0, 0.05),   # cílové okno
            margin=0.5,
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
            approach_bonus = torch.clamp(delta_distance, min=0.0) * 2.0  # zesil efekt
            self.prev_distance = distance.clone()

        # --- Penalizace za vzdálení ---
        away_penalty = torch.clamp(distance - 1.0, min=0.0) * 0.1  # pokud >1m, mírná penalizace

        # --- Celková odměna ---
        total_reward = base_reward + approach_bonus - away_penalty

        # omez hodnoty (pro stabilitu)
        total_reward = torch.clamp(total_reward, 0.0, 5.0)
        #print("Total Reward:", total_reward)
        return total_reward


class OrientationReward(HumanoidBaseReward):
    """Reward function for cube orientation alignment."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the orientation reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # Pozice a orientace
        ee_name = "endeffector"
        right_hand_pos = right_palm_position(states, self.robot_name, ee_name)
        right_hand_ori = right_palm_orientation(states, self.robot_name, ee_name)
        cube1_state = states.objects["cube_1"].root_state[:, :3]
        cube1_orient = states.objects["cube_1"].body_state[:, 0, 3:7]

        # vzdálenost od cíle
        distance = torch.norm(right_hand_pos - cube1_state, dim=1)

        # výpočet shody orientace (kvaternionový dot produkt)
        dot_product = torch.abs(torch.sum(right_hand_ori * cube1_orient, dim=1))

        # základní orientační odměna
        orient_reward = humanoid_reward_util.tolerance(
            dot_product,
            bounds=(0.98, 1.0),   # téměř perfektní shoda
            margin=0.2,           # plynulý přechod
            sigmoid="gaussian"
        )
        #orient_reward = torch.as_tensor(orient_reward_np, device=dot_product.device, dtype=dot_product.dtype)

        # váhový koeficient závislý na vzdálenosti
        # (čím blíž, tím víc se orientace počítá)
        distance_weight = torch.exp(-10.0 * torch.clamp(distance - 0.05, min=0.0))
        # - pokud distance < 0.05 → váha ≈ 1
        # - pokud distance = 0.1  → váha ≈ 0.6
        # - pokud distance = 0.2  → váha ≈ 0.14
        orient_reward *= distance_weight
        #print("right hand ori",right_hand_ori)
        #print("cube ori",cube1_orient)
        #print("Orientation Reward:", orient_reward)
        return orient_reward




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
class ReachposoriCfg(HumanoidTaskCfg):
    """Cube task for humanoid robots."""
    success_bar = 0.9
    episode_length = 100
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
