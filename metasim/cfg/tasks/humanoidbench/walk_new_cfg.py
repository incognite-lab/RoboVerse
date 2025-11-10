"""Walking task for humanoid robots."""

from metasim.cfg.checkers import _WalkChecker
from metasim.utils import configclass
import torch
from .base_cfg import BaseLocomotionReward, HumanoidTaskCfg, HumanoidBaseReward
from metasim.types import EnvState
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    neck_height_tensor,
    robot_local_velocity_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
    robot_rotation_tensor,
)
def quat_to_euler_rpy(q: torch.Tensor) -> torch.Tensor:
    """
    Převede tenzor kvaternionů (w, x, y, z) na Eulerovy úhly (roll, pitch, yaw).

    Args:
        q (torch.Tensor): Tenzor kvaternionů s tvarem [..., 4].

    Returns:
        torch.Tensor: Tenzor Eulerových úhlů s tvarem [..., 3].
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotace)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotace)
    sinp = 2 * (w * y - z * x)
    # Ošetření proti gimbal lock (asin(1) nebo asin(-1))
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.pi / 2, sinp), # 90 stupňů
        torch.asin(sinp)
    )

    # Yaw (z-axis rotace)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Vrátí RPY ve sloupci
    return torch.stack([roll, pitch, yaw], dim=-1)
class CommandFollowXYReward(HumanoidBaseReward):
    """Reward function for following command direction xy."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the command follow reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the command follow reward."""
        # Get robot velocity
        cmd = states.sencors["command0"]
        robot_vel_x,robot_vel_y,_ = robot_local_velocity_tensor(states, self.robot_name).unbind(dim=1)
        err_x = cmd[:,0] - robot_vel_x
        err_y = cmd[:,1] - robot_vel_y
        err_vel_xy = err_x**2 + err_y**2
        R_vel_xy = torch.exp(-5.0 * err_vel_xy)
        return R_vel_xy
class CommandFollowYawReward(HumanoidBaseReward):
    """Reward function for following command yaw."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the command follow reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the command follow reward."""
        # Get robot yaw rate
        cmd = states.sencors["command0"]
        q = robot_rotation_tensor(states, self.robot_name)  # (B,4)
        w, x, y, z = q.unbind(-1)  # rozbalíme komponenty quaternionu

        # vypočítáme yaw
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y*y + z*z))  # (B,)

        err_yaw = torch.abs(cmd[:,2] - yaw)
        R_yaw = torch.exp(-30.0 * err_yaw)
        return R_yaw
class SingleFootContactReward(HumanoidBaseReward):
    """Reward function for single foot contact."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the single foot contact reward."""
        super().__init__(robot_name)
        self.time_both_foot_on_ground = None


    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the single foot contact reward."""
        cmd = states.sencors["command0"]

        left_foot_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_foot_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_foot_contact = states.robots[self.robot_name].body_states[:, left_foot_idx, 2]<0.1
        right_foot_contact = states.robots[self.robot_name].body_states[:, right_foot_idx, 2]<0.1
        is_standing = torch.norm(cmd, dim=1) < 0.01 # Tvar: [num_envs,]

        is_single_contact = (left_foot_contact & ~right_foot_contact) | (~left_foot_contact & right_foot_contact)
        is_double_contact = left_foot_contact & right_foot_contact
        if self.time_both_foot_on_ground is None:
            self.time_both_foot_on_ground = torch.zeros(self.num_envs, device=self.device)
        # Logika pro "grace period"
        # Resetuj časovač, pokud je jen jedna noha na zemi
        self.time_both_foot_on_ground = torch.where(is_single_contact, 0.0, self.time_both_foot_on_ground + 1.0)
        # Resetuj časovač, pokud nestojí (aby se nepočítal při stání)
        self.time_both_foot_on_ground = torch.where(is_standing, 0.0, self.time_both_foot_on_ground)

        # Podmínka pro odměnu (zjednodušená)
        # Odměna = 1 pokud je single contact NEBO pokud je double contact jen na < 3 kroky
        reward_condition = is_single_contact | (is_double_contact & (self.time_both_foot_on_ground < 3.0))

        # Finální odměna
        # 1.0 pokud stojí, jinak 1.0/0.0 podle 'reward_condition'
        reward = torch.where(
            is_standing,
            1.0,
            reward_condition.float()
        )
        return reward
class BaseHeightReward(HumanoidBaseReward):
    """Base class for height rewards."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the height reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the height reward."""
        neck_height = neck_height_tensor(states, self.robot_name)
        neck_height_ref = self._stand_neck_height
        err_height = torch.abs(neck_height - neck_height_ref)
        R_height = torch.exp(-20.0 * err_height)
        return R_height
class FeetAirTimeReward(HumanoidBaseReward):
    """Reward function for feet airtime."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the feet airtime reward."""
        super().__init__(robot_name)
        self.foot_airtime = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)# Tenzor pro sledování stavu kontaktu z *předchozího* kroku


    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the feet airtime reward."""

        if self.command[:] == torch.tensor([0.0,0.0,0.0]):
            #return zeros tensor for all envs becoause standing
            self.foot_airtime = torch.zeros(2, device="cpu")
            return torch.ones(states.num_envs, device=states.device)
        left_foot_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_foot_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_foot_contact = states.robots[self.robot_name].body_states[:, left_foot_idx, 2]<0.1
        right_foot_contact = states.robots[self.robot_name].body_states[:, right_foot_idx, 2]<0.1
        current_contact = torch.stack([left_foot_contact, right_foot_contact], dim=1)
        just_touched_down = current_contact & (~self.prev_contact)

        self.foot_airtime = torch.where(current_contact, 0.0, self.foot_airtime + 1.0)
        reward_values = self.foot_airtime - 4.0
        foot_rewards = torch.where(
            just_touched_down,
            reward_values,
            0.0  # Nulová odměna, pokud nedošlo k dopadu
        )
        total_step_reward = torch.sum(foot_rewards, dim=1)

        self.prev_contact = current_contact
        return total_step_reward
class FeetOrientationReward(HumanoidBaseReward):
    """Reward function for feet orientation."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the feet orientation reward."""
        super().__init__(robot_name)
        self.ROTATION_THRESHOLD = 0.1
        self.REWARD_SCALING_FACTOR = 3.0
    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """
        Vypočítá odměnu za orientaci chodidel pro všechna prostředí.

        Args:
            states (list[EnvState]): Aktuální stav simulace.
            command (torch.Tensor): Aktuální příkaz [c_x, c_y, c_yaw]
                                      pro každé prostředí (shape: [num_envs, 3]).
        """

        # --- 1. Získání kvaternionů chodidel ---
        # Tvar: [num_envs, 4]
        left_foot_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_foot_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_foot_quat = states.robots[self.robot_name].body_state[:, left_foot_idx, 3:7]
        right_foot_quat = states.robots[self.robot_name].body_state[:, right_foot_idx, 3:7]

        # --- 2. Převod na Eulerovy úhly (RPY) ---
        # Tvar: [num_envs, 3]
        left_rpy = quat_to_euler_rpy(left_foot_quat)
        right_rpy = quat_to_euler_rpy(right_foot_quat)

        # Cílová orientace je [0, 0, 0], takže chyba je absolutní hodnota úhlů
        abs_left_rpy = torch.abs(left_rpy)
        abs_right_rpy = torch.abs(right_rpy)

        # --- 3. Výpočet chyb ---

        # Chyba RPY (Roll + Pitch + Yaw) pro obě nohy
        # Použije se, když se robot NEOTÁČÍ
        # Tvar: [num_envs,]
        total_error_rpy = torch.sum(abs_left_rpy, dim=1) + torch.sum(abs_right_rpy, dim=1)

        # Chyba RP (Roll + Pitch) pro obě nohy
        # Použije se, když se robot OTÁČÍ
        # Tvar: [num_envs,]
        total_error_rp = torch.sum(abs_left_rpy[:, :2], dim=1) + torch.sum(abs_right_rpy[:, :2], dim=1)

        # --- 4. Výběr chyby na základě příkazu ---
        # Příkaz k otáčení, Tvar: [num_envs,]
        command_yaw = self.command[:, 2]

        # Maska, Tvar: [num_envs,]
        is_rotating = torch.abs(command_yaw) > self.ROTATION_THRESHOLD

        # Vektorizovaný výběr chyby
        # Tvar: [num_envs,]
        total_error = torch.where(
            is_rotating,
            total_error_rp,   # Penalizuj pouze RP, pokud se otáčíme
            total_error_rpy   # Penalizuj RPY, pokud se neotáčíme
        )

        # --- 5. Výpočet finální odměny ---
        # e^(-k * chyba)
        reward = torch.exp(-self.REWARD_SCALING_FACTOR * total_error)

        return reward
class FeetPositionReward(BaseLocomotionReward):
    """Reward function for feet position."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the feet position reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the feet position reward."""
        command = self.command
        R_feet_pos = torch.ones(self.num_envs, device=self.device)
        left_foot_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_foot_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_foot_pos = states.robots[self.robot_name].body_state[:, left_foot_idx, :3]
        right_foot_pos = states.robots[self.robot_name].body_state[:, right_foot_idx, :3]
        is_standing = torch.norm(command, dim=1) < 0.01
        treshold_haigh = 0.1
        err_feet_pos_heigh = torch.abs(left_foot_pos[:, 2]) + torch.abs(right_foot_pos[:, 2])
        err_feet_pos_heigh_normalise = torch.clamp(err_feet_pos_heigh - treshold_haigh, min=0.0)
        err_left_feet = torch.norm(command[:,:2]-left_foot_pos[:,:2])
        err_right_feet = torch.norm(command[:,:2]-right_foot_pos[:,:2])
        treshold = 0.5
        err_left_normalise = torch.clamp(err_left_feet - treshold, min=0.0)
        err_right_normalise = torch.clamp(err_right_feet - treshold, min=0.0)
        err_feet_pos_xyz = err_left_normalise + err_right_normalise + err_feet_pos_heigh_normalise
        R_feet_pos_xy = torch.exp(-3.0 * err_feet_pos_xyz)
        reward = torch.where(
            is_standing,
            torch.ones_like(R_feet_pos_xy),
            R_feet_pos_xy
        )
        return reward

@configclass
class WalkNewCfg(HumanoidTaskCfg):
    """Walking task for humanoid robots."""
    W_VEL_XY = 0.15
    W_YAW_ORIENT = 0.1
    W_RP_ORIENT = 0.2
    W_CONTACT = 0.1
    W_BASE_HEIGHT = 0.05
    W_FEET_AIRTIME = 1.0  #Vysoká váha, protože jde o řídkou odměnu
    W_FEET_ORIENT = 0.05
    W_FEET_POS = 0.05
    W_ARM = 0.03
    W_BASE_ACCEL = 0.1
    W_ACTION_DIFF = 0.02
    W_TORQUE = 0.02
    commmand = torch.tensor([1.0,0.0,0.0])
    name = "walk_new"

    episode_length = 1000
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/h1_v2.pkl"
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/initial_state_v2.json"
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"

    checker = _WalkChecker()
    reward_functions = [CommandFollowXYReward()
                        # CommandFollowYawReward(),
                        # SingleFootContactReward(),
                        # BaseHeightReward(),
                        # FeetAirTimeReward(),
                        # FeetOrientationReward(),
                        # FeetPositionReward()
                        ]
    reward_weights = [W_VEL_XY,
                      W_YAW_ORIENT,
                      W_CONTACT,
                      W_BASE_HEIGHT,
                      W_FEET_AIRTIME,
                      W_FEET_ORIENT,
                      W_FEET_POS
                      ]

    def extra_spec(self):
        print("dddddddd")
        """This task does not require any extra observations."""
        return {}
