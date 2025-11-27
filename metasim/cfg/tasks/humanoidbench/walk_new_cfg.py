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
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotace)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotace)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.full_like(sinp, torch.pi / 2), sinp),  # ✅ opraveno
        torch.asin(sinp)
    )

    # Yaw (z-axis rotace)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)
class CommandFollowXYReward(HumanoidBaseReward):
    """Reward function for following command direction xy."""

    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the command follow reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the command follow reward."""
        # Get robot velocity
        cmd = states.sensors["command0"]

        robot_vel = robot_local_velocity_tensor(states, self.robot_name).unbind(dim=1)
        robot_vel_x = robot_vel[0]
        robot_vel_y = robot_vel[1]
        err_x = torch.clamp(cmd[:, 0] - robot_vel_x, min=0.0)
        err_y = torch.clamp(cmd[:, 1] - robot_vel_y, min=0.0)
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
        cmd = states.sensors["command0"]


        q = robot_rotation_tensor(states, self.robot_name)  # (B,4)
        w, x, y, z = q.unbind(-1)  # rozbalíme komponenty quaternionu

        # vypočítáme yaw
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y*y + z*z))  # (B,)

        err_yaw = torch.clamp(cmd[:,2] - yaw, min=0.0)
        R_yaw = torch.exp(-300.0 * err_yaw)
        return R_yaw
class SingleFootContactReward(HumanoidBaseReward):
    """Reward for having single-foot contact during walking."""

    def __init__(self, robot_name="g1_with_hands", dt: float = 0.02):
        super().__init__(robot_name)
        self.dt = dt
        self.time_both_foot_on_ground = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        CONTACT_HEIGHT = 0.1
        cmd = states.sensors["command0"]
        is_standing = torch.norm(cmd, dim=1) < 0.01  # zero command = stand

        left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")

        left_contact = states.robots[self.robot_name].body_state[:, left_idx, 2] < CONTACT_HEIGHT
        right_contact = states.robots[self.robot_name].body_state[:, right_idx, 2] < CONTACT_HEIGHT
        is_single_contact = (left_contact ^ right_contact)
        is_double_contact = left_contact & right_contact

        if self.time_both_foot_on_ground is None:
            self.time_both_foot_on_ground = torch.zeros_like(is_double_contact, dtype=torch.float32)

        # 0.2 s grace → 0.2 / dt = počet kroků
        grace_steps = int(0.2 / self.dt)
        self.time_both_foot_on_ground = torch.where(
            is_single_contact, 0.0, self.time_both_foot_on_ground + 1.0
        )

        reward_condition = is_single_contact | (is_double_contact & (self.time_both_foot_on_ground < grace_steps))
        reward = torch.where(is_standing, 1.0, reward_condition.float())
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
    """Reward for adequate airtime and rhythmic stepping."""

    def __init__(self, robot_name="g1_with_hands", dt: float = 0.02):
        super().__init__(robot_name)
        self.dt = dt
        self.foot_airtime = None
        self.prev_contact = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        CONTACT_HEIGHT = 0.1
        WANT_AIRTIME = 1  # seconds
        cmd = states.sensors["command0"]
        is_standing = torch.norm(cmd, dim=1) < 0.01

        num_envs = states.robots[self.robot_name].body_state.size(0)
        if self.foot_airtime is None:
            self.foot_airtime = torch.zeros(num_envs, 2, dtype=torch.float32)
        if self.prev_contact is None:
            self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool)

        left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_contact = states.robots[self.robot_name].body_state[:, left_idx, 2] < CONTACT_HEIGHT
        right_contact = states.robots[self.robot_name].body_state[:, right_idx, 2] < CONTACT_HEIGHT
        current_contact = torch.stack([left_contact, right_contact], dim=1)

        just_touched_down = current_contact & (~self.prev_contact)

        # airtime += dt while in air
        self.foot_airtime = torch.where(current_contact, 0.0, self.foot_airtime + self.dt)

        # reward = (t_air - 0.4) for feet that just touched down
        reward_values = torch.abs(self.foot_airtime - WANT_AIRTIME)
        foot_rewards = torch.where(just_touched_down, reward_values, 0.0)
        total_reward = torch.sum(foot_rewards, dim=1)

        # 1 for standing command (constant)
        reward = torch.where(is_standing, torch.ones_like(total_reward), total_reward)

        self.prev_contact = current_contact
        return reward
class FeetOrientationReward(HumanoidBaseReward):
    """Reward for keeping feet level and aligned."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.ROTATION_THRESHOLD = 0.1
        self.SCALE = 3.0  # odpovídá paperu: e^(−Σ|r_feet−ref|)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        cmd = states.sensors["command0"]
        command_yaw = cmd[:, 2]

        left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_q = states.robots[self.robot_name].body_state[:, left_idx, 3:7]
        right_q = states.robots[self.robot_name].body_state[:, right_idx, 3:7]

        left_rpy = quat_to_euler_rpy(left_q)
        right_rpy = quat_to_euler_rpy(right_q)
        abs_left = torch.abs(left_rpy)
        abs_right = torch.abs(right_rpy)

        total_err_rpy = torch.sum(abs_left, dim=1) + torch.sum(abs_right, dim=1)
        total_err_rp = torch.sum(abs_left[:, :2], dim=1) + torch.sum(abs_right[:, :2], dim=1)
        is_rotating = torch.abs(command_yaw) > self.ROTATION_THRESHOLD

        total_err = torch.where(is_rotating, total_err_rp, total_err_rpy)
        reward = torch.exp(-self.SCALE * total_err)
        return reward
class FeetPositionReward(BaseLocomotionReward):
    """Reward for keeping feet near neutral position during standing."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        cmd = states.sensors["command0"]
        is_standing = torch.norm(cmd, dim=1) < 0.01

        left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
        right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
        left_pos = states.robots[self.robot_name].body_state[:, left_idx, :3]
        right_pos = states.robots[self.robot_name].body_state[:, right_idx, :3]

        feet_dist = torch.norm(left_pos[:, :2] - right_pos[:, :2], dim=1)
        R_feet = torch.exp(-3.0 * feet_dist)

        # Reward active only when standing
        reward = torch.where(is_standing, R_feet, torch.ones_like(R_feet))
        return reward



class ArmPoseReward(HumanoidBaseReward):
    """Reward for keeping arm joints near a nominal reference configuration."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        # Reference arm joint angles [rad]; můžeš doladit podle svého modelu
        self.ref_angles = {
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.0471,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.0471,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[self.robot_name]
        joint_names = robot.joint_names
        joint_pos = robot.joint_pos
        q_dict = {name: joint_pos[:, idx] for idx, name in enumerate(joint_names)}
        # dict: joint_name -> tensor (B,)
        errors = []
        for jname, ref in self.ref_angles.items():
            if jname in q_dict:
                err = torch.abs(q_dict[jname] - ref)
                errors.append(err)
        if len(errors) == 0:
            return torch.ones(states.num_envs, device=states.device)
        err_total = torch.stack(errors, dim=1).sum(dim=1)
        return torch.exp(-3.0 * err_total)

class BaseAccelerationReward(HumanoidBaseReward):
    """Reward for minimizing base linear acceleration."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_base_vel = None

    def __call__(self, states: list[EnvState], robot_name: str = None,dt: float = 0.02) -> torch.FloatTensor:
        if self.prev_base_vel is None:
            self.prev_base_vel = robot_velocity_tensor(states, self.robot_name)


        base_vel = robot_velocity_tensor(states,self.robot_name)
        base_acc = (base_vel - self.prev_base_vel)/dt


        accel_sum = torch.sum(torch.abs(base_acc), dim=1)
        reward = torch.exp(-0.01 * accel_sum)
        self.prev_base_vel = base_vel.clone()
        return reward


class TorqueReward(HumanoidBaseReward):
    """Reward for minimizing actuator torque usage.
    ---->zerous in sapien3<------
    """

    def __init__(self, robot_name="g1_with_hands", torque_limit: float = 100.0):
        super().__init__(robot_name)
        self.torque_limit = torque_limit

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        tau = actuator_forces_tensor(states, self.robot_name)  # (B, N)
        mean_torque_ratio = torch.mean(torch.abs(tau), dim=1)
        reward = torch.exp(-0.02 * mean_torque_ratio)
        return reward
class ActionDifferenceReward(HumanoidBaseReward):
    """Reward for smoothness in consecutive actions."""

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_action = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        curr_action = states.robots[self.robot_name].joint_pos
        #TODO
        if self.prev_action is None:
            self.prev_action = curr_action.clone()
        diff = torch.abs(curr_action - self.prev_action).sum(dim=1)
        reward = torch.exp(-0.02 * diff)
        self.prev_action = curr_action.clone()
        return reward
class RollPitchOrientationReward(HumanoidBaseReward):
    """
    Reward penalizing torso roll & pitch deviations.
    Implements Table I: e^{-30 * qd(q_rp, c_rp)} where c_rp = identity (roll=0,pitch=0).
    """

    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: EnvState, robot_name: str = None) -> torch.FloatTensor:
        # torso orientation quaternion (w,x,y,z)
        q = robot_rotation_tensor(states, self.robot_name)  # (B,4)

        # Extract roll & pitch only, zero yaw
        # Convert to rpy
        rpy = quat_to_euler_rpy(q)  # (B,3)

        # Zero yaw → only roll, pitch matter
        rpy_rp = torch.stack([rpy[:, 0], rpy[:, 1], torch.zeros_like(rpy[:, 2])], dim=1)

        # Convert rp-only back to quaternion
        roll = rpy_rp[:, 0]
        pitch = rpy_rp[:, 1]

        cy = torch.ones_like(roll)          # yaw = 0 => cos(yaw/2)=1
        sy = torch.zeros_like(roll)

        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        # quaternion from (roll,pitch,0)
        # standard ZYX convention
        qr = torch.stack([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ], dim=1)  # (B,4)

        # reference orientation: identity quaternion (roll=0,pitch=0,yaw=0)
        q_ref = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q)

        # quaternion distance (qd)
        # qd(q1, q2) = 1 - |dot(q1,q2)|
        dot = torch.sum(qr * q_ref, dim=1)
        qd = 1.0 - torch.abs(dot)

        # reward
        reward = torch.exp(-30.0 * qd)
        return reward
@configclass
class WalkNewCfg(HumanoidTaskCfg):
    """Walking task for humanoid robots."""
    W_VEL_XY = 0.24
    W_YAW_ORIENT = 0.1
    W_RP_ORIENT = 0.2
    W_CONTACT = 0.1
    W_BASE_HEIGHT = 0.05
    W_FEET_AIRTIME = 1.0  #Vysoká váha, protože jde o řídkou odměnu
    W_FEET_ORIENT = 0.05
    W_FEET_POS = 0.05
    W_ARM = 0.03
    W_BASE_ACCEL = 0.01
    W_ACTION_DIFF = 0.02
    W_TORQUE = 0.02
    commmand = torch.tensor([1.0,0.0,0.0])
    name = "walk_new"

    episode_length = 1000
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/h1_v2.pkl"
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/initial_state_v2.json"
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"

    checker = _WalkChecker()
    reward_functions = [CommandFollowXYReward(),
                        CommandFollowYawReward(),
                        SingleFootContactReward(),
                        BaseHeightReward(),
                        FeetAirTimeReward(),
                        FeetOrientationReward(),
                        FeetPositionReward(),
                        ArmPoseReward(),
                        BaseAccelerationReward(),
                        TorqueReward(),
                        ActionDifferenceReward(),
                        RollPitchOrientationReward()
                        ]
    reward_weights = [W_VEL_XY,
                      W_YAW_ORIENT,
                      W_CONTACT,
                      W_BASE_HEIGHT,
                      W_FEET_AIRTIME,
                      W_FEET_ORIENT,
                      W_FEET_POS,
                      W_ARM,
                      W_BASE_ACCEL,
                      W_TORQUE,
                      W_ACTION_DIFF,
                      W_RP_ORIENT
                      ]

    def extra_spec(self):

        """This task does not require any extra observations."""
        return {}
