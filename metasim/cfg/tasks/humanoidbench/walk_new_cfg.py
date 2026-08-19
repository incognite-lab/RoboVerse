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
# def quat_to_euler_rpy(q: torch.Tensor) -> torch.Tensor:
#     """
#     Převede tenzor kvaternionů (w, x, y, z) na Eulerovy úhly (roll, pitch, yaw).
#     """
#     w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

#     # Roll (x-axis rotace)
#     sinr_cosp = 2 * (w * x + y * z)
#     cosr_cosp = 1 - 2 * (x * x + y * y)
#     roll = torch.atan2(sinr_cosp, cosr_cosp)

#     # Pitch (y-axis rotace)
#     sinp = 2 * (w * y - z * x)
#     pitch = torch.where(
#         torch.abs(sinp) >= 1,
#         torch.copysign(torch.full_like(sinp, torch.pi / 2), sinp),  # ✅ opraveno
#         torch.asin(sinp)
#     )

#     # Yaw (z-axis rotace)
#     siny_cosp = 2 * (w * z + x * y)
#     cosy_cosp = 1 - 2 * (y * y + z * z)
#     yaw = torch.atan2(siny_cosp, cosy_cosp)

#     return torch.stack([roll, pitch, yaw], dim=-1)
# class CommandFollowXYReward(HumanoidBaseReward):
#     """Reward function for following command direction xy."""

#     def __init__(self, robot_name="g1_with_hands"):
#         """Initialize the command follow reward."""
#         super().__init__(robot_name)

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         """Compute the command follow reward."""
#         # Get robot velocity
#         cmd = states.sensors["command0"]

#         robot_vel = robot_local_velocity_tensor(states, self.robot_name).unbind(dim=1)
#         robot_vel_x = robot_vel[0]
#         robot_vel_y = robot_vel[1]
#         err_x = torch.clamp(cmd[:, 0] - robot_vel_x, min=0.0)
#         err_y = torch.clamp(cmd[:, 1] - robot_vel_y, min=0.0)
#         # err_x = cmd[:,0] - robot_vel_x
#         # err_y = cmd[:,1] - robot_vel_y
#         err_vel_xy = err_x**2 + err_y**2
#         R_vel_xy = torch.exp(-5.0 * err_vel_xy)
#         return R_vel_xy
# class CommandFollowYawReward(HumanoidBaseReward):
#     """Reward function for following command yaw."""

#     def __init__(self, robot_name="g1_with_hands"):
#         """Initialize the command follow reward."""
#         super().__init__(robot_name)

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         """Compute the command follow reward."""
#         # Get robot yaw rate
#         cmd = states.sensors["command0"]


#         q = robot_rotation_tensor(states, self.robot_name)  # (B,4)
#         w, x, y, z = q.unbind(-1)  # rozbalíme komponenty quaternionu

#         # vypočítáme yaw
#         yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y*y + z*z))  # (B,)

#         err_yaw = torch.clamp(cmd[:,2] - yaw, min=0.0)
#         R_yaw = torch.exp(-300.0 * err_yaw)
#         return R_yaw
# class SingleFootContactReward(HumanoidBaseReward):
#     """Reward for having single-foot contact during walking."""

#     def __init__(self, robot_name="g1_with_hands", dt: float = 0.02):
#         super().__init__(robot_name)
#         self.dt = dt
#         self.time_both_foot_on_ground = None

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         CONTACT_HEIGHT = 0.1
#         cmd = states.sensors["command0"]
#         is_standing = torch.norm(cmd, dim=1) < 0.01  # zero command = stand

#         left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
#         right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")

#         left_contact = states.robots[self.robot_name].body_state[:, left_idx, 2] < CONTACT_HEIGHT
#         right_contact = states.robots[self.robot_name].body_state[:, right_idx, 2] < CONTACT_HEIGHT
#         is_single_contact = (left_contact ^ right_contact)
#         is_double_contact = left_contact & right_contact

#         if self.time_both_foot_on_ground is None:
#             self.time_both_foot_on_ground = torch.zeros_like(is_double_contact, dtype=torch.float32)

#         # 0.2 s grace → 0.2 / dt = počet kroků
#         grace_steps = int(0.2 / self.dt)
#         self.time_both_foot_on_ground = torch.where(
#             is_single_contact, 0.0, self.time_both_foot_on_ground + 1.0
#         )

#         reward_condition = is_single_contact | (is_double_contact & (self.time_both_foot_on_ground < grace_steps))
#         reward = torch.where(is_standing, 1.0, reward_condition.float())
#         return reward
# class BaseHeightReward(HumanoidBaseReward):
#     """Base class for height rewards."""

#     def __init__(self, robot_name="g1_with_hands"):
#         """Initialize the height reward."""
#         super().__init__(robot_name)

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         """Compute the height reward."""
#         neck_height = neck_height_tensor(states, self.robot_name)
#         neck_height_ref = self._stand_neck_height
#         err_height = torch.abs(neck_height - neck_height_ref)
#         R_height = torch.exp(-20.0 * err_height)
#         return R_height
# class FeetAirTimeReward(HumanoidBaseReward):
#     """Reward for adequate airtime and rhythmic stepping."""

#     def __init__(self, robot_name="g1_with_hands", dt: float = 0.02):
#         super().__init__(robot_name)
#         self.dt = dt
#         self.foot_airtime = None
#         self.prev_contact = None

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         CONTACT_HEIGHT = 0.1
#         WANT_AIRTIME = 1  # seconds
#         cmd = states.sensors["command0"]
#         is_standing = torch.norm(cmd, dim=1) < 0.01

#         num_envs = states.robots[self.robot_name].body_state.size(0)
#         if self.foot_airtime is None:
#             self.foot_airtime = torch.zeros(num_envs, 2, dtype=torch.float32)
#         if self.prev_contact is None:
#             self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool)

#         left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
#         right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
#         left_contact = states.robots[self.robot_name].body_state[:, left_idx, 2] < CONTACT_HEIGHT
#         right_contact = states.robots[self.robot_name].body_state[:, right_idx, 2] < CONTACT_HEIGHT
#         current_contact = torch.stack([left_contact, right_contact], dim=1)

#         just_touched_down = current_contact & (~self.prev_contact)

#         # airtime += dt while in air
#         self.foot_airtime = torch.where(current_contact, 0.0, self.foot_airtime + self.dt)

#         # reward = (t_air - 0.4) for feet that just touched down
#         reward_values = torch.abs(self.foot_airtime - WANT_AIRTIME)
#         foot_rewards = torch.where(just_touched_down, reward_values, 0.0)
#         total_reward = torch.sum(foot_rewards, dim=1)

#         # 1 for standing command (constant)
#         reward = torch.where(is_standing, torch.ones_like(total_reward), total_reward)

#         self.prev_contact = current_contact
#         return reward
# class FeetOrientationReward(HumanoidBaseReward):
#     """Reward for keeping feet level and aligned."""

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)
#         self.ROTATION_THRESHOLD = 0.1
#         self.SCALE = 3.0  # odpovídá paperu: e^(−Σ|r_feet−ref|)

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         cmd = states.sensors["command0"]
#         command_yaw = cmd[:, 2]

#         left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
#         right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
#         left_q = states.robots[self.robot_name].body_state[:, left_idx, 3:7]
#         right_q = states.robots[self.robot_name].body_state[:, right_idx, 3:7]

#         left_rpy = quat_to_euler_rpy(left_q)
#         right_rpy = quat_to_euler_rpy(right_q)
#         abs_left = torch.abs(left_rpy)
#         abs_right = torch.abs(right_rpy)

#         total_err_rpy = torch.sum(abs_left, dim=1) + torch.sum(abs_right, dim=1)
#         total_err_rp = torch.sum(abs_left[:, :2], dim=1) + torch.sum(abs_right[:, :2], dim=1)
#         is_rotating = torch.abs(command_yaw) > self.ROTATION_THRESHOLD

#         total_err = torch.where(is_rotating, total_err_rp, total_err_rpy)
#         reward = torch.exp(-self.SCALE * total_err)
#         return reward
# class FeetPositionReward(BaseLocomotionReward):
#     """Reward for keeping feet near neutral position during standing."""

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         cmd = states.sensors["command0"]
#         is_standing = torch.norm(cmd, dim=1) < 0.01

#         left_idx = states.robots[self.robot_name].body_names.index("left_ankle_roll_link")
#         right_idx = states.robots[self.robot_name].body_names.index("right_ankle_roll_link")
#         left_pos = states.robots[self.robot_name].body_state[:, left_idx, :3]
#         right_pos = states.robots[self.robot_name].body_state[:, right_idx, :3]

#         feet_dist = torch.norm(left_pos[:, :2] - right_pos[:, :2], dim=1)
#         R_feet = torch.exp(-3.0 * feet_dist)

#         # Reward active only when standing
#         reward = torch.where(is_standing, R_feet, torch.ones_like(R_feet))
#         return reward



# class ArmPoseReward(HumanoidBaseReward):
#     """Reward for keeping arm joints near a nominal reference configuration."""

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)
#         # Reference arm joint angles [rad]; můžeš doladit podle svého modelu
#         self.ref_angles = {
#             "left_shoulder_pitch_joint": 0.0,
#             "left_shoulder_roll_joint": 0.0,
#             "left_shoulder_yaw_joint": 0.0,
#             "left_elbow_joint": 1.0471,
#             "left_wrist_roll_joint": 0.0,
#             "left_wrist_pitch_joint": 0.0,
#             "left_wrist_yaw_joint": 0.0,
#             "right_shoulder_pitch_joint": 0.0,
#             "right_shoulder_roll_joint": 0.0,
#             "right_shoulder_yaw_joint": 0.0,
#             "right_elbow_joint": 1.0471,
#             "right_wrist_roll_joint": 0.0,
#             "right_wrist_pitch_joint": 0.0,
#             "right_wrist_yaw_joint": 0.0,
#         }

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         robot = states.robots[self.robot_name]
#         joint_names = robot.joint_names
#         joint_pos = robot.joint_pos
#         q_dict = {name: joint_pos[:, idx] for idx, name in enumerate(joint_names)}
#         # dict: joint_name -> tensor (B,)
#         errors = []
#         for jname, ref in self.ref_angles.items():
#             if jname in q_dict:
#                 err = torch.abs(q_dict[jname] - ref)
#                 errors.append(err)
#         if len(errors) == 0:
#             return torch.ones(states.num_envs, device=states.device)
#         err_total = torch.stack(errors, dim=1).sum(dim=1)
#         return torch.exp(-3.0 * err_total)

# class BaseAccelerationReward(HumanoidBaseReward):
#     """Reward for minimizing base linear acceleration."""

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)
#         self.prev_base_vel = None

#     def __call__(self, states: list[EnvState], robot_name: str = None,dt: float = 0.02) -> torch.FloatTensor:
#         if self.prev_base_vel is None:
#             self.prev_base_vel = robot_velocity_tensor(states, self.robot_name)


#         base_vel = robot_velocity_tensor(states,self.robot_name)
#         base_acc = (base_vel - self.prev_base_vel)/dt


#         accel_sum = torch.sum(torch.abs(base_acc), dim=1)
#         reward = torch.exp(-0.01 * accel_sum)
#         self.prev_base_vel = base_vel.clone()
#         return reward


# class TorqueReward(HumanoidBaseReward):
#     """Reward for minimizing actuator torque usage.
#     ---->zerous in sapien3<------
#     """

#     def __init__(self, robot_name="g1_with_hands", torque_limit: float = 100.0):
#         super().__init__(robot_name)
#         self.torque_limit = torque_limit

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         tau = actuator_forces_tensor(states, self.robot_name)  # (B, N)
#         mean_torque_ratio = torch.mean(torch.abs(tau), dim=1)
#         reward = torch.exp(-0.02 * mean_torque_ratio)
#         return reward
# class ActionDifferenceReward(HumanoidBaseReward):
#     """Reward for smoothness in consecutive actions."""

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)
#         self.prev_action = None

#     def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
#         curr_action = states.robots[self.robot_name].joint_pos
#         #TODO
#         if self.prev_action is None:
#             self.prev_action = curr_action.clone()
#         diff = torch.abs(curr_action - self.prev_action).sum(dim=1)
#         reward = torch.exp(-0.02 * diff)
#         self.prev_action = curr_action.clone()
#         return reward
# class RollPitchOrientationReward(HumanoidBaseReward):
#     """
#     Reward penalizing torso roll & pitch deviations.
#     Implements Table I: e^{-30 * qd(q_rp, c_rp)} where c_rp = identity (roll=0,pitch=0).
#     """

#     def __init__(self, robot_name="g1_with_hands"):
#         super().__init__(robot_name)

#     def __call__(self, states: EnvState, robot_name: str = None) -> torch.FloatTensor:
#         # torso orientation quaternion (w,x,y,z)
#         q = robot_rotation_tensor(states, self.robot_name)  # (B,4)

#         # Extract roll & pitch only, zero yaw
#         # Convert to rpy
#         rpy = quat_to_euler_rpy(q)  # (B,3)

#         # Zero yaw → only roll, pitch matter
#         rpy_rp = torch.stack([rpy[:, 0], rpy[:, 1], torch.zeros_like(rpy[:, 2])], dim=1)

#         # Convert rp-only back to quaternion
#         roll = rpy_rp[:, 0]
#         pitch = rpy_rp[:, 1]

#         cy = torch.ones_like(roll)          # yaw = 0 => cos(yaw/2)=1
#         sy = torch.zeros_like(roll)

#         cr = torch.cos(roll * 0.5)
#         sr = torch.sin(roll * 0.5)
#         cp = torch.cos(pitch * 0.5)
#         sp = torch.sin(pitch * 0.5)

#         # quaternion from (roll,pitch,0)
#         # standard ZYX convention
#         qr = torch.stack([
#             cy * cp * cr + sy * sp * sr,
#             cy * cp * sr - sy * sp * cr,
#             cy * sp * cr + sy * cp * sr,
#             sy * cp * cr - cy * sp * sr,
#         ], dim=1)  # (B,4)

#         # reference orientation: identity quaternion (roll=0,pitch=0,yaw=0)
#         q_ref = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q)

#         # quaternion distance (qd)
#         # qd(q1, q2) = 1 - |dot(q1,q2)|
#         dot = torch.sum(qr * q_ref, dim=1)
#         qd = 1.0 - torch.abs(dot)

#         # reward
#         reward = torch.exp(-30.0 * qd)
#         return reward


class WalkingReward(HumanoidBaseReward):
    """Vectorized reference-free walking reward with more human-like gait shaping.

    Added human-like terms:
    - feet lateral separation target
    - wide stance penalty
    - knee flexion reward during swing
    - straight swing leg penalty
    - hip roll / hip yaw penalty
    - lateral velocity penalty
    - swing leg hip-pitch + knee coordination reward
    """

    def __init__(self, robot_name="g1_with_hands", dt: float = 0.02):
        super().__init__(robot_name)
        self.dt = dt

        self.contact_force_threshold = 5.0
        self.swing_height = 0.16
        self.stand_command_threshold = 0.05

        # Human-like gait targets.
        # These values will likely need tuning for your robot scale.
        self.target_feet_lateral_dist = 0.20   # [m], desired left-right foot spacing
        self.max_feet_lateral_dist = 0.30      # [m], above this counts as too wide
        self.min_feet_lateral_dist = 0.10      # [m], below this counts as too narrow/crossing

        self.target_swing_knee = 0.55          # [rad], about 31 degrees
        self.min_swing_knee = 0.25             # [rad], minimum useful knee flexion in swing

        self.prev_action = None
        self.prev_base_vel = None
        self.prev_joint_vel = None
        self.foot_air_time = None
        self.prev_contact = None

        self._body_cache_key = None
        self._joint_cache_key = None
        self._device = None

    def _resolve_robot_name(self, states: EnvState, robot_name: str | None) -> str:
        if robot_name is not None:
            return robot_name
        if self.robot_name in states.robots:
            return self.robot_name
        return next(iter(states.robots.keys()))

    @staticmethod
    def _find_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
        for name in candidates:
            if name in names:
                return names.index(name)
        return None

    def _cache_body_indices(self, robot):
        body_names = list(robot.body_names)
        cache_key = tuple(body_names)
        if self._body_cache_key == cache_key:
            return

        self.left_foot_idx = self._find_index(
            body_names,
            ("left_ankle_roll_link", "left_ankle_link", "left_foot_link")
        )
        self.right_foot_idx = self._find_index(
            body_names,
            ("right_ankle_roll_link", "right_ankle_link", "right_foot_link")
        )

        self.left_shoulder_idx = self._find_index(
            body_names,
            ("left_shoulder_roll_link", "left_shoulder_pitch_link")
        )
        self.right_shoulder_idx = self._find_index(
            body_names,
            ("right_shoulder_roll_link", "right_shoulder_pitch_link")
        )

        if self.left_foot_idx is None or self.right_foot_idx is None:
            raise ValueError("WalkingReward needs left and right foot body indices.")

        self._body_cache_key = cache_key

    def _cache_joint_indices(self, robot, device: torch.device):
        joint_names = list(robot.joint_names) if robot.joint_names is not None else []
        cache_key = (tuple(joint_names), device)
        if self._joint_cache_key == cache_key:
            return

        finger_words = ("hand", "thumb", "index", "middle")
        non_finger = [
            idx for idx, name in enumerate(joint_names)
            if not any(word in name for word in finger_words)
        ]
        if len(non_finger) == 0:
            non_finger = list(range(robot.joint_pos.shape[1]))

        self.non_finger_idx = torch.tensor(non_finger, dtype=torch.long, device=device)

        def joint_index(name: str) -> int | None:
            return joint_names.index(name) if name in joint_names else None

        # Leg joints used for human-like gait shaping.
        self.left_hip_pitch_idx = joint_index("left_hip_pitch_joint")
        self.right_hip_pitch_idx = joint_index("right_hip_pitch_joint")

        self.left_hip_roll_idx = joint_index("left_hip_roll_joint")
        self.right_hip_roll_idx = joint_index("right_hip_roll_joint")

        self.left_hip_yaw_idx = joint_index("left_hip_yaw_joint")
        self.right_hip_yaw_idx = joint_index("right_hip_yaw_joint")

        self.left_knee_idx = joint_index("left_knee_joint")
        self.right_knee_idx = joint_index("right_knee_joint")

        # Arm reference pose.
        arm_refs = {
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.6,
        }

        arm_indices = []
        arm_values = []
        for name, value in arm_refs.items():
            if name in joint_names:
                arm_indices.append(joint_names.index(name))
                arm_values.append(value)

        self.arm_idx = torch.tensor(arm_indices, dtype=torch.long, device=device)
        self.arm_ref = torch.tensor(arm_values, dtype=robot.joint_pos.dtype, device=device)

        self._joint_cache_key = cache_key
        self._device = device

    def _ensure_state_buffers(self, robot):
        num_envs = robot.joint_pos.shape[0]
        device = robot.joint_pos.device
        dtype = robot.joint_pos.dtype

        actions = robot.joint_pos_target if robot.joint_pos_target is not None else robot.joint_pos

        if (
            self.prev_action is None
            or self.prev_action.shape != actions.shape
            or self.prev_action.device != device
        ):
            self.prev_action = actions.detach().clone()

        if (
            self.prev_joint_vel is None
            or self.prev_joint_vel.shape != robot.joint_vel.shape
            or self.prev_joint_vel.device != device
        ):
            self.prev_joint_vel = robot.joint_vel.detach().clone()

        if (
            self.prev_base_vel is None
            or self.prev_base_vel.shape != robot.root_state[:, 7:10].shape
            or self.prev_base_vel.device != device
        ):
            self.prev_base_vel = robot.root_state[:, 7:10].detach().clone()

        if (
            self.foot_air_time is None
            or self.foot_air_time.shape != (num_envs, 2)
            or self.foot_air_time.device != device
        ):
            self.foot_air_time = torch.zeros(num_envs, 2, dtype=dtype, device=device)

        if (
            self.prev_contact is None
            or self.prev_contact.shape != (num_envs, 2)
            or self.prev_contact.device != device
        ):
            self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, states: EnvState):
        robot_name = self._resolve_robot_name(states, None)
        robot = states.robots[robot_name]

        self._cache_body_indices(robot)
        self._cache_joint_indices(robot, robot.joint_pos.device)
        self._ensure_state_buffers(robot)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=robot.joint_pos.device)

        actions = robot.joint_pos_target if robot.joint_pos_target is not None else robot.joint_pos
        contact = self._foot_floor_contact_from_forces(
            robot,
            robot.joint_pos.device,
            robot.joint_pos.dtype,
            robot.joint_pos.shape[0],
        )
        if contact is None:
            contact = torch.zeros(
                robot.joint_pos.shape[0],
                2,
                dtype=torch.bool,
                device=robot.joint_pos.device,
            )

        self.prev_action[env_ids] = actions[env_ids].detach()
        self.prev_joint_vel[env_ids] = robot.joint_vel[env_ids].detach()
        self.prev_base_vel[env_ids] = robot.root_state[env_ids, 7:10].detach()
        self.foot_air_time[env_ids] = 0.0
        self.prev_contact[env_ids] = contact[env_ids]

    def _command(
        self,
        states: EnvState,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        cmd = states.sensors.get("command0", None)
        if cmd is None:
            cmd = torch.zeros(num_envs, 3, dtype=dtype, device=device)
            cmd[:, 0] = 1.0
            return cmd
        return cmd.to(device=device, dtype=dtype)

    @staticmethod
    def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )
        return yaw

    @staticmethod
    def _local_xy_velocity(root_state: torch.Tensor) -> torch.Tensor:
        vel_world = root_state[:, 7:10]
        q = root_state[:, 3:7]

        yaw = WalkingReward._yaw_from_quat(q)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx = vel_world[:, 0] * cos_yaw + vel_world[:, 1] * sin_yaw
        vy = -vel_world[:, 0] * sin_yaw + vel_world[:, 1] * cos_yaw

        return torch.stack((vx, vy), dim=-1)

    @staticmethod
    def _roll_pitch_from_quat(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, x, y, z = q.unbind(-1)

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        return roll, pitch

    def _foot_floor_contact_from_forces(
        self,
        robot,
        device: torch.device,
        dtype: torch.dtype,
        num_envs: int,
    ) -> torch.Tensor | None:
        contact_data = getattr(robot, "contact", None)
        if contact_data is None:
            return None
        if not all(key in contact_data for key in ("link_a", "link_b", "valid_mask")):
            return None

        link_a = contact_data["link_a"].to(device=device)
        link_b = contact_data["link_b"].to(device=device)
        valid_mask = contact_data["valid_mask"].to(device=device).bool()

        forces = contact_data.get("force_b", contact_data.get("force", None))
        if forces is not None:
            force_mag = torch.linalg.norm(forces.to(device=device, dtype=dtype), dim=-1)
            valid_mask = valid_mask & (force_mag > self.contact_force_threshold)

        num_robot_bodies = len(robot.body_names)
        # Genesis contact indices can be either local body ids or global ids with
        # the ground at 0 and robot links shifted by one. Check both variants.
        divisor = max(num_robot_bodies + 1, 1)
        link_a_mod = torch.remainder(link_a, divisor)
        link_b_mod = torch.remainder(link_b, divisor)

        ground_a = link_a_mod == 0
        ground_b = link_b_mod == 0

        foot_contacts = []
        for foot_idx in (self.left_foot_idx, self.right_foot_idx):
            foot_a = (link_a_mod == foot_idx) | (link_a_mod == foot_idx + 1)
            foot_b = (link_b_mod == foot_idx) | (link_b_mod == foot_idx + 1)
            foot_floor_contact = valid_mask & ((foot_a & ground_b) | (foot_b & ground_a))
            foot_contacts.append(torch.any(foot_floor_contact, dim=1))

        contact = torch.stack(foot_contacts, dim=1)
        if contact.shape[0] != num_envs:
            return None
        return contact

    @staticmethod
    def _world_xy_to_base_xy(xy_world: torch.Tensor, root_state: torch.Tensor) -> torch.Tensor:
        """Convert world XY points to base-local XY frame.

        xy_world:   (B, 2)
        root_state: (B, root_state_dim)
        """
        root_xy = root_state[:, :2]
        rel = xy_world - root_xy

        q = root_state[:, 3:7]
        yaw = WalkingReward._yaw_from_quat(q)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        local_x = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        local_y = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw

        return torch.stack((local_x, local_y), dim=-1)

    def __call__(self, states: EnvState, robot_name: str = None) -> torch.FloatTensor:
        robot_name = self._resolve_robot_name(states, robot_name)
        robot = states.robots[robot_name]

        device = robot.joint_pos.device
        dtype = robot.joint_pos.dtype
        num_envs = robot.joint_pos.shape[0]

        self._cache_body_indices(robot)
        self._cache_joint_indices(robot, device)
        self._ensure_state_buffers(robot)

        root_state = robot.root_state
        body_state = robot.body_state

        cmd = self._command(states, num_envs, device, dtype)
        is_standing_cmd = torch.linalg.norm(cmd, dim=1) < self.stand_command_threshold

        # ---------------------------------------------------------------------
        # 1) Command tracking
        # ---------------------------------------------------------------------
        local_vel_xy = self._local_xy_velocity(root_state)

        target_xy = torch.where(
            is_standing_cmd.unsqueeze(1),
            torch.zeros_like(cmd[:, :2]),
            cmd[:, :2]
        )

        vel_error = torch.sum(torch.square(target_xy - local_vel_xy), dim=1)
        velocity_reward = torch.exp(-2.0 * vel_error)

        yaw_rate = root_state[:, 12]
        target_yaw_rate = torch.where(
            is_standing_cmd,
            torch.zeros_like(cmd[:, 2]),
            cmd[:, 2]
        )

        yaw_error = torch.square(target_yaw_rate - yaw_rate)
        yaw_reward = torch.exp(-1.5 * yaw_error)

        # ---------------------------------------------------------------------
        # 2) Upright and height
        # ---------------------------------------------------------------------
        upright = torch.clamp(torso_upright_tensor(states, robot_name), min=-1.0, max=1.0)
        upright_reward = torch.exp(-5.0 * torch.square(1.0 - upright))

        if self.left_shoulder_idx is not None and self.right_shoulder_idx is not None:
            base_height = 0.5 * (
                body_state[:, self.left_shoulder_idx, 2]
                + body_state[:, self.right_shoulder_idx, 2]
            )
            height_ref = self._stand_neck_height
        else:
            base_height = root_state[:, 2]
            height_ref = 0.85

        height_reward = torch.exp(-25.0 * torch.square(base_height - height_ref))

        # ---------------------------------------------------------------------
        # 3) Foot contact, clearance, airtime
        # ---------------------------------------------------------------------
        foot_state = body_state[:, [self.left_foot_idx, self.right_foot_idx], :]
        feet_z = foot_state[:, :, 2]
        feet_xy_vel = foot_state[:, :, 7:9]

        contact = self._foot_floor_contact_from_forces(robot, device, dtype, num_envs)
        if contact is None:
            contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        contact_count = contact.to(dtype).sum(dim=1)

        single_contact = contact_count == 1.0
        double_contact = contact_count == 2.0
        no_contact = contact_count == 0.0

        moving_contact_reward = single_contact.to(dtype) + 0.35 * double_contact.to(dtype) - 0.80 * no_contact.to(dtype)
        standing_contact_reward = double_contact.to(dtype) + 0.4 * single_contact.to(dtype)

        contact_reward = torch.where(
            is_standing_cmd,
            standing_contact_reward,
            moving_contact_reward
        )

        foot_roll, foot_pitch = self._roll_pitch_from_quat(foot_state[:, :, 3:7])
        foot_tilt_error = torch.square(foot_roll) + torch.square(foot_pitch)
        flat_foot_each = torch.exp(-25.0 * foot_tilt_error) * contact.to(dtype)
        flat_foot_reward = flat_foot_each.sum(dim=1) / torch.clamp(contact_count, min=1.0)
        flat_foot_reward = torch.where(
            contact_count > 0.0,
            flat_foot_reward,
            torch.zeros_like(flat_foot_reward)
        )
        flat_foot_penalty_each = (1.0 - torch.exp(-25.0 * foot_tilt_error)) * contact.to(dtype)
        flat_foot_penalty = flat_foot_penalty_each.sum(dim=1) / torch.clamp(contact_count, min=1.0)
        flat_foot_penalty = torch.where(
            contact_count > 0.0,
            flat_foot_penalty,
            torch.ones_like(flat_foot_penalty)
        )

        swing = (~contact).to(dtype)
        swing_count = swing.sum(dim=1)

        clearance_each = torch.exp(
            -80.0 * torch.square(feet_z - self.swing_height)
        ) * swing

        clearance_reward = clearance_each.sum(dim=1) / torch.clamp(swing_count, min=1.0)
        clearance_reward = torch.where(
            swing_count > 0.0,
            clearance_reward,
            torch.zeros_like(clearance_reward)
        )
        clearance_reward = torch.where(
            is_standing_cmd,
            torch.ones_like(clearance_reward),
            clearance_reward
        )

        touchdown = contact & (~self.prev_contact)

        next_air_time = self.foot_air_time + self.dt
        airtime_target = 0.35

        airtime_reward_each = torch.exp(
            -12.0 * torch.square(next_air_time - airtime_target)
        ) * touchdown.to(dtype)

        airtime_reward = airtime_reward_each.sum(dim=1)
        airtime_reward = torch.where(
            is_standing_cmd,
            torch.ones_like(airtime_reward),
            airtime_reward
        )

        self.foot_air_time = torch.where(
            contact,
            torch.zeros_like(self.foot_air_time),
            next_air_time
        )
        self.prev_contact = contact.detach().clone()

        # Foot slip while in contact.
        slip_penalty = torch.sum(
            torch.sum(torch.square(feet_xy_vel), dim=2) * contact.to(dtype),
            dim=1
        )

        # ---------------------------------------------------------------------
        # 4) Human-like foot spacing
        # ---------------------------------------------------------------------
        left_foot_xy_world = foot_state[:, 0, :2]
        right_foot_xy_world = foot_state[:, 1, :2]

        left_foot_xy_local = self._world_xy_to_base_xy(left_foot_xy_world, root_state)
        right_foot_xy_local = self._world_xy_to_base_xy(right_foot_xy_world, root_state)

        # Local y is lateral direction. This avoids penalizing normal step length.
        feet_lateral_dist = torch.abs(left_foot_xy_local[:, 1] - right_foot_xy_local[:, 1])

        step_length = torch.abs(left_foot_xy_local[:, 0] - right_foot_xy_local[:, 0])
        target_step_length = torch.clamp(
            0.35 * torch.abs(cmd[:, 0]),
            min=0.10,
            max=0.35,
        )
        step_length_reward = torch.exp(
            -10.0 * torch.square(step_length - target_step_length)
        )
        step_length_reward = torch.where(
            is_standing_cmd,
            torch.ones_like(step_length_reward),
            step_length_reward,
        )

        feet_separation_penalty = torch.square(
            feet_lateral_dist - self.target_feet_lateral_dist
        )

        wide_stance_penalty = torch.square(
            torch.relu(feet_lateral_dist - self.max_feet_lateral_dist)
        )

        narrow_stance_penalty = torch.square(
            torch.relu(self.min_feet_lateral_dist - feet_lateral_dist)
        )

        # During standing, allow stronger target spacing.
        # During walking, keep it active but slightly softer through reward weights below.
        feet_spacing_penalty = feet_separation_penalty + 2.0 * wide_stance_penalty + narrow_stance_penalty

        # Penalize sideways body motion. This discourages waddling.
        lateral_vel_penalty = torch.square(local_vel_xy[:, 1])

        # ---------------------------------------------------------------------
        # 5) Human-like leg joint usage
        # ---------------------------------------------------------------------
        # Knee flexion reward during swing.
        if self.left_knee_idx is not None and self.right_knee_idx is not None:
            knee_q = robot.joint_pos[:, [self.left_knee_idx, self.right_knee_idx]]

            # Use abs because some robots define knee flexion as positive,
            # others as negative.
            knee_flex = torch.abs(knee_q)

            knee_flex_reward_each = torch.exp(
                -8.0 * torch.square(knee_flex - self.target_swing_knee)
            ) * swing

            knee_flex_reward = knee_flex_reward_each.sum(dim=1) / torch.clamp(swing_count, min=1.0)
            knee_flex_reward = torch.where(
                swing_count > 0.0,
                knee_flex_reward,
                torch.zeros_like(knee_flex_reward)
            )
            knee_flex_reward = torch.where(
                is_standing_cmd,
                torch.ones_like(knee_flex_reward),
                knee_flex_reward
            )

            straight_swing_leg_penalty_each = torch.square(
                torch.relu(self.min_swing_knee - knee_flex)
            ) * swing

            straight_swing_leg_penalty = straight_swing_leg_penalty_each.sum(dim=1) / torch.clamp(
                swing_count,
                min=1.0
            )
            straight_swing_leg_penalty = torch.where(
                swing_count > 0.0,
                straight_swing_leg_penalty,
                torch.zeros_like(straight_swing_leg_penalty)
            )

        else:
            knee_flex = None
            knee_flex_reward = torch.zeros(num_envs, dtype=dtype, device=device)
            straight_swing_leg_penalty = torch.zeros(num_envs, dtype=dtype, device=device)

        # Penalize using hip roll and hip yaw as the main locomotion strategy.
        hip_side_indices = [
            idx for idx in [
                self.left_hip_roll_idx,
                self.right_hip_roll_idx,
                self.left_hip_yaw_idx,
                self.right_hip_yaw_idx,
            ]
            if idx is not None
        ]

        if len(hip_side_indices) > 0:
            hip_side_q = robot.joint_pos[:, hip_side_indices]
            hip_side_penalty = torch.mean(torch.square(hip_side_q), dim=1)
        else:
            hip_side_penalty = torch.zeros(num_envs, dtype=dtype, device=device)

        # Encourage swing motion to come from hip pitch + knee flexion,
        # not from whole-leg roll/yaw rotation.
        if (
            self.left_hip_pitch_idx is not None
            and self.right_hip_pitch_idx is not None
            and self.left_knee_idx is not None
            and self.right_knee_idx is not None
        ):
            hip_pitch = robot.joint_pos[:, [self.left_hip_pitch_idx, self.right_hip_pitch_idx]]
            knee_q_for_motion = robot.joint_pos[:, [self.left_knee_idx, self.right_knee_idx]]

            swing_leg_motion = torch.abs(hip_pitch) + 0.7 * torch.abs(knee_q_for_motion)

            swing_leg_motion_reward_each = torch.exp(
                -4.0 * torch.square(swing_leg_motion - 0.8)
            ) * swing

            swing_leg_motion_reward = swing_leg_motion_reward_each.sum(dim=1) / torch.clamp(
                swing_count,
                min=1.0
            )
            swing_leg_motion_reward = torch.where(
                swing_count > 0.0,
                swing_leg_motion_reward,
                torch.zeros_like(swing_leg_motion_reward)
            )
            swing_leg_motion_reward = torch.where(
                is_standing_cmd,
                torch.ones_like(swing_leg_motion_reward),
                swing_leg_motion_reward
            )
        else:
            swing_leg_motion_reward = torch.zeros(num_envs, dtype=dtype, device=device)

        # ---------------------------------------------------------------------
        # 6) Smoothness and energy regularization
        # ---------------------------------------------------------------------
        actions = robot.joint_pos_target if robot.joint_pos_target is not None else robot.joint_pos

        action_delta = actions[:, self.non_finger_idx] - self.prev_action[:, self.non_finger_idx]
        action_rate_penalty = torch.mean(torch.square(action_delta), dim=1)
        self.prev_action = actions.detach().clone()

        joint_vel = robot.joint_vel[:, self.non_finger_idx]
        joint_vel_penalty = torch.mean(torch.square(joint_vel / 8.0), dim=1)

        joint_acc = (
            robot.joint_vel[:, self.non_finger_idx]
            - self.prev_joint_vel[:, self.non_finger_idx]
        ) / self.dt

        joint_acc_penalty = torch.mean(torch.square(joint_acc / 120.0), dim=1)
        self.prev_joint_vel = robot.joint_vel.detach().clone()

        base_acc = (root_state[:, 7:10] - self.prev_base_vel) / self.dt
        base_acc_penalty = torch.mean(torch.square(base_acc / 30.0), dim=1)
        self.prev_base_vel = root_state[:, 7:10].detach().clone()

        if robot.joint_effort_target is None:
            torque_penalty = torch.zeros(num_envs, dtype=dtype, device=device)
        else:
            torque = robot.joint_effort_target[:, self.non_finger_idx]
            torque_penalty = torch.mean(torch.square(torque / 120.0), dim=1)

        if self.arm_idx.numel() > 0:
            arm_error = robot.joint_pos[:, self.arm_idx] - self.arm_ref.unsqueeze(0)
            arm_pose_penalty = torch.mean(torch.square(arm_error), dim=1)
        else:
            arm_pose_penalty = torch.zeros(num_envs, dtype=dtype, device=device)

        fall_penalty = ((root_state[:, 2] < 0.45) | (upright < 0.35)).to(dtype)

        # ---------------------------------------------------------------------
        # 7) Final reward
        # ---------------------------------------------------------------------
        reward = (
            # Original locomotion objective.
            2.20 * velocity_reward
            + 0.35 * yaw_reward
            + 0.45 * upright_reward
            + 0.25 * height_reward
            + 0.45 * contact_reward
            + 0.30 * flat_foot_reward
            + 0.12 * clearance_reward
            + 0.08 * airtime_reward

            # Human-like gait shaping.
            + 0.20 * knee_flex_reward
            + 0.10 * swing_leg_motion_reward
            + 0.15 * step_length_reward

            - 0.18 * feet_spacing_penalty
            - 0.10 * hip_side_penalty
            - 0.20 * straight_swing_leg_penalty
            - 0.08 * lateral_vel_penalty
            - 0.25 * flat_foot_penalty

            # Original regularization.
            - 0.08 * slip_penalty
            - 0.035 * action_rate_penalty
            - 0.020 * joint_vel_penalty
            - 0.015 * joint_acc_penalty
            - 0.020 * base_acc_penalty
            - 0.005 * torque_penalty
            - 0.030 * arm_pose_penalty
            - 1.00 * fall_penalty
        )

        return torch.clamp(reward, min=-2.0, max=5.0)


@configclass
class WalkNewCfg(HumanoidTaskCfg):
    """Walking task for humanoid robots."""
    commmand = torch.tensor([1.0,0.0,0.0])
    name = "walk_new"
    needs_contact_state = True

    episode_length = 1000
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/h1_v2.pkl"
    # traj_filepath = "roboverse_data/trajs/humanoidbench/walk/v2/initial_state_v2.json"
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"

    checker = _WalkChecker()
    reward_functions = [WalkingReward()]
    reward_weights = [1.0]

    def extra_spec(self):

        """This task does not require any extra observations."""
        return {}
