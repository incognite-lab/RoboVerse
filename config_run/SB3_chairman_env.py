
from __future__ import annotations

from multiprocessing.util import debug
from typing import Literal

import torch
from loguru import logger as log
import numpy as np

import cv2


from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from metasim.utils.chair_navigation import (
    CHAIR_FINAL_DISTANCE,
    CHAIR_STAGING_DISTANCE,
    chair_back_direction_xy,
    world_vector_to_body_xy,
)
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

try:
    from .policy import G1MotionPolicy
except ImportError:
    # ``main.py`` is commonly executed directly from the config_run directory.
    from policy import G1MotionPolicy

VIZUALIZATION = False
#from roboverse_learn.rl.rsl_rl.rsl_rl import env


def _quaternion_error_vector(
    current_wxyz: torch.Tensor,
    target_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Shortest target orientation error expressed in the current frame."""
    current = torch.nn.functional.normalize(current_wxyz, dim=-1)
    target = torch.nn.functional.normalize(target_wxyz, dim=-1)
    cw, cx, cy, cz = current.unbind(dim=-1)
    tw, tx, ty, tz = target.unbind(dim=-1)

    # inverse(current) * target
    error_w = cw * tw + cx * tx + cy * ty + cz * tz
    error_x = cw * tx - cx * tw - cy * tz + cz * ty
    error_y = cw * ty + cx * tz - cy * tw - cz * tx
    error_z = cw * tz - cx * ty + cy * tx - cz * tw
    error_xyz = torch.stack((error_x, error_y, error_z), dim=-1)
    canonical_sign = torch.where(error_w < 0.0, -1.0, 1.0).unsqueeze(-1)
    return error_xyz * canonical_sign

class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    LOCOMOTION_COMMAND_NAMES = ("walk_vx", "walk_vy", "walk_yaw_rate")
    MAIN_ROBOT_LINK_NAMES = (
        'left_shoulder_pitch_link',
        'left_shoulder_roll_link',
        'left_shoulder_yaw_link',
        'left_elbow_link',
        'left_wrist_roll_link',
        'left_wrist_pitch_link',
        'left_wrist_yaw_link',
        'left_hand_palm_link',
        'left_hand_thumb_0_link',
        'left_hand_thumb_1_link',
        'left_hand_thumb_2_link',
        'left_hand_middle_0_link',
        'left_hand_middle_1_link',
        'left_hand_index_0_link',
        'left_hand_index_1_link',
        'right_shoulder_pitch_link',
        'right_shoulder_roll_link',
        'right_shoulder_yaw_link',
        'right_elbow_link',
        'right_wrist_roll_link',
        'right_wrist_pitch_link',
        'right_wrist_yaw_link',
        'endeffector',
        'left_endeffector',
        'right_hand_thumb_0_link',
        'right_hand_thumb_1_link',
        'right_hand_thumb_2_link',
        'right_hand_middle_0_link',
        'right_hand_middle_1_link',
        'right_hand_index_0_link',
        'right_hand_index_1_link',
        'torso_link',
    )

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        robot_cfg = env.scenario.robots[0]
        joint_limits = robot_cfg.joint_limits
        self.robot_name = robot_cfg.name
        self.robot_joint_names = tuple(joint_limits.keys())
        self.leg_joint_names = G1MotionPolicy.JOINT_NAMES
        missing_leg_joints = [name for name in self.leg_joint_names if name not in joint_limits]
        if missing_leg_joints:
            raise ValueError(
                "SB3_chairman_env requires a full G1 robot with the 12 walking-policy leg joints. "
                f"Missing joints: {', '.join(missing_leg_joints)}"
            )
        if robot_cfg.fix_base_link:
            raise ValueError(
                "Walking cannot move a fixed pelvis. Set fix_base_link: false in the Chairman YAML config."
            )

        leg_joint_set = set(self.leg_joint_names)
        self.upper_body_joint_names = tuple(
            name for name in self.robot_joint_names if name not in leg_joint_set
        )
        self.action_names = self.upper_body_joint_names + self.LOCOMOTION_COMMAND_NAMES
        self._upper_default_targets = np.asarray(
            [robot_cfg.default_joint_positions[name] for name in self.upper_body_joint_names],
            dtype=np.float32,
        )

        upper_low = [joint_limits[name][0] for name in self.upper_body_joint_names]
        upper_high = [joint_limits[name][1] for name in self.upper_body_joint_names]
        self.action_space = spaces.Box(
            low=np.asarray(upper_low + (-G1MotionPolicy.MAX_COMMAND).tolist(), dtype=np.float32),
            high=np.asarray(upper_high + G1MotionPolicy.MAX_COMMAND.tolist(), dtype=np.float32),
            shape=(len(self.action_names),),
            dtype=np.float32,
        )

        num_joints = len(joint_limits)

        states = env.env.handler.get_states()
        robot_state = states.robots[self.robot_name]
        self.main_robot_link_names = list(self.MAIN_ROBOT_LINK_NAMES)
        self.indexes = [robot_state.body_names.index(link) for link in self.main_robot_link_names]

        state_joint_names = (
            robot_state.joint_names.tolist()
            if hasattr(robot_state.joint_names, "tolist")
            else list(robot_state.joint_names)
        )
        self.sim_joint_names = tuple(str(name) for name in state_joint_names)
        missing_sim_joints = [name for name in self.robot_joint_names if name not in self.sim_joint_names]
        if missing_sim_joints:
            raise ValueError(
                "Configured G1 joints are missing from the simulator state: "
                f"{', '.join(missing_sim_joints)}"
            )
        self._leg_state_indices = np.asarray(
            [self.sim_joint_names.index(name) for name in self.leg_joint_names], dtype=np.int64
        )
        self._upper_state_indices = np.asarray(
            [self.sim_joint_names.index(name) for name in self.upper_body_joint_names], dtype=np.int64
        )
        self._leg_state_indices_torch = torch.as_tensor(
            self._leg_state_indices, dtype=torch.long, device=env.env.handler.device
        )
        self._pelvis_index = robot_state.body_names.index("pelvis")

        physics_dt = getattr(env.scenario.sim_params, "dt", None)
        if physics_dt is None:
            physics_dt = 0.01 if env.scenario.sim == "genesis" else G1MotionPolicy.CONTROL_DT
        self._physics_dt = float(physics_dt)
        self._env_control_dt = self._physics_dt * int(env.scenario.decimation)
        self._motion_decimation = max(
            1, int(round(G1MotionPolicy.CONTROL_DT / self._env_control_dt))
        )
        self._effective_motion_dt = self._env_control_dt * self._motion_decimation
        if not np.isclose(self._effective_motion_dt, G1MotionPolicy.CONTROL_DT, rtol=0.0, atol=1e-6):
            log.warning(
                "The simulator control period ({:.6f}s) cannot reproduce the motion policy's "
                "20ms period exactly; using {:.6f}s.",
                self._env_control_dt,
                self._effective_motion_dt,
            )
        self.motion_policy = G1MotionPolicy(
            device=env.env.handler.device,
            control_dt=self._effective_motion_dt,
        )
        self._motion_step = 0
        self._cached_leg_targets = np.tile(
            G1MotionPolicy.DEFAULT_ANGLES[None, :], (env.num_envs, 1)
        ).astype(np.float32)
        self.last_requested_locomotion_command = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.last_locomotion_command = np.zeros((env.num_envs, 3), dtype=np.float32)
        self._printed_motion_step = False
        self._printed_one_second_motion_diagnostic = False
        self._printed_motion_failure_diagnostic = False
        applied_properties = getattr(env.env.handler, "applied_actuator_properties", {})
        self._leg_dof_indices = [
            int(applied_properties[name]["dof_index"])
            for name in self.leg_joint_names
            if name in applied_properties
        ]
        self._leg_torque_limits = np.asarray(
            [
                max(
                    abs(float(applied_properties[name]["force_lower"])),
                    abs(float(applied_properties[name]["force_upper"])),
                )
                for name in self.leg_joint_names
                if name in applied_properties
            ],
            dtype=np.float32,
        )
        self._motion_diagnostic_steps = np.zeros(env.num_envs, dtype=np.int64)
        self._max_leg_tracking_error = np.zeros(
            (env.num_envs, len(self.leg_joint_names)), dtype=np.float32
        )
        self._max_leg_torque_utilization = np.zeros_like(self._max_leg_tracking_error)
        log.info(
            "Chairman action layout: {} upper-body joints + 3 walking commands = {} actions; "
            "motion.pt runs every {} environment step(s).",
            len(self.upper_body_joint_names),
            len(self.action_names),
            self._motion_decimation,
        )

        # Full Chairman checker uses stages 0..6.  Clipping them into four bins
        # made stages 3..6 observationally indistinguishable.
        self.num_stages = 7

        num_robot_bodies = len(self.main_robot_link_names)

        # extra obs:
        # robot_body_states        = num_robot_bodies * 7
        # pelvis vel (lin+ang)     = 6
        # chair world pos          = 3
        # chair world vel          = 3
        # robot->chair world vec   = 3
        # robot->chair body vec    = 2
        # dist to chair            = 1
        # dist to final target     = 1
        # staging target body vec  = 2
        # final target body vec    = 2
        # chair back dir in body   = 2
        # hand target body vectors = 6
        # hand orientation errors  = 6
        # hand body velocities     = 6
        # fingertip chair forces   = 6
        # previous walk command    = 3
        # stage one hot            = self.num_stages
        # arm errors (L,R)         = 2
        extra_obs_dim = (
            num_robot_bodies * 7 +
            6 +
            3 +
            3 +
            3 +
            2 +
            1 +
            1 +
            2 +
            2 +
            2 +
            6 +
            6 +
            6 +
            6 +
            3 +
            self.num_stages +
            2
        )
        self.left_endffector = None
        self.right_endffector = None
        obs_shape = num_joints + extra_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float32,
        )
        self.env = env
        self.render_mode = None
        self.timesteps = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.env.handler.device,
        )
        self.action = None #TODO debug holder
        self.finger_current_positions = {} #TODO debug holder
        self._contact_base_to_tip = None
        self._contact_chair_ids = None
        self._contact_num_bodies = None
        super().__init__(env.num_envs, self.observation_space, self.action_space)
        self._log_motion_configuration_once(robot_cfg)
        if VIZUALIZATION:
            self._init_joint_viz()

    def _log_motion_configuration_once(self, robot_cfg) -> None:
        """Print the effective locomotion timing and leg controller values once."""
        decimation = int(self.env.scenario.decimation)
        log.info(
            "G1 motion timing (printed once): URDF={}, fixed_base={}, physics dt={:.6f}s "
            "({:.1f} Hz), env decimation={}, env dt={:.6f}s ({:.1f} Hz), "
            "motion stride={}, motion dt={:.6f}s ({:.1f} Hz)",
            robot_cfg.urdf_path,
            robot_cfg.fix_base_link,
            self._physics_dt,
            1.0 / self._physics_dt,
            decimation,
            self._env_control_dt,
            1.0 / self._env_control_dt,
            self._motion_decimation,
            self._effective_motion_dt,
            1.0 / self._effective_motion_dt,
        )

        applied = getattr(self.env.env.handler, "applied_actuator_properties", {})
        rows = [
            "joint | sim index | policy default | reset default | limits | "
            "configured Kp/Kd | Genesis Kp/Kd | torque range"
        ]
        for policy_index, name in enumerate(self.leg_joint_names):
            actuator = robot_cfg.actuators[name]
            actual = applied.get(name)
            if actual is None:
                actual_gains = "not available"
                actual_torque = "not available"
            else:
                actual_gains = f"{actual['kp']:.3f}/{actual['kd']:.3f}"
                actual_torque = (
                    f"[{actual['force_lower']:.3f}, {actual['force_upper']:.3f}]"
                )
            lower, upper = robot_cfg.joint_limits[name]
            rows.append(
                f"{name} | {self._leg_state_indices[policy_index]} | "
                f"{G1MotionPolicy.DEFAULT_ANGLES[policy_index]:.3f} | "
                f"{robot_cfg.default_joint_positions[name]:.3f} | "
                f"[{lower:.3f}, {upper:.3f}] | "
                f"{actuator.stiffness:.3f}/{actuator.damping:.3f} | "
                f"{actual_gains} | {actual_torque}"
            )
        log.info("G1 leg controller configuration (printed once):\n{}", "\n".join(rows))

    def _fingertip_chair_forces(self, states, robot) -> torch.Tensor:
        """Return normalized chair-contact force for three fingertips per hand."""
        device = robot.joint_pos.device
        result = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=device)
        contact = getattr(robot, "contact", None)
        if contact is None or contact["link_a"].shape[1] == 0:
            return result

        if self._contact_base_to_tip is None:
            extras = getattr(states, "extras", {})
            global_map = extras.get("global_link_map", {})
            num_bodies = int(extras.get("num_bodies_per_env", 1000))
            tip_names = {
                "left_hand_thumb_2": 0,
                "left_hand_index_1": 1,
                "left_hand_middle_1": 2,
                "right_hand_thumb_2": 3,
                "right_hand_index_1": 4,
                "right_hand_middle_1": 5,
            }
            base_to_tip = torch.full(
                (num_bodies,), -1, dtype=torch.long, device=device
            )
            chair_ids = []
            for base_index, (object_name, link_name) in global_map.items():
                if object_name == self.robot_name:
                    for tip_name, tip_id in tip_names.items():
                        if tip_name in link_name:
                            base_to_tip[int(base_index)] = tip_id
                elif object_name == "chair":
                    chair_ids.append(int(base_index))

            if not chair_ids:
                return result
            self._contact_base_to_tip = base_to_tip
            self._contact_chair_ids = torch.tensor(
                chair_ids, dtype=torch.long, device=device
            )
            self._contact_num_bodies = num_bodies

        link_a = contact["link_a"]
        link_b = contact["link_b"]
        valid = contact["valid_mask"]
        forces = contact.get("force_b", contact.get("force", None))
        if forces is None:
            return result

        base_a = link_a % self._contact_num_bodies
        base_b = link_b % self._contact_num_bodies
        a_is_chair = torch.isin(base_a, self._contact_chair_ids)
        b_is_chair = torch.isin(base_b, self._contact_chair_ids)
        tip_a = self._contact_base_to_tip[base_a]
        tip_b = self._contact_base_to_tip[base_b]
        contact_tip = torch.where(
            b_is_chair,
            tip_a,
            torch.where(a_is_chair, tip_b, torch.full_like(tip_a, -1)),
        )
        force_magnitude = torch.norm(forces, dim=-1)
        valid_tip_contact = valid & (contact_tip >= 0)
        for tip_id in range(6):
            tip_force = torch.where(
                valid_tip_contact & (contact_tip == tip_id),
                force_magnitude,
                torch.zeros_like(force_magnitude),
            )
            result[:, tip_id] = torch.max(tip_force, dim=1).values

        # The stage checker requires 2 N.  Normalizing at the same value makes
        # one mean full-scale observation correspond to checker success.
        return torch.clamp(result / 2.0, min=0.0, max=1.0)

    def add_extra_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extend obs with extra task-relevant data.

        Přidané informace:
        - vybrané body states robota
        - rychlost pelvisu
        - world pozice a rychlost židle
        - vektor robot -> židle ve world frame
        - vektor robot -> židle v body frame robota
        - vzdálenost k židli
        - vzdálenost k finálnímu cíli
        - vektory k mezibodu a finálnímu cíli v body frame
        - směr za opěradlo židle v body frame
        - 3D chyby pozice a orientace obou rukou
        - rychlosti obou rukou a síly kontaktů šesti konečků prstů
        - one-hot aktuální stage
        - chyba ramen vůči cílové póze podle stage
        """
        handler = self.env.env.handler
        states = handler.get_states()
        robot_name = self.env.scenario.robots[0].name

        robot = states.robots[robot_name]
        chair = states.objects["chair"]

        # --------------------------------------------------
        # 1) vybrané body states robota (pos + quat)
        # --------------------------------------------------
        robot_body_states = (
            robot.body_state[:, self.indexes, :7]
            .reshape(self.num_envs, -1)
            .cpu()
            .numpy()
        )

        # --------------------------------------------------
        # 2) pelvis state
        # --------------------------------------------------
        pelvis_idx = robot.body_names.index("pelvis")
        pelvis_pos = robot.body_state[:, pelvis_idx, :3]
        pelvis_quat = robot.body_state[:, pelvis_idx, 3:7]   # w,x,y,z
        velocity_pelvis = robot.body_state[:, pelvis_idx, 7:13].cpu().numpy()  # lin(3)+ang(3)

        # --------------------------------------------------
        # 3) chair base state
        # --------------------------------------------------
        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_quat = chair.body_state[:, chair_idx, 3:7]  # w,x,y,z
        chair_vel = chair.body_state[:, chair_idx, 7:10]

        chair_pos_np = chair_pos.cpu().numpy()
        chair_vel_np = chair_vel.cpu().numpy()

        # --------------------------------------------------
        # 4) robot -> chair vector ve world frame
        # --------------------------------------------------
        vec_world = chair_pos - pelvis_pos
        vec_world_np = vec_world.cpu().numpy()

        dist_to_chair = torch.norm(vec_world[:, :2], dim=-1, keepdim=True)
        dist_to_chair_np = dist_to_chair.cpu().numpy()

        chair_back_world = chair_back_direction_xy(chair_quat)
        staging_pos_xy = chair_pos[:, :2] + CHAIR_STAGING_DISTANCE * chair_back_world
        final_pos_xy = chair_pos[:, :2] + CHAIR_FINAL_DISTANCE * chair_back_world
        staging_vec_world = staging_pos_xy - pelvis_pos[:, :2]
        final_vec_world = final_pos_xy - pelvis_pos[:, :2]
        dist_to_final = torch.norm(final_vec_world, dim=-1, keepdim=True)
        dist_to_final_np = dist_to_final.cpu().numpy()

        # --------------------------------------------------
        # 5) robot -> chair vector v body frame robota
        #    používáme yaw-aligned body frame pelvisu
        # --------------------------------------------------
        chair_rel_body = world_vector_to_body_xy(vec_world[:, :2], pelvis_quat)
        staging_rel_body = world_vector_to_body_xy(staging_vec_world, pelvis_quat)
        final_rel_body = world_vector_to_body_xy(final_vec_world, pelvis_quat)
        chair_back_body = world_vector_to_body_xy(chair_back_world, pelvis_quat)
        chair_rel_body_np = chair_rel_body.cpu().numpy()
        staging_rel_body_np = staging_rel_body.cpu().numpy()
        final_rel_body_np = final_rel_body.cpu().numpy()
        chair_back_body_np = chair_back_body.cpu().numpy()

        # --------------------------------------------------
        # 6) stage one-hot
        # --------------------------------------------------
        current_stage_tensor = handler.task.reward_functions[0].actual_stage
        if current_stage_tensor is None:
            current_stages = np.zeros(self.num_envs, dtype=np.int32)
        else:
            current_stages = current_stage_tensor.cpu().numpy().astype(np.int32)

        stage_one_hot = np.zeros((self.num_envs, self.num_stages), dtype=np.float32)
        safe_stages = np.clip(current_stages, 0, self.num_stages - 1)
        stage_one_hot[np.arange(self.num_envs), safe_stages] = 1.0

        # --------------------------------------------------
        # 7) arm posture errors vůči cíli podle stage
        # --------------------------------------------------
        if self.left_endffector is None:
            self.left_endffector = robot.body_names.index("left_endeffector")
            self.right_endffector = robot.body_names.index("endeffector")

        pos_left = robot.body_state[:, self.left_endffector, :3]
        pos_right = robot.body_state[:, self.right_endffector, :3]
        quat_left = robot.body_state[:, self.left_endffector, 3:7]
        quat_right = robot.body_state[:, self.right_endffector, 3:7]
        vel_left = robot.body_state[:, self.left_endffector, 7:10]
        vel_right = robot.body_state[:, self.right_endffector, 7:10]

        target_left_idx = chair.body_names.index("target_hand_left")
        target_right_idx = chair.body_names.index("target_hand_right")

        target_left_pos = chair.body_state[:, target_left_idx, :3]
        target_right_pos = chair.body_state[:, target_right_idx, :3]
        target_left_quat = chair.body_state[:, target_left_idx, 3:7]
        target_right_quat = chair.body_state[:, target_right_idx, 3:7]

        arm_err_left = torch.norm(pos_left - target_left_pos, dim=-1, keepdim=True)
        arm_err_right = torch.norm(pos_right - target_right_pos, dim=-1, keepdim=True)
        arm_err_np = torch.cat([arm_err_left, arm_err_right], dim=-1).cpu().numpy()

        left_target_delta = target_left_pos - pos_left
        right_target_delta = target_right_pos - pos_right
        left_target_body = torch.cat(
            (
                world_vector_to_body_xy(left_target_delta[:, :2], pelvis_quat),
                left_target_delta[:, 2:3],
            ),
            dim=-1,
        )
        right_target_body = torch.cat(
            (
                world_vector_to_body_xy(right_target_delta[:, :2], pelvis_quat),
                right_target_delta[:, 2:3],
            ),
            dim=-1,
        )
        hand_target_body_np = torch.cat(
            (left_target_body, right_target_body), dim=-1
        ).cpu().numpy()

        hand_orientation_error_np = torch.cat(
            (
                _quaternion_error_vector(quat_left, target_left_quat),
                _quaternion_error_vector(quat_right, target_right_quat),
            ),
            dim=-1,
        ).cpu().numpy()

        left_vel_body = torch.cat(
            (
                world_vector_to_body_xy(vel_left[:, :2], pelvis_quat),
                vel_left[:, 2:3],
            ),
            dim=-1,
        )
        right_vel_body = torch.cat(
            (
                world_vector_to_body_xy(vel_right[:, :2], pelvis_quat),
                vel_right[:, 2:3],
            ),
            dim=-1,
        )
        hand_velocity_body_np = torch.cat(
            (left_vel_body, right_vel_body), dim=-1
        ).cpu().numpy()
        fingertip_force_np = self._fingertip_chair_forces(states, robot).cpu().numpy()
        locomotion_command_np = self.last_locomotion_command.astype(
            np.float32, copy=False
        )

        # --------------------------------------------------
        # 8) složení extra observace
        # --------------------------------------------------
        extra_obs = np.concatenate([
            robot_body_states,     # 3 body * (pos+quat)
            velocity_pelvis,       # 6
            chair_pos_np,          # 3
            chair_vel_np,          # 3
            vec_world_np,          # 3
            chair_rel_body_np,     # 2
            dist_to_chair_np,      # 1
            dist_to_final_np,      # 1
            staging_rel_body_np,   # 2
            final_rel_body_np,     # 2
            chair_back_body_np,    # 2
            hand_target_body_np,   # 6
            hand_orientation_error_np,  # 6
            hand_velocity_body_np, # 6
            fingertip_force_np,    # 6
            locomotion_command_np, # 3: previous/current command for Markov state
            stage_one_hot,         # 7
            arm_err_np,            # 2
        ], axis=1)

        obs = obs.reshape(self.num_envs, -1)
        return np.concatenate([obs, extra_obs], axis=1).astype(np.float32)

    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        states = self.env.env.handler.get_states()
        gyrodata = states.sensors["gyro0"].cpu().numpy()  # shape (num_envs, 3)
        gyrodata = gyrodata.reshape(self.num_envs, 3)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata], axis=1).astype(np.float32)


    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        self._reset_motion_state()
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)
        self.timesteps.zero_()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Keep the complete Chairman action.  In particular, the last three
        # values are the physical [vx, vy, yaw_rate] command consumed by
        # motion.pt.  Replacing the action with zeros here made locomotion
        # independent of the PPO policy, so a walking reward could not teach
        # the agent to approach or stop.
        robot_targets = self._compose_robot_targets(actions)

        # --- RYCHLÁ CESTA PRO GENESIS ---
        if self.env.scenario.sim == 'genesis':
            # Akce si uložíme rovnou jako numpy array, žádné slovníky!
            self.raw_actions = robot_targets
            self.action_dicts = None

        # --- POMALÁ CESTA PRO OSTATNÍ SIMULÁTORY ---
        else:
            self.raw_actions = None
            self.action_dicts = [
                {
                    self.robot_name: {
                        "dof_pos_target": dict(zip(self.sim_joint_names, target))
                    }
                }
                for target in robot_targets
            ]

    def _compose_robot_targets(self, actions: np.ndarray) -> np.ndarray:
        """Combine SB3 upper-body targets with leg targets produced by motion.pt."""
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, len(self.action_names))
        if actions.shape != expected_shape:
            raise ValueError(f"Expected Chairman actions with shape {expected_shape}, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Chairman policy produced NaN or infinite actions")

        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        upper_targets = actions[:, :len(self.upper_body_joint_names)]
        requested_command = actions[:, -len(self.LOCOMOTION_COMMAND_NAMES):].copy()
        command = np.clip(
            requested_command,
            -G1MotionPolicy.MAX_COMMAND,
            G1MotionPolicy.MAX_COMMAND,
        )
        previous_command = self.last_locomotion_command.copy()
        self.last_requested_locomotion_command = requested_command.copy()
        self.last_locomotion_command = command.copy()

        # Command-rate rewards live with the task rewards, but walking commands
        # are high-level actions and are not otherwise present in EnvState.
        for reward_fn in self.env.scenario.task.reward_functions:
            set_context = getattr(reward_fn, "set_control_context", None)
            if set_context is not None:
                set_context(
                    command,
                    previous_command,
                    device=self.env.env.handler.device,
                )

        states = self.env.env.handler.get_states()
        robot_state = states.robots[self.robot_name]

        if self._motion_step % self._motion_decimation == 0:
            joint_positions = robot_state.joint_pos.index_select(1, self._leg_state_indices_torch)
            joint_velocities = robot_state.joint_vel.index_select(1, self._leg_state_indices_torch)
            pelvis_state = robot_state.body_state[:, self._pelvis_index, :]

            self._cached_leg_targets = self.motion_policy.predict_joint_positions(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                angular_velocity=pelvis_state[:, 10:13],
                angular_velocity_frame="world",
                base_quaternion_wxyz=pelvis_state[:, 3:7],
                command=command,
            )
            self._log_first_motion_step_once(
                joint_positions,
                joint_velocities,
                requested_command,
                command,
            )
        self._motion_step += 1

        full_targets = robot_state.joint_pos.detach().cpu().numpy().astype(np.float32, copy=True)
        full_targets[:, self._leg_state_indices] = self._cached_leg_targets
        full_targets[:, self._upper_state_indices] = upper_targets
        return full_targets

    def _log_first_motion_step_once(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        requested_command: np.ndarray,
        applied_command: np.ndarray,
    ) -> None:
        """Print the first actual policy input/output for at most two envs."""
        if self._printed_motion_step:
            return
        self._printed_motion_step = True

        q = joint_positions.detach().cpu().numpy()
        dq = joint_velocities.detach().cpu().numpy()
        action = self.motion_policy.last_action
        rows = []
        for env_id in range(min(self.num_envs, 2)):
            rows.append(
                f"env {env_id}: requested command={requested_command[env_id].tolist()}, "
                f"applied command={applied_command[env_id].tolist()}"
            )
            last_observation = getattr(self.motion_policy, "last_observation", None)
            if last_observation is not None:
                observation = last_observation[env_id]
                body_angular_velocity = (
                    observation[0:3] / G1MotionPolicy.ANGULAR_VELOCITY_SCALE
                ).tolist()
                rows.append(
                    "policy IMU observation: "
                    f"body angular velocity={body_angular_velocity}, "
                    f"projected gravity={observation[3:6].tolist()}, "
                    f"scaled command={observation[6:9].tolist()}, "
                    f"phase sin/cos={observation[45:47].tolist()}"
                )
            rows.append("joint | q [rad] | dq [rad/s] | policy action | target [rad]")
            for joint_index, name in enumerate(self.leg_joint_names):
                rows.append(
                    f"{name} | {q[env_id, joint_index]:.5f} | "
                    f"{dq[env_id, joint_index]:.5f} | "
                    f"{action[env_id, joint_index]:.5f} | "
                    f"{self._cached_leg_targets[env_id, joint_index]:.5f}"
                )
        log.info("First G1 motion-policy step (printed once):\n{}", "\n".join(rows))

    def _update_motion_diagnostics(self, unsuccessful: torch.Tensor) -> None:
        """Measure PD tracking and torque saturation, and print concise snapshots.

        One snapshot is emitted after one simulated second and another at the
        first unsuccessful termination.  This distinguishes a bad IMU input
        from a controller that cannot follow the policy targets.
        """
        handler = self.env.env.handler
        if (
            len(self._leg_dof_indices) != len(self.leg_joint_names)
            or not hasattr(handler, "robot_inst")
            or not hasattr(handler.robot_inst, "get_dofs_control_force")
        ):
            return

        states = handler.get_states()
        robot_state = states.robots[self.robot_name]
        q = (
            robot_state.joint_pos.index_select(1, self._leg_state_indices_torch)
            .detach()
            .cpu()
            .numpy()
        )
        dq = (
            robot_state.joint_vel.index_select(1, self._leg_state_indices_torch)
            .detach()
            .cpu()
            .numpy()
        )
        torque = (
            handler.robot_inst.get_dofs_control_force(
                dofs_idx_local=self._leg_dof_indices
            )
            .detach()
            .cpu()
            .numpy()
        )
        tracking_error = np.abs(self._cached_leg_targets - q)
        torque_utilization = np.abs(torque) / np.maximum(self._leg_torque_limits[None, :], 1e-6)
        self._motion_diagnostic_steps += 1
        self._max_leg_tracking_error = np.maximum(
            self._max_leg_tracking_error, tracking_error
        )
        self._max_leg_torque_utilization = np.maximum(
            self._max_leg_torque_utilization, torque_utilization
        )

        unsuccessful_np = unsuccessful.detach().cpu().numpy().astype(bool)
        one_second_steps = max(1, int(round(1.0 / self._env_control_dt)))
        report_one_second = (
            not self._printed_one_second_motion_diagnostic
            and np.any(self._motion_diagnostic_steps >= one_second_steps)
        )
        report_failure = (
            not self._printed_motion_failure_diagnostic and unsuccessful_np.any()
        )
        if not report_one_second and not report_failure:
            return

        if report_failure:
            env_ids = np.flatnonzero(unsuccessful_np).tolist()
            title = "first unsuccessful termination"
            self._printed_motion_failure_diagnostic = True
        else:
            env_ids = [int(np.argmax(self._motion_diagnostic_steps))]
            title = "one-second snapshot"
            self._printed_one_second_motion_diagnostic = True

        pelvis = robot_state.body_state[:, self._pelvis_index, :].detach().cpu().numpy()
        observation = self.motion_policy.last_observation
        rows = [
            f"G1 motion diagnostics ({title}); torque utilization >= 1.0 means saturation:"
        ]
        for env_id in env_ids:
            body_omega = (
                observation[env_id, 0:3]
                / G1MotionPolicy.ANGULAR_VELOCITY_SCALE
            )
            rows.append(
                f"env {env_id}: simulated time={self._motion_diagnostic_steps[env_id] * self._env_control_dt:.3f}s, "
                f"pelvis z={pelvis[env_id, 2]:.4f}, quaternion WXYZ={pelvis[env_id, 3:7].tolist()}, "
                f"world omega={pelvis[env_id, 10:13].tolist()}, body omega={body_omega.tolist()}, "
                f"projected gravity={observation[env_id, 3:6].tolist()}"
            )
            rows.append(
                "joint | q | target | abs error | dq | torque [Nm] | limit [Nm] | max utilization"
            )
            for joint_index, name in enumerate(self.leg_joint_names):
                rows.append(
                    f"{name} | {q[env_id, joint_index]:.4f} | "
                    f"{self._cached_leg_targets[env_id, joint_index]:.4f} | "
                    f"{tracking_error[env_id, joint_index]:.4f} | "
                    f"{dq[env_id, joint_index]:.3f} | {torque[env_id, joint_index]:.2f} | "
                    f"{self._leg_torque_limits[joint_index]:.1f} | "
                    f"{self._max_leg_torque_utilization[env_id, joint_index]:.3f}"
                )
        log.warning("{}", "\n".join(rows))

    def _reset_motion_state(self, env_ids=None) -> None:
        """Reset recurrent walking-policy state together with simulator environments."""
        if env_ids is None:
            self.motion_policy.reset()
            self._cached_leg_targets[:] = G1MotionPolicy.DEFAULT_ANGLES
            self.last_requested_locomotion_command.fill(0.0)
            self.last_locomotion_command.fill(0.0)
            self._motion_step = 0
            self._motion_diagnostic_steps.fill(0)
            self._max_leg_tracking_error.fill(0.0)
            self._max_leg_torque_utilization.fill(0.0)
            return

        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        self.motion_policy.reset(env_ids.tolist())
        self._cached_leg_targets[env_ids] = G1MotionPolicy.DEFAULT_ANGLES
        self.last_requested_locomotion_command[env_ids] = 0.0
        self.last_locomotion_command[env_ids] = 0.0
        self._motion_diagnostic_steps[env_ids] = 0
        self._max_leg_tracking_error[env_ids] = 0.0
        self._max_leg_torque_utilization[env_ids] = 0.0

    def step_wait(self):
        """Wait for the step to complete."""
        #------------------------------------
        #--------------DEBUG-----------------
        #------------------------------------
        # debug = 0
        # if debug == 0:
        #     actions = self.debug0()
        #     obs, rewards, unsuccess, timeout, _ = self.env.step(actions)
        # elif debug == 1:
        #     if self.timesteps[0] % 20 == 0:
        #         self.action = self.ik_solver()
        #     obs, rewards, unsuccess, timeout, _ = self.env.step([{"g1_slider": {"dof_pos_target": self.action}}])
        # elif debug == 2:
        #     obs, rewards, unsuccess, timeout, _ = self.env.step(self.debug2())
        # elif debug == 3:
        #     obs, rewards, unsuccess, timeout, _ = self.env.step(self.debug_hold_still_arms_up())
        #end debug

        if self.env.scenario.sim == 'genesis' and self.raw_actions is not None:
            actions_to_pass = self.raw_actions
        else:
            actions_to_pass = self.action_dicts

        # # Provedení kroku s vybraným formátem akcí
        obs, rewards, unsuccess, timeout, _ = self.env.step(actions_to_pass)
        obs = obs.cpu().numpy()
        obs = self.add_extra_to_obs(obs)

        # --- Done flag ---
        dones = timeout.to(unsuccess.device) | unsuccess

        # Read controller forces before a failed environment is reset.
        self._update_motion_diagnostics(unsuccess)

        # --- Update time counters ---
        self.timesteps += (~unsuccess).float()

        # --- Připrav info dicty ---
        infos = [{} for _ in range(self.num_envs)]

        # --- Masky ---
        unsuccess_mask = unsuccess.cpu().numpy().astype(bool)
        timeout_mask = timeout.cpu().numpy().astype(bool)

        # --- Reset neúspěšných envů ---
        if unsuccess_mask.any():
            self.timesteps[unsuccess_mask] = 0.0
            unsuccess_ids = np.nonzero(unsuccess_mask)[0].tolist()
            obs, _ = self.env.reset(env_ids=unsuccess_ids)
            self._reset_motion_state(unsuccess_ids)
            obs = obs.cpu().numpy()
            obs = self.add_extra_to_obs(obs)
            for i in unsuccess_ids:
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = False
        success = self.env.env.handler.task.just_finished.to(dones.device)
        if success.any():
            dones = dones | success
            self.timesteps[success] = 0.0
            success_ids = success.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
            obs, _ = self.env.reset(env_ids=success_ids)
            self._reset_motion_state(success_ids)
            obs = obs.cpu().numpy()
            obs = self.add_extra_to_obs(obs)
            for i in success_ids:
                infos[i]["is_success"] = True
                infos[i]["TimeLimit.truncated"] = False

        # --- Reset neúspěšných envů ---
        if timeout_mask.any():
            self.timesteps[timeout_mask] = 0.0
            timeout_ids = np.nonzero(timeout_mask)[0].tolist()
            obs, _ = self.env.reset(env_ids=timeout_ids)
            self._reset_motion_state(timeout_ids)
            obs = obs.cpu().numpy()
            obs = self.add_extra_to_obs(obs)
            for i in timeout_ids:
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = True
        if VIZUALIZATION:
            self._update_joint_viz()
            print(f"Step rewards: {rewards.cpu().numpy()}, Unsuccess: {unsuccess.cpu().numpy()}, Timeout: {timeout.cpu().numpy()}")

        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        """Get images from the environment."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set an attribute of the environment."""
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call a method of the environment."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if the environment is wrapped by a given wrapper class."""
        raise NotImplementedError


    def ik_solver(self) -> dict:
        import numpy as np
        from ikpy.chain import Chain
        from scipy.spatial.transform import Rotation as R

        # 1. SETUP PROSTŘEDÍ
        env = self.env
        robot_cfg = env.scenario.robots[0]
        states = env.env.handler.get_states()
        robot_state = states.robots[robot_cfg.name]

        # Inicializace výstupního slovníku (všechny klouby na 0)
        full_joint_dict = {name: 0.0 for name in list(robot_cfg.joint_limits.keys())}

        # 2. VÝPOČET BÁZE (TORSO)
        # Získáme globální pozici a orientaci torsa, která je společná pro obě ruce
        torso_idx = robot_state.body_names.index("torso_link")
        base_pos = robot_state.body_state[0, torso_idx, :3].cpu().numpy()
        q_raw = robot_state.body_state[0, torso_idx, 3:7].cpu().numpy()

        # Fix Quaternionu (WXYZ -> XYZW) pro MuJoCo
        if abs(q_raw[0]) > 0.9:
            base_quat = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
        else:
            base_quat = q_raw

        rot_base = R.from_quat(base_quat)

        # Vytvoření inverzní matice báze (pro převod Svět -> Lokální řetězec)
        base_matrix = np.eye(4)
        base_matrix[:3, :3] = rot_base.as_matrix()
        base_matrix[:3, 3] = base_pos
        base_inv = np.linalg.inv(base_matrix)

        # 3. DEFINICE KONFIGURACE PRO OBĚ RUCE
        # Zde definujeme cesty k URDF a názvy targetů pro smyčku
        configs = [
            {
                "side": "left",
                "urdf": "/home/roboversepc/code/RoboVerse/roboverse_data/robots/g1/urdf/g1_rotslider_for_IK_left.urdf",
                "target_name": "target_hand_left",
                # Rotace pro levou ruku (dle tvého funkčního kódu)
                #"rot_fix": R.from_euler('z', -90, degrees=True),
                #"rot_approach": R.from_euler('y', -90, degrees=True)
            },
            {
                "side": "right",
                "urdf": "/home/roboversepc/code/RoboVerse/roboverse_data/robots/g1/urdf/g1_rotslider_for_IK_right.urdf",
                "target_name": "target_hand_right",
                # Rotace pro pravou ruku (Zrcadlově? Často bývá approach 'y' +90 nebo stejně, nutno vyzkoušet)
                # Zde nechávám stejné nastavení, pokud je URDF symetrické, možná bude třeba upravit approach.
                "rot_fix": R.from_euler('z', -90, degrees=True), # Pravděpodobně opačně pro druhou ruku
                #"rot_approach": R.from_euler('y', -90, degrees=True)
            }
        ]

        # 4. HLAVNÍ SMYČKA PRO LEVOU A PRAVOU RUKU
        chair = states.objects["chair"]

        for cfg in configs:
            # A) Načtení řetězce
            chain = Chain.from_urdf_file(cfg["urdf"], base_elements=["torso_link"])

            # B) Získání Targetu
            try:
                target_idx = chair.body_names.index(cfg["target_name"])
            except ValueError:
                print(f"⚠️ Target {cfg['target_name']} nenalezen, přeskakuji {cfg['side']} ruku.")
                continue

            target_pos = chair.body_state[0, target_idx, :3].cpu().numpy()
            q_tgt_raw = chair.body_state[0, target_idx, 3:7].cpu().numpy()
            # Fix target quaternionu
            target_quat = [q_tgt_raw[1], q_tgt_raw[2], q_tgt_raw[3], q_tgt_raw[0]]

            # C) Výpočet finální orientace cíle
            r_target = R.from_quat(target_quat)
            #final_rot = r_target * cfg["rot_fix"] #* cfg["rot_approach"]

            # D) Transformace do lokálního prostoru (Chain Frame)
            target_matrix_world = np.eye(4)
            target_matrix_world[:3, :3] = r_target.as_matrix()
            target_matrix_world[:3, 3] = target_pos

            target_in_chain = base_inv @ target_matrix_world

            # E) Výpočet IK
            ik_joints = chain.inverse_kinematics_frame(
                target_in_chain,
                orientation_mode="all",
                optimizer="least_squares"
            )

            # F) Mapování na názvy kloubů (Links -> Joints)
            for i, link in enumerate(chain.links):
                link_name = link.name
                val = ik_joints[i]

                # Logika pro nalezení správného klíče
                if link_name in full_joint_dict:
                    full_joint_dict[link_name] = val
                elif link_name.replace("_link", "_joint") in full_joint_dict:
                    full_joint_dict[link_name.replace("_link", "_joint")] = val

        return full_joint_dict

    def debug0(self):
        # [{"g1_slider": {"dof_pos_target": self.action}}]
        actions = [{"g1_slider": {"dof_pos_target":
                                  {
                        "baseslide_joint": 0.0, #y
                        "baseslide_joint2": -0.8, #x
                        "baserot_joint": 0.0,
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.34,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 1.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 1.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                        # Left hand fingers
                        "left_hand_thumb_0_joint": 0.0,
                        "left_hand_thumb_1_joint": 0.0,
                        "left_hand_thumb_2_joint": 0.0,
                        "left_hand_middle_0_joint": 0.0,
                        "left_hand_middle_1_joint": 0.0,
                        "left_hand_index_0_joint": 0.0,
                        "left_hand_index_1_joint": 0.0,
                        # Right hand fingers
                        "right_hand_thumb_0_joint": 0.0,
                        "right_hand_thumb_1_joint": 0.0,
                        "right_hand_thumb_2_joint": 0.0,
                        "right_hand_middle_0_joint": 0.0,
                        "right_hand_middle_1_joint": 0.0,
                        "right_hand_index_0_joint": 0.0,
                        "right_hand_index_1_joint": 0.0
                                  }
                                    }}]
        return actions
    def debug2(self):
        """
        Drží tělo ve fixní pozici a postupně zavírá prsty.
        Zastaví pohyb prstu pouze pokud síla kontaktu na ŠPIČCE přesáhne práh.

        Implementace využívá optimalizované tensorové operace kompatibilní s Genesis.
        """
        FORCE_THRESHOLD = 1.0
        STEP_SIZE = 0.02
        robot_name = "g1_slider"

        # 1. DEFINICE CÍLOVÝCH HODNOT (Limity kam až se mají prsty zavřít)
        finger_limits = {
            # --- LEVÁ RUKA ---
            "left_hand_thumb_0_joint": 0.396, "left_hand_thumb_1_joint": 0.7, "left_hand_thumb_2_joint": 1.0,
            "left_hand_middle_0_joint": -1.5, "left_hand_middle_1_joint": -1.7,
            "left_hand_index_0_joint": -1.5, "left_hand_index_1_joint": -1.7,
            # --- PRAVÁ RUKA ---
            "right_hand_thumb_0_joint": -0.396, "right_hand_thumb_1_joint": -0.7, "right_hand_thumb_2_joint": -1.0,
            "right_hand_middle_0_joint": 1.5, "right_hand_middle_1_joint": 1.7,
            "right_hand_index_0_joint": 1.5, "right_hand_index_1_joint": 1.7
        }

        # Base pose zbytku těla
        base_pose = {
            'baseslide_joint': 1.0338146e-05, 'baseslide_joint2': 0.042001966, 'baserot_joint': -7.5084286e-06,
            'waist_yaw_joint': -4.4286382e-05, 'waist_roll_joint': 8.922054e-05, 'waist_pitch_joint': 0.008053156,
            'left_shoulder_pitch_joint': -0.7364295, 'right_shoulder_pitch_joint': -0.72817403,
            'left_shoulder_roll_joint': 0.5458939, 'right_shoulder_roll_joint': -0.5318888,
            'left_shoulder_yaw_joint': -0.57718706, 'right_shoulder_yaw_joint': 0.5727658,
            'left_elbow_joint': 0.2027091, 'right_elbow_joint': 0.18453476,
            'left_wrist_roll_joint': 0.8287475, 'right_wrist_roll_joint': -0.80135757,
            'left_wrist_pitch_joint': 0.56613743, 'right_wrist_pitch_joint': 0.5907418,
            'left_wrist_yaw_joint': -0.24424289, 'right_wrist_yaw_joint': 0.24177466
        }

        # 2. DEFINICE SKUPIN PRSTŮ (Které klouby patří ke kterému "senzoru" na špičce)
        # Mapování: ID skupiny -> (Seznam kloubů, Klíčové slovo pro link špičky)
        finger_groups = {
            0: (["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"], "left_hand_thumb_2_link"),
            1: (["left_hand_index_0_joint", "left_hand_index_1_joint"], "left_hand_index_1_link"),
            2: (["left_hand_middle_0_joint", "left_hand_middle_1_joint"], "left_hand_middle_1_link"),
            3: (["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"], "right_hand_thumb_2_link"),
            4: (["right_hand_index_0_joint", "right_hand_index_1_joint"], "right_hand_index_1_link"),
            5: (["right_hand_middle_0_joint", "right_hand_middle_1_joint"], "right_hand_middle_1_link"),
        }

        handler = self.env.env.handler
        device = handler.device
        states = handler.get_states()
        num_envs = self.num_envs

        # --- INICIALIZACE STAVU (běží jen poprvé) ---
        if not hasattr(self, "_debug2_cache_init"):
            # 1. Inicializace aktuálních pozic prstů (Tenzor [num_envs, num_finger_joints])
            self._finger_joint_names = list(finger_limits.keys())
            self._finger_joint_limits = torch.tensor([finger_limits[n] for n in self._finger_joint_names], device=device)
            self._current_finger_pos = torch.zeros((num_envs, len(self._finger_joint_names)), device=device)

            # 2. Mapování ID linků na ID skupiny prstů (stejné jako v GraspForceReward)
            global_map = states.extras.get("global_link_map", {})
            num_bodies = states.extras.get("num_bodies_per_env", 1000)

            # Vytvoříme mapu: Global Link ID -> Finger Group ID (0-5) nebo -1
            self._idx_to_group = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            chair_ids = []

            for idx, (o_name, l_name) in global_map.items():
                if o_name == robot_name:
                    for grp_id, (_, tip_link_name) in finger_groups.items():
                        # Hledáme přesnou shodu špičky prstu
                        if tip_link_name in l_name:
                            self._idx_to_group[idx] = grp_id
                elif o_name == "chair":
                    chair_ids.append(idx)

            self._chair_ids = torch.tensor(chair_ids, device=device)
            self._num_bodies = num_bodies
            self._debug2_cache_init = True

            log.info("Debug2: Initialization complete.")

        # --- DETEKCE KONTAKTŮ (Tenzorově) ---
        # Získáme kontakty přímo z robota
        contact_data = states.robots[robot_name].contact

        # Maska blokovaných skupin prstů [num_envs, 6] (6 skupin)
        blocked_groups = torch.zeros((num_envs, len(finger_groups)), dtype=torch.bool, device=device)

        if contact_data is not None:
            link_a = contact_data['link_a']
            link_b = contact_data['link_b']
            valid_mask = contact_data['valid_mask']

            # Získání sil
            forces = contact_data.get('force_b', contact_data.get('force', None))
            if forces is None:
                forces = torch.zeros((*link_a.shape, 3), device=device)
            force_mags = torch.norm(forces, dim=-1)

            # Modulo pro získání base indexů (pro multi-env)
            base_a = link_a % self._num_bodies
            base_b = link_b % self._num_bodies

            # Zjištění, kdo je židle a kdo je prst
            a_is_chair = torch.isin(base_a, self._chair_ids)
            b_is_chair = torch.isin(base_b, self._chair_ids)

            # Mapování na skupiny (pokud není prst, vrátí -1)
            group_a = self._idx_to_group[base_a]
            group_b = self._idx_to_group[base_b]

            # Která skupina prstů se dotýká židle?
            # Pokud b je židle -> kontakt je na a. Pokud a je židle -> kontakt je na b.
            contact_group = torch.where(b_is_chair, group_a, torch.where(a_is_chair, group_b, torch.tensor(-1, device=device)))

            # Validní kontakt: (Je to kontakt prst-židle) AND (Validní v simulaci) AND (Síla > Threshold)
            valid_interaction = (contact_group >= 0) & valid_mask & (force_mags > FORCE_THRESHOLD)

            # Vyplnění masky blokovaných skupin
            # Pro každou skupinu zjistíme, zda má v daném envu validní silný kontakt
            for grp_id in range(len(finger_groups)):
                # Má tento env nějaký kontakt pro tuto skupinu?
                has_contact = (valid_interaction & (contact_group == grp_id)).any(dim=1)
                blocked_groups[:, grp_id] = has_contact

        # --- AKTUALIZACE POZIC ---
        # Iterujeme přes definované klouby a aktualizujeme pozice
        for i, joint_name in enumerate(self._finger_joint_names):
            # Zjistíme, do které skupiny tento kloub patří
            target_grp = -1
            for grp_id, (joints, _) in finger_groups.items():
                if joint_name in joints:
                    target_grp = grp_id
                    break

            # Získáme limit pro tento kloub
            limit = self._finger_joint_limits[i]

            # Maska: Kde MŮŽEME hýbat? (Kde NENÍ skupina blokovaná)
            can_move = ~blocked_groups[:, target_grp]

            # Logika pohybu směrem k limitu
            current_val = self._current_finger_pos[:, i]

            # Vypočítáme krok (plus nebo mínus podle směru k limitu)
            direction = torch.sign(limit - current_val)
            step = direction * STEP_SIZE

            # Nová hodnota (před oříznutím)
            next_val = current_val + step

            # Ošetření přehmitu (clamp mezi current a limit nefunguje jednoduše, uděláme to logicky)
            # Pokud jsme blízko limitu, nastavíme limit
            close_to_limit = torch.abs(current_val - limit) < STEP_SIZE
            next_val = torch.where(close_to_limit, limit, next_val)

            # Aplikujeme změnu jen tam, kde není blokace
            self._current_finger_pos[:, i] = torch.where(can_move, next_val, current_val)

        # --- SESTAVENÍ AKCE ---
        # Protože vracíme list[dict] pro SB3 wrapper, musíme převést tensor zpět (nebo použít tensor přímo, pokud wrapper podporuje)
        # Zde sestavíme full dictionary, kde base_pose je statická a prsty dynamické.

        # Poznámka: Aby to bylo rychlé, ideálně bychom měli posílat Tensor přímo do handleru,
        # ale SB3VecEnv ve vaší implementaci očekává list dictů. Uděláme to tedy hybridně.

        actions = []
        cpu_finger_pos = self._current_finger_pos.cpu().numpy()

        for env_id in range(num_envs):
            # Kopie base pose
            dof_targets = base_pose.copy()

            # Update prstů pro tento env
            for i, name in enumerate(self._finger_joint_names):
                dof_targets[name] = float(cpu_finger_pos[env_id, i])

            actions.append({robot_name: {"dof_pos_target": dof_targets}})

        return actions
    def _init_joint_viz(self):
        """Inicializuje okno a parametry pro živou vizualizaci kloubů."""
        self.viz_joint_limits = self.env.scenario.robots[0].joint_limits
        self.viz_joint_names = list(self.viz_joint_limits.keys())
        self.viz_num_joints = len(self.viz_joint_names)

        # Rozměry vykreslovacího plátna
        self.viz_bar_height = 14
        self.viz_margin_y = 30
        self.viz_width = 800
        self.viz_height = self.viz_num_joints * self.viz_margin_y + 40
        self.viz_text_width = 250
        self.viz_bar_width = self.viz_width - self.viz_text_width - 50

    def _update_joint_viz(self):
        """Vykreslí aktuální a cílové pozice kloubů pro Env 0."""
        states = self.env.env.handler.get_states()
        robot_name = self.env.scenario.robots[0].name

        # Získáme data pouze pro prvního robota (Env 0)
        curr_pos = states.robots[robot_name].joint_pos[0].cpu().numpy()

        # Targety bereme přímo ze states, což je nejspolehlivější
        targ_pos = states.robots[robot_name].joint_pos_target[0].cpu().numpy()

        # Vytvoření černého plátna
        img = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)

        for i, j_name in enumerate(self.viz_joint_names):
            low, high = self.viz_joint_limits[j_name]
            c_val = curr_pos[i]
            t_val = targ_pos[i]

            y_offset = 20 + i * self.viz_margin_y

            # 1. Název kloubu
            cv2.putText(img, j_name, (10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Přepočet hodnoty na pixely
            range_val = high - low
            if range_val == 0: range_val = 1e-6

            def val_to_px(v):
                v_clamped = np.clip(v, low, high)
                norm = (v_clamped - low) / range_val
                return self.viz_text_width + int(norm * self.viz_bar_width)

            px_low = self.viz_text_width
            px_high = self.viz_text_width + self.viz_bar_width
            px_curr = val_to_px(c_val)
            px_targ = val_to_px(t_val)

            # 2. Vykreslení limitů (Šedé pozadí)
            cv2.rectangle(img, (px_low, y_offset), (px_high, y_offset + self.viz_bar_height), (50, 50, 50), -1)

            # 3. Vykreslení aktuální pozice (Zelený bar)
            # Pokud je hodnota záporná/kladná, bar roste od minima (zleva)
            cv2.rectangle(img, (px_low, y_offset), (px_curr, y_offset + self.viz_bar_height), (0, 180, 0), -1)

            # 4. Vykreslení cílové pozice akce (Červená svislá čára)
            cv2.line(img, (px_targ, y_offset - 4), (px_targ, y_offset + self.viz_bar_height + 4), (0, 0, 255), 2)

            # 5. Textové hodnoty pod barem (C=Current, T=Target, L=Limits)
            val_text = f"C: {c_val: .2f} | T: {t_val: .2f}  [lim: {low:.2f} to {high:.2f}]"
            cv2.putText(img, val_text, (self.viz_text_width, y_offset + self.viz_bar_height + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Zobrazení okna
        cv2.imshow("Live Joint Info (Env 0)", img)
        cv2.waitKey(1) # 1 ms pauza nutná pro překreslení okna OpenCV
    def debug_hold_still_arms_up(self) -> np.ndarray:
        """
        Debug akce pro simple slider robota:
        - robot stojí na místě
        - ruce drží nahoře

        Vrací numpy array tvaru [num_envs, action_dim],
        což je přesně formát, který Genesis větev ve wrapperu umí poslat dál.
        """
        robot_cfg = self.env.scenario.robots[0]
        joint_names = list(robot_cfg.joint_limits.keys())

        # výchozí targety pro aktuální simple robota
        target_dict = {
            "baseslide_joint": 0.0,
            "baseslide_joint2": 0.0,
            "baserot_joint": 0.0,
            "left_shoulder_pitch_joint": -1.86,
            "right_shoulder_pitch_joint": -1.86,
        }

        # složení akce přesně ve stejném pořadí jako action_space
        single_action = np.array(
            [target_dict[name] for name in joint_names],
            dtype=np.float32
        )

        # stejná akce pro všechna prostředí
        actions = np.tile(single_action[None, :], (self.num_envs, 1))
        return actions
