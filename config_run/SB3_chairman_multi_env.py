"""SB3 wrapper extensions required by concurrent stage-wise ChairMan PPO.

The physical action/observation implementation stays in
``SB3_chairman_env.py``. This subclass adds an explicit stage-transition API
for the multi-policy collector. Intermediate stage transitions are not Gym
episode boundaries; the trainer creates policy-local boundaries from the
metadata returned here.
"""

from __future__ import annotations

import numpy as np
import torch

from metasim.utils.chair_navigation import (
    CHAIR_FINAL_DISTANCE,
    CHAIR_STAGING_DISTANCE,
    chair_back_direction_xy,
    world_vector_to_body_xy,
)

try:
    from .SB3_chairman_env import (
        StableBaseline3VecEnv as _ChairmanVecEnv,
        _quaternion_error_vector,
    )
except ImportError:  # ``python config_run/main_multi.py ...``
    from SB3_chairman_env import (
        StableBaseline3VecEnv as _ChairmanVecEnv,
        _quaternion_error_vector,
    )


class StableBaseline3VecEnv(_ChairmanVecEnv):
    """ChairMan VecEnv with walking commands and policy-routing metadata.

    Each stage PPO outputs targets only for non-leg joints followed by
    ``[walk_vx, walk_vy, walk_yaw_rate]``. The inherited ``G1MotionPolicy``
    consumes those three commands and produces all 12 leg targets.
    """

    NUM_POLICY_STAGES = 6

    def __init__(self, env):
        self.verbose_motion_diagnostics = bool(
            getattr(env.scenario.task, "verbose_motion_diagnostics", False)
        )
        super().__init__(env)
        self.num_upper_body_actions = len(self.upper_body_joint_names)
        self.walk_command_slice = slice(
            self.num_upper_body_actions, self.num_upper_body_actions + 3
        )
        expected_actions = self.num_upper_body_actions + len(
            self.LOCOMOTION_COMMAND_NAMES
        )
        if self.action_space.shape != (expected_actions,):
            raise RuntimeError(
                "ChairMan multi-policy action space must contain only upper-body "
                f"targets plus three walking commands, got {self.action_space.shape}"
            )
        if len(self.leg_joint_names) != 12:
            raise RuntimeError(
                f"G1MotionPolicy must own 12 leg joints, got {len(self.leg_joint_names)}"
            )
        self.last_stage_policy_upper_body_actions = np.zeros(
            (self.num_envs, self.num_upper_body_actions), dtype=np.float32
        )
        self.last_stage_policy_walk_commands = np.zeros(
            (self.num_envs, 3), dtype=np.float32
        )
        self._stage_before_step = self.get_current_stages()
        self.torch_device = torch.device(env.env.handler.device)
        self._upper_state_indices_torch = torch.as_tensor(
            self._upper_state_indices, dtype=torch.long, device=self.torch_device
        )
        self._action_low_torch = torch.as_tensor(
            self.action_space.low, dtype=torch.float32, device=self.torch_device
        )
        self._action_high_torch = torch.as_tensor(
            self.action_space.high, dtype=torch.float32, device=self.torch_device
        )
        self._walk_max_command_torch = torch.as_tensor(
            self.motion_policy.MAX_COMMAND,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self._cached_leg_targets_torch = torch.as_tensor(
            self.motion_policy.DEFAULT_ANGLES,
            dtype=torch.float32,
            device=self.torch_device,
        ).expand(self.num_envs, -1).clone()
        self.last_requested_locomotion_command_torch = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.torch_device
        )
        self.last_locomotion_command_torch = torch.zeros_like(
            self.last_requested_locomotion_command_torch
        )
        self.last_stage_policy_upper_body_actions_torch = torch.zeros(
            (self.num_envs, self.num_upper_body_actions),
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.last_stage_policy_walk_commands_torch = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.torch_device
        )

    def _log_motion_configuration_once(self, robot_cfg) -> None:
        if self.verbose_motion_diagnostics:
            super()._log_motion_configuration_once(robot_cfg)

    def _log_first_motion_step_once(self, *args, **kwargs) -> None:
        if self.verbose_motion_diagnostics:
            super()._log_first_motion_step_once(*args, **kwargs)
        else:
            self._printed_motion_step = True

    def _update_motion_diagnostics(self, unsuccessful) -> None:
        if self.verbose_motion_diagnostics:
            super()._update_motion_diagnostics(unsuccessful)

    def _compose_robot_targets(self, actions: np.ndarray) -> np.ndarray:
        """Delegate all leg control to the pretrained walking policy."""
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected stage-policy actions with shape {expected_shape}, "
                f"got {actions.shape}"
            )
        self.last_stage_policy_upper_body_actions[:] = actions[
            :, : self.num_upper_body_actions
        ]
        self.last_stage_policy_walk_commands[:] = actions[
            :, self.walk_command_slice
        ]
        # SB3_chairman_env._compose_robot_targets invokes G1MotionPolicy for
        # the 12 legs and combines its output with PPO upper-body targets.
        return super()._compose_robot_targets(actions)

    def get_current_stages(self) -> np.ndarray:
        """Return a detached CPU copy of the current stage for every env."""
        stages = self.env.env.handler.task.reward_functions[0].actual_stage
        if stages is None:
            return np.zeros(self.num_envs, dtype=np.int64)
        return stages.detach().cpu().numpy().astype(np.int64, copy=True)

    def get_current_stages_torch(self) -> torch.Tensor:
        """Return current routing stages without synchronizing the GPU."""
        stages = self.env.env.handler.task.reward_functions[0].actual_stage
        if stages is None:
            return torch.zeros(
                self.num_envs, dtype=torch.long, device=self.torch_device
            )
        return stages.detach().to(device=self.torch_device, dtype=torch.long).clone()

    def add_extra_to_obs_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Build the complete Chairman observation directly on the simulator GPU."""
        handler = self.env.env.handler
        states = handler.get_states()
        robot = states.robots[self.robot_name]
        chair = states.objects["chair"]
        obs = obs.to(device=self.torch_device, dtype=torch.float32).reshape(
            self.num_envs, -1
        )

        robot_body_states = robot.body_state[:, self.indexes, :7].reshape(
            self.num_envs, -1
        )
        pelvis_pos = robot.body_state[:, self._pelvis_index, :3]
        pelvis_quat = robot.body_state[:, self._pelvis_index, 3:7]
        pelvis_velocity = robot.body_state[:, self._pelvis_index, 7:13]

        chair_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_idx, :3]
        chair_quat = chair.body_state[:, chair_idx, 3:7]
        chair_vel = chair.body_state[:, chair_idx, 7:10]
        vec_world = chair_pos - pelvis_pos
        dist_to_chair = torch.linalg.vector_norm(
            vec_world[:, :2], dim=-1, keepdim=True
        )
        chair_back_world = chair_back_direction_xy(chair_quat)
        staging_pos_xy = (
            chair_pos[:, :2] + CHAIR_STAGING_DISTANCE * chair_back_world
        )
        final_pos_xy = chair_pos[:, :2] + CHAIR_FINAL_DISTANCE * chair_back_world
        staging_vec_world = staging_pos_xy - pelvis_pos[:, :2]
        final_vec_world = final_pos_xy - pelvis_pos[:, :2]
        dist_to_final = torch.linalg.vector_norm(
            final_vec_world, dim=-1, keepdim=True
        )
        chair_rel_body = world_vector_to_body_xy(vec_world[:, :2], pelvis_quat)
        staging_rel_body = world_vector_to_body_xy(staging_vec_world, pelvis_quat)
        final_rel_body = world_vector_to_body_xy(final_vec_world, pelvis_quat)
        chair_back_body = world_vector_to_body_xy(chair_back_world, pelvis_quat)

        safe_stages = self.get_current_stages_torch().clamp(
            0, self.num_stages - 1
        )
        stage_one_hot = torch.nn.functional.one_hot(
            safe_stages, num_classes=self.num_stages
        ).to(dtype=torch.float32)

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
        arm_errors = torch.cat(
            (
                torch.linalg.vector_norm(
                    pos_left - target_left_pos, dim=-1, keepdim=True
                ),
                torch.linalg.vector_norm(
                    pos_right - target_right_pos, dim=-1, keepdim=True
                ),
            ),
            dim=-1,
        )
        left_target_delta = target_left_pos - pos_left
        right_target_delta = target_right_pos - pos_right
        hand_target_body = torch.cat(
            (
                world_vector_to_body_xy(left_target_delta[:, :2], pelvis_quat),
                left_target_delta[:, 2:3],
                world_vector_to_body_xy(right_target_delta[:, :2], pelvis_quat),
                right_target_delta[:, 2:3],
            ),
            dim=-1,
        )
        hand_orientation_error = torch.cat(
            (
                _quaternion_error_vector(quat_left, target_left_quat),
                _quaternion_error_vector(quat_right, target_right_quat),
            ),
            dim=-1,
        )
        hand_velocity_body = torch.cat(
            (
                world_vector_to_body_xy(vel_left[:, :2], pelvis_quat),
                vel_left[:, 2:3],
                world_vector_to_body_xy(vel_right[:, :2], pelvis_quat),
                vel_right[:, 2:3],
            ),
            dim=-1,
        )
        fingertip_forces = self._fingertip_chair_forces(states, robot)

        extra_obs = torch.cat(
            (
                robot_body_states,
                pelvis_velocity,
                chair_pos,
                chair_vel,
                vec_world,
                chair_rel_body,
                dist_to_chair,
                dist_to_final,
                staging_rel_body,
                final_rel_body,
                chair_back_body,
                hand_target_body,
                hand_orientation_error,
                hand_velocity_body,
                fingertip_forces,
                self.last_locomotion_command_torch,
                stage_one_hot,
                arm_errors,
            ),
            dim=1,
        )
        result = torch.cat((obs, extra_obs), dim=1)
        if tuple(result.shape[1:]) != self.observation_space.shape:
            raise RuntimeError(
                "Torch Chairman observation has unexpected shape "
                f"{tuple(result.shape)}, expected ({self.num_envs}, "
                f"{self.observation_space.shape[0]})"
            )
        return result

    def _compose_robot_targets_torch(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.torch_device
        )
        expected_shape = (self.num_envs, self.action_space.shape[0])
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"Expected stage-policy actions with shape {expected_shape}, "
                f"got {tuple(actions.shape)}"
            )
        actions = torch.maximum(
            torch.minimum(actions, self._action_high_torch), self._action_low_torch
        )
        upper_targets = actions[:, : self.num_upper_body_actions]
        requested_command = actions[:, self.walk_command_slice]
        command = torch.maximum(
            torch.minimum(requested_command, self._walk_max_command_torch),
            -self._walk_max_command_torch,
        )
        previous_command = self.last_locomotion_command_torch.clone()
        self.last_requested_locomotion_command_torch.copy_(requested_command)
        self.last_locomotion_command_torch.copy_(command)
        self.last_stage_policy_upper_body_actions_torch.copy_(upper_targets)
        self.last_stage_policy_walk_commands_torch.copy_(requested_command)

        for reward_fn in self.env.scenario.task.reward_functions:
            set_context = getattr(reward_fn, "set_control_context", None)
            if set_context is not None:
                set_context(command, previous_command, device=self.torch_device)

        states = self.env.env.handler.get_states()
        robot_state = states.robots[self.robot_name]
        if self._motion_step % self._motion_decimation == 0:
            joint_positions = robot_state.joint_pos.index_select(
                1, self._leg_state_indices_torch
            )
            joint_velocities = robot_state.joint_vel.index_select(
                1, self._leg_state_indices_torch
            )
            pelvis_state = robot_state.body_state[:, self._pelvis_index, :]
            self._cached_leg_targets_torch = (
                self.motion_policy.predict_joint_positions_torch(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                    angular_velocity=pelvis_state[:, 10:13],
                    angular_velocity_frame="world",
                    base_quaternion_wxyz=pelvis_state[:, 3:7],
                    command=command,
                )
            )
        self._motion_step += 1
        full_targets = robot_state.joint_pos.detach().clone()
        full_targets.index_copy_(
            1, self._leg_state_indices_torch, self._cached_leg_targets_torch
        )
        full_targets.index_copy_(
            1, self._upper_state_indices_torch, upper_targets
        )
        return full_targets

    def _reset_motion_state_torch(self, env_ids: torch.Tensor | None = None) -> None:
        defaults = getattr(self.motion_policy, "_default_angles_torch", None)
        if defaults is None:
            defaults = torch.as_tensor(
                self.motion_policy.DEFAULT_ANGLES,
                dtype=torch.float32,
                device=self.torch_device,
            )
        if env_ids is None:
            self.motion_policy.reset()
            self._cached_leg_targets_torch.copy_(
                defaults.expand(self.num_envs, -1)
            )
            self.last_requested_locomotion_command_torch.zero_()
            self.last_locomotion_command_torch.zero_()
            self._motion_step = 0
            return
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.torch_device, dtype=torch.long)
        cpu_ids = env_ids.detach().cpu().tolist()
        self.motion_policy.reset(cpu_ids)
        self._cached_leg_targets_torch.index_copy_(
            0, env_ids, defaults.expand(env_ids.numel(), -1)
        )
        self.last_requested_locomotion_command_torch.index_fill_(0, env_ids, 0.0)
        self.last_locomotion_command_torch.index_fill_(0, env_ids, 0.0)

    def torch_reset(self) -> torch.Tensor:
        """Reset and return a GPU observation for MultiPPOTrainer."""
        obs, _ = self.env.reset()
        self._reset_motion_state_torch()
        self.timesteps.zero_()
        return self.add_extra_to_obs_torch(obs)

    def torch_step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Step Chairman while keeping the rollout data on the simulator GPU."""
        stage_before = self.get_current_stages_torch()
        robot_targets = self._compose_robot_targets_torch(actions)
        obs, rewards, unsuccessful, timeout, _ = self.env.step(robot_targets)
        unsuccessful = unsuccessful.to(device=self.torch_device, dtype=torch.bool)
        timeout = timeout.to(device=self.torch_device, dtype=torch.bool)
        rewards = rewards.to(device=self.torch_device, dtype=torch.float32)
        dones = unsuccessful | timeout
        self.timesteps += (~unsuccessful).float()

        task = self.env.env.handler.task
        completed_source = getattr(task, "completed_stage_events", None)
        completed = (
            torch.full_like(stage_before, -1)
            if completed_source is None
            else completed_source.detach().to(
                device=self.torch_device, dtype=torch.long
            ).clone()
        )
        stage_after_event = self.get_current_stages_torch()
        success = task.just_finished.to(
            device=self.torch_device, dtype=torch.bool
        ).clone()
        dones |= success
        reset_mask = unsuccessful | timeout | success
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            self.timesteps.index_fill_(0, reset_ids, 0.0)
            cpu_ids = reset_ids.detach().cpu().tolist()
            obs, _ = self.env.reset(env_ids=cpu_ids)
            self._reset_motion_state_torch(reset_ids)

        observation = self.add_extra_to_obs_torch(obs)
        metadata = {
            "stage_before": stage_before,
            "stage_after_event": stage_after_event,
            "stage_after": stage_after_event,
            "completed_stage": completed,
            "physical_done": unsuccessful | timeout | success,
            "task_success": success,
            "timeout": timeout,
        }
        return observation, rewards, dones, metadata

    def reset(self):
        observation = super().reset()
        self._stage_before_step = self.get_current_stages()
        unexpected = (self._stage_before_step < 0) | (
            self._stage_before_step >= self.NUM_POLICY_STAGES
        )
        if unexpected.any():
            raise RuntimeError(
                "ChairMan multi-policy reset selected a stage without a policy: "
                f"{np.unique(self._stage_before_step[unexpected])}"
            )
        return observation

    def step_async(self, actions: np.ndarray) -> None:
        # Capture this before simulation/checker execution. The checker may
        # increment actual_stage during the step.
        self._stage_before_step = self.get_current_stages()
        super().step_async(actions)

    def step_wait(self):
        observation, rewards, dones, infos = super().step_wait()
        stage_after = self.get_current_stages()

        task = self.env.env.handler.task
        completed_events = getattr(task, "completed_stage_events", None)
        if completed_events is None:
            completed = np.full(self.num_envs, -1, dtype=np.int64)
        else:
            completed = (
                completed_events.detach().cpu().numpy().astype(np.int64, copy=True)
            )

        dones = np.asarray(dones, dtype=bool)
        for env_id, info in enumerate(infos):
            before = int(self._stage_before_step[env_id])
            after = int(stage_after[env_id])
            completed_stage = int(completed[env_id])
            stage_changed = completed_stage == before

            info["stage_before"] = before
            info["stage_after"] = after
            info["completed_stage"] = completed_stage
            info["stage_changed"] = bool(stage_changed)
            # This is terminal solely for the policy that generated the
            # action. It does not imply a simulator reset.
            info["policy_terminal"] = bool(dones[env_id] or stage_changed)
            info["physical_done"] = bool(dones[env_id])
            info["task_success"] = bool(info.get("is_success", False))

        return observation, rewards, dones, infos


ChairmanMultiVecEnv = StableBaseline3VecEnv
