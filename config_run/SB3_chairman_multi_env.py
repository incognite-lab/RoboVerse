"""SB3 wrapper extensions required by concurrent stage-wise ChairMan PPO.

The physical action/observation implementation stays in
``SB3_chairman_env.py``. This subclass adds an explicit stage-transition API
for the multi-policy collector. Intermediate stage transitions are not Gym
episode boundaries; the trainer creates policy-local boundaries from the
metadata returned here.
"""

from __future__ import annotations

import numpy as np

try:
    from .SB3_chairman_env import StableBaseline3VecEnv as _ChairmanVecEnv
except ImportError:  # ``python config_run/main_multi.py ...``
    from SB3_chairman_env import StableBaseline3VecEnv as _ChairmanVecEnv


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
