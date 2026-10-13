"""ChairMan multi-policy wrapper with a compact relative observation space.

Observation layout::

    [joint positions,
     pelvis linear/angular velocity,
     chair position relative to the pelvis in a yaw-aligned frame,
     chair orientation relative to the pelvis]

The stage is deliberately not part of the observation: ChairMan routes every
stage to its own policy.  Stage state remains available to the trainer through
the APIs inherited from :mod:`SB3_chairman_multi_env`.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from metasim.utils.chair_navigation import world_vector_to_body_xy
from metasim.utils.math import quat_inv, quat_mul, quat_unique

try:
    from .SB3_chairman_multi_env import StableBaseline3VecEnv as _FullChairmanMultiVecEnv
except ImportError:  # ``python config_run/main_multi_diff_obs.py ...``
    from SB3_chairman_multi_env import StableBaseline3VecEnv as _FullChairmanMultiVecEnv


class _MinimalChairmanMultiVecEnv(_FullChairmanMultiVecEnv):
    """Shared implementation for compact ChairMan observations."""

    TASK_OBSERVATION_DIM: int
    OBSERVATION_VARIANT: str

    def __init__(self, env):
        super().__init__(env)
        self.num_joint_observations = len(self.robot_joint_names)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_joint_observations + self.TASK_OBSERVATION_DIM,),
            dtype=np.float32,
        )

    @property
    def stage_observation_indices(self) -> None:
        """Compact observations have no stage one-hot columns."""
        return None

    def _task_observation_torch(
        self,
        pelvis_pos: torch.Tensor,
        pelvis_quat: torch.Tensor,
        pelvis_velocity: torch.Tensor,
        chair_pos: torch.Tensor,
        chair_quat: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def add_extra_to_obs_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Construct a compact observation without CPU synchronization."""
        states = self.env.env.handler.get_states()
        robot = states.robots[self.robot_name]
        chair = states.objects["chair"]

        joint_positions = obs.to(
            device=self.torch_device, dtype=torch.float32
        ).reshape(self.num_envs, -1)
        if joint_positions.shape[1] != self.num_joint_observations:
            raise RuntimeError(
                "ChairMan joint observation has unexpected width "
                f"{joint_positions.shape[1]}, expected {self.num_joint_observations}"
            )

        pelvis_state = robot.body_state[:, self._pelvis_index]
        chair_idx = chair.body_names.index("base_link")
        chair_state = chair.body_state[:, chair_idx]
        task_observation = self._task_observation_torch(
            pelvis_pos=pelvis_state[:, :3],
            pelvis_quat=pelvis_state[:, 3:7],
            pelvis_velocity=pelvis_state[:, 7:13],
            chair_pos=chair_state[:, :3],
            chair_quat=chair_state[:, 3:7],
        )
        result = torch.cat((joint_positions, task_observation), dim=1)
        if tuple(result.shape) != (self.num_envs, self.observation_space.shape[0]):
            raise RuntimeError(
                f"{self.OBSERVATION_VARIANT} observation has shape "
                f"{tuple(result.shape)}, expected "
                f"({self.num_envs}, {self.observation_space.shape[0]})"
            )
        return result

    def add_extra_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """CPU/SB3 counterpart of :meth:`add_extra_to_obs_torch`."""
        observation = self.add_extra_to_obs_torch(
            torch.as_tensor(obs, dtype=torch.float32, device=self.torch_device)
        )
        return observation.detach().cpu().numpy().astype(np.float32, copy=False)


class StableBaseline3VecEnv(_MinimalChairmanMultiVecEnv):
    """Compact observation using chair pose relative to the pelvis."""

    # pelvis velocity (6) + relative chair position (3) + quaternion (4)
    TASK_OBSERVATION_DIM = 13
    OBSERVATION_VARIANT = "minimal_relative"

    def _task_observation_torch(
        self,
        pelvis_pos: torch.Tensor,
        pelvis_quat: torch.Tensor,
        pelvis_velocity: torch.Tensor,
        chair_pos: torch.Tensor,
        chair_quat: torch.Tensor,
    ) -> torch.Tensor:
        chair_delta_world = chair_pos - pelvis_pos
        chair_relative_position = torch.cat(
            (
                world_vector_to_body_xy(
                    chair_delta_world[:, :2], pelvis_quat
                ),
                chair_delta_world[:, 2:3],
            ),
            dim=1,
        )
        chair_relative_quat = quat_unique(
            quat_mul(quat_inv(pelvis_quat), chair_quat)
        )
        return torch.cat(
            (
                pelvis_velocity,
                chair_relative_position,
                chair_relative_quat,
            ),
            dim=1,
        )


ChairmanMultiRelativeVecEnv = StableBaseline3VecEnv
