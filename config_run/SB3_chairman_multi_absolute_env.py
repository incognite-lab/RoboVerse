"""ChairMan multi-policy wrapper with a compact absolute observation space.

Observation layout::

    [joint positions,
     pelvis world position and orientation,
     pelvis linear/angular velocity,
     chair world position and orientation]
"""

from __future__ import annotations

import torch

from metasim.utils.math import quat_unique

try:
    from .SB3_chairman_multi_relative_env import _MinimalChairmanMultiVecEnv
except ImportError:  # ``python config_run/main_multi_diff_obs.py ...``
    from SB3_chairman_multi_relative_env import _MinimalChairmanMultiVecEnv


class StableBaseline3VecEnv(_MinimalChairmanMultiVecEnv):
    """Compact observation using pelvis and chair poses in the world frame."""

    # pelvis position (3) + quaternion (4) + velocity (6)
    # + chair position (3) + quaternion (4)
    TASK_OBSERVATION_DIM = 20
    OBSERVATION_VARIANT = "minimal_absolute"

    def _task_observation_torch(
        self,
        pelvis_pos: torch.Tensor,
        pelvis_quat: torch.Tensor,
        pelvis_velocity: torch.Tensor,
        chair_pos: torch.Tensor,
        chair_quat: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                pelvis_pos,
                quat_unique(pelvis_quat),
                pelvis_velocity,
                chair_pos,
                quat_unique(chair_quat),
            ),
            dim=1,
        )


ChairmanMultiAbsoluteVecEnv = StableBaseline3VecEnv
