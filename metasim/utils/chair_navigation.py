"""Shared planar geometry for the Chairman walking stage."""

from __future__ import annotations

import torch


# ``foldable_chair_debug.urdf`` has its handle/backrest on local +Y.  Both
# navigation targets are therefore placed on that side of the chair.
CHAIR_STAGING_DISTANCE = 1.50
CHAIR_FINAL_DISTANCE = 0.65
CHAIR_FINAL_TOLERANCE = 0.15


def _normalize_xy(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.norm(vector, dim=-1, keepdim=True), min=1.0e-6)


def chair_back_direction_xy(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return chair local +Y projected into the world XY plane."""
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    local_y_world = torch.stack(
        (2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z)),
        dim=-1,
    )
    return _normalize_xy(local_y_world)


def forward_direction_xy(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return body local +X projected into the world XY plane."""
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    local_x_world = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)),
        dim=-1,
    )
    return _normalize_xy(local_x_world)


def world_vector_to_body_xy(
    vector_world_xy: torch.Tensor,
    body_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Express a planar world vector in the body's yaw-aligned XY frame."""
    forward = forward_direction_xy(body_quaternion_wxyz)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    return torch.stack(
        (
            torch.sum(vector_world_xy * forward, dim=-1),
            torch.sum(vector_world_xy * left, dim=-1),
        ),
        dim=-1,
    )


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """C1-continuous interpolation from zero to one for input in [0, 1]."""
    value = torch.clamp(value, min=0.0, max=1.0)
    return value * value * (3.0 - 2.0 * value)
