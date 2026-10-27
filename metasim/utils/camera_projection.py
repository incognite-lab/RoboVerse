"""Independent camera FOV axes on a renderer exposing only vertical FOV.

Render a square-pixel frustum covering both requested axes, then resample its
centered region into the requested output. No Full-HD intermediate is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math


def resolve_camera_fovs(width, height, *, horizontal=None, vertical=None,
                        legacy_vertical=None, default_horizontal=45.0):
    """Resolve degrees; one explicit axis derives the other for square pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    for name, value in (("fov_horizontal", horizontal), ("fov_vertical", vertical), ("fov", legacy_vertical)):
        if value is not None and (not math.isfinite(value) or not 0 < value < 180):
            raise ValueError(f"{name} must be finite and strictly between 0 and 180 degrees")
    if vertical is not None and legacy_vertical is not None:
        if not math.isclose(vertical, legacy_vertical, abs_tol=1e-6, rel_tol=0):
            raise ValueError("fov and fov_vertical disagree; remove fov or use the same vertical angle")
    vertical = vertical if vertical is not None else legacy_vertical
    if horizontal is None and vertical is None:
        horizontal = default_horizontal
    if vertical is None:
        vertical = math.degrees(2 * math.atan(math.tan(math.radians(horizontal / 2)) * height / width))
    if horizontal is None:
        horizontal = math.degrees(2 * math.atan(math.tan(math.radians(vertical / 2)) * width / height))
    return float(horizontal), float(vertical)


@dataclass(frozen=True)
class CameraProjection:
    width: int
    height: int
    horizontal_fov: float
    vertical_fov: float
    render_width: int
    render_height: int
    crop_x: float

    @property
    def needs_resampling(self):
        return ((self.render_width, self.render_height) != (self.width, self.height)
                or not math.isclose(self.crop_x, 1.0, abs_tol=1e-12, rel_tol=0))

    @property
    def intrinsics(self):
        return [
            [self.width / (2 * math.tan(math.radians(self.horizontal_fov / 2))), 0., self.width / 2],
            [0., self.height / (2 * math.tan(math.radians(self.vertical_fov / 2))), self.height / 2],
            [0., 0., 1.],
        ]

    def resample_rgb(self, rgb):
        """Map NHWC/HWC renderer RGB to output size, retaining device/dtype."""
        if not self.needs_resampling:
            return rgb
        import torch
        from torch.nn import functional as F

        if rgb.ndim not in (3, 4) or tuple(rgb.shape[-3:-1]) != (self.render_height, self.render_width):
            raise ValueError(f"Expected renderer RGB with height/width {(self.render_height, self.render_width)}")
        single = rgb.ndim == 3
        batch = rgb.unsqueeze(0) if single else rgb
        grid = _sampling_grid(self, rgb.device).expand(batch.shape[0], -1, -1, -1)
        sampled = F.grid_sample(batch.permute(0, 3, 1, 2).float(), grid,
                                mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1)
        if rgb.dtype == torch.uint8:
            sampled = sampled.round().clamp(0, 255)
        sampled = sampled.to(dtype=rgb.dtype).contiguous()
        return sampled[0] if single else sampled


@lru_cache(maxsize=32)
def _sampling_grid(projection, device):
    import torch
    from torch.nn import functional as F

    # align_corners=False samples the output pixel centers. Fractional crop_x
    # corrects integer rounding of the internal renderer width, so BOTH FOVs
    # describe the final projection, not merely an approximate aspect ratio.
    theta = torch.tensor([[[projection.crop_x, 0., 0.], [0., 1., 0.]]], device=device, dtype=torch.float32)
    return F.affine_grid(theta, (1, 3, projection.height, projection.width), align_corners=False)


def make_camera_projection(width, height, horizontal_fov, vertical_fov):
    horizontal_fov, vertical_fov = resolve_camera_fovs(
        width, height, horizontal=horizontal_fov, vertical=vertical_fov,
    )
    aspect = math.tan(math.radians(horizontal_fov / 2)) / math.tan(math.radians(vertical_fov / 2))
    # Keep at least one renderer sample per output pixel on each axis.
    render_height = max(height, math.ceil(width / aspect - 1e-9))
    render_width = max(width, math.ceil(render_height * aspect - 1e-9))
    if max(render_width, render_height) > 16384:
        raise ValueError("Requested FOV combination needs an internal camera dimension > 16384; reduce the aspect ratio")
    crop_x = min(1., aspect * render_height / render_width)
    return CameraProjection(width, height, horizontal_fov, vertical_fov, render_width, render_height, crop_x)
