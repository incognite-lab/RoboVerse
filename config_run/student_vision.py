"""Shared aspect-preserving center crop and area resize for DAgger students.

Simulation already supplies RGB; deployment additionally converts BGR to RGB.
Area integration runs on the tensor's device, without a GPU -> CPU round trip.
"""
from functools import lru_cache
import math

import torch


def image_size_from_config(config):
    """Return (height, width); omitted settings preserve the 128x128 pipeline."""
    size = (int(config.get("dagger_image_height", 128)), int(config.get("dagger_image_width", 128)))
    if min(size) < 32:
        raise ValueError("dagger_image_height/width must be >= 32")
    return size


def center_crop_bounds(height, width, output_size):
    """Crop to the target aspect ratio without stretching the camera image."""
    out_height, out_width = output_size
    if min(height, width, out_height, out_width) <= 0:
        raise ValueError("Image and output dimensions must be positive")
    if width * out_height > height * out_width:
        crop_height, crop_width = height, max(1, height * out_width // out_height)
    else:
        crop_height, crop_width = max(1, width * out_height // out_width), width
    return ((height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width)


@lru_cache(maxsize=32)
def _area_samples(source, target, device):
    # Pixel footprints, rather than adaptive_avg_pool2d's integer bins: for
    # 360 -> 128 the latter does NOT reproduce OpenCV INTER_AREA.
    scale = source / target
    left = torch.arange(target, dtype=torch.float64) * scale
    right = (torch.arange(target, dtype=torch.float64) + 1) * scale
    start = left.floor().to(torch.long)
    samples = []
    for offset in range(math.ceil(scale) + 1):
        index = start + offset
        overlap = (torch.minimum(right, index + 1)
                   - torch.maximum(left, index)).clamp(min=0) / scale
        samples.append((index.clamp(0, source - 1).to(device),
                        overlap.to(device=device, dtype=torch.float32)))
    return samples


def _resize_axis(image, axis, target):
    source = image.shape[axis]
    if source == target:
        return image
    shape = [1] * image.ndim
    shape[axis] = target
    result = None
    for indices, weights in _area_samples(source, target, image.device):
        term = image.index_select(axis, indices).float() * weights.view(shape)
        result = term if result is None else result + term
    return result


def preprocess_student_rgb(rgb, device=None, output_size=(128, 128)):
    """NHWC RGB(A) -> NCHW uint8 RGB at the configured (height, width).

    Same crop and area filtering as deploy_dagger_student_real.preprocess_frame.
    uint8 rounding can differ from OpenCV by one intensity level.
    """
    if rgb.ndim != 4 or rgb.shape[-1] < 3 or min(rgb.shape[1:3]) <= 0:
        raise ValueError(f"Expected nonempty NHWC RGB camera tensor, got {tuple(rgb.shape)}")
    if rgb.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 camera pixels, got {rgb.dtype}")
    if tuple(rgb.shape[1:3]) == tuple(output_size):
        # Full-resolution path: only RGB channels, device and tensor layout.
        return rgb[..., :3].to(device=device).permute(0, 3, 1, 2).contiguous()
    y0, x0, height, width = center_crop_bounds(*rgb.shape[1:3], output_size)
    image = rgb[:, y0:y0 + height, x0:x0 + width, :3].to(device=device)
    image = _resize_axis(image, 1, output_size[0])
    image = _resize_axis(image, 2, output_size[1])
    if image.dtype != torch.uint8:
        image = image.round().clamp(0, 255).to(torch.uint8)
    return image.permute(0, 3, 1, 2).contiguous()
