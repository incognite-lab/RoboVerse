"""Sub-module containing the camera configuration."""

from __future__ import annotations

import math
from typing import Literal

from metasim.utils.configclass import configclass


def _tuple_or_none(value, length: int, name: str):
    if value is None:
        return None
    out = tuple(float(v) for v in value)
    if len(out) != length:
        raise ValueError(f"{name} must have {length} values, got {len(out)}")
    return out


@configclass
class BaseCameraCfg:
    """Base camera configuration."""

    name: str = "camera0"
    """Name of the camera. Defaults to "camera0". Different cameras should have different names, so if you add multiple cameras, make sure to give them unique names."""
    data_types: list[Literal["rgb", "depth", "instance_seg", "instance_id_seg"]] = ["rgb", "depth"]
    """List of sensor types to enable for the camera. Defaults to ["rgb", "depth"]."""
    width: int = 256
    """Width of the image in pixels. Defaults to 256."""
    height: int = 256
    """Height of the image in pixels. Defaults to 256."""
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Position of the camera in the world frame. Defaults to (0.0, 0.0, 1.0)."""
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look at point of the camera in the world frame. Defaults to (0.0, 0.0, 0.0)."""
    mount_to: str | tuple[str, str] | None = None
    """Mount the camera to a specific object or robot. Defaults to None."""
    mount_link: str | tuple[str, str] | None = None
    """Specify the link to mount the camera to. Defaults to None."""
    mount_pos: tuple[float, float, float] | None = None
    """Position of the camera on the mount. Defaults to None."""
    mount_quat: tuple[float, float, float, float] | None = None
    """Quaternion of the camera on the mount. Defaults to None."""

    def __post_init__(self):
        self.pos = _tuple_or_none(self.pos, 3, "pos")
        self.look_at = _tuple_or_none(self.look_at, 3, "look_at")
        self.mount_pos = _tuple_or_none(self.mount_pos, 3, "mount_pos")
        self.mount_quat = _tuple_or_none(self.mount_quat, 4, "mount_quat")


@configclass
class PinholeCameraCfg(BaseCameraCfg):
    """Pinhole camera configuration."""

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0 cm."""
    focus_distance: float = 400.0
    """Distance from the camera to the focus plane (in m). Defaults to 400.0."""
    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm."""
    clipping_range: tuple[float, float] = (0.05, 1e5)
    """Near and far clipping distances (in m). Defaults to (0.05, 1e5)."""

    def __post_init__(self):
        super().__post_init__()
        self.clipping_range = _tuple_or_none(self.clipping_range, 2, "clipping_range")

    @property
    def vertical_aperture(self) -> float:
        """Vertical aperture (in cm)."""
        return self.horizontal_aperture * self.height / self.width

    @property
    def horizontal_fov(self) -> float:
        """Horizontal field of view, in degrees."""
        return 2 * math.atan(self.horizontal_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view, in degrees."""
        return 2 * math.atan(self.vertical_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def intrinsics(self) -> list[list[float]]:
        """Intrinsics matrix of the camera. Type is 3x3 nested list of floats."""
        fx = self.width * self.focal_length / self.horizontal_aperture
        fy = self.height * self.focal_length / self.vertical_aperture
        cx = self.width * 0.5
        cy = self.height * 0.5
        return [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]


@configclass
class NyxGaussianSplatCameraCfg(PinholeCameraCfg):
    """Genesis/Nyx camera that sees a Gaussian splat."""

    gaussian_splat_path: str = ""
    gaussian_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaussian_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    gaussian_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gaussian_z_up: bool = True
    """Splat transform in Genesis/Nerfstudio Z-up coordinates."""
    fov: float | None = None
    """Optional vertical field of view in degrees. Defaults to PinholeCameraCfg.vertical_fov."""
    spp: int = 16
    render_mode: str = "FastPathTracer"
    tone_mapper: str | None = "Off"
    render_sim_geometry: bool = True
    standalone_floor: bool = False
    standalone_floor_pos: tuple[float, float, float] = (0.0, 0.0, -0.03)
    standalone_floor_size: tuple[float, float, float] = (20.0, 20.0, 0.04)
    standalone_floor_color: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 1.0)
    standalone_light_dir: tuple[float, float, float] = (-0.4, -0.5, -1.0)
    standalone_light_intensity: float = 5.0
    standalone_metasim_objects: bool = False
    standalone_metasim_robot: bool = False
    open_nyx_window: bool = False
    debug_image_stats: bool = True
    debug_show: bool = False
    debug_show_all_envs: bool = False
    debug_calibrate: bool = False
    debug_detach_from_mount: bool = False
    debug_control_target: str = "camera"
    debug_translation_step: float = 0.05
    debug_rotation_step_deg: float = 5.0
    debug_scale_step: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.gaussian_position = _tuple_or_none(self.gaussian_position, 3, "gaussian_position")
        self.gaussian_rotation_xyzw = _tuple_or_none(self.gaussian_rotation_xyzw, 4, "gaussian_rotation_xyzw")
        self.gaussian_scale = _tuple_or_none(self.gaussian_scale, 3, "gaussian_scale")
        self.standalone_floor_pos = _tuple_or_none(self.standalone_floor_pos, 3, "standalone_floor_pos")
        self.standalone_floor_size = _tuple_or_none(self.standalone_floor_size, 3, "standalone_floor_size")
        self.standalone_floor_color = _tuple_or_none(self.standalone_floor_color, 4, "standalone_floor_color")
        self.standalone_light_dir = _tuple_or_none(self.standalone_light_dir, 3, "standalone_light_dir")
        if not self.gaussian_splat_path:
            raise ValueError("gaussian_splat_path must point to a .ply or .spz file")
        if self.debug_control_target not in ("camera", "splat"):
            raise ValueError("debug_control_target must be 'camera' or 'splat'")


REALSENSE_CAMERA = PinholeCameraCfg(
    name="realsense_camera",
    data_types=["rgb", "depth"],
    width=640,
    height=360,
    focal_length=1.88,
    horizontal_aperture=float(2 * 1.88 * math.tan(71.28 / 180 * math.pi / 2)),
)
