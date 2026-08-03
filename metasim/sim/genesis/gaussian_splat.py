from __future__ import annotations

import importlib
import os
import xml.etree.ElementTree as ET

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg, _FileBasedMixin
from metasim.cfg.sensors import NyxGaussianSplatCameraCfg

try:
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
except ImportError:
    npr = None
    nps = None
    NyxCameraOptions = None

KEY_LEFT = 2424832
KEY_UP = 2490368
KEY_RIGHT = 2555904
KEY_DOWN = 2621440


def uses_gaussian_splat_camera(cameras) -> bool:
    return any(isinstance(camera, NyxGaussianSplatCameraCfg) for camera in cameras)


def ensure_nyx_available() -> None:
    if NyxCameraOptions is None or nps is None or npr is None:
        raise ImportError(
            "NyxGaussianSplatCameraCfg requires the Genesis Nyx plugin. "
            "Install it with `pip install gs-nyx-plugin` or use PinholeCameraCfg."
        )


def _patch_genesis_engine_namespace_for_nyx_import() -> None:
    engine_mod = importlib.import_module("genesis.engine")
    entities_mod = importlib.import_module("genesis.engine.entities")
    base_entity_mod = importlib.import_module("genesis.engine.entities.base_entity")

    setattr(gs, "engine", engine_mod)
    setattr(engine_mod, "entities", entities_mod)
    setattr(entities_mod, "base_entity", base_entity_mod)


def patch_nyx_renderer_for_genesis_vgeom_api() -> None:
    try:
        _patch_genesis_engine_namespace_for_nyx_import()
        from gs_nyx_plugin.nyx_renderer import NyxPyRenderer
    except Exception:
        return

    if getattr(NyxPyRenderer, "_metasim_vgeom_api_patch", False):
        return

    original_update_geometry_tensors = NyxPyRenderer._update_geometry_tensors

    def _metasim_update_geometry_tensors(self, env_index):
        if getattr(self, "_num_rigid_geoms", 0) == 0 and getattr(self, "_num_deform_verts", 0) == 0:
            return

        if hasattr(self._rigid_solver, "vgeoms_state"):
            return original_update_geometry_tensors(self, env_index)

        if not (
            hasattr(self._rigid_solver, "get_vgeoms_pos")
            and hasattr(self._rigid_solver, "get_vgeoms_quat")
        ):
            return original_update_geometry_tensors(self, env_index)

        geom_pos_tensor = self._rigid_solver.get_vgeoms_pos()
        geom_rot_tensor = self._rigid_solver.get_vgeoms_quat()

        if geom_pos_tensor.dim() == 2:
            geom_pos_tensor = geom_pos_tensor.unsqueeze(0)
        if geom_rot_tensor.dim() == 2:
            geom_rot_tensor = geom_rot_tensor.unsqueeze(0)

        self._geom_pos_tensor_cuda.copy_(geom_pos_tensor[env_index].to(torch.float32))
        self._geom_rot_tensor_cuda.copy_(geom_rot_tensor[env_index].to(torch.float32))

    NyxPyRenderer._update_geometry_tensors = _metasim_update_geometry_tensors
    NyxPyRenderer._metasim_vgeom_api_patch = True


def make_nyx_compatible_urdf(urdf_path: str) -> str:
    if not urdf_path or not urdf_path.endswith(".urdf"):
        return urdf_path

    src_path = os.path.abspath(urdf_path)
    if not os.path.exists(src_path):
        return urdf_path

    src_dir = os.path.dirname(src_path)
    tree = ET.parse(src_path)
    changed = False

    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if (
            os.path.isabs(filename)
            or filename.startswith("package://")
            or filename.startswith("http://")
            or filename.startswith("https://")
        ):
            continue

        mesh.set("filename", os.path.abspath(os.path.join(src_dir, filename)))
        changed = True

    if not changed:
        return urdf_path

    cache_dir = os.path.join("/tmp", "metasim_nyx_urdf")
    os.makedirs(cache_dir, exist_ok=True)
    out_name = src_path.strip(os.sep).replace(os.sep, "__")
    out_path = os.path.join(cache_dir, out_name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def prepare_urdfs_for_nyx(robot, objects, cameras) -> None:
    if not uses_gaussian_splat_camera(cameras):
        return

    robot.urdf_path = make_nyx_compatible_urdf(robot.urdf_path)
    for obj in objects:
        if isinstance(obj, _FileBasedMixin) and obj.urdf_path:
            obj.urdf_path = make_nyx_compatible_urdf(obj.urdf_path)


def _patch_one_nyx_rigid_solver(rigid_solver) -> bool:
    if rigid_solver is None:
        return False

    if not hasattr(rigid_solver, "vgeoms_state") and hasattr(rigid_solver, "_init_vgeom_fields"):
        try:
            rigid_solver._init_vgeom_fields()
        except Exception:
            pass

    if not hasattr(rigid_solver, "vverts_state") and hasattr(rigid_solver, "_init_vvert_fields"):
        try:
            rigid_solver._init_vvert_fields()
        except Exception:
            pass

    if hasattr(rigid_solver, "vgeoms_state"):
        if hasattr(rigid_solver, "update_vgeoms"):
            try:
                rigid_solver.update_vgeoms()
            except Exception:
                pass
        return True

    source = None
    for candidate_name in ("data_manager", "_data_manager", "_data"):
        candidate = getattr(rigid_solver, candidate_name, None)
        if candidate is not None and hasattr(candidate, "vgeoms_state"):
            source = candidate
            break

    if source is None:
        nested_solver = getattr(rigid_solver, "_solver", None)
        for candidate_name in ("data_manager", "_data_manager", "_data"):
            candidate = getattr(nested_solver, candidate_name, None)
            if candidate is not None and hasattr(candidate, "vgeoms_state"):
                source = candidate
                break

    if source is None:
        return False

    for attr in ("vgeoms_state", "vgeoms_info", "vverts_state", "vverts_info", "vfaces_info"):
        if not hasattr(rigid_solver, attr) and hasattr(source, attr):
            setattr(rigid_solver, attr, getattr(source, attr))
    if hasattr(rigid_solver, "update_vgeoms"):
        try:
            rigid_solver.update_vgeoms()
        except Exception:
            pass
    return hasattr(rigid_solver, "vgeoms_state")


def patch_nyx_rigid_solver_compat(scene_inst, camera_inst=None) -> None:
    patch_nyx_renderer_for_genesis_vgeom_api()

    patched = _patch_one_nyx_rigid_solver(getattr(scene_inst, "rigid_solver", None))

    sensor_manager = getattr(camera_inst, "_manager", None)
    sim = getattr(sensor_manager, "_sim", None)
    scene = getattr(sim, "scene", None)
    patched = _patch_one_nyx_rigid_solver(getattr(scene, "rigid_solver", None)) or patched

    shared_metadata = getattr(camera_inst, "_shared_metadata", None)
    renderer = getattr(shared_metadata, "renderer", None)
    renderer_solver = getattr(renderer, "_rigid_solver", None)
    patched = _patch_one_nyx_rigid_solver(renderer_solver) or patched

    renderer_has_public_vgeom_api = (
        renderer_solver is not None
        and hasattr(renderer_solver, "get_vgeoms_pos")
        and hasattr(renderer_solver, "get_vgeoms_quat")
    )
    if (
        camera_inst is not None
        and renderer_solver is not None
        and not hasattr(renderer_solver, "vgeoms_state")
        and not renderer_has_public_vgeom_api
    ):
        solver_attrs = [attr for attr in dir(renderer_solver) if "vgeom" in attr or "data" in attr]
        raise AttributeError(
            "Could not expose vgeoms_state for gs-nyx-plugin. "
            f"Rigid solver attrs containing vgeom/data: {solver_attrs}"
        )


class NyxGaussianSplatRuntime:
    def __init__(self):
        self.splat_assets: dict[str, object] = {}
        self.splat_z_up: dict[str, bool] = {}
        self.standalone_scenes: dict[str, object] = {}
        self.standalone_entities: dict[str, dict[str, object]] = {}
        self.camera_mounts: dict[str, dict[str, object]] = {}
        self.calibration_state: dict[str, dict[str, tuple[float, ...]]] = {}
        self.camera_debug_state: dict[str, dict[str, tuple[float, ...]]] = {}
        self.debug_control_targets: dict[str, str] = {}
        self.debug_help_printed: set[str] = set()
        self.debug_frame_counts: dict[str, int] = {}

    def _add_standalone_floor(self, scene_inst, camera: NyxGaussianSplatCameraCfg) -> None:
        if not camera.standalone_floor:
            return

        scene_inst.add_entity(
            gs.morphs.Box(
                pos=camera.standalone_floor_pos,
                size=camera.standalone_floor_size,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=camera.standalone_floor_color, roughness=0.8),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

    @staticmethod
    def _scalar_scale(scale) -> float:
        if isinstance(scale, (tuple, list)):
            return float(scale[0])
        return float(scale)

    def _add_standalone_robot(self, scene_inst, robot) -> None:
        if not getattr(robot, "urdf_path", None):
            log.warning("Nyx standalone robot requested, but robot has no urdf_path.")
            return None

        return scene_inst.add_entity(
            gs.morphs.URDF(
                file=robot.urdf_path,
                pos=getattr(robot, "default_position", (0.0, 0.0, 0.0)),
                quat=getattr(robot, "default_orientation", (1.0, 0.0, 0.0, 0.0)),
                merge_fixed_links=getattr(robot, "collapse_fixed_joints", False),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

    def _add_standalone_object(self, scene_inst, obj) -> None:
        pos = getattr(obj, "default_position", (0.0, 0.0, 0.0))
        quat = getattr(obj, "default_orientation", (1.0, 0.0, 0.0, 0.0))

        if isinstance(obj, PrimitiveCubeCfg):
            return scene_inst.add_entity(
                gs.morphs.Box(pos=pos, quat=quat, size=obj.size, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=obj.color),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
        elif isinstance(obj, PrimitiveSphereCfg):
            return scene_inst.add_entity(
                gs.morphs.Sphere(pos=pos, quat=quat, radius=obj.radius, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=obj.color),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
        elif isinstance(obj, RigidObjCfg):
            if not obj.urdf_path:
                log.warning(f"Nyx standalone object '{obj.name}' has no urdf_path, skipping.")
                return None
            return scene_inst.add_entity(
                gs.morphs.URDF(
                    file=obj.urdf_path,
                    pos=pos,
                    quat=quat,
                    fixed=True,
                    collision=False,
                    scale=self._scalar_scale(obj.scale),
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
        elif isinstance(obj, ArticulationObjCfg):
            if not obj.urdf_path:
                log.warning(f"Nyx standalone object '{obj.name}' has no urdf_path, skipping.")
                return None
            return scene_inst.add_entity(
                gs.morphs.URDF(
                    file=obj.urdf_path,
                    pos=pos,
                    quat=quat,
                    fixed=True,
                    collision=False,
                    scale=self._scalar_scale(obj.scale),
                    merge_fixed_links=obj.colapse_fixed_joints,
                    batch_fixed_verts=obj.batch_fixed_verts,
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
        else:
            log.warning(f"Nyx standalone scene does not support object type {type(obj)}, skipping {obj.name}.")
            return None

    def _add_standalone_metasim_geometry(self, scene_inst, camera: NyxGaussianSplatCameraCfg, robot, objects) -> None:
        entity_map = self.standalone_entities.setdefault(camera.name, {})
        if camera.standalone_metasim_objects:
            for obj in objects or []:
                entity = self._add_standalone_object(scene_inst, obj)
                if entity is not None:
                    entity_map[obj.name] = entity
        if camera.standalone_metasim_robot and robot is not None:
            entity = self._add_standalone_robot(scene_inst, robot)
            if entity is not None:
                entity_map[robot.name] = entity

    @staticmethod
    def _first_env(tensor):
        if tensor is None:
            return None
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if getattr(tensor, "dim", lambda: 0)() > 1:
            tensor = tensor[0]
        return tensor

    @staticmethod
    def _quat_rotate_wxyz(quat, vec) -> torch.Tensor:
        quat_t = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
        vec_t = torch.as_tensor(vec, dtype=gs.tc_float, device=gs.device)
        quat_t = quat_t / torch.clamp(torch.linalg.norm(quat_t), min=1e-8)
        w, x, y, z = quat_t.unbind(-1)
        q_vec = torch.stack((x, y, z))
        uv = torch.cross(q_vec, vec_t, dim=0)
        uuv = torch.cross(q_vec, uv, dim=0)
        return vec_t + 2.0 * (w * uv + uuv)

    @staticmethod
    def _quat_multiply_wxyz(lhs, rhs) -> torch.Tensor:
        lhs_t = torch.as_tensor(lhs, dtype=gs.tc_float, device=gs.device)
        rhs_t = torch.as_tensor(rhs, dtype=gs.tc_float, device=gs.device)
        lw, lx, ly, lz = lhs_t.unbind(-1)
        rw, rx, ry, rz = rhs_t.unbind(-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            )
        )

    def _mounted_camera_pose(self, camera_name: str):
        mount = self.camera_mounts.get(camera_name)
        if mount is None:
            return None

        link = mount["link"]
        link_pos = self._first_env(link.get_pos(envs_idx=[0]))
        link_quat = self._first_env(link.get_quat(envs_idx=[0]))
        mount_pos = mount["pos"]
        mount_quat = mount["quat"]

        cam_pos_t = link_pos + self._quat_rotate_wxyz(link_quat, mount_pos)
        cam_quat = self._quat_multiply_wxyz(link_quat, mount_quat)
        forward = self._quat_rotate_wxyz(cam_quat, (1.0, 0.0, 0.0))
        up = self._quat_rotate_wxyz(cam_quat, (0.0, 0.0, 1.0))
        lookat_t = cam_pos_t + forward
        return (
            tuple(float(v) for v in cam_pos_t.detach().cpu().tolist()),
            tuple(float(v) for v in lookat_t.detach().cpu().tolist()),
            tuple(float(v) for v in up.detach().cpu().tolist()),
        )

    def update_mounted_camera_pose(self, camera_inst, camera_name: str) -> None:
        pose = self._mounted_camera_pose(camera_name)
        if pose is None:
            return
        pos, lookat, up = pose
        camera_inst.update_camera_pose(pos=pos, lookat=lookat, up=up)

    def _sync_standalone_entity(self, source_entity, target_entity) -> None:
        try:
            target_entity.set_pos(
                self._first_env(source_entity.get_pos(envs_idx=[0])),
                zero_velocity=True,
                skip_forward=True,
            )
            target_entity.set_quat(
                self._first_env(source_entity.get_quat(envs_idx=[0])),
                zero_velocity=True,
                skip_forward=True,
            )
        except Exception as exc:
            log.warning(f"Could not sync standalone root pose for {source_entity}: {exc}")

        try:
            if getattr(source_entity, "n_qs", None) == getattr(target_entity, "n_qs", None):
                target_entity.set_qpos(
                    self._first_env(source_entity.get_qpos(envs_idx=[0])),
                    zero_velocity=True,
                    skip_forward=False,
                )
            elif getattr(source_entity, "n_dofs", 0) == getattr(target_entity, "n_dofs", 0) and source_entity.n_dofs > 0:
                target_entity.set_dofs_position(
                    self._first_env(source_entity.get_dofs_position(envs_idx=[0])),
                    zero_velocity=True,
                )
        except Exception as exc:
            log.warning(f"Could not sync standalone joint state for {source_entity}: {exc}")

    def sync_standalone_scene(self, camera_name: str, source_entities: dict[str, object]) -> None:
        target_entities = self.standalone_entities.get(camera_name, {})
        for name, target_entity in target_entities.items():
            source_entity = source_entities.get(name)
            if source_entity is not None:
                self._sync_standalone_entity(source_entity, target_entity)

    def make_camera(
        self,
        scene_inst,
        camera: NyxGaussianSplatCameraCfg,
        mount_entity=None,
        mount_link=None,
        robot=None,
        objects=None,
    ):
        ensure_nyx_available()
        patch_nyx_renderer_for_genesis_vgeom_api()

        splat_path = os.path.abspath(camera.gaussian_splat_path)
        if not os.path.exists(splat_path):
            raise FileNotFoundError(f"Gaussian splat file not found: {splat_path}")

        splat = nps.LightFieldAsset()
        splat.type = nps.ELightFieldType.GaussianField
        splat.uri = splat_path

        splat_position = nps.float3(*camera.gaussian_position)
        splat_rotation = nps.quaternion(*camera.gaussian_rotation_xyzw)
        splat_scale = nps.float3(*camera.gaussian_scale)
        if camera.gaussian_z_up:
            splat_position = nps.float3_z_up_to_y_up_a(splat_position)
            splat_rotation = nps.quaternion_z_up_to_y_up_a(splat_rotation)
            splat_scale = nps.float3_z_up_to_y_up_a(splat_scale)

        splat.position = splat_position
        splat.rotation = splat_rotation
        splat.scale = splat_scale

        pos = np.array(camera.pos, dtype=np.float64)
        lookat = np.array(camera.look_at, dtype=np.float64)
        if np.linalg.norm(lookat - pos) < 1e-8:
            lookat = pos + np.array([1.0, 0.0, 0.0], dtype=np.float64)
            log.warning(
                f"Nyx camera '{camera.name}' had identical pos and look_at; "
                f"using look_at={tuple(float(v) for v in lookat)}."
            )

        options = {
            "res": (camera.width, camera.height),
            "fov": camera.fov if camera.fov is not None else camera.vertical_fov,
            "near": camera.clipping_range[0],
            "far": camera.clipping_range[1],
            "spp": camera.spp,
            "render_mode": getattr(npr.ERenderMode, camera.render_mode),
            "light_fields": (splat,),
            "open_window": camera.open_nyx_window,
            "window_size": (camera.width, camera.height),
        }
        if camera.standalone_floor:
            options["lights"] = [
                {
                    "type": "directional",
                    "dir": camera.standalone_light_dir,
                    "color": (1.0, 1.0, 1.0),
                    "intensity": camera.standalone_light_intensity,
                    "shadow": False,
                },
            ]

        if camera.tone_mapper is not None:
            tone_mapper = "Off" if camera.tone_mapper is False else camera.tone_mapper
            options["tone_mapper"] = getattr(nps.EToneMapper, tone_mapper)

        if mount_entity is not None and mount_link is not None:
            self.camera_mounts[camera.name] = {
                "entity": mount_entity,
                "link": mount_link,
                "pos": tuple(float(v) for v in (camera.mount_pos or (0.05, 0.0, 0.0))),
                "quat": tuple(float(v) for v in (camera.mount_quat or (1.0, 0.0, 0.0, 0.0))),
            }

        if mount_entity is None or mount_link is None:
            options["pos"] = tuple(float(v) for v in pos)
            options["lookat"] = tuple(float(v) for v in lookat)
        elif not camera.render_sim_geometry:
            options["pos"] = tuple(float(v) for v in pos)
            options["lookat"] = tuple(float(v) for v in lookat)

        if not camera.render_sim_geometry:
            # First-step viewer mode: match config_run/view_gaussian_splat_genesis.py exactly.
            # We keep the splat camera in an empty Genesis scene so Nyx does not export or
            # render MetaSim robot/ground geometry over the Gaussian field.
            standalone_scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.01),
                show_viewer=False,
            )
            self._add_standalone_floor(standalone_scene, camera)
            self._add_standalone_metasim_geometry(standalone_scene, camera, robot, objects)
            camera_inst = standalone_scene.add_sensor(NyxCameraOptions(**options))
            standalone_scene.build(n_envs=1)
            self.standalone_scenes[camera.name] = standalone_scene
        elif mount_entity is not None and mount_link is not None:
            pos_t = torch.tensor(camera.mount_pos or (0.05, 0.0, 0.0), dtype=gs.tc_float, device=gs.device)
            quat_t = torch.tensor(camera.mount_quat or (1.0, 0.0, 0.0, 0.0), dtype=gs.tc_float, device=gs.device)
            options["entity_idx"] = mount_entity.idx
            options["link_idx_local"] = mount_link.idx_local
            options["offset_T"] = gu.trans_quat_to_T(pos_t, quat_t).detach().cpu().numpy()
            camera_inst = scene_inst.add_sensor(NyxCameraOptions(**options))
        else:
            camera_inst = scene_inst.add_sensor(NyxCameraOptions(**options))

        self.splat_assets[camera.name] = splat
        self.splat_z_up[camera.name] = camera.gaussian_z_up
        self.calibration_state[camera.name] = {
            "position": tuple(float(v) for v in camera.gaussian_position),
            "rotation_xyzw": tuple(float(v) for v in camera.gaussian_rotation_xyzw),
            "scale": tuple(float(v) for v in camera.gaussian_scale),
        }
        self.camera_debug_state[camera.name] = {
            "pos": tuple(float(v) for v in pos),
            "lookat": tuple(float(v) for v in lookat),
            "up": (0.0, 0.0, 1.0),
        }
        self.debug_control_targets[camera.name] = camera.debug_control_target
        return camera_inst

    def step_standalone_scene(
        self,
        camera_name: str,
        source_entities: dict[str, object] | None = None,
        camera_inst=None,
    ) -> None:
        scene = self.standalone_scenes.get(camera_name)
        if scene is not None:
            if source_entities is not None:
                self.sync_standalone_scene(camera_name, source_entities)
            if camera_inst is not None:
                self.update_mounted_camera_pose(camera_inst, camera_name)
            scene.step()

    @staticmethod
    def _quat_multiply_xyzw(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]):
        lx, ly, lz, lw = lhs
        rx, ry, rz, rw = rhs
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    @staticmethod
    def _axis_angle_quat_xyzw(axis: tuple[float, float, float], angle_rad: float):
        half = angle_rad * 0.5
        sin_half = np.sin(half)
        return (
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half),
        )

    @staticmethod
    def _normalize_quat_xyzw(quat: tuple[float, float, float, float]):
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(float(v / norm) for v in quat)

    def _set_splat_transform(
        self,
        camera_name: str,
        position: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float],
        scale: tuple[float, float, float],
    ) -> None:
        splat = self.splat_assets.get(camera_name)
        if splat is None:
            return

        splat_position = nps.float3(*position)
        splat_rotation = nps.quaternion(*rotation_xyzw)
        splat_scale = nps.float3(*scale)
        if self.splat_z_up.get(camera_name, True):
            splat_position = nps.float3_z_up_to_y_up_a(splat_position)
            splat_rotation = nps.quaternion_z_up_to_y_up_a(splat_rotation)
            splat_scale = nps.float3_z_up_to_y_up_a(splat_scale)

        splat.position = splat_position
        splat.rotation = splat_rotation
        splat.scale = splat_scale

        self.calibration_state[camera_name] = {
            "position": position,
            "rotation_xyzw": rotation_xyzw,
            "scale": scale,
        }

    def print_calibration_state(self, camera_name: str) -> None:
        state = self.calibration_state[camera_name]
        camera_state = self.camera_debug_state.get(camera_name)
        pos = [round(v, 4) for v in state["position"]]
        rot = [round(v, 6) for v in state["rotation_xyzw"]]
        scale = [round(v, 4) for v in state["scale"]]
        print(
            f"[Nyx calibration {camera_name}] "
            f"gaussian_position: {pos}, gaussian_rotation_xyzw: {rot}, gaussian_scale: {scale}"
        )
        if camera_state is not None:
            cam_pos = [round(v, 4) for v in camera_state["pos"]]
            cam_lookat = [round(v, 4) for v in camera_state["lookat"]]
            print(f"[Nyx calibration {camera_name}] camera pos: {cam_pos}, look_at: {cam_lookat}")

    def _handle_camera_key(self, camera_inst, camera: NyxGaussianSplatCameraCfg, key: int) -> bool:
        camera_name = camera.name
        state = self.camera_debug_state.get(camera_name)
        if state is None:
            return False

        key_char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) <= 255 else ""
        pos = np.array(state["pos"], dtype=np.float64)
        lookat = np.array(state["lookat"], dtype=np.float64)
        world_up = np.array(state.get("up", (0.0, 0.0, 1.0)), dtype=np.float64)
        forward = lookat - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            lookat = pos + forward
        else:
            forward = forward / forward_norm

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        else:
            right = right / right_norm

        trans_step = float(camera.debug_translation_step)
        rot_step = float(np.deg2rad(camera.debug_rotation_step_deg))
        moved = False

        move_vectors = {
            "w": forward * trans_step,
            "s": -forward * trans_step,
            "a": -right * trans_step,
            "d": right * trans_step,
            "r": world_up * trans_step,
            "f": -world_up * trans_step,
        }
        arrow_moves = {
            KEY_UP: forward * trans_step,
            KEY_DOWN: -forward * trans_step,
        }
        if key_char in move_vectors:
            delta = move_vectors[key_char]
            pos += delta
            lookat += delta
            moved = True
        elif key in arrow_moves:
            delta = arrow_moves[key]
            pos += delta
            lookat += delta
            moved = True
        elif key_char in ("j", "l", "i", "k") or key in (KEY_LEFT, KEY_RIGHT):
            yaw = (
                rot_step
                if key_char == "j" or key == KEY_LEFT
                else -rot_step
                if key_char == "l" or key == KEY_RIGHT
                else 0.0
            )
            pitch = rot_step if key_char == "i" else -rot_step if key_char == "k" else 0.0
            direction = lookat - pos
            if abs(yaw) > 0.0:
                c, s = np.cos(yaw), np.sin(yaw)
                rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                direction = rot_z @ direction
            if abs(pitch) > 0.0:
                direction = (
                    direction * np.cos(pitch)
                    + np.cross(right, direction) * np.sin(pitch)
                    + right * np.dot(right, direction) * (1.0 - np.cos(pitch))
                )
            lookat = pos + direction
            moved = True

        if not moved:
            return False

        pos_tuple = tuple(float(v) for v in pos)
        lookat_tuple = tuple(float(v) for v in lookat)
        up_tuple = tuple(float(v) for v in world_up)
        self.camera_debug_state[camera_name] = {"pos": pos_tuple, "lookat": lookat_tuple, "up": up_tuple}
        camera_inst.update_camera_pose(pos=pos_tuple, lookat=lookat_tuple, up=up_tuple)
        self.print_calibration_state(camera_name)
        return True

    def _handle_debug_key(self, camera_inst, camera: NyxGaussianSplatCameraCfg, key: int) -> None:
        if key < 0:
            return

        camera_name = camera.name
        key_char = chr(key & 0xFF).lower()
        if key_char == "p":
            self.print_calibration_state(camera_name)
            return
        if key_char == "t":
            current = self.debug_control_targets.get(camera_name, camera.debug_control_target)
            self.debug_control_targets[camera_name] = "splat" if current == "camera" else "camera"
            print(f"[Nyx calibration {camera_name}] control target: {self.debug_control_targets[camera_name]}")
            return

        control_target = self.debug_control_targets.get(camera_name, camera.debug_control_target)
        if control_target == "camera" and self._handle_camera_key(camera_inst, camera, key):
            return

        state = self.calibration_state.get(camera_name)
        if state is None:
            return

        pos = list(state["position"])
        rot = state["rotation_xyzw"]
        scale = list(state["scale"])
        trans_step = float(camera.debug_translation_step)
        rot_step = float(np.deg2rad(camera.debug_rotation_step_deg))
        scale_step = float(camera.debug_scale_step)
        changed = False

        move_keys = {
            "a": (1, -trans_step),
            "d": (1, trans_step),
            "w": (0, trans_step),
            "s": (0, -trans_step),
            "r": (2, trans_step),
            "f": (2, -trans_step),
        }
        rotation_keys = {
            "j": ((0.0, 0.0, 1.0), rot_step),
            "l": ((0.0, 0.0, 1.0), -rot_step),
            "i": ((0.0, 1.0, 0.0), rot_step),
            "k": ((0.0, 1.0, 0.0), -rot_step),
            "u": ((1.0, 0.0, 0.0), rot_step),
            "o": ((1.0, 0.0, 0.0), -rot_step),
        }

        if key_char in move_keys:
            axis, delta = move_keys[key_char]
            pos[axis] += delta
            changed = True
        elif key_char in rotation_keys:
            axis, angle = rotation_keys[key_char]
            delta_quat = self._axis_angle_quat_xyzw(axis, angle)
            rot = self._normalize_quat_xyzw(self._quat_multiply_xyzw(delta_quat, rot))
            changed = True
        elif key_char in ("=", "+"):
            scale = [v * (1.0 + scale_step) for v in scale]
            changed = True
        elif key_char in ("-", "_"):
            scale = [v * max(0.01, 1.0 - scale_step) for v in scale]
            changed = True
        if not changed:
            return

        self._set_splat_transform(camera_name, tuple(pos), rot, tuple(scale))
        self.print_calibration_state(camera_name)

    def show_debug_frame(self, camera_inst, camera: NyxGaussianSplatCameraCfg, rgb: torch.Tensor) -> None:
        if not (camera.debug_show or camera.debug_calibrate):
            return

        try:
            import cv2
        except ImportError:
            log.warning("Nyx debug_show/debug_calibrate requires opencv-python (`import cv2` failed).")
            return

        if camera.name not in self.debug_help_printed:
            print(
                f"[Nyx calibration {camera.name}] controls: "
                "W/S or Up/Down forward/back, A/D left/right, R/F up/down, "
                "J/L or Left/Right yaw, I/K pitch, T toggle camera/splat, P print values."
            )
            self.debug_help_printed.add(camera.name)

        frames = rgb if rgb.dim() == 4 else rgb.unsqueeze(0)
        frames = frames.detach().cpu()
        frame_count = self.debug_frame_counts.get(camera.name, 0)
        self.debug_frame_counts[camera.name] = frame_count + 1
        if camera.debug_image_stats and frame_count % 30 == 0:
            stats_frame = frames.to(torch.float32)
            pose_info = ""
            attached_pos = getattr(camera_inst, "_attached_pos", None)
            attached_lookat = getattr(camera_inst, "_attached_lookat", None)
            if attached_pos is not None and attached_lookat is not None:
                pose_info = (
                    f", env0_cam_pos={[round(float(v), 3) for v in attached_pos[0].detach().cpu().tolist()]}"
                    f", env0_cam_lookat={[round(float(v), 3) for v in attached_lookat[0].detach().cpu().tolist()]}"
                )
            print(
                f"[Nyx frame {camera.name}] "
                f"dtype={rgb.dtype}, shape={tuple(rgb.shape)}, "
                f"min={float(stats_frame.min()):.2f}, max={float(stats_frame.max()):.2f}, "
                f"mean={float(stats_frame.mean()):.2f}, std={float(stats_frame.std()):.2f}"
                f"{pose_info}"
            )

        num_frames = frames.shape[0] if camera.debug_show_all_envs else 1
        key = -1
        for env_idx in range(num_frames):
            frame = frames[env_idx]
            if frame.dtype != torch.uint8:
                max_value = float(frame.max()) if frame.numel() else 1.0
                if max_value <= 1.0:
                    frame = frame.clamp(0.0, 1.0) * 255.0
                frame = frame.clamp(0.0, 255.0).to(torch.uint8)

            frame_np = np.ascontiguousarray(frame.numpy())
            if frame_np.shape[-1] == 4:
                frame_np = frame_np[:, :, :3]
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            window_name = (
                f"Genesis Nyx Gaussian Splat env {env_idx}"
                if camera.debug_show_all_envs
                else "Genesis Nyx Gaussian Splat"
            )
            cv2.imshow(window_name, frame_np)

        key = cv2.waitKeyEx(1)
        if camera.debug_calibrate:
            self._handle_debug_key(camera_inst, camera, key)
