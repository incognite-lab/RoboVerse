from __future__ import annotations

import argparse
import importlib
import os
import time

import genesis as gs
import numpy as np
import torch
from PIL import Image

import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions


KEY_LEFT = 2424832
KEY_UP = 2490368
KEY_RIGHT = 2555904
KEY_DOWN = 2621440


def patch_genesis_engine_namespace_for_nyx_import() -> None:
    engine_mod = importlib.import_module("genesis.engine")
    entities_mod = importlib.import_module("genesis.engine.entities")
    base_entity_mod = importlib.import_module("genesis.engine.entities.base_entity")

    setattr(gs, "engine", engine_mod)
    setattr(engine_mod, "entities", entities_mod)
    setattr(entities_mod, "base_entity", base_entity_mod)


def patch_nyx_renderer_for_current_genesis() -> None:
    patch_genesis_engine_namespace_for_nyx_import()
    from gs_nyx_plugin.nyx_renderer import NyxPyRenderer

    if getattr(NyxPyRenderer, "_standalone_viewer_vgeom_patch", False):
        return

    original_update_geometry_tensors = NyxPyRenderer._update_geometry_tensors

    def patched_update_geometry_tensors(self, env_index):
        if getattr(self, "_num_rigid_geoms", 0) == 0 and getattr(self, "_num_deform_verts", 0) == 0:
            return

        if hasattr(self._rigid_solver, "vgeoms_state"):
            return original_update_geometry_tensors(self, env_index)

        if hasattr(self._rigid_solver, "get_vgeoms_pos") and hasattr(self._rigid_solver, "get_vgeoms_quat"):
            geom_pos_tensor = self._rigid_solver.get_vgeoms_pos()
            geom_rot_tensor = self._rigid_solver.get_vgeoms_quat()

            if geom_pos_tensor.dim() == 2:
                geom_pos_tensor = geom_pos_tensor.unsqueeze(0)
            if geom_rot_tensor.dim() == 2:
                geom_rot_tensor = geom_rot_tensor.unsqueeze(0)

            self._geom_pos_tensor_cuda.copy_(geom_pos_tensor[env_index].to(torch.float32))
            self._geom_rot_tensor_cuda.copy_(geom_rot_tensor[env_index].to(torch.float32))
            return

        return original_update_geometry_tensors(self, env_index)

    NyxPyRenderer._update_geometry_tensors = patched_update_geometry_tensors
    NyxPyRenderer._standalone_viewer_vgeom_patch = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Gaussian splat with Genesis + Nyx.")
    parser.add_argument("--splat", default="./config_run/splat/splat.ply", help="Path to .ply or .spz splat file.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--lookat", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument(
        "--splat-pos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Splat translation before Nyx loads the scene. Default is intentionally huge to test whether Nyx applies it.",
    )
    parser.add_argument("--splat-rot-xyzw", type=float, nargs=4, default=(0.0, 0.0, 0.0, 1.0))
    parser.add_argument("--splat-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--splat-z-up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", default="", help="Optional output image path for the current frame when pressing P.")
    args = parser.parse_args()

    splat_path = os.path.abspath(args.splat)
    if not os.path.exists(splat_path):
        raise FileNotFoundError(splat_path)

    gs.init(backend=gs.gpu, logging_level=gs._logging.WARNING)
    patch_nyx_renderer_for_current_genesis()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )

    splat = nps.LightFieldAsset()
    splat.type = nps.ELightFieldType.GaussianField
    splat.uri = splat_path

    splat_pos_input = tuple(args.splat_pos)
    splat_pos = nps.float3(*splat_pos_input)
    splat_rot = nps.quaternion(*args.splat_rot_xyzw)
    splat_scale = nps.float3(*args.splat_scale)
    if args.splat_z_up:
        splat_pos = nps.float3_z_up_to_y_up_a(splat_pos)
        splat_rot = nps.quaternion_z_up_to_y_up_a(splat_rot)
        splat_scale = nps.float3_z_up_to_y_up_a(splat_scale)

    splat.position = splat_pos
    splat.rotation = splat_rot
    splat.scale = splat_scale

    print(f"Requested splat position: {list(splat_pos_input)}")
    print(
        "Nyx splat position after coordinate conversion: "
        f"[{splat.position.x:.6f}, {splat.position.y:.6f}, {splat.position.z:.6f}]"
    )

    cam = scene.add_sensor(
        NyxCameraOptions(
            res=(args.width, args.height),
            pos=tuple(args.pos),
            lookat=tuple(args.lookat),
            fov=args.fov,
            spp=args.spp,
            render_mode=npr.ERenderMode.FastPathTracer,
            tone_mapper=nps.EToneMapper.Off,
            light_fields=(splat,),
            open_window=args.native_window,
            window_size=(args.width, args.height),
        )
    )

    scene.build(n_envs=1)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("This viewer needs OpenCV. Install `opencv-python` in the active environment.") from exc

    pos = np.array(args.pos, dtype=np.float64)
    lookat = np.array(args.lookat, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(lookat - pos) < 1e-8:
        lookat = pos + np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cam.update_camera_pose(pos=tuple(pos.tolist()), lookat=tuple(lookat.tolist()), up=tuple(up.tolist()))
        print("Camera pos and lookat were identical; using lookat = pos + [1, 0, 0].")
    move_step = 0.05
    turn_step = np.deg2rad(5.0)

    print(
        "Controls: W/S or Up/Down forward/back, A/D left/right, R/F up/down, "
        "J/L or Left/Right yaw, I/K pitch, P print/save, Q quit."
    )
    print(f"Loaded splat: {splat_path}")

    last_stats = 0.0
    while True:
        scene.step()
        rgb = cam.read().rgb[0]
        frame = rgb.detach().cpu()

        now = time.time()
        if now - last_stats > 1.0:
            stats = frame.to(torch.float32)
            print(
                f"[frame] min={float(stats.min()):.2f} max={float(stats.max()):.2f} "
                f"mean={float(stats.mean()):.2f} std={float(stats.std()):.2f} "
                f"pos={[round(v, 3) for v in pos.tolist()]} "
                f"lookat={[round(v, 3) for v in lookat.tolist()]}"
            )
            last_stats = now

        frame_np = np.ascontiguousarray(frame.numpy())
        if frame_np.shape[-1] == 4:
            frame_np = frame_np[:, :, :3]
        cv2.imshow("Genesis Nyx Gaussian Splat", cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

        key = cv2.waitKeyEx(1)
        if key < 0:
            continue
        key_char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) <= 255 else ""
        if key_char == "q":
            break

        forward = lookat - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            lookat = pos + forward
        else:
            forward /= forward_norm
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        else:
            right /= right_norm
        changed = False

        moves = {
            "w": forward * move_step,
            "s": -forward * move_step,
            "a": -right * move_step,
            "d": right * move_step,
            "r": up * move_step,
            "f": -up * move_step,
        }
        arrow_moves = {
            KEY_UP: forward * move_step,
            KEY_DOWN: -forward * move_step,
        }
        if key_char in moves:
            delta = moves[key_char]
            pos += delta
            lookat += delta
            changed = True
        elif key in arrow_moves:
            delta = arrow_moves[key]
            pos += delta
            lookat += delta
            changed = True
        elif key_char in ("j", "l", "i", "k") or key in (KEY_LEFT, KEY_RIGHT):
            yaw = turn_step if key_char == "j" or key == KEY_LEFT else -turn_step if key_char == "l" or key == KEY_RIGHT else 0.0
            pitch = turn_step if key_char == "i" else -turn_step if key_char == "k" else 0.0
            direction = lookat - pos
            if yaw != 0.0:
                c, s = np.cos(yaw), np.sin(yaw)
                rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                direction = rot_z @ direction
            if pitch != 0.0:
                direction = (
                    direction * np.cos(pitch)
                    + np.cross(right, direction) * np.sin(pitch)
                    + right * np.dot(right, direction) * (1.0 - np.cos(pitch))
                )
            lookat = pos + direction
            changed = True
        elif key_char == "p":
            print(f"camera pos: {[round(v, 5) for v in pos.tolist()]}")
            print(f"camera look_at: {[round(v, 5) for v in lookat.tolist()]}")
            print(
                "To make this visual point the world origin by moving the PLY itself, "
                f"shift the PLY by: {[round(float(-v), 5) for v in pos.tolist()]}"
            )
            if args.save:
                os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                Image.fromarray(frame_np).save(args.save)
                print(f"Saved {args.save}")

        if changed:
            cam.update_camera_pose(pos=tuple(pos.tolist()), lookat=tuple(lookat.tolist()), up=tuple(up.tolist()))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
