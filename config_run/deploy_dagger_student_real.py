#!/usr/bin/env python3
"""Run a trained DAgger vision student on the real G1 external-control wrapper.

The student was trained in simulation with image input plus joint positions and
outputs absolute joint targets in the same order as the simulated action space:
base joints first, then arm joints. The real wrapper accepts arm targets and,
optionally, normalized walking velocity commands.

This script is intentionally conservative:
- it runs in dry-run mode unless --enable-robot is passed;
- it rate-limits arm target changes;
- base velocity is disabled unless --enable-base-velocity is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config_run.dagger_vp.student_net import VisionStudent  # noqa: E402


SERVER_URL = "http://192.168.124.101:8080"

SIM_ACTION_JOINTS = [
    "baseslide_joint",
    "baseslide_joint2",
    "baserot_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

SIM_JOINT_LIMITS = {
    "baseslide_joint": (-1.5, 1.5),
    "baseslide_joint2": (-1.6, 0.1),
    "baserot_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}


def infer_model_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer (num_actions, num_joints) from a VisionStudent checkpoint."""
    actor_out = None
    joint_in = None
    for key, value in state_dict.items():
        if key.endswith("actor.4.weight"):
            actor_out = int(value.shape[0])
        elif key.endswith("joint_head.0.weight"):
            joint_in = int(value.shape[1])

    if actor_out is None:
        actor_weights = [
            value for key, value in state_dict.items()
            if "actor" in key and key.endswith(".weight") and value.ndim == 2
        ]
        if actor_weights:
            actor_out = int(actor_weights[-1].shape[0])
    if joint_in is None:
        joint_weights = [
            value for key, value in state_dict.items()
            if "joint_head" in key and key.endswith(".weight") and value.ndim == 2
        ]
        if joint_weights:
            joint_in = int(joint_weights[0].shape[1])

    if actor_out is None or joint_in is None:
        raise RuntimeError("Could not infer VisionStudent dimensions from checkpoint.")
    return actor_out, joint_in


def load_student(model_path: Path, device: str) -> tuple[VisionStudent, int, int]:
    state_dict = torch.load(model_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Expected a state_dict checkpoint, got {type(state_dict)}")
    num_actions, num_joints = infer_model_dims(state_dict)
    model = VisionStudent(num_actions=num_actions, num_joints=num_joints).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"WARNING: checkpoint load was not clean: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, num_actions, num_joints


def sim_joint_to_real_name(sim_name: str) -> str:
    return sim_name.removesuffix("_joint")


def preprocess_frame(frame_bgr: np.ndarray, width: int, height: int, device: str) -> torch.Tensor:
    frame = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_chw = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(frame_chw).to(device=device, dtype=torch.float32)
    return tensor.unsqueeze(0) / 255.0


def build_joint_input(
    arm_positions: dict[str, float],
    policy_joint_names: list[str],
    device: str,
) -> torch.Tensor:
    values = []
    missing = []
    for sim_name in policy_joint_names:
        real_name = sim_joint_to_real_name(sim_name)
        if real_name not in arm_positions:
            missing.append(real_name)
            values.append(0.0)
        else:
            values.append(float(arm_positions[real_name]))
    if missing:
        raise RuntimeError("Robot did not report required arm joints: " + ", ".join(missing))
    return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)


def clip_action(action: np.ndarray, action_names: list[str]) -> np.ndarray:
    clipped = action.astype(np.float32, copy=True)
    for idx, name in enumerate(action_names):
        low, high = SIM_JOINT_LIMITS.get(name, (-np.inf, np.inf))
        clipped[idx] = float(np.clip(clipped[idx], low, high))
    return clipped


def rate_limit_targets(
    previous: dict[str, float],
    target: dict[str, float],
    max_delta: float,
) -> dict[str, float]:
    limited = {}
    for joint, value in target.items():
        old = previous.get(joint, value)
        limited[joint] = old + float(np.clip(value - old, -max_delta, max_delta))
    return limited


def make_arm_targets(action: np.ndarray, action_names: list[str]) -> dict[str, float]:
    targets = {}
    for idx, sim_name in enumerate(action_names):
        if sim_name.startswith("base"):
            continue
        targets[sim_joint_to_real_name(sim_name)] = float(action[idx])
    return targets


def make_velocity_command(
    action: np.ndarray,
    action_names: list[str],
    x_scale: float,
    y_scale: float,
    yaw_scale: float,
) -> dict[str, float]:
    values = {name: float(action[idx]) for idx, name in enumerate(action_names)}
    x = values.get("baseslide_joint2", 0.0) / max(abs(SIM_JOINT_LIMITS["baseslide_joint2"][0]), 1e-6)
    y = values.get("baseslide_joint", 0.0) / max(abs(SIM_JOINT_LIMITS["baseslide_joint"][1]), 1e-6)
    yaw = values.get("baserot_joint", 0.0) / max(abs(SIM_JOINT_LIMITS["baserot_joint"][1]), 1e-6)
    return {
        "x": float(np.clip(x * x_scale, -1.0, 1.0)),
        "y": float(np.clip(y * y_scale, -1.0, 1.0)),
        "yaw": float(np.clip(yaw * yaw_scale, -1.0, 1.0)),
    }


def import_robot_client(client_dir: Path | None):
    if client_dir is not None:
        sys.path.insert(0, str(client_dir.resolve()))
    try:
        from external_control_client import BalancedStandingArms, LockedStandingArms, WalkingArms
    except ImportError as exc:
        raise RuntimeError(
            "Could not import external_control_client. Pass --client-dir pointing "
            "to the directory containing external_control_client.py."
        ) from exc
    return {
        "locked": LockedStandingArms,
        "balanced": BalancedStandingArms,
        "walking": WalkingArms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to student_model_*.pth")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--client-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["locked", "balanced", "walking"], default="balanced")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--self-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-robot", action="store_true", help="Actually send commands to the robot")
    parser.add_argument("--enable-base-velocity", action="store_true")
    parser.add_argument("--max-arm-delta", type=float, default=0.08, help="Max radian target change per command")
    parser.add_argument("--x-scale", type=float, default=0.25)
    parser.add_argument("--y-scale", type=float, default=0.25)
    parser.add_argument("--yaw-scale", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, num_actions, num_joints = load_student(args.model, device)
    if num_actions > len(SIM_ACTION_JOINTS):
        raise RuntimeError(
            f"Checkpoint outputs {num_actions} actions, but only {len(SIM_ACTION_JOINTS)} are mapped."
        )
    action_names = SIM_ACTION_JOINTS[:num_actions]
    policy_joint_names = action_names[3:3 + num_joints]
    if len(policy_joint_names) != num_joints:
        raise RuntimeError(
            f"Checkpoint expects {num_joints} joint inputs, but action mapping provides "
            f"{len(policy_joint_names)} after the 3 base joints."
        )

    camera_source: int | str
    camera_source = int(args.camera) if str(args.camera).isdigit() else args.camera
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.camera}")

    client_classes = import_robot_client(args.client_dir)
    robot_cls = client_classes[args.mode]
    robot = robot_cls(args.server_url, self_collision=args.self_collision)

    print(f"Loaded {args.model}")
    print(f"Policy actions: {action_names}")
    print(f"Policy joint inputs: {policy_joint_names}")
    print("DRY RUN: commands will not be sent." if not args.enable_robot else "LIVE: commands will be sent.")

    previous_targets: dict[str, float] = {}
    period = 1.0 / args.hz
    started = time.monotonic()
    step_idx = 0

    try:
        robot.start()
        previous_targets = robot.get_arm_positions()

        while args.duration <= 0.0 or (time.monotonic() - started) < args.duration:
            loop_started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera did not return a frame.")

            arm_positions = robot.get_arm_positions()
            image_tensor = preprocess_frame(frame, args.image_width, args.image_height, device)
            joint_tensor = build_joint_input(arm_positions, policy_joint_names, device)

            with torch.no_grad():
                action = model(image_tensor, joint_tensor).squeeze(0).detach().cpu().numpy()
            action = clip_action(action, action_names)

            arm_targets = make_arm_targets(action, action_names)
            arm_targets = rate_limit_targets(previous_targets, arm_targets, args.max_arm_delta)
            previous_targets.update(arm_targets)

            velocity = None
            if args.enable_base_velocity and args.mode == "walking":
                velocity = make_velocity_command(
                    action, action_names, args.x_scale, args.y_scale, args.yaw_scale
                )

            if args.enable_robot:
                robot.step(arms=arm_targets, velocity=velocity)

            if step_idx % args.print_every == 0:
                print(
                    f"step={step_idx} "
                    f"arms={{{', '.join(f'{k}: {v:.3f}' for k, v in arm_targets.items())}}} "
                    f"velocity={velocity}"
                )
            step_idx += 1

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if args.enable_robot and args.mode == "walking":
            try:
                robot.step(velocity={"x": 0.0, "y": 0.0, "yaw": 0.0})
            except Exception:
                pass
        robot.close()
        capture.release()


if __name__ == "__main__":
    main()
