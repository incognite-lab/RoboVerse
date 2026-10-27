"""Self-describing recurrent checkpoints, isolated from legacy student files."""
from __future__ import annotations

import math
from pathlib import Path
import random

import torch

from .student_net import RecurrentVisionStudent
from .inputs import IMU_BODY_NAME

FORMAT = "chairman_recurrent_v1"
IMU_LAYOUT = "body_angular_velocity_xyz_projected_gravity_xyz"


def control_dt(base):
    handler = base.env.handler
    return float(handler.scene_inst.dt) * int(handler.scenario.decimation)


def output_directory(config):
    # Changing only the mode must never overwrite the legacy student's files.
    root = Path(config.get("model_save_path", "./config_run/output/dagger_separate"))
    return root if root.name == "recurrent" else root / "recurrent"


def evaluation_path(path):
    root = Path(path)
    return root / "recurrent" if root.is_dir() and (root / "recurrent").is_dir() else root


def model_kwargs(config, num_joints):
    return dict(
        num_joints=num_joints,
        hidden_size=int(config.get("dagger_new_hidden_size", 256)),
        num_layers=int(config.get("dagger_new_num_layers", 1)),
        proprio_embedding_size=int(config.get("dagger_new_proprio_embedding_size", 128)),
        visual_batch_size=int(config.get("dagger_new_visual_batch_size", 0)),
        checkpoint_visual=bool(config.get("dagger_new_checkpoint_visual", True)),
    )


def save_checkpoint(path, student, optimizer, next_step, metadata):
    path = Path(path)
    payload = dict(format=FORMAT, student_state_dict=student.state_dict(), metadata=metadata,
                   next_step=int(next_step))
    torch.save(payload, path)
    # A weights-only evaluation file, separate from the larger optimizer state.
    torch.save(dict(format=FORMAT, optimizer_state_dict=optimizer.state_dict(),
                    next_step=int(next_step), random_state=random.getstate(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               f"{path}.training.pth")


def load_student(path, env, base, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Expected a train_dagger_new checkpoint; legacy student weights are incompatible")
    meta = payload["metadata"]
    if meta["action_names"] != list(env.action_names):
        raise ValueError("Recurrent checkpoint action names/order differ from this environment")
    if set(meta["joint_names"]) != set(env.robot_joint_names):
        raise ValueError("Recurrent checkpoint joint names differ from this environment")
    if meta["imu_layout"] != IMU_LAYOUT:
        raise ValueError("Unsupported checkpoint IMU convention")
    if "imu_body_name" not in meta:
        # Old recurrent checkpoints used pelvis. The current URDF fixes the
        # sensor to it with zero relative rotation: gyro/gravity are identical.
        # Verify equivalence before accepting legacy weights on this robot.
        robot = base.env.handler.get_states().robots[env.robot_name]
        names = list(robot.body_names)
        if "pelvis" not in names or IMU_BODY_NAME not in names:
            raise ValueError("Cannot validate the IMU frame of this legacy recurrent checkpoint")
        old = robot.body_state[:, names.index("pelvis")]
        new = robot.body_state[:, names.index(IMU_BODY_NAME)]
        old_q = torch.nn.functional.normalize(old[:, 3:7], dim=-1)
        new_q = torch.nn.functional.normalize(new[:, 3:7], dim=-1)
        same_rotation = ((old_q * new_q).sum(-1).abs() - 1).abs().max() < 1e-6
        if not same_rotation or not torch.allclose(old[:, 10:13], new[:, 10:13], atol=1e-6, rtol=1e-5):
            raise ValueError("Legacy checkpoint used pelvis IMU; its frame differs from imu_in_pelvis")
        meta = dict(meta, imu_body_name=IMU_BODY_NAME)
    elif meta["imu_body_name"] != IMU_BODY_NAME:
        raise ValueError(f"Checkpoint IMU body must be {IMU_BODY_NAME!r}, got {meta['imu_body_name']!r}")
    if not math.isclose(meta["control_dt"], control_dt(base), rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("Recurrent checkpoint control period differs; use the training decimation/dt")
    student = RecurrentVisionStudent(**meta["model_kwargs"]).to(device)
    student.load_state_dict(payload["student_state_dict"])
    student.eval()
    return student, meta, int(payload["next_step"])


def restore_optimizer(path, optimizer, device):
    state_path = Path(f"{path}.training.pth")
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume requires optimizer state: {state_path}")
    payload = torch.load(state_path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Unsupported recurrent training state")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["random_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and len(payload["cuda_rng_states"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    return int(payload["next_step"])
