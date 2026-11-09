"""Self-describing recurrent checkpoints, isolated from legacy student files."""
from __future__ import annotations

import math
from pathlib import Path
import random
from uuid import uuid4

import torch

from .student_net import RecurrentVisionStudent
from .inputs import IMU_BODY_NAME

FORMAT = "chairman_recurrent_v2"
LEGACY_FORMAT = "chairman_recurrent_v1"
IMU_LAYOUT = "body_angular_velocity_xyz_projected_gravity_xyz"


def control_dt(base):
    handler = base.env.handler
    return float(handler.scene_inst.dt) * int(handler.scenario.decimation)


def output_directory(config):
    # Changing only the mode must never overwrite the legacy student's files.
    root = Path(config.get("model_save_path", "./config_run/output/dagger_separate_stable_v2"))
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
        architecture_version=2,
        visual_normalization="batchnorm" if config.get("dagger_new_pretrained_visual", False) else "groupnorm",
    )


def save_checkpoint(path, student, optimizer, next_step, metadata):
    path = Path(path)
    checkpoint_id = uuid4().hex
    payload = dict(format=FORMAT, student_state_dict=student.state_dict(), metadata=metadata,
                   next_step=int(next_step), checkpoint_id=checkpoint_id)
    atomic_save(payload, path)
    # A weights-only evaluation file, separate from the larger optimizer state.
    atomic_save(dict(format=FORMAT, optimizer_state_dict=optimizer.state_dict(),
                    next_step=int(next_step), checkpoint_id=checkpoint_id, random_state=random.getstate(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               f"{path}.training.pth")


def atomic_save(payload, path):
    """Leave the previous checkpoint/archive intact if writing is interrupted."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_student(path, env, base, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") not in (FORMAT, LEGACY_FORMAT):
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
    meta = dict(meta, model_kwargs=dict(meta["model_kwargs"]))
    meta["model_kwargs"].setdefault("architecture_version", 1 if payload["format"] == LEGACY_FORMAT else 2)
    student = RecurrentVisionStudent(**meta["model_kwargs"]).to(device)
    student.load_state_dict(payload["student_state_dict"])
    student.eval()
    return student, meta, int(payload["next_step"])


def restore_optimizer(path, optimizer, device, student=None):
    state_path = Path(f"{path}.training.pth")
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume requires optimizer state: {state_path}")
    payload = torch.load(state_path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Unsupported recurrent training state")
    # Tensor storage is not materialized: only read metadata and the generation
    # ID. Independent atomic writes cannot make two files a transaction; a
    # crash between them must be detected even when both have the same step.
    weights = torch.load(path, map_location='meta', weights_only=True)
    if weights.get('next_step') != payload.get('next_step'):
        raise ValueError('Student and optimizer checkpoint steps differ')
    if weights.get('checkpoint_id') != payload.get('checkpoint_id'):
        raise ValueError('Student and optimizer belong to different checkpoint generations')
    state = payload["optimizer_state_dict"]
    if len(state['param_groups']) == 1 and len(optimizer.param_groups) == 2:
        if student is None:
            raise ValueError("Migrating a single-group optimizer requires the student parameter order")
        old_ids = state['param_groups'][0]['params']
        parameters = list(student.parameters())
        if len(old_ids) != len(parameters):
            raise ValueError("Legacy optimizer parameters differ from the student")
        by_parameter = {id(parameter): state['state'].get(old_id)
                        for parameter, old_id in zip(parameters, old_ids, strict=True)}
        migrated = optimizer.state_dict()
        for live_group, saved_group in zip(optimizer.param_groups, migrated['param_groups'], strict=True):
            for key, value in state['param_groups'][0].items():
                if key not in ('params', 'lr', 'name'):
                    saved_group[key] = value
            for parameter, new_id in zip(live_group['params'], saved_group['params'], strict=True):
                if by_parameter[id(parameter)] is not None:
                    migrated['state'][new_id] = by_parameter[id(parameter)]
        state = migrated
    optimizer.load_state_dict(state)
    random.setstate(payload["random_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and len(payload["cuda_rng_states"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    return int(payload["next_step"])
