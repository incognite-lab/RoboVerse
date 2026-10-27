"""Opt-in recurrent DAgger collector, sequence trainer and evaluations."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from loguru import logger as log

from .buffer import SequenceBuffer, train_sequence_step
from .checkpoint import (
    IMU_LAYOUT, control_dt, evaluation_path, load_student, model_kwargs,
    output_directory, restore_optimizer, save_checkpoint,
)
from .inputs import IMU_BODY_NAME, RolloutState, StudentInputs
from .student_net import ACTION_GROUP_SIZES, RecurrentVisionStudent

try:
    from ..student_vision import image_size_from_config
    from ..dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from ..dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from ..separate_dagger import _device, _student_checkpoints
    from ..separate_ppo_trainer import load_policy_router
except ImportError:  # python config_run/main_separate.py ...
    from student_vision import image_size_from_config
    from dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from separate_dagger import _device, _student_checkpoints
    from separate_ppo_trainer import load_policy_router


def validate_action_layout(env):
    expected = [name for name in env.robot_joint_names
                if name.startswith("waist_")]
    for side in ("left", "right"):
        expected += [name for name in env.robot_joint_names
                     if name.startswith(f"{side}_") and "_hand_" not in name
                     and not any(word in name for word in ("hip_", "knee_", "ankle_"))]
    for side in ("left", "right"):
        expected += [name for name in env.robot_joint_names if name.startswith(f"{side}_hand_")]
    expected += list(env.LOCOMOTION_COMMAND_NAMES)
    if len(expected) != 34 or list(env.action_names) != expected:
        raise ValueError("New student requires the standard 31 upper-body targets + 3 walking commands")
    for name, size in ACTION_GROUP_SIZES:
        if len(env.action_groups[name]) != size:
            raise ValueError(f"Unexpected {name} action layout")


def train_dagger(config, base, env):
    device = _device()
    validate_action_layout(env)
    sequence_length = int(config.get("dagger_new_sequence_length", 16))
    burn_in = int(config.get("dagger_new_burn_in", 8))
    batch_size = int(config.get("dagger_new_batch_size", min(8, max(1, int(config.get("dagger_batch_size", 512))
                                                                  // max(1, sequence_length)))))
    capacity = int(config.get("dagger_buffer_steps", 3000))
    store_count = int(config.get("dagger_store_per_step", min(32, env.num_envs)))
    train_every = int(config.get("dagger_train_every", 20))
    updates = int(config.get("dagger_updates_per_train", 10))
    log_every = int(config.get("dagger_log_every", 100))
    grad_clip = float(config.get("dagger_new_grad_clip", 1.0))
    if min(sequence_length, batch_size, train_every, updates, log_every) <= 0 or burn_in < 0:
        raise ValueError("Sequence/training counts must be positive and burn-in nonnegative")
    if capacity < sequence_length + burn_in:
        raise ValueError(
            f"dagger_buffer_steps={capacity} must be >= dagger_new_burn_in={burn_in} "
            f"+ dagger_new_sequence_length={sequence_length} = {burn_in + sequence_length}"
        )
    if store_count <= 0:
        raise ValueError(f"dagger_store_per_step must be positive, got {store_count}")
    if store_count > env.num_envs:
        log.warning("dagger_store_per_step={} exceeds num_envs={}; storing all {} environments instead",
                    store_count, env.num_envs, env.num_envs)
        store_count = env.num_envs
    if not np.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("dagger_new_grad_clip must be positive and finite")
    seed = int(config.get("dagger_new_seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    expert_path = config.get("expert_model_path") or config.get("load_model_path")
    if not expert_path:
        raise ValueError("train_dagger_new requires expert_model_path (the six-policy teacher bundle)")
    teacher, _ = load_policy_router(env, expert_path, device=device,
                                    checkpoint=config.get("load_model_checkpoint"))
    resume = config.get("train_or_eval") == "load_and_train_dagger_new"
    start_step = 0
    if resume:
        if not config.get("load_student_path"):
            raise ValueError("load_and_train_dagger_new requires load_student_path")
        student, metadata, start_step = load_student(config["load_student_path"], env, base, device)
        camera_name = metadata["camera_name"]
    else:
        camera_name = str(config.get("dagger_camera_name", "camera0"))
        kwargs = model_kwargs(config, len(env.robot_joint_names))
        student = RecurrentVisionStudent(
            **kwargs, pretrained_visual=bool(config.get("dagger_new_pretrained_visual", False))
        ).to(device)
        # Fixed physical scales, rather than normalization changing underneath
        # an active recurrent rollout. Saved buffers are reused in evaluation.
        std = torch.ones(student.proprio_dim, device=device)
        std[student.num_joints:2 * student.num_joints] = 20.0
        std[2 * student.num_joints:2 * student.num_joints + 3] = 4.0
        student.set_proprio_normalization(torch.zeros_like(std), std)
        metadata = dict(model_kwargs=kwargs, joint_names=list(env.robot_joint_names),
                        action_names=list(env.action_names), imu_layout=IMU_LAYOUT, imu_body_name=IMU_BODY_NAME,
                        camera_name=camera_name, control_dt=control_dt(base))
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config.get("learning_rate", 3e-4)))
    if resume:
        restored_step = restore_optimizer(config["load_student_path"], optimizer, device)
        if restored_step != start_step:
            raise ValueError("Student and optimizer checkpoint steps differ")
        log.info("Resuming recurrent weights/optimizer at {}; simulator, replay and GRU start fresh", start_step)
    end_step = dagger_target_steps(config, start_step)
    observation = env.reset()
    state = RolloutState(student, env.num_envs)
    adapter = StudentInputs(base, env, device, camera_name, metadata["joint_names"],
                            image_size=image_size_from_config(config))
    first = adapter.read(state.previous_actions)
    image_shape = list(first[0].shape[1:])
    if resume and image_shape != metadata["image_shape"]:
        raise ValueError("Resume camera image shape differs from training")
    metadata["image_shape"] = image_shape
    metadata["image_preprocessing"] = "center_crop_to_aspect_area_resize_rgb_v2"
    # Keep slots fixed throughout the run; resampling IDs each step would
    # create sequences made from unrelated environments.
    selected = torch.randperm(env.num_envs, device=device)[:store_count]
    buffer = SequenceBuffer(
        capacity, store_count, image_shape, student.num_joints,
        (student.num_layers, student.hidden_size), device=device,
        storage_device=config.get("dagger_new_buffer_device", "cpu"),
        pin_memory=bool(config.get("dagger_buffer_pin_memory", False)),
    )
    save_dir = output_directory(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(config.get("tensorboard_log", str(save_dir / "tensorboard"))) / "recurrent"
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(log_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")))
    writer.add_text("Run/config", json.dumps(config, indent=2, default=str), start_step)
    writer.add_text("Run/recurrent", "Replay uses stored rollout hidden states plus burn-in (truncated BPTT). "
                    "Only episode resets zero memory. Resume restarts simulator/replay. "
                    "Mixed-teacher training success is not pure-student evaluation.", start_step)
    metrics = DaggerMetrics(env.get_current_stages(), env.NUM_POLICY_STAGES, control_dt(base))
    hardware = HardwareMonitor(device, interval=config.get("dagger_gpu_poll_seconds", 10))
    save_frequency = int(config.get("model_save_freq", 5000))
    next_step = start_step
    log.info("Recurrent DAgger: {}..{}, sequence={} + burn-in={}, batch={} sequences, replay={:.1f} MiB on {}",
             start_step, end_step, sequence_length, burn_in, batch_size, buffer.storage_bytes / 2**20,
             buffer.storage_device)
    try:
        for step in range(start_step, end_step):
            beta = beta_at_step(config, step)
            with metrics.measure("camera_and_inputs"):
                values = adapter.read(state.previous_actions)
            with metrics.measure("inference_and_actions"):
                student.eval()
                with torch.no_grad():
                    expert = teacher.predict(observation, deterministic=True)
                    prediction, hidden = student(values[0].float() / 255, *values[1:],
                                                 hidden=state.hidden, reset_mask=state.reset_mask)
                proposed = prediction.cpu().numpy()
                metrics.record_predictions(proposed, expert, env.action_space.low, env.action_space.high)
                execute_expert = random.random() < beta
                actions = np.clip(expert if execute_expert else proposed,
                                  env.action_space.low, env.action_space.high)
            with metrics.measure("buffer_store"):
                buffer.add(tuple(value.index_select(0, selected) for value in values),
                           torch.as_tensor(expert, device=device).index_select(0, selected),
                           state.hidden.index_select(1, selected), state.reset_mask.index_select(0, selected))
            with metrics.measure("env_step"):
                observation, rewards, dones, infos = env.step(actions)
                state.after_step(hidden, actions, dones)
            metrics.observe(dones, infos, store_count, execute_expert)
            if (step + 1) % train_every == 0 and buffer.filled_steps >= sequence_length + burn_in:
                with metrics.measure("student_updates"):
                    losses = [train_sequence_step(student, optimizer, buffer, batch_size,
                                                 sequence_length, burn_in, grad_clip) for _ in range(updates)]
                metrics.record_updates(losses, batch_size * sequence_length)
                writer.add_scalar("DAgger/MSE_Loss", float(np.mean(losses)), step + 1)
            next_step = step + 1
            beta_next = beta_at_step(config, next_step)
            if metrics.steps % log_every == 0:
                metrics.write(writer, next_step, buffer, beta_next, optimizer.param_groups[0]["lr"], hardware, end_step)
                writer.add_scalar("Recurrent/buffer_total_mib", buffer.storage_bytes / 2**20, next_step)
                log.info("Recurrent DAgger {}/{} beta={:.4f}, episodes={}, success={}",
                         next_step, end_step, beta_next, metrics.episodes, metrics.successful_episodes)
            if save_frequency > 0 and next_step % save_frequency == 0:
                save_checkpoint(save_dir / f"student_model_step_{step}.pth", student, optimizer, next_step, metadata)
    finally:
        try:
            save_checkpoint(save_dir / "student_model_final.pth", student, optimizer, next_step, metadata)
            if metrics.steps > metrics.last_steps:
                metrics.write(writer, next_step, buffer, beta_at_step(config, next_step),
                              optimizer.param_groups[0]["lr"], hardware, end_step)
        finally:
            hardware.close()
            writer.close()


def evaluate_checkpoint(config, base, env, path, device, video=False):
    student, metadata, checkpoint_step = load_student(path, env, base, device)
    env.reset()
    state = RolloutState(student, env.num_envs)
    adapter = StudentInputs(base, env, device, metadata["camera_name"], metadata["joint_names"],
                            image_size=tuple(metadata["image_shape"][-2:]))
    writer = None
    if video:
        import cv2
        video_path = Path(config.get("video_save_path", "./config_run/output/eval_videos/recurrent.mp4"))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        _, height, width = metadata["image_shape"]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(config.get("video_fps", 30)), (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot write video to {video_path}")
    target = int(config.get("eval_episodes", 1 if video else env.num_envs))
    max_steps = int(config.get("eval_max_steps", 2000) if video else config.get("eval_total_step_cap", 5000))
    if target <= 0 or max_steps <= 0:
        if writer is not None:
            writer.release()
        raise ValueError("Evaluation episode count and step cap must be positive")
    returns = np.zeros(env.num_envs)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    finished_returns, finished_lengths, successes = [], [], []
    num_stages = int(config.get("num_eval_stages", 6))
    if num_stages <= 0:
        if writer is not None:
            writer.release()
        raise ValueError("num_eval_stages must be positive")
    stage_counts = np.zeros(num_stages, dtype=np.int64)
    try:
        for _ in range(max_steps):
            values = adapter.read(state.previous_actions)
            if list(values[0].shape[1:]) != metadata["image_shape"]:
                raise ValueError("Evaluation camera image shape differs from training")
            if writer is not None:
                frame = cv2.cvtColor(values[0][0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                for _ in range(max(1, int(config.get("video_slowdown", 1)))):
                    writer.write(frame)
            stages = np.clip(env.get_current_stages(), 0, num_stages - 1)
            with torch.no_grad():
                prediction, hidden = student(values[0].float() / 255, *values[1:],
                                             hidden=state.hidden, reset_mask=state.reset_mask)
            actions = np.clip(prediction.cpu().numpy(), env.action_space.low, env.action_space.high)
            _, rewards, dones, infos = env.step(actions)
            state.after_step(hidden, actions, dones)
            returns += rewards
            lengths += 1
            for index in np.flatnonzero(dones):
                if len(successes) < target:
                    finished_returns.append(float(returns[index]))
                    finished_lengths.append(int(lengths[index]))
                    successes.append(int(infos[index].get("is_success", False)))
                    stage_counts[int(stages[index])] += 1
                returns[index] = lengths[index] = 0
            if len(successes) >= target:
                break
        if len(successes) < target and config.get("eval_count_timeouts_as_failures", True):
            # Count only live, unfinished attempts, not freshly auto-reset rows.
            for index in np.flatnonzero(lengths > 0)[:target - len(successes)]:
                finished_returns.append(float(returns[index]))
                finished_lengths.append(int(lengths[index]))
                successes.append(0)
                stage_counts[int(np.clip(env.get_current_stages()[index], 0, num_stages - 1))] += 1
    finally:
        if writer is not None:
            writer.release()
    row = dict(checkpoint=Path(path).name, x_step=checkpoint_step, episodes_finished=len(successes),
               mean_reward=float(np.mean(finished_returns)) if successes else 0.,
               std_reward=float(np.std(finished_returns)) if successes else 0.,
               success_rate=float(np.mean(successes)) if successes else 0., success_count=sum(successes),
               mean_length=float(np.mean(finished_lengths)) if successes else 0.,
               std_length=float(np.std(finished_lengths)) if successes else 0.)
    row.update({f"end_stage_{stage}_count": int(count) for stage, count in enumerate(stage_counts)})
    log.info("Recurrent evaluation {}: success={}/{}, reward={:.4f}", path, sum(successes), len(successes), row["mean_reward"])
    return row


def eval_dagger(config, base, env, video=False):
    validate_action_layout(env)
    root = evaluation_path(config.get("load_model_path", "./config_run/output/dagger_separate"))
    checkpoints = _student_checkpoints(root)
    if video:
        # A directory selects its final/latest checkpoint for video.
        checkpoints = checkpoints[-1:]
    rows = [evaluate_checkpoint(config, base, env, path, _device(), video) for _, path in checkpoints]
    output = root.parent if root.is_file() else root
    with (output / "eval_dagger_new_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not video:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib is unavailable; recurrent evaluation CSV was saved")
            return rows
        for key, label in (("mean_reward", "Average reward"), ("success_rate", "Success rate"),
                           ("mean_length", "Average episode length")):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot([row["x_step"] for row in rows], [row[key] for row in rows], marker="o")
            axis.set(xlabel="DAgger vector step", ylabel=label, title="Recurrent student")
            axis.grid(True, alpha=.3)
            figure.tight_layout()
            figure.savefig(output / f"eval_dagger_new_{key}.png")
            plt.close(figure)
    return rows


def run_dagger(config, base, env):
    mode = config.get("train_or_eval")
    if mode in ("train_dagger_new", "load_and_train_dagger_new"):
        return train_dagger(config, base, env)
    if mode in ("eval_dagger_new", "eval_dagger_new_video"):
        return eval_dagger(config, base, env, video=mode == "eval_dagger_new_video")
    raise ValueError(f"Unsupported recurrent DAgger mode: {mode}")
