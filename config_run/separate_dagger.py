"""DAgger training and evaluation with a six-policy ChairMan teacher."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from loguru import logger as log

try:
    from .dagger_vp.checkpoint import (
        load_dagger_checkpoint,
        save_dagger_checkpoint,
    )
    from .dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step
    from .dagger_vp.student_net import VisionStudent
    from .dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from .dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from .separate_ppo_trainer import load_policy_router
except ImportError:
    from dagger_vp.checkpoint import load_dagger_checkpoint, save_dagger_checkpoint
    from dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step
    from dagger_vp.student_net import VisionStudent
    from dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from separate_ppo_trainer import load_policy_router


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _student_dimensions(env):
    num_actions = int(env.action_space.shape[0])
    num_joints = len(env.robot_joint_names)
    if num_joints <= 0:
        raise ValueError("DAgger student requires robot joint positions")
    return num_actions, num_joints


def _student_inputs(metasim_env, observation, device, num_joints, camera_name):
    handler = metasim_env.env.handler
    if getattr(getattr(handler, "scenario", None), "defer_camera_rendering", False):
        cameras = handler.get_camera_states()
    else:
        cameras = handler.get_states().cameras
    if camera_name not in cameras:
        raise KeyError(
            f"DAgger camera {camera_name!r} is unavailable; choose from {sorted(cameras)}"
        )
    rgb = cameras[camera_name].rgb
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected NHWC RGB camera tensor, got {tuple(rgb.shape)}")
    rgb = rgb[..., :3].to(device=device)
    images_uint8 = rgb.permute(0, 3, 1, 2).contiguous().to(torch.uint8)
    images = images_uint8.float() / 255.0

    observation = np.asarray(observation)
    if observation.ndim != 2 or observation.shape[1] < num_joints:
        raise ValueError(
            f"Full observation shape {observation.shape} cannot provide joint slice [0:{num_joints}]"
        )
    # MetaSim observations start with all joint positions in simulator order,
    # including the legs. Task features follow this joint-position block.
    joints = torch.as_tensor(
        observation[:, :num_joints], dtype=torch.float32, device=device
    )
    return images_uint8, images, joints


def _new_student(env, device):
    num_actions, num_joints = _student_dimensions(env)
    return (
        VisionStudent(num_actions=num_actions, num_joints=num_joints).to(device),
        num_actions,
        num_joints,
    )


def _load_student_weights(student, path, device):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"DAgger student checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    # Evaluation consumes the plain .pth written by save_dagger_checkpoint.
    # Accept a training-state dictionary as a convenience as well.
    if isinstance(state, dict) and "student_state_dict" in state:
        state = state["student_state_dict"]
    student.load_state_dict(state)
    student.eval()


def train_dagger(config, metasim_env, env):
    """Train one vision student from the composed six-policy expert action."""
    expert_path = config.get("expert_model_path") or config.get("load_model_path")
    if not expert_path:
        raise ValueError(
            "train_dagger requires expert_model_path pointing to a "
            "chairman_separate six-policy run directory"
        )

    device = _device()
    teacher, manifest = load_policy_router(
        env,
        expert_path,
        device=device,
        checkpoint=config.get("load_model_checkpoint"),
    )
    log.info(
        "Loaded six-policy DAgger teacher at {:,} transitions from {}",
        int(manifest.get("global_timesteps", 0)),
        expert_path,
    )

    student, num_actions, num_joints = _new_student(env, device)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=float(config.get("learning_rate", 3.0e-4))
    )
    mode = config.get("train_or_eval")
    start_step = 0
    beta = beta_at_step(config, 0)
    if mode == "load_and_train_dagger":
        checkpoint = config.get("load_student_path")
        if not checkpoint:
            raise ValueError("load_and_train_dagger requires load_student_path")
        start_step, beta = load_dagger_checkpoint(
            checkpoint, student, optimizer, device, config
        )
        log.info(
            "Resuming DAgger student from {} at step {} with beta={:.6f}",
            checkpoint, start_step, beta,
        )

    end_step = dagger_target_steps(config, start_step)
    camera_name = str(config.get("dagger_camera_name", "camera0"))
    observation = env.reset()
    first_uint8, _, _ = _student_inputs(
        metasim_env, observation, device, num_joints, camera_name
    )
    image_shape = tuple(int(value) for value in first_uint8.shape[1:])
    buffer_device = str(config.get("dagger_buffer_device", "cpu"))
    pin_memory = bool(
        config.get(
            "dagger_buffer_pin_memory",
            buffer_device == "cpu" and device.startswith("cuda"),
        )
    )
    store_per_step = int(config.get("dagger_store_per_step", 32))
    buffer = DAggerBuffer(
        max_samples=int(config.get("dagger_buffer_steps", 4000))
        * store_per_step,
        img_shape=image_shape,
        num_joints=num_joints,
        num_actions=num_actions,
        device=device,
        storage_device=buffer_device,
        pin_memory=pin_memory,
    )

    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(config.get(
        "tensorboard_log", "./config_run/output/dagger_separate/tensorboard"
    ))
    save_dir = Path(config.get(
        "model_save_path", "./config_run/output/dagger_separate"
    ))
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    writer = SummaryWriter(str(log_dir / run_name))
    writer.add_text("Run/config", "```json\n" + json.dumps(config, indent=2, default=str) + "\n```", start_step)
    log.info("DAgger TensorBoard run: {}", log_dir / run_name)

    iterations = end_step - start_step
    save_frequency = int(config.get("model_save_freq", 5000))
    train_every = int(config.get("dagger_train_every", 20))
    updates_per_train = int(config.get("dagger_updates_per_train", 10))
    batch_size = int(config.get("dagger_batch_size", 512))
    log_every = int(config.get("dagger_log_every", 100))
    if log_every <= 0:
        raise ValueError("dagger_log_every must be positive")
    if iterations <= 0 or train_every <= 0 or updates_per_train <= 0:
        raise ValueError("DAgger iteration and update counts must be positive")
    if store_per_step <= 0 or store_per_step > env.num_envs:
        raise ValueError(
            f"dagger_store_per_step must be in [1, {env.num_envs}], got {store_per_step}"
        )

    completed_total = successful_total = 0
    completed_recent = successful_recent = 0
    next_step = start_step
    handler = metasim_env.env.handler
    control_dt = float(handler.scene_inst.dt) * int(handler.scenario.decimation)
    metrics = DaggerMetrics(env.get_current_stages(), env.NUM_POLICY_STAGES, control_dt)
    hardware = HardwareMonitor(device, interval=config.get("dagger_gpu_poll_seconds", 10))
    writer.add_text(
        "Run/metric_definitions",
        "Counters restart per run, including resume. Stage completions count observed attempts; "
        "curriculum starts do not count as completions. Durations start at stage entry/reset, "
        "not at the saved snapshot's original history. Success includes teacher/student mixing "
        "and curriculum resets; this is not student-only evaluation from stage 0. "
        "Timing is wall time with asynchronous GPU attribution; no extra CUDA synchronization. "
        "GPU device metrics include other processes; torch memory metrics describe this process.",
        start_step,
    )
    log.info(
        "Starting separate-teacher DAgger: steps={}..{}, image_shape={}, "
        "buffer={} samples on {}",
        start_step, end_step, image_shape, buffer.max_samples,
        buffer.storage_device,
    )
    try:
        for step in range(start_step, end_step):
            with metrics.measure("camera_and_inputs"):
                images_uint8, images, joints = _student_inputs(
                    metasim_env, observation, device, num_joints, camera_name
                )
            with metrics.measure("inference_and_actions"):
                with torch.no_grad():
                    expert_actions = teacher.predict(observation, deterministic=True)
                    expert_tensor = torch.as_tensor(
                        expert_actions, dtype=torch.float32, device=device
                    )
                    student.eval()
                    student_tensor = student(images, joints)
                    student_actions = student_tensor.cpu().numpy()

                metrics.record_predictions(student_actions, expert_actions, env.action_space.low, env.action_space.high)
                execute_expert = random.random() < beta
                actions = expert_actions if execute_expert else student_actions
                actions = np.clip(actions, env.action_space.low, env.action_space.high)
            with metrics.measure("buffer_store"):
                buffer.add_batch(
                    images_uint8, joints, expert_tensor, store_count=store_per_step
                )
            with metrics.measure("env_step"):
                observation, rewards, dones, infos = env.step(actions)

            failure_masks = {}
            if np.any(dones):
                sources = getattr(handler.task, "failure_masks", {})
                if sources:
                    names = list(sources)
                    rows = torch.stack([sources[name] for name in names]).detach().cpu().numpy()
                    failure_masks = dict(zip(names, rows))
            metrics.observe(dones, infos, store_per_step, execute_expert, failure_masks)
            completed = int(np.count_nonzero(dones))
            successful = sum(
                int(info.get("is_success", False))
                for done, info in zip(dones, infos)
                if done
            )
            completed_total += completed
            successful_total += successful
            completed_recent += completed
            successful_recent += successful

            if step > start_step and step % train_every == 0:
                with metrics.measure("student_updates"):
                    losses = [
                        train_dagger_step(student, optimizer, buffer, batch_size)
                        for _ in range(updates_per_train)
                    ]
                metrics.record_updates(losses, min(batch_size, buffer.size))
                mean_loss = float(np.mean(losses)) if losses else 0.0
                total_rate = successful_total / max(1, completed_total)
                recent_rate = successful_recent / max(1, completed_recent)
                writer.add_scalar("DAgger/MSE_Loss", mean_loss, step)
                writer.add_scalar("DAgger/Beta_Mix_Ratio", beta, step)
                writer.add_scalar(
                    "DAgger/Env_Mean_Reward", float(np.mean(rewards)), step
                )
                writer.add_scalar("DAgger/Buffer_Size", buffer.size, step)
                writer.add_scalar(
                    "DAgger/Total_Completed_Envs", completed_total, step
                )
                writer.add_scalar(
                    "DAgger/Total_Successful_Envs", successful_total, step
                )
                writer.add_scalar("DAgger/Total_Success_Rate", total_rate, step)
                writer.add_scalar(
                    "DAgger/Recent_Success_Rate", recent_rate, step
                )
                log.info(
                    "DAgger {}/{} beta={:.4f} loss={:.6f} buffer={} "
                    "success={}/{} ({:.1%}), recent={}/{} ({:.1%})",
                    step, end_step, beta, mean_loss, buffer.size,
                    successful_total, completed_total, total_rate,
                    successful_recent, completed_recent, recent_rate,
                )
                completed_recent = successful_recent = 0

            beta = beta_at_step(config, step + 1)
            next_step = step + 1
            if metrics.steps % log_every == 0:
                with metrics.measure("logging"):
                    metrics.write(writer, next_step, buffer, beta, optimizer.param_groups[0]["lr"], hardware, end_step)
            if save_frequency > 0 and step > start_step and step % save_frequency == 0:
                path = save_dir / f"student_model_step_{step}.pth"
                with metrics.measure("checkpoint"):
                    save_dagger_checkpoint(
                        str(path), student, optimizer, next_step, beta
                    )
                log.info("Saved DAgger checkpoint to {}", path)
    finally:
        final_path = save_dir / "student_model_final.pth"
        try:
            with metrics.measure("checkpoint"):
                save_dagger_checkpoint(
                    str(final_path), student, optimizer, next_step, beta
                )
            if metrics.steps > metrics.last_steps:
                metrics.write(writer, next_step, buffer, beta, optimizer.param_groups[0]["lr"], hardware, end_step)
        finally:
            hardware.close()
            writer.close()
        log.info("Saved final DAgger student to {}", final_path)


def eval_dagger_video(config, metasim_env, env):
    """Evaluate one student and save its first-person camera stream."""
    import cv2

    device = _device()
    student, _, num_joints = _new_student(env, device)
    _load_student_weights(student, config.get("load_model_path"), device)
    camera_name = str(config.get("dagger_camera_name", "camera0"))
    observation = env.reset()
    max_steps = int(config.get("eval_max_steps", 2000))
    target_episodes = int(config.get("eval_episodes", 1))
    slowdown = max(1, int(config.get("video_slowdown", 1)))
    video_path = Path(config.get(
        "video_save_path",
        "./config_run/output/eval_videos/chairman_separate_dagger.mp4",
    ))
    video_path.parent.mkdir(parents=True, exist_ok=True)

    images_uint8, _, _ = _student_inputs(
        metasim_env, observation, device, num_joints, camera_name
    )
    height, width = images_uint8.shape[-2:]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.get("video_fps", 30.0)), (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    episodes = successes = 0
    try:
        for step in range(max_steps):
            images_uint8, images, joints = _student_inputs(
                metasim_env, observation, device, num_joints, camera_name
            )
            frame = images_uint8[0].permute(1, 2, 0).cpu().numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(slowdown):
                writer.write(frame)
            with torch.no_grad():
                actions = student(images, joints).cpu().numpy()
            actions = np.clip(actions, env.action_space.low, env.action_space.high)
            observation, rewards, dones, infos = env.step(actions)
            for index in np.flatnonzero(dones):
                episodes += 1
                successes += int(infos[index].get("is_success", False))
                log.info(
                    "DAgger video episode {} finished at step {}, success={}",
                    episodes, step + 1, bool(infos[index].get("is_success", False)),
                )
                if episodes >= target_episodes:
                    break
            if episodes >= target_episodes:
                break
    finally:
        writer.release()
    log.info(
        "Saved DAgger video to {}; success={}/{}",
        video_path, successes, episodes,
    )


def _student_checkpoints(path):
    root = Path(path)
    if root.is_file():
        return [(0, root)]
    if not root.is_dir():
        raise FileNotFoundError(f"DAgger model path does not exist: {root}")
    result = []
    pattern = re.compile(r"student_model_step_(\d+)\.pth$")
    for candidate in root.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match:
            result.append((int(match.group(1)), candidate))
        elif candidate.name == "student_model_final.pth":
            result.append((10**18, candidate))
    result.sort(key=lambda item: item[0])
    if not result:
        raise FileNotFoundError(f"No DAgger student checkpoints found below {root}")
    return result


def eval_dagger(config, metasim_env, env):
    """Evaluate every student checkpoint and write compact CSV/plots."""
    device = _device()
    checkpoints = _student_checkpoints(config.get("load_model_path"))
    camera_name = str(config.get("dagger_camera_name", "camera0"))
    target_episodes = int(config.get("eval_episodes", env.num_envs))
    max_steps = int(config.get("eval_total_step_cap", 5000))
    num_stages = int(config.get("num_eval_stages", 6))
    rows = []

    for checkpoint_step, path in checkpoints:
        student, _, num_joints = _new_student(env, device)
        _load_student_weights(student, str(path), device)
        observation = env.reset()
        returns = np.zeros(env.num_envs, dtype=np.float64)
        lengths = np.zeros(env.num_envs, dtype=np.int64)
        completed_returns = []
        completed_lengths = []
        completed_successes = []
        end_stage_counts = np.zeros(num_stages, dtype=np.int64)

        for _ in range(max_steps):
            _, images, joints = _student_inputs(
                metasim_env, observation, device, num_joints, camera_name
            )
            stages = np.clip(env.get_current_stages(), 0, num_stages - 1)
            with torch.no_grad():
                actions = student(images, joints).cpu().numpy()
            actions = np.clip(actions, env.action_space.low, env.action_space.high)
            observation, rewards, dones, infos = env.step(actions)
            returns += np.asarray(rewards, dtype=np.float64)
            lengths += 1
            for index in np.flatnonzero(dones):
                if len(completed_returns) >= target_episodes:
                    break
                completed_returns.append(float(returns[index]))
                completed_lengths.append(int(lengths[index]))
                completed_successes.append(
                    int(infos[index].get("is_success", False))
                )
                end_stage_counts[int(stages[index])] += 1
                returns[index] = 0.0
                lengths[index] = 0
            if len(completed_returns) >= target_episodes:
                break

        if len(completed_returns) < target_episodes and bool(
            config.get("eval_count_timeouts_as_failures", True)
        ):
            remaining = target_episodes - len(completed_returns)
            for index in range(min(remaining, env.num_envs)):
                completed_returns.append(float(returns[index]))
                completed_lengths.append(int(lengths[index]))
                completed_successes.append(0)
                stage = int(np.clip(env.get_current_stages()[index], 0, num_stages - 1))
                end_stage_counts[stage] += 1

        finite_step = (
            max((row["x_step"] for row in rows), default=-1) + 1
            if checkpoint_step == 10**18 else checkpoint_step
        )
        row = {
            "checkpoint": path.name,
            "x_step": finite_step,
            "mean_reward": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "std_reward": float(np.std(completed_returns)) if completed_returns else 0.0,
            "success_rate": float(np.mean(completed_successes)) if completed_successes else 0.0,
            "success_count": int(np.sum(completed_successes)),
            "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "std_length": float(np.std(completed_lengths)) if completed_lengths else 0.0,
            "episodes_finished": len(completed_returns),
        }
        for stage in range(num_stages):
            row[f"end_stage_{stage}_count"] = int(end_stage_counts[stage])
        rows.append(row)
        log.info(
            "DAgger eval {}: reward={:.4f}, success={}/{} ({:.1%}), "
            "length={:.1f}, end_stages={}",
            path.name, row["mean_reward"], row["success_count"],
            row["episodes_finished"], row["success_rate"],
            row["mean_length"], end_stage_counts.tolist(),
        )

    output_dir = Path(config.get("load_model_path"))
    if output_dir.is_file():
        output_dir = output_dir.parent
    csv_path = output_dir / "eval_dagger_results.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        steps = [row["x_step"] for row in rows]
        plots = (
            ("mean_reward", "Average reward", "eval_dagger_reward_plot.png"),
            ("success_rate", "Success rate", "eval_dagger_success_plot.png"),
            ("mean_length", "Average episode length", "eval_dagger_length_plot.png"),
        )
        for key, title, filename in plots:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, [row[key] for row in rows], marker="o")
            plt.xlabel("DAgger training step")
            plt.ylabel(title)
            plt.title(f"ChairMan separate DAgger – {title}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / filename)
            plt.close()
    except ImportError:
        log.warning("matplotlib is unavailable; CSV was saved without plots")
    log.info("Saved DAgger evaluation results to {}", csv_path)


def run_dagger(config, metasim_env, env):
    mode = config.get("train_or_eval")
    if mode in ("train_dagger", "load_and_train_dagger"):
        return train_dagger(config, metasim_env, env)
    if mode == "eval_dagger":
        return eval_dagger(config, metasim_env, env)
    if mode == "eval_dagger_video":
        return eval_dagger_video(config, metasim_env, env)
    raise ValueError(f"Unsupported DAgger mode: {mode}")
