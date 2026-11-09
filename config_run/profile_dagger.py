"""Short, isolated ChairMan separate DAgger benchmark (uses the real teacher).

Example:
    python config_run/profile_dagger.py --num-envs 100 --steps 101 --cprofile

Run from the repository root in the Genesis/Nyx Python environment. Outputs
are isolated from training checkpoints. Replay capacity and student settings
are inherited from the config unless explicitly overridden. --sync-timing
adds CUDA-synchronized phase timing for bottleneck analysis.
"""
from __future__ import annotations

import argparse
import cProfile
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import pstats
import random
import sys
import time
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config_run"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="chairman_separate/train_dagger")
    parser.add_argument("--config-file", type=Path, help="Use an exact generated candidate YAML")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--student", choices=("auto", "legacy", "recurrent"), default="auto")
    parser.add_argument("--sequence-batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--updates-per-train", type=int)
    parser.add_argument("--train-every", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--replay-device", choices=("cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Suppress repetitive asset INFO logs")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--visual-warmup-updates", type=int, default=0,
                        help="Benchmark fine-tuning by default; use N to explicitly measure frozen-backbone warmup")
    parser.add_argument("--buffer-steps", type=int)
    parser.add_argument("--render-mode", choices=("RefPathTracer", "FastPathTracer"))
    parser.add_argument("--spp", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--scale-collection", action="store_true",
                        help="Store all envs and scale batch/replay/beta consistently with the config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--sync-timing", action="store_true",
                        help="Synchronize CUDA phase boundaries for diagnostic wall-time attribution")
    parser.add_argument("--verify-body-states", action="store_true")
    parser.add_argument("--eager-cameras", action="store_true",
                        help="Render during physics state reads, for comparison with deferred rendering")
    args = parser.parse_args()
    if (any(value is not None and value <= 0 for value in (args.num_envs, args.steps, args.buffer_steps, args.threads))
            or args.warmup < 0 or args.visual_warmup_updates < 0):
        parser.error("env count, steps and buffer steps must be positive; warmup nonnegative")

    import numpy as np
    import torch
    from main_separate import build_env, load_config_from_yaml
    import separate_dagger
    import yaml
    config = yaml.safe_load(args.config_file.read_text()) if args.config_file else load_config_from_yaml(args.config)
    if args.student == "auto":
        args.student = "recurrent" if "_new" in config.get("train_or_eval", "") else "legacy"
    args.num_envs = args.num_envs or int(config["num_envs"])
    args.threads = args.threads or int(config.get("dagger_torch_threads", 4))
    if args.quiet:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if args.student == "recurrent":
        from config_run.dagger_recurrent import runner as training_module
    else:
        training_module = separate_dagger
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    output = args.output or ROOT / "config_run/output/dagger_profile" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    if args.scale_collection:
        from config_run.dagger_vp.scaling import scale_dagger_config
        config = scale_dagger_config(config, args.num_envs)
    config.update(
        num_envs=args.num_envs,
        train_or_eval="train_dagger_new" if args.student == "recurrent" else "train_dagger",
        headless=True,
        total_timesteps=args.steps,
        dagger_store_per_step=min(args.num_envs, int(config.get("dagger_store_per_step", 32))),
        model_save_freq=0,
        model_save_path=str(output / "student"),
        tensorboard_log=str(output / "tensorboard"),
        log_termination_reasons=False,
        snapshot_save_probability=0.0,
        dagger_defer_camera_rendering=not args.eager_cameras,
        dagger_log_every=args.log_every,
        dagger_gpu_poll_seconds=1,
        # Benchmark renderer/learning throughput under a fixed expert rollout;
        # a short run must not accelerate the production beta schedule to zero.
        beta_schedule="linear",
        beta_start=1.0,
        beta_end=1.0,
        # Benchmark only collection/learning, not independent evaluation.
        dagger_teacher_baseline=False,
        dagger_eval_every=0,
        # A frozen encoder would understate the steady-state training cost.
        dagger_new_visual_warmup_updates=args.visual_warmup_updates,
    )
    if args.buffer_steps is not None:
        config['dagger_buffer_steps'] = args.buffer_steps
    if args.student == "recurrent":
        for key, value in (("dagger_new_batch_size", args.sequence_batch_size),
                           ("dagger_new_sequence_length", args.sequence_length),
                           ("dagger_new_burn_in", args.burn_in),
                           ("dagger_new_buffer_device", args.replay_device)):
            if value is not None:
                config[key] = value
    elif args.replay_device is not None:
        config['dagger_buffer_device'] = args.replay_device
    if args.updates_per_train is not None:
        config["dagger_updates_per_train"] = args.updates_per_train
    if args.train_every is not None:
        config["dagger_train_every"] = args.train_every
    for camera in config.get("cameras", {}).values():
        if args.render_mode:
            camera["params"]["render_mode"] = args.render_mode
        if args.spp is not None:
            camera["params"]["spp"] = args.spp
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # build_env replaces robot config strings with configuration objects.
    # Keep the exact serializable input, rather than a reference it mutates.
    effective_config = deepcopy(config)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    start = time.perf_counter()
    base, env = build_env(config)
    build_seconds = time.perf_counter() - start
    timers = {}
    phase_timers, steady_phase_timers, steady_timers = {}, {}, {}
    current_phase = None
    steady_iteration = False
    originals = []
    monitors = []
    original_monitor = training_module.HardwareMonitor
    originals.append((training_module, 'HardwareMonitor', original_monitor))
    def monitored(device, interval=10.):
        monitor = original_monitor(device, interval=1., retain_samples=True)
        monitors.append(monitor)
        return monitor
    training_module.HardwareMonitor = monitored

    def synchronize():
        if args.sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    def record(target, label, elapsed):
        entry = target.setdefault(label, {"seconds": 0.0, "calls": 0})
        entry["seconds"] += elapsed
        entry["calls"] += 1

    original_metrics = training_module.DaggerMetrics
    originals.append((training_module, 'DaggerMetrics', original_metrics))
    def timed_metrics(*a, **kw):
        metrics = original_metrics(*a, **kw)
        original_measure = metrics.measure
        @contextmanager
        def phase_measure(label):
            nonlocal current_phase, steady_iteration
            if label == 'camera_and_inputs':
                steady_iteration = metrics.steps >= args.log_every
            previous = current_phase
            current_phase = label
            synchronize()
            started = time.perf_counter()
            try:
                with original_measure(label):
                    yield
                    synchronize()
            finally:
                elapsed = time.perf_counter() - started
                record(phase_timers, label, elapsed)
                if steady_iteration:
                    record(steady_phase_timers, label, elapsed)
                current_phase = previous
        metrics.measure = phase_measure
        return metrics
    training_module.DaggerMetrics = timed_metrics

    def measure(obj, name, label, sync=False, phases=None):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        def measured(*a, **kw):
            if phases is not None and current_phase not in phases:
                return original(*a, **kw)
            actual_label = label(current_phase) if callable(label) else label
            if sync:
                synchronize()
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                if sync:
                    synchronize()
                elapsed = time.perf_counter() - start
                record(timers, actual_label, elapsed)
                if steady_iteration:
                    record(steady_timers, actual_label, elapsed)
        setattr(obj, name, measured)

    profile = cProfile.Profile() if args.cprofile else None
    try:
        env.reset()
        if args.verify_body_states:
            handler = base.env.handler
            # Include reordered/subset environments to catch row and link-order
            # regressions. Check nonzero velocities again after stepping below.
            def verify():
                from metasim.cfg.objects import ArticulationObjCfg
                for name, cfg in handler.object_dict.items():
                    if not isinstance(cfg, ArticulationObjCfg):
                        continue
                    entity = handler.object_inst_dict[name]
                    for ids in (None, list(reversed(range(args.num_envs)))[:2]):
                        expected = torch.cat([
                            torch.stack([getattr(link, getter)(envs_idx=ids) for link in entity.links], dim=1)
                            for getter in ("get_pos", "get_quat", "get_vel", "get_ang")
                        ], dim=-1)
                        torch.testing.assert_close(handler.get_body_states(name, ids), expected)
            verify()
        for _ in range(args.warmup):
            env.step(np.zeros((args.num_envs, *env.action_space.shape), dtype=np.float32))
        if args.verify_body_states:
            verify()
            print("Body states match individual link reads before/after stepping, including reordered envs.", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        handler = base.env.handler
        measure(env, "step", "env_step")
        measure(handler, "_simulate", "physics")
        measure(handler, "get_body_states", "body_states")
        for name, camera in handler.camera_inst_dict.items():
            measure(camera, "read", f"camera_read/{name}")
        if args.student == "recurrent":
            measure(training_module.StudentInputs, "read", "student_inputs")
            measure(training_module, "train_sequence_step", "student_update")
            measure(training_module.SequenceBuffer, 'sample', 'replay_sample', sync=True)
            measure(training_module.RecurrentVisionStudent, 'forward',
                    lambda phase: 'student_rollout_forward' if phase == 'inference_and_actions' else 'student_training_forward',
                    sync=True, phases=('inference_and_actions', 'student_updates'))
            measure(torch.autograd, 'backward', 'backward', sync=True, phases=('student_updates',))
            measure(torch.optim.Adam, 'step', 'optimizer_step', sync=True, phases=('student_updates',))
            loader = training_module.load_policy_router
            originals.append((training_module, 'load_policy_router', loader))
            def timed_loader(*a, **kw):
                teacher, metadata = loader(*a, **kw)
                measure(teacher, 'predict', 'teacher_predict', sync=True)
                return teacher, metadata
            training_module.load_policy_router = timed_loader
        else:
            measure(separate_dagger, "_student_inputs", "student_inputs")
            measure(separate_dagger, "train_dagger_step", "student_update")
        if profile:
            profile.enable()
        start = time.perf_counter()
        training_module.train_dagger(config, base, env)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if profile:
            profile.disable()
            profile.dump_stats(str(output / "training.prof"))
            with (output / "profile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(100)
        summary = {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "num_envs": args.num_envs, "vector_steps": args.steps,
            "transitions": args.steps * args.num_envs,
            "build_seconds": build_seconds, "train_seconds_including_setup_and_save": seconds,
            "vector_steps_per_second": args.steps / seconds,
            "transitions_per_second": args.steps * args.num_envs / seconds,
            "peak_torch_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "cprofile": args.cprofile,
            "student": args.student,
            "image_shape": [3, int(config.get('dagger_image_height', 128)), int(config.get('dagger_image_width', 128))],
            "effective_config": effective_config,
            "timers_inclusive_do_not_sum": timers,
            "steady_timers_inclusive_do_not_sum": steady_timers,
            "phase_timings_nonoverlapping": phase_timers,
            "steady_phase_timings_nonoverlapping": steady_phase_timers,
            "timing_cuda_synchronized": args.sync_timing,
            "steady_measured_vector_steps": max(0, args.steps - args.log_every),
        }
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_root = output / "tensorboard"
        if args.student == "recurrent":
            event_root = event_root / "recurrent"
        event_run = next(event_root.iterdir())
        events = EventAccumulator(str(event_run), size_guidance={"scalars": 0})
        events.Reload()
        rates = events.Scalars("Speed/stored_samples_per_second")
        # Discard initial warmup/first optimizer work and the final short window.
        from config_run.dagger_vp.scaling import steady_sample_rate
        summary["steady_stored_samples_per_second"] = steady_sample_rate(rates, args.log_every)
        summary["store_per_step"] = config["dagger_store_per_step"]
        summary["batch_size"] = config["dagger_batch_size"]
        hardware = [sample for monitor in monitors for sample in (monitor.samples or [])]
        summary['hardware_samples'] = hardware
        summary['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        summary['peak_torch_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        summary['sampled_peak_device_memory_bytes'] = max(
            (sample.get('GPU/device_memory_used_mib', 0) * 2**20 for sample in hardware), default=None)
        # Use actual per-window learning progress to reject collection-only wins.
        tags = events.Tags()['scalars']
        for tag, key in (("Speed/trained_samples_per_second", "steady_trained_samples_per_second"),
                         ("Student/window_mse_loss", "steady_mse_loss")):
            summary[key] = steady_sample_rate(events.Scalars(tag), args.log_every) if tag in tags else None
        summary['mean_gpu_utilization_percent'] = mean(
            [s['GPU/utilization_percent'] for s in hardware if 'GPU/utilization_percent' in s]
        ) if any('GPU/utilization_percent' in s for s in hardware) else None
        summary['student_update_calls'] = timers.get('student_update', {}).get('calls', 0)
        if args.student == "recurrent":
            summary["sequence_batch_size"] = config.get('dagger_new_batch_size', 8)
            summary["sequence_length"] = config.get('dagger_new_sequence_length', 16)
            summary["burn_in"] = config.get('dagger_new_burn_in', 8)
            for tag in ("Speed/vector_steps_per_second", "Speed/trained_samples_per_second",
                        "GPU/utilization_percent", "GPU/device_memory_used_mib", "Student/window_mse_loss"):
                if tag in events.Tags()["scalars"]:
                    summary[tag] = [{"step": event.step, "value": event.value} for event in events.Scalars(tag)]
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Results: {output}", flush=True)
    finally:
        if profile:
            profile.disable()
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        env.close()


if __name__ == "__main__":
    main()
