"""Short, isolated ChairMan separate DAgger benchmark (uses the real teacher).

Example:
    python config_run/profile_dagger.py --num-envs 100 --steps 101 --cprofile

Run from the repository root in the Genesis/Nyx Python environment. Outputs
are isolated from training checkpoints. Replay capacity is deliberately small;
use --buffer-steps 3000 to reproduce the training configuration's allocation.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config_run"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="chairman_separate/train_dagger")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--buffer-steps", type=int, default=100)
    parser.add_argument("--render-mode", choices=("RefPathTracer", "FastPathTracer"))
    parser.add_argument("--spp", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--scale-collection", action="store_true",
                        help="Store all envs and scale batch/replay/beta consistently with the config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--verify-body-states", action="store_true")
    parser.add_argument("--eager-cameras", action="store_true",
                        help="Render during physics state reads, for comparison with deferred rendering")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.buffer_steps) <= 0 or args.warmup < 0:
        parser.error("env count, steps and buffer steps must be positive; warmup nonnegative")

    import numpy as np
    import torch
    from main_separate import build_env, load_config_from_yaml
    import separate_dagger

    output = args.output or ROOT / "config_run/output/dagger_profile" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    config = load_config_from_yaml(args.config)
    if args.scale_collection:
        from config_run.dagger_vp.scaling import scale_dagger_config
        config = scale_dagger_config(config, args.num_envs)
    config.update(
        num_envs=args.num_envs,
        train_or_eval="train_dagger",
        total_timesteps=args.steps,
        dagger_store_per_step=min(args.num_envs, int(config.get("dagger_store_per_step", 32))),
        dagger_buffer_steps=args.buffer_steps,
        model_save_freq=0,
        model_save_path=str(output / "student"),
        tensorboard_log=str(output / "tensorboard"),
        log_termination_reasons=False,
        snapshot_save_probability=0.0,
        dagger_defer_camera_rendering=not args.eager_cameras,
        dagger_log_every=args.log_every,
        # Benchmark renderer/learning throughput under a fixed expert rollout;
        # a short run must not accelerate the production beta schedule to zero.
        beta_schedule="linear",
        beta_start=1.0,
        beta_end=1.0,
    )
    for camera in config.get("cameras", {}).values():
        if args.render_mode:
            camera["params"]["render_mode"] = args.render_mode
        if args.spp is not None:
            camera["params"]["spp"] = args.spp
    (output / "config.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    start = time.perf_counter()
    base, env = build_env(config)
    build_seconds = time.perf_counter() - start
    timers = {}
    originals = []

    def measure(obj, name, label):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                entry = timers.setdefault(label, {"seconds": 0.0, "calls": 0})
                entry["seconds"] += time.perf_counter() - start
                entry["calls"] += 1
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
        measure(separate_dagger, "_student_inputs", "student_inputs")
        measure(separate_dagger, "train_dagger_step", "student_update")
        if profile:
            profile.enable()
        start = time.perf_counter()
        separate_dagger.train_dagger(config, base, env)
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
            "timers_inclusive_do_not_sum": timers,
        }
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_run = next((output / "tensorboard").iterdir())
        events = EventAccumulator(str(event_run), size_guidance={"scalars": 0})
        events.Reload()
        rates = events.Scalars("Speed/stored_samples_per_second")
        # Discard initial warmup/first optimizer work and the final short window.
        from config_run.dagger_vp.scaling import steady_sample_rate
        summary["steady_stored_samples_per_second"] = steady_sample_rate(rates, args.log_every)
        summary["store_per_step"] = config["dagger_store_per_step"]
        summary["batch_size"] = config["dagger_batch_size"]
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
