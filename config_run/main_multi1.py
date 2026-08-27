"""Stage-wise PPO teachers and student training for the ChairMan task.

The original :mod:`main` trains one PPO policy over the complete staged task.
This entry point keeps the same YAML-driven PPO/DAgger/GRPO workflows, while
using one independent PPO teacher for each actionable ChairMan stage (0..5).

Training is sequential and on-policy: completing the currently trained stage
ends that PPO episode, saves a simulator snapshot for the next stage, and
resets into the same stage.  Full-task inference disables those artificial
episode boundaries and dispatches each environment row to the teacher selected
by its current stage.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from loguru import logger as log
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from metasim.cfg.lights import CylinderLightCfg, DistantLightCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import (
    CommandCfg,
    GyroSensorCfg,
    NyxGaussianSplatCameraCfg,
    PinholeCameraCfg,
)
from metasim.wrapper.gym_vec_env import MetaSimVecEnv

try:
    from .callbacks import TensorboardMetricsCallback
    from .utils import ObsSaver
except ImportError:  # ``python config_run/main_multi.py ...``
    from callbacks import TensorboardMetricsCallback
    from utils import ObsSaver


NUM_CHAIRMAN_POLICIES = 6
MANIFEST_NAME = "multi_policy_manifest.json"


def load_config_from_yaml(config_name: str) -> dict:
    config_path = Path(__file__).resolve().parent / "configs" / f"{config_name}.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_lights_from_config(lights_config: dict):
    lights = []
    for light_config in lights_config.values():
        light_type = light_config.get("type")
        params = dict(light_config.get("params", {}))
        if light_type == "DistantLightCfg":
            lights.append(DistantLightCfg(**params))
        elif light_type == "CylinderLightCfg":
            lights.append(CylinderLightCfg(**params))
        else:
            log.warning(f"Unknown light type: {light_type}, skipping")
    return lights


def get_sensors_from_config(sensors_config: dict):
    sensors = []
    for sensor_config in sensors_config.values():
        sensor_type = sensor_config.get("type")
        params = dict(sensor_config.get("params", {}))
        if sensor_type == "GyroSensorCfg":
            sensor = GyroSensorCfg(**params)
            if isinstance(sensor.pos, str):
                sensor.pos = tuple(map(float, sensor.pos.strip("()").split(",")))
            sensors.append(sensor)
        elif sensor_type == "CommandCfg":
            sensors.append(CommandCfg(**params))
        else:
            log.warning(f"Unknown sensor type: {sensor_type}, skipping")
    return sensors


def get_cameras_from_config(cameras_config: dict):
    cameras = []
    for camera_config in cameras_config.values():
        camera_type = camera_config.get("type")
        params = dict(camera_config.get("params", {}))
        if camera_type == "PinholeCameraCfg":
            cameras.append(PinholeCameraCfg(**params))
        elif camera_type == "NyxGaussianSplatCameraCfg":
            cameras.append(NyxGaussianSplatCameraCfg(**params))
        else:
            log.warning(f"Unknown camera type: {camera_type}, skipping")
    return cameras


def _numpy_pickle_compatibility() -> None:
    # Checkpoints produced with older NumPy versions can reference this module.
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.numeric"] = np.core.numeric


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)


def _read_stages(metasim_env, num_envs: int) -> np.ndarray:
    try:
        stages = metasim_env.env.handler.task.reward_functions[0].actual_stage
        if stages is None:
            return np.zeros(num_envs, dtype=np.int64)
        return stages.detach().cpu().numpy().astype(np.int64, copy=True)
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("ChairMan actual_stage is not available") from exc


def _chair_xy(metasim_env) -> np.ndarray:
    states = metasim_env.env.handler.get_states()
    chair = states.objects["chair"]
    base_idx = chair.body_names.index("base_link")
    return chair.body_state[:, base_idx, :2].detach().cpu().numpy().copy()


def _student_inputs(metasim_env, device: str):
    """Return the same RGB/joint inputs for DAgger, GRPO, and evaluation."""
    states = metasim_env.env.handler.get_states()
    robot_name = metasim_env.scenario.robots[0].name
    images_u8 = states.cameras["camera0"].rgb.permute(0, 3, 1, 2).contiguous()
    images_f32 = images_u8.to(device=device, dtype=torch.float32) / 255.0
    joints_f32 = states.robots[robot_name].joint_pos.to(
        device=device, dtype=torch.float32
    )
    return images_u8, images_f32, joints_f32


def _build_scenario(config: dict, *, dagger_mode: int, training_stage: int | None):
    task_name = config.get("task")
    if task_name != "chairmanmulti":
        raise ValueError(
            "main_multi.py requires task: chairmanmulti. "
            f"The config currently contains task: {task_name!r}."
        )

    scenario_kwargs = dict(
        task=task_name,
        robots=config.get("robots"),
        try_add_table=config.get("try_add_table", config.get("add_table", True)),
        sim=config.get("sim"),
        num_envs=config.get("num_envs", 1),
        headless=config.get("headless", False),
        sensors=get_sensors_from_config(config.get("sensors", {})),
        cameras=get_cameras_from_config(config.get("cameras", {})),
        force=config.get("force", False),
        force_x_min=config.get("force_x_min", 0.0),
        force_x_max=config.get("force_x_max", 0.0),
        force_y_min=config.get("force_y_min", 0.0),
        force_y_max=config.get("force_y_max", 0.0),
    )
    if config.get("lights"):
        scenario_kwargs["lights"] = get_lights_from_config(config["lights"])
    scenario = ScenarioCfg(**scenario_kwargs)

    scenario.env_spacing = config.get("env_spacing", 2.0)
    scenario.robots[0].fix_base_link = config.get("fix_base_link", False)
    scenario.task.decimation = config.get("decimation", 1)
    scenario.task.training_stage = training_stage
    scenario.dagger = dagger_mode

    if scenario.robots[0].fix_base_link:
        raise ValueError("ChairMan multi-policy walking requires fix_base_link: false")
    if scenario.robots[0].name != "g1_with_hands":
        raise ValueError(
            "ChairMan multi-policy uses the full walking robot; set "
            "robots: [g1_with_hands]."
        )
    return scenario


def _build_env(config: dict, *, dagger_mode: int = 0, training_stage: int | None = None):
    scenario = _build_scenario(
        config, dagger_mode=dagger_mode, training_stage=training_stage
    )
    try:
        from .SB3_chairman_env import StableBaseline3VecEnv
    except ImportError:
        from SB3_chairman_env import StableBaseline3VecEnv

    metasim_env = MetaSimVecEnv(
        scenario,
        task_name=config.get("task"),
        num_envs=config.get("num_envs", 1),
        sim=config.get("sim"),
    )
    return scenario, metasim_env, StableBaseline3VecEnv(metasim_env)


def _policy_kwargs(config: dict) -> dict:
    if config.get("net_arch_pivf", False):
        net_arch = {
            "pi": config.get("net_arch_pi", [128, 128, 128]),
            "vf": config.get("net_arch_vf", [128, 128, 128]),
        }
    else:
        net_arch = config.get("net_arch", [128, 128, 128])
    return {
        "net_arch": net_arch,
        "log_std_init": config.get("log_std_init", 0.0),
    }


def _learning_rate(config: dict):
    initial = float(config.get("learning_rate", 3e-4))
    if config.get("learning_schedule", "constant") != "linear":
        return initial
    final = float(config.get("final_learning_rate", 0.0))
    return lambda progress_remaining: final + (initial - final) * progress_remaining


def _new_ppo(env, config: dict) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=_learning_rate(config),
        n_steps=config.get("n_steps", 128),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        tensorboard_log=config.get("tensorboard_log", "./ppo_tensorboard/"),
        policy_kwargs=_policy_kwargs(config),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _stage_value(config: dict, name: str, stage: int, default):
    value = config.get(name, default)
    if isinstance(value, dict):
        return value.get(stage, value.get(str(stage), default))
    if isinstance(value, (list, tuple)):
        if len(value) != NUM_CHAIRMAN_POLICIES:
            raise ValueError(
                f"{name} must contain {NUM_CHAIRMAN_POLICIES} values, got {len(value)}"
            )
        return value[stage]
    return value


class StageTrainingCallback(BaseCallback):
    """Save one stage policy and stop when its rolling success target is met."""

    def __init__(self, stage: int, stage_dir: Path, config: dict):
        super().__init__(verbose=1)
        self.stage = stage
        self.stage_dir = stage_dir
        self.save_freq = int(config.get("model_save_freq", 1_000_000))
        self.log_freq = int(config.get("stage_log_interval", self.save_freq))
        self.threshold = config.get("stage_advance_success_rate", 0.9)
        if self.threshold is not None:
            self.threshold = float(self.threshold)
        window = int(config.get("stage_success_window", 1000))
        self.success_window = deque(maxlen=max(1, window))
        self.min_episodes = int(config.get("stage_min_episodes", window))
        default_min_steps = int(config.get("n_steps", 128)) * int(
            config.get("num_envs", 1)
        )
        self.min_timesteps = int(config.get("stage_min_timesteps", default_min_steps))
        self.last_save = 0
        self.last_log = 0
        self.threshold_reached = False

    @property
    def rolling_success_rate(self) -> float:
        return float(np.mean(self.success_window)) if self.success_window else 0.0

    def _on_step(self) -> bool:
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [{} for _ in range(len(dones))])
        for done, info in zip(dones, infos):
            if done:
                self.success_window.append(bool(info.get("is_success", False)))

        if self.num_timesteps - self.last_save >= self.save_freq:
            path = self.stage_dir / f"model_{self.num_timesteps}"
            self.model.save(path)
            self.last_save = self.num_timesteps
            log.info(f"Stage {self.stage}: checkpoint saved to {path}.zip")

        if self.num_timesteps - self.last_log >= self.log_freq:
            self.last_log = self.num_timesteps
            log.info(
                f"Stage {self.stage} | timesteps={self.num_timesteps} | "
                f"rolling_success={self.rolling_success_rate:.3f} | "
                f"episodes_in_window={len(self.success_window)}"
            )

        enough_data = (
            self.num_timesteps >= self.min_timesteps
            and len(self.success_window) >= self.min_episodes
        )
        if (
            self.threshold is not None
            and enough_data
            and self.rolling_success_rate >= self.threshold
        ):
            self.threshold_reached = True
            log.info(
                f"Stage {self.stage} learned: rolling success "
                f"{self.rolling_success_rate:.3f} >= {self.threshold:.3f}"
            )
            return False
        return True


def _manifest_payload(config: dict, run_dir: Path, stages: dict[int, dict]) -> dict:
    return {
        "format_version": 1,
        "task": config.get("task"),
        "num_stage_policies": NUM_CHAIRMAN_POLICIES,
        "actionable_stages": list(range(NUM_CHAIRMAN_POLICIES)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stages": {str(stage): value for stage, value in stages.items()},
        "run_dir": str(run_dir.resolve()),
    }


def _write_manifest(config: dict, run_dir: Path, stages: dict[int, dict]) -> None:
    path = run_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(_manifest_payload(config, run_dir, stages), indent=2),
        encoding="utf-8",
    )


def _train_ppo(config: dict, *, resume: bool = False) -> None:
    scenario, _, env = _build_env(config, training_stage=0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = Path(config.get("model_save_path", "./output/ppo_multi"))
    run_dir = run_root / f"run_{timestamp}_{config.get('task')}_multi"
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_paths = None
    if resume:
        resume_paths, _ = _resolve_policy_bundle(
            config.get("load_model_path"), NUM_CHAIRMAN_POLICIES
        )

    stages_manifest: dict[int, dict] = {}
    try:
        for stage in range(NUM_CHAIRMAN_POLICIES):
            scenario.task.training_stage = stage
            env.reset()

            stage_dir = run_dir / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_timesteps = int(
                _stage_value(
                    config,
                    "stage_total_timesteps",
                    stage,
                    config.get("total_timesteps", 1_000_000),
                )
            )
            if stage_timesteps <= 0:
                raise ValueError(f"Stage {stage} has invalid timestep budget {stage_timesteps}")

            if resume_paths is None:
                model = _new_ppo(env, config)
            else:
                model = PPO.load(
                    resume_paths[stage],
                    env=env,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            stage_callback = StageTrainingCallback(stage, stage_dir, config)
            metrics_callback = TensorboardMetricsCallback(
                log_dir=str(
                    Path(config.get("tensorboard_log", "./ppo_tensorboard"))
                    / run_dir.name
                    / f"stage_{stage}"
                ),
                log_interval=config.get("tensorboard_log_interval", 100_000),
                max_stage=NUM_CHAIRMAN_POLICIES,
                verbose=1,
            )

            log.info(
                f"Training independent PPO policy for ChairMan stage {stage} "
                f"with a maximum of {stage_timesteps} timesteps"
            )
            model.learn(
                total_timesteps=stage_timesteps,
                callback=[stage_callback, metrics_callback],
                progress_bar=config.get("progress_bar", True),
                reset_num_timesteps=not resume,
            )

            final_path = stage_dir / "model_final"
            model.save(final_path)
            stages_manifest[stage] = {
                "model": str(Path(f"stage_{stage}") / "model_final.zip"),
                "timesteps": int(model.num_timesteps),
                "rolling_success_rate": stage_callback.rolling_success_rate,
                "threshold_reached": stage_callback.threshold_reached,
            }
            _write_manifest(config, run_dir, stages_manifest)
            log.info(f"Stage {stage} final policy saved to {final_path}.zip")

            require_threshold = bool(config.get("require_stage_success_threshold", True))
            threshold_enabled = stage_callback.threshold is not None
            if (
                stage < NUM_CHAIRMAN_POLICIES - 1
                and require_threshold
                and threshold_enabled
                and not stage_callback.threshold_reached
            ):
                raise RuntimeError(
                    f"Stage {stage} exhausted its budget without reaching the "
                    "configured success threshold. The next policy was not trained. "
                    "Increase stage_total_timesteps or set "
                    "require_stage_success_threshold: false to advance anyway."
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(f"All stage policies trained. Multi-policy bundle: {run_dir}")
    finally:
        env.close()


def _existing_model_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    zipped = Path(f"{path}.zip")
    return zipped if zipped.is_file() else None


def _latest_stage_model(stage_dir: Path) -> Path | None:
    for name in ("model_final.zip", "best_model.zip"):
        candidate = stage_dir / name
        if candidate.is_file():
            return candidate
    checkpoints = []
    for candidate in stage_dir.glob("model_*.zip"):
        match = re.fullmatch(r"model_(\d+)\.zip", candidate.name)
        if match:
            checkpoints.append((int(match.group(1)), candidate))
    return max(checkpoints, default=(None, None), key=lambda item: item[0])[1]


def _resolve_policy_bundle(
    bundle_path: str | os.PathLike[str] | None,
    num_stages: int = NUM_CHAIRMAN_POLICIES,
) -> tuple[dict[int, str], Path]:
    if not bundle_path:
        raise ValueError("A multi-policy bundle path is required")
    root = Path(bundle_path).expanduser()
    if root.is_file() and root.name == MANIFEST_NAME:
        root = root.parent

    if root.is_dir() and not (root / MANIFEST_NAME).exists():
        direct_stage_dirs = all((root / f"stage_{i}").is_dir() for i in range(num_stages))
        if not direct_stage_dirs:
            run_dirs = sorted(
                (
                    child
                    for child in root.iterdir()
                    if child.is_dir() and (child / MANIFEST_NAME).is_file()
                ),
                key=lambda child: child.stat().st_mtime,
            )
            if run_dirs:
                root = run_dirs[-1]

    manifest_path = root / MANIFEST_NAME
    resolved: dict[int, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_count = int(manifest.get("num_stage_policies", -1))
        if manifest_count != num_stages:
            raise ValueError(
                f"Bundle contains {manifest_count} policies, expected {num_stages}"
            )
        for stage in range(num_stages):
            entry = manifest.get("stages", {}).get(str(stage), {})
            model_value = entry.get("model")
            if not model_value:
                raise ValueError(f"Bundle manifest has no model for stage {stage}")
            candidate = Path(model_value)
            if not candidate.is_absolute():
                candidate = root / candidate
            model_path = _existing_model_path(candidate)
            if model_path is None:
                raise FileNotFoundError(
                    f"Stage {stage} model from manifest does not exist: {candidate}"
                )
            resolved[stage] = str(model_path)
    else:
        for stage in range(num_stages):
            stage_dir = root / f"stage_{stage}"
            model_path = _latest_stage_model(stage_dir)
            if model_path is None:
                for candidate in (
                    root / f"stage_{stage}.zip",
                    root / f"policy_stage_{stage}.zip",
                ):
                    if candidate.is_file():
                        model_path = candidate
                        break
            if model_path is None:
                raise FileNotFoundError(
                    f"Could not find a PPO model for stage {stage} below {root}"
                )
            resolved[stage] = str(model_path)
    return resolved, root


def _load_stage_router(config: dict, env, metasim_env, bundle_path=None):
    paths, root = _resolve_policy_bundle(
        bundle_path or config.get("load_model_path"), NUM_CHAIRMAN_POLICIES
    )
    _numpy_pickle_compatibility()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = {
        stage: PPO.load(path, env=env, device=device)
        for stage, path in paths.items()
    }
    log.info(
        "Loaded stage teachers: "
        + ", ".join(f"S{stage}={paths[stage]}" for stage in sorted(paths))
    )
    return StagePolicyRouter(policies, metasim_env, env.action_space), root


class StagePolicyRouter:
    """Dispatch a vector batch to independent PPO policies by current stage."""

    def __init__(self, policies: dict[int, PPO], metasim_env, action_space):
        expected = set(range(NUM_CHAIRMAN_POLICIES))
        if set(policies) != expected:
            raise ValueError(
                f"Expected policies {sorted(expected)}, got {sorted(policies)}"
            )
        self.policies = policies
        self.metasim_env = metasim_env
        self.action_space = action_space

    def predict(
        self,
        observations: np.ndarray,
        *,
        deterministic: bool = True,
        stages: np.ndarray | None = None,
    ) -> np.ndarray:
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[None, :]
        if stages is None:
            stages = _read_stages(self.metasim_env, observations.shape[0])
        stages = np.asarray(stages, dtype=np.int64)
        if stages.shape != (observations.shape[0],):
            raise ValueError(
                f"Stage batch has shape {stages.shape}, expected {(observations.shape[0],)}"
            )

        actions = np.zeros(
            (observations.shape[0], self.action_space.shape[0]), dtype=np.float32
        )
        invalid = (stages < 0) | (stages >= NUM_CHAIRMAN_POLICIES)
        # Stage 6 is the terminal sentinel and normally gets reset in the same
        # wrapper step.  Zero action is safe if it is observed transiently.
        unexpected = invalid & (stages != NUM_CHAIRMAN_POLICIES)
        if unexpected.any():
            raise ValueError(f"Unexpected ChairMan stages: {np.unique(stages[unexpected])}")

        for stage, policy in self.policies.items():
            mask = stages == stage
            if not mask.any():
                continue
            stage_actions, _ = policy.predict(
                observations[mask], deterministic=deterministic
            )
            actions[mask] = np.asarray(stage_actions, dtype=np.float32)
        return np.clip(actions, self.action_space.low, self.action_space.high)


def _evaluate(
    env,
    metasim_env,
    action_provider: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: dict,
) -> tuple[dict, list[dict]]:
    obs = env.reset()
    requested = int(config.get("eval_episodes", env.num_envs))
    if requested > env.num_envs:
        log.warning(
            f"eval_episodes={requested} exceeds num_envs={env.num_envs}; "
            f"evaluating {env.num_envs} one-shot episodes"
        )
    episode_count = min(requested, env.num_envs)
    max_steps = int(config.get("eval_total_step_cap", config.get("eval_max_steps", 5000)))
    count_timeouts = bool(config.get("eval_count_timeouts_as_failures", True))
    num_stages = int(config.get("num_eval_stages", NUM_CHAIRMAN_POLICIES))

    active = np.zeros(env.num_envs, dtype=bool)
    active[:episode_count] = True
    finished = np.zeros(env.num_envs, dtype=bool)
    rewards_sum = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    stage_steps = np.zeros((env.num_envs, num_stages), dtype=np.int64)
    last_stage = np.zeros(env.num_envs, dtype=np.int64)
    initial_chair = _chair_xy(metasim_env)
    last_chair = initial_chair.copy()
    rows: list[dict] = []

    for _ in range(max_steps):
        running = active & ~finished
        if not running.any():
            break
        current_chair = _chair_xy(metasim_env)
        last_chair[running] = current_chair[running]
        stages = _read_stages(metasim_env, env.num_envs)
        clipped_stages = np.clip(stages, 0, num_stages - 1)
        for env_id in np.flatnonzero(running):
            stage_steps[env_id, clipped_stages[env_id]] += 1
            last_stage[env_id] = clipped_stages[env_id]

        actions = np.asarray(action_provider(obs, stages), dtype=np.float32)
        actions[~running] = 0.0
        obs, rewards, dones, infos = env.step(actions)
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        rewards_sum[running] += rewards[running]
        lengths[running] += 1

        for env_id in np.flatnonzero(running & dones):
            success = bool(infos[env_id].get("is_success", False))
            rows.append(
                {
                    "env_id": int(env_id),
                    "episode_reward": float(rewards_sum[env_id]),
                    "episode_length": int(lengths[env_id]),
                    "success": int(success),
                    "end_stage": int(last_stage[env_id]),
                    "chair_displacement_xy": float(
                        np.linalg.norm(last_chair[env_id] - initial_chair[env_id])
                    ),
                    "stage_steps": stage_steps[env_id].copy(),
                    "timeout": 0,
                }
            )
            finished[env_id] = True

    if count_timeouts:
        timeout_chair = _chair_xy(metasim_env)
        for env_id in np.flatnonzero(active & ~finished):
            rows.append(
                {
                    "env_id": int(env_id),
                    "episode_reward": float(rewards_sum[env_id]),
                    "episode_length": int(lengths[env_id]),
                    "success": 0,
                    "end_stage": int(last_stage[env_id]),
                    "chair_displacement_xy": float(
                        np.linalg.norm(timeout_chair[env_id] - initial_chair[env_id])
                    ),
                    "stage_steps": stage_steps[env_id].copy(),
                    "timeout": 1,
                }
            )
            finished[env_id] = True

    if not rows:
        mean_stage_steps = np.zeros(num_stages, dtype=np.float64)
        end_counts = np.zeros(num_stages, dtype=np.int64)
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "success_count": 0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "mean_chair_displacement_xy": 0.0,
            "std_chair_displacement_xy": 0.0,
            "episodes_evaluated": 0,
            "requested_eval_episodes": episode_count,
            "mean_stage_steps": mean_stage_steps,
            "end_stage_counts": end_counts,
        }, rows

    reward_values = np.asarray([row["episode_reward"] for row in rows])
    length_values = np.asarray([row["episode_length"] for row in rows])
    success_values = np.asarray([row["success"] for row in rows])
    chair_values = np.asarray([row["chair_displacement_xy"] for row in rows])
    mean_stage_steps = np.mean(
        np.stack([row["stage_steps"] for row in rows]), axis=0
    )
    end_counts = np.bincount(
        [row["end_stage"] for row in rows], minlength=num_stages
    )[:num_stages]
    return {
        "mean_reward": float(np.mean(reward_values)),
        "std_reward": float(np.std(reward_values)),
        "success_rate": float(np.mean(success_values)),
        "success_count": int(np.sum(success_values)),
        "mean_length": float(np.mean(length_values)),
        "std_length": float(np.std(length_values)),
        "mean_chair_displacement_xy": float(np.mean(chair_values)),
        "std_chair_displacement_xy": float(np.std(chair_values)),
        "episodes_evaluated": len(rows),
        "requested_eval_episodes": episode_count,
        "mean_stage_steps": mean_stage_steps,
        "end_stage_counts": end_counts,
    }, rows


def _save_evaluation(
    output_dir: Path,
    method: str,
    labels: list[str],
    x_values: list[float],
    results: list[dict],
    episode_rows: list[dict],
    task_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_stages = NUM_CHAIRMAN_POLICIES
    summary_rows = []
    for label, x_value, result in zip(labels, x_values, results):
        row = {
            "checkpoint": label,
            "x_step": x_value,
            **{
                key: value
                for key, value in result.items()
                if key not in ("mean_stage_steps", "end_stage_counts")
            },
        }
        for stage in range(num_stages):
            row[f"mean_stage_{stage}_steps"] = float(result["mean_stage_steps"][stage])
            row[f"end_stage_{stage}_count"] = int(result["end_stage_counts"][stage])
        summary_rows.append(row)

    if summary_rows:
        summary_path = output_dir / f"eval_{method}_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    if episode_rows:
        flat_episode_rows = []
        for row in episode_rows:
            flat = {key: value for key, value in row.items() if key != "stage_steps"}
            for stage, count in enumerate(row["stage_steps"]):
                flat[f"stage_{stage}_steps"] = int(count)
            flat_episode_rows.append(flat)
        episode_path = output_dir / f"eval_{method}_episodes.csv"
        with episode_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(flat_episode_rows[0]))
            writer.writeheader()
            writer.writerows(flat_episode_rows)

    if not results:
        return
    import matplotlib.pyplot as plt

    def line_plot(values, std_values, ylabel, title, filename, ylim=None):
        x_np = np.asarray(x_values, dtype=np.float64)
        values_np = np.asarray(values, dtype=np.float64)
        plt.figure(figsize=(10, 5))
        plt.plot(x_np, values_np, marker="o")
        if std_values is not None:
            std_np = np.asarray(std_values, dtype=np.float64)
            plt.fill_between(x_np, values_np - std_np, values_np + std_np, alpha=0.2)
        plt.title(f"{method.upper()} Evaluation - {title} ({task_name})")
        plt.xlabel("Checkpoint")
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xticks(x_np, labels, rotation=30, ha="right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    line_plot(
        [result["mean_reward"] for result in results],
        [result["std_reward"] for result in results],
        "Average reward",
        "Average Reward",
        f"eval_{method}_reward_plot.png",
    )
    line_plot(
        [result["success_rate"] for result in results],
        None,
        "Success rate",
        "Success Rate",
        f"eval_{method}_success_plot.png",
        (-0.05, 1.05),
    )
    line_plot(
        [result["mean_length"] for result in results],
        [result["std_length"] for result in results],
        "Average steps per episode",
        "Episode Length",
        f"eval_{method}_length_plot.png",
    )
    line_plot(
        [result["mean_chair_displacement_xy"] for result in results],
        [result["std_chair_displacement_xy"] for result in results],
        "Chair displacement XY [m]",
        "Chair Displacement",
        f"eval_{method}_chair_displacement_plot.png",
    )

    stage_matrix = np.stack([result["mean_stage_steps"] for result in results])
    x = np.arange(len(results))
    bottom = np.zeros(len(results), dtype=np.float64)
    plt.figure(figsize=(max(10, len(results) * 1.2), 6))
    for stage in range(num_stages):
        plt.bar(x, stage_matrix[:, stage], bottom=bottom, label=f"Stage {stage}")
        bottom += stage_matrix[:, stage]
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Average steps per episode")
    plt.title(f"{method.upper()} Evaluation - Episode Steps by Stage ({task_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"eval_{method}_stage_stacked_bar.png")
    plt.close()

    log.info(f"{method.upper()} evaluation artifacts saved to {output_dir}")


def _discover_bundles(path: str) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file() or (root / MANIFEST_NAME).is_file() or all(
        (root / f"stage_{stage}").is_dir() for stage in range(NUM_CHAIRMAN_POLICIES)
    ):
        _, resolved_root = _resolve_policy_bundle(root)
        return [resolved_root]

    if root.is_dir():
        bundles = sorted(
            child for child in root.iterdir() if (child / MANIFEST_NAME).is_file()
        )
        if bundles:
            return bundles

    _, resolved_root = _resolve_policy_bundle(root)
    return [resolved_root]


def _eval_ppo(config: dict) -> None:
    _, metasim_env, env = _build_env(config, training_stage=None)
    bundle_root = Path(config.get("load_model_path")).expanduser()
    bundles = _discover_bundles(str(bundle_root))
    results = []
    all_episode_rows = []
    labels = []
    try:
        for bundle_index, bundle in enumerate(bundles):
            router, _ = _load_stage_router(config, env, metasim_env, bundle)
            result, episode_rows = _evaluate(
                env,
                metasim_env,
                lambda obs, stages: router.predict(obs, stages=stages),
                config,
            )
            label = bundle.name
            for row in episode_rows:
                row["checkpoint"] = label
            labels.append(label)
            results.append(result)
            all_episode_rows.extend(episode_rows)
            log.info(
                f"PPO bundle {label}: success={result['success_rate']:.2%}, "
                f"reward={result['mean_reward']:.3f}, length={result['mean_length']:.1f}"
            )
            del router
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        env.close()
    _save_evaluation(
        bundle_root if bundle_root.is_dir() else bundle_root.parent,
        "ppo_multi",
        labels,
        list(range(len(results))),
        results,
        all_episode_rows,
        config.get("task"),
    )


def _eval_ppo_video(config: dict) -> None:
    _, metasim_env, env = _build_env(config, training_stage=None)
    router, _ = _load_stage_router(config, env, metasim_env)
    video_path = config.get("video_save_path", "./output/ppo_multi.mp4")
    _ensure_parent(video_path)
    saver = ObsSaver(video_path=video_path)
    slowdown = int(config.get("video_slowdown", 3))
    obs = env.reset()
    previous_stage = None
    try:
        for step in range(int(config.get("eval_max_steps", 1000))):
            stages = _read_stages(metasim_env, env.num_envs)
            current_stage = int(stages[0])
            if current_stage != previous_stage:
                log.info(f"PPO teacher switch at step {step}: {previous_stage} -> {current_stage}")
                previous_stage = current_stage
            actions = router.predict(obs, stages=stages)
            obs, rewards, dones, infos = env.step(actions)
            states = metasim_env.env.handler.get_states()
            for _ in range(slowdown):
                saver.add(states)
            if bool(dones[0]) and config.get("video_stop_on_done", True):
                log.info(
                    f"Episode finished at step {step + 1}; "
                    f"success={infos[0].get('is_success', False)}"
                )
                break
    finally:
        saver.save()
        env.close()
    log.info(f"PPO multi-policy video saved to {video_path}")


def _train_dagger(config: dict) -> None:
    from torch.utils.tensorboard import SummaryWriter
    try:
        from .dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step
        from .dagger_vp.student_net import VisionStudent
    except ImportError:
        from dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step
        from dagger_vp.student_net import VisionStudent

    _, metasim_env, env = _build_env(config, dagger_mode=1, training_stage=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expert_path = config.get("expert_model_path", config.get("load_model_path"))
    router, _ = _load_stage_router(config, env, metasim_env, expert_path)
    env.reset()
    images_u8, _, joints = _student_inputs(metasim_env, device)
    num_actions = env.action_space.shape[0]
    num_joints = joints.shape[1]
    student = VisionStudent(num_actions=num_actions, num_joints=num_joints).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=config.get("learning_rate", 3e-4)
    )

    buffer_device = config.get("dagger_buffer_device", "cpu")
    pin_memory = config.get(
        "dagger_buffer_pin_memory",
        buffer_device == "cpu" and device.startswith("cuda"),
    )
    store_per_step = int(config.get("dagger_store_per_step", 32))
    buffer = DAggerBuffer(
        max_samples=int(config.get("dagger_buffer_steps", 4000)) * store_per_step,
        img_shape=tuple(images_u8.shape[1:]),
        num_joints=num_joints,
        num_actions=num_actions,
        device=device,
        storage_device=buffer_device,
        pin_memory=pin_memory,
    )

    log_dir = Path(config.get("tensorboard_log", "./dagger_tensorboard"))
    save_dir = Path(config.get("model_save_path", "./output/dagger_models"))
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    save_freq = int(config.get("model_save_freq", 5000))
    total_iterations = int(config.get("total_timesteps", 100_000))
    beta = float(config.get("beta_start", 1.0))
    beta_decay = float(config.get("beta_decay", 0.9995))
    train_every = int(config.get("dagger_train_every", 20))
    updates_per_train = int(config.get("dagger_updates_per_train", 10))
    batch_size = int(config.get("dagger_batch_size", 512))
    expert_obs = env.reset()
    completed = successful = 0

    log.info("Starting DAgger with stage-routed PPO teachers")
    try:
        for step in range(total_iterations):
            images_u8, images_f32, joint_inputs = _student_inputs(metasim_env, device)
            stages = _read_stages(metasim_env, env.num_envs)
            with torch.no_grad():
                expert_actions = router.predict(expert_obs, stages=stages)
                expert_actions_t = torch.as_tensor(
                    expert_actions, device=device, dtype=torch.float32
                )
                student.eval()
                student_actions_t = student(images_f32, joint_inputs)

            # Standard DAgger mixing is sampled independently per environment.
            use_expert = np.random.random(env.num_envs) < beta
            env_actions = student_actions_t.detach().cpu().numpy()
            env_actions[use_expert] = expert_actions[use_expert]
            env_actions = np.clip(env_actions, env.action_space.low, env.action_space.high)

            buffer.add_batch(
                images_u8.to(torch.uint8),
                joint_inputs,
                expert_actions_t,
                store_count=store_per_step,
            )
            expert_obs, rewards, dones, infos = env.step(env_actions)
            completed += int(np.sum(dones))
            successful += sum(
                int(info.get("is_success", False))
                for done, info in zip(dones, infos)
                if done
            )

            if step > 0 and step % train_every == 0:
                losses = [
                    train_dagger_step(student, optimizer, buffer, batch_size=batch_size)
                    for _ in range(updates_per_train)
                ]
                mean_loss = float(np.mean(losses)) if losses else 0.0
                success_rate = successful / completed if completed else 0.0
                writer.add_scalar("DAgger/MSE_Loss", mean_loss, step)
                writer.add_scalar("DAgger/Beta_Mix_Ratio", beta, step)
                writer.add_scalar("DAgger/Env_Mean_Reward", float(np.mean(rewards)), step)
                writer.add_scalar("DAgger/Buffer_Size", buffer.size, step)
                writer.add_scalar("DAgger/Success_Rate", success_rate, step)
                for stage in range(NUM_CHAIRMAN_POLICIES):
                    writer.add_scalar(
                        f"DAgger/TeacherStage_{stage}_Ratio",
                        float(np.mean(stages == stage)),
                        step,
                    )
                log.info(
                    f"DAgger {step}/{total_iterations} | beta={beta:.4f} | "
                    f"loss={mean_loss:.6f} | success={success_rate:.3f} | "
                    f"buffer={buffer.size}"
                )

            if step > 0 and step % save_freq == 0:
                checkpoint = save_dir / f"student_model_step_{step}.pth"
                torch.save(student.state_dict(), checkpoint)
                log.info(f"DAgger checkpoint saved to {checkpoint}")
            beta = max(0.0, beta * beta_decay)

        final_path = save_dir / "student_model_final.pth"
        torch.save(student.state_dict(), final_path)
        log.info(f"DAgger final student saved to {final_path}")
    finally:
        writer.close()
        env.close()


def _student_checkpoints(model_dir: Path, method: str):
    if method == "dagger":
        step_regex = re.compile(r"student_model_step_(\d+)\.pth$")
        final_name = "student_model_final.pth"
    else:
        step_regex = re.compile(r"student_grpo_step_(\d+)\.pth$")
        final_name = "student_grpo_final.pth"
    checkpoints = []
    for path in model_dir.iterdir():
        match = step_regex.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
        elif path.name == final_name:
            checkpoints.append((10**18, path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _eval_student(config: dict, method: str) -> None:
    if method == "dagger":
        try:
            from .dagger_vp.student_net import VisionStudent
        except ImportError:
            from dagger_vp.student_net import VisionStudent
    else:
        try:
            from .grpo.student_net_stochastic import VisionStudent
        except ImportError:
            from grpo.student_net_stochastic import VisionStudent

    _, metasim_env, env = _build_env(config, dagger_mode=2, training_stage=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(config.get("load_model_path")).expanduser()
    if not model_dir.is_dir():
        env.close()
        raise ValueError(f"load_model_path must be a checkpoint directory: {model_dir}")
    checkpoints = _student_checkpoints(model_dir, method)
    if not checkpoints:
        env.close()
        raise FileNotFoundError(f"No {method.upper()} checkpoints found in {model_dir}")

    env.reset()
    _, _, joints = _student_inputs(metasim_env, device)
    num_actions = env.action_space.shape[0]
    num_joints = joints.shape[1]
    deterministic = bool(config.get("eval_deterministic", True))
    low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)
    results = []
    labels = []
    x_values = []
    all_episode_rows = []
    try:
        for step, checkpoint in checkpoints:
            student = VisionStudent(num_actions=num_actions, num_joints=num_joints).to(device)
            state_dict = torch.load(checkpoint, map_location=device)
            missing, unexpected = student.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                log.warning(
                    f"{checkpoint.name}: missing={missing}, unexpected={unexpected}"
                )
            student.eval()

            def action_provider(_obs, _stages):
                _, images, joint_inputs = _student_inputs(metasim_env, device)
                with torch.no_grad():
                    if method == "dagger":
                        actions_t = student(images, joint_inputs)
                    else:
                        actions_t, _, _, _ = student.act(
                            images, joint_inputs, deterministic=deterministic
                        )
                    actions_t = torch.max(torch.min(actions_t, high_t), low_t)
                return actions_t.detach().cpu().numpy()

            result, rows = _evaluate(env, metasim_env, action_provider, config)
            label = checkpoint.name
            for row in rows:
                row["checkpoint"] = label
            results.append(result)
            labels.append(label)
            x_values.append(
                (max(x_values) + 1 if x_values else 0)
                if step == 10**18
                else step
            )
            all_episode_rows.extend(rows)
            log.info(
                f"{method.upper()} {label}: success={result['success_rate']:.2%}, "
                f"reward={result['mean_reward']:.3f}"
            )
            del student
    finally:
        env.close()
    _save_evaluation(
        model_dir,
        method,
        labels,
        x_values,
        results,
        all_episode_rows,
        config.get("task"),
    )


def _eval_student_video(config: dict, method: str) -> None:
    import cv2

    if method == "dagger":
        try:
            from .dagger_vp.student_net import VisionStudent
        except ImportError:
            from dagger_vp.student_net import VisionStudent
    else:
        try:
            from .grpo.student_net_stochastic import VisionStudent
        except ImportError:
            from grpo.student_net_stochastic import VisionStudent

    _, metasim_env, env = _build_env(config, dagger_mode=2, training_stage=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env.reset()
    images_u8, _, joints = _student_inputs(metasim_env, device)
    student = VisionStudent(
        num_actions=env.action_space.shape[0], num_joints=joints.shape[1]
    ).to(device)
    checkpoint = torch.load(config.get("load_model_path"), map_location=device)
    missing, unexpected = student.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        log.warning(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    student.eval()

    video_path = config.get("video_save_path", f"./output/{method}_multi.mp4")
    _ensure_parent(video_path)
    _, _, height, width = images_u8.shape
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )
    low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)
    deterministic = bool(config.get("eval_deterministic", True))
    show_window = bool(config.get("show_fpv_window", False))
    try:
        for step in range(int(config.get("eval_max_steps", 1000))):
            images_u8, images, joint_inputs = _student_inputs(metasim_env, device)
            frame = images_u8[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            if show_window:
                cv2.imshow(f"{method.upper()} student", frame_bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            with torch.no_grad():
                if method == "dagger":
                    actions_t = student(images, joint_inputs)
                else:
                    actions_t, _, _, _ = student.act(
                        images, joint_inputs, deterministic=deterministic
                    )
                actions_t = torch.max(torch.min(actions_t, high_t), low_t)
            _, rewards, dones, infos = env.step(actions_t.detach().cpu().numpy())
            if bool(dones[0]) and config.get("video_stop_on_done", True):
                log.info(
                    f"{method.upper()} episode finished at step {step + 1}; "
                    f"success={infos[0].get('is_success', False)}"
                )
                break
    finally:
        writer.release()
        if show_window:
            cv2.destroyAllWindows()
        env.close()
    log.info(f"{method.upper()} video saved to {video_path}")


def _train_grpo(config: dict) -> None:
    import copy
    import gc
    from torch.utils.tensorboard import SummaryWriter
    try:
        from .grpo.grpo_trainer import collect_parallel_episodes, build_grpo_batch, grpo_update
        from .grpo.student_net_stochastic import VisionStudent
    except ImportError:
        from grpo.grpo_trainer import collect_parallel_episodes, build_grpo_batch, grpo_update
        from grpo.student_net_stochastic import VisionStudent

    _, metasim_env, env = _build_env(config, dagger_mode=2, training_stage=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env.reset()
    _, _, joints = _student_inputs(metasim_env, device)
    student = VisionStudent(
        num_actions=env.action_space.shape[0], num_joints=joints.shape[1]
    ).to(device)
    checkpoint = torch.load(config.get("load_model_path"), map_location=device)
    missing, unexpected = student.load_state_dict(checkpoint, strict=False)
    log.info(f"Loading DAgger student: missing={missing}, unexpected={unexpected}")
    reference = copy.deepcopy(student).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        student.parameters(), lr=config.get("grpo_learning_rate", 1e-5)
    )
    log_dir = Path(config.get("tensorboard_log", "./grpo_tensorboard"))
    save_dir = Path(config.get("model_save_path", "./output/grpo_models"))
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    total_updates = int(config.get("grpo_total_updates", 2000))
    group_size = int(config.get("grpo_group_size", 4))
    rollout_count = int(config.get("grpo_rollouts_per_batch", env.num_envs))
    if rollout_count % group_size:
        env.close()
        writer.close()
        raise ValueError("grpo_rollouts_per_batch must be divisible by grpo_group_size")

    try:
        for update in range(total_updates):
            episodes = collect_parallel_episodes(
                env=env,
                metasim_env=metasim_env,
                policy=student,
                device=device,
                num_episodes=rollout_count,
                max_steps=int(config.get("grpo_max_episode_steps", 1000)),
                success_bonus=float(config.get("grpo_success_bonus", 20.0)),
            )
            batch, rollout_stats = build_grpo_batch(episodes, group_size)
            del episodes
            update_stats = grpo_update(
                policy=student,
                ref_model=reference,
                optimizer=optimizer,
                batch=batch,
                device=device,
                kl_coef=config.get("grpo_kl_coef", 0.05),
                clip_eps=config.get("grpo_clip_eps", 0.2),
                ent_coef=config.get("grpo_ent_coef", 1e-3),
                epochs=config.get("grpo_update_epochs", 4),
                minibatch_size=config.get("grpo_minibatch_size", 2048),
                max_grad_norm=config.get("grpo_max_grad_norm", 1.0),
            )
            for key, value in rollout_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"GRPO/{key}", value, update)
            for key, value in update_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"GRPO/{key}", value, update)
            log.info(
                f"GRPO {update}/{total_updates} | "
                f"return={rollout_stats['mean_return']:.3f} | "
                f"success={rollout_stats['success_rate']:.3f} | "
                f"loss={update_stats['loss']:.6f}"
            )
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_freq = int(config.get("model_save_freq", 50))
            if update > 0 and update % save_freq == 0:
                path = save_dir / f"student_grpo_step_{update}.pth"
                torch.save(student.state_dict(), path)
        final_path = save_dir / "student_grpo_final.pth"
        torch.save(student.state_dict(), final_path)
        log.info(f"GRPO final student saved to {final_path}")
    finally:
        writer.close()
        env.close()


def main() -> None:
    if len(sys.argv) < 2:
        config_name = "chairman_multi/train_ppo"
    elif len(sys.argv) == 2:
        config_name = sys.argv[1]
    else:
        raise ValueError("Provide at most one YAML config path relative to config_run/configs")

    config = load_config_from_yaml(config_name)
    mode = config.get("train_or_eval")
    log.info(f"Loaded config {config_name}; mode={mode}")

    dispatch = {
        "train": lambda: _train_ppo(config),
        "load_and_train": lambda: _train_ppo(config, resume=True),
        "eval": lambda: _eval_ppo(config),
        "eval_video": lambda: _eval_ppo_video(config),
        "train_dagger": lambda: _train_dagger(config),
        "eval_dagger": lambda: _eval_student(config, "dagger"),
        "eval_dagger_video": lambda: _eval_student_video(config, "dagger"),
        "train_grpo": lambda: _train_grpo(config),
        "eval_grpo": lambda: _eval_student(config, "grpo"),
        "eval_grpo_video": lambda: _eval_student_video(config, "grpo"),
    }
    if mode not in dispatch:
        raise ValueError(
            f"Unsupported main_multi.py mode {mode!r}. Supported modes: "
            + ", ".join(sorted(dispatch))
        )
    dispatch[mode]()


if __name__ == "__main__":
    main()
