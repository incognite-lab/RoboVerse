"""Concurrent PPO trainer for six body-part policies acting on one ChairMan task."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from stable_baselines3 import PPO
from stable_baselines3.common.utils import update_learning_rate
from torch.utils.tensorboard import SummaryWriter
try:
    from .multi_ppo_trainer import FlatBatch, RaggedStageRollout, StageStep
    from .SB3_chairman_separate import POLICY_NAMES
except ImportError:
    from multi_ppo_trainer import FlatBatch, RaggedStageRollout, StageStep
    from SB3_chairman_separate import POLICY_NAMES

MANIFEST_NAME = "separate_policy_manifest.json"

def _policy_value(config, policy, key, default):
    local = config.get("policies", {}).get(policy, {})
    return local.get(key, config.get(key, default))

def _policy_kwargs(config, policy):
    if config.get("net_arch_pivf", False):
        arch = {"pi": _policy_value(config, policy, "net_arch_pi", [128, 128]),
                "vf": _policy_value(config, policy, "net_arch_vf", [128, 128])}
    else:
        arch = _policy_value(config, policy, "net_arch", [128, 128])
    return {"net_arch": arch, "log_std_init": float(_policy_value(config, policy, "log_std_init", -1.0))}

def _lr(config, policy):
    initial = float(_policy_value(config, policy, "learning_rate", 3e-4))
    final = float(_policy_value(config, policy, "final_learning_rate", 0.0))
    if _policy_value(config, policy, "learning_schedule", "constant") != "linear":
        return initial
    return lambda progress: final + (initial - final) * progress

def _new_model(env, config, name, device):
    return PPO(
        "MlpPolicy", env.policy_env(name), verbose=0, learning_rate=_lr(config, name),
        n_steps=2, batch_size=2, n_epochs=int(_policy_value(config, name, "n_epochs", 4)),
        gamma=float(_policy_value(config, name, "gamma", 0.99)),
        gae_lambda=float(_policy_value(config, name, "gae_lambda", 0.95)),
        clip_range=float(_policy_value(config, name, "clip_range", 0.2)),
        ent_coef=float(_policy_value(config, name, "ent_coef", 0.0)),
        vf_coef=float(_policy_value(config, name, "vf_coef", 0.5)),
        max_grad_norm=float(_policy_value(config, name, "max_grad_norm", 0.5)),
        target_kl=_policy_value(config, name, "target_kl", None),
        normalize_advantage=bool(_policy_value(config, name, "normalize_advantage", True)),
        policy_kwargs=_policy_kwargs(config, name), device=device,
        seed=int(config.get("seed", 0)) + POLICY_NAMES.index(name),
    )

def _load_model(path, env, name, device):
    local = env.policy_env(name)
    return PPO.load(path, env=local, device=device, custom_objects={
        "observation_space": local.observation_space,
        "action_space": local.action_space,
        "_last_obs": None, "_last_episode_starts": None,
    })

class SeparatePPOTrainer:
    """All six policies collect every physical step and solve the full episode."""
    def __init__(self, env, config, resume_path=None, resume_checkpoint=None):
        required = ("policy_observations_torch", "compose_actions_torch", "separate_rewards_torch")
        if not all(hasattr(env, name) for name in required):
            raise TypeError("SeparatePPOTrainer requires SB3_chairman_separate.StableBaseline3VecEnv")
        self.env, self.config = env, dict(config)
        self.names = tuple(env.POLICY_NAMES)
        self.num_envs = int(env.num_envs)
        self.device = str(env.torch_device)
        self.torch_device = torch.device(self.device)
        self.n_steps = int(config.get("n_steps", 128))
        self.batch_size = int(config.get("batch_size", 8192))
        self.n_epochs = int(config.get("n_epochs", 4))
        self.total_timesteps = int(config.get("total_timesteps", 1_000_000))
        if self.n_steps <= 0 or self.batch_size <= 1:
            raise ValueError("n_steps must be positive and batch_size must exceed one")
        root = Path(config.get("model_save_path", "./config_run/output/ppo_models_separate"))
        self.run_dir = root / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}_chairman_separate"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tb = Path(config.get("tensorboard_log", root / "tensorboard")) / self.run_dir.name
        tb.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb.resolve()))
        self.global_timesteps = self.global_env_steps = self.last_save = 0
        self.samples = {name: 0 for name in self.names}
        self.updates = {name: 0 for name in self.names}
        self.lr_samples = {name: 0 for name in self.names}
        self.num_stages = int(getattr(env, "NUM_POLICY_STAGES", 6))
        self.cumulative_stage_completions = torch.zeros(
            self.num_stages, dtype=torch.long, device=self.torch_device)
        self.episode_returns = {
            name: torch.zeros(self.num_envs, device=self.torch_device)
            for name in self.names}
        self.episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.torch_device)
        if resume_path:
            paths, manifest = resolve_policy_bundle(resume_path, resume_checkpoint)
            validate_bundle_layout(env, manifest)
            self.models = {name: _load_model(paths[name], env, name, self.device) for name in self.names}
            self.global_timesteps = int(manifest.get("global_timesteps", 0))
            self.global_env_steps = int(manifest.get("global_env_steps", 0))
            for name in self.names:
                saved = manifest.get("policies", {}).get(name, {})
                self.samples[name] = int(saved.get("samples", 0))
                self.updates[name] = int(saved.get("updates", 0))
                self.lr_samples[name] = int(saved.get("lr_trained_samples", self.samples[name]))
        else:
            self.models = {name: _new_model(env, config, name, self.device) for name in self.names}

    @property
    def progress_remaining(self):
        return max(0.0, 1.0 - self.global_timesteps / max(1, self.total_timesteps))

    def _lr_progress(self, name):
        budget = int(_policy_value(self.config, name, "policy_lr_timesteps", self.total_timesteps))
        return max(0.0, 1.0 - self.lr_samples[name] / max(1, budget))

    def _synchronize_device(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def _collect(self, full_obs):
        rollouts = {name: RaggedStageRollout(
            self.num_envs,
            float(_policy_value(self.config, name, "gamma", 0.99)),
            float(_policy_value(self.config, name, "gae_lambda", 0.95))) for name in self.names}
        ids = torch.arange(self.num_envs, device=self.torch_device)
        scalar = lambda: torch.zeros((), device=self.torch_device)
        reward_sums = {name: scalar() for name in self.names}
        reward_sq_sums = {name: scalar() for name in self.names}
        action_abs_sums = {name: scalar() for name in self.names}
        completed_return_sums = {name: scalar() for name in self.names}
        metrics = {
            "steps": 0,
            "stage_occupancy": torch.zeros(self.num_stages, device=self.torch_device),
            "stage_current": torch.zeros(self.num_stages, device=self.torch_device),
            "stage_completed": torch.zeros(self.num_stages, dtype=torch.long, device=self.torch_device),
            "episode_ends": scalar(),
            "successes": scalar(),
            "timeouts": scalar(),
            "failures": scalar(),
            "episode_length_sum": scalar(),
            "failure_counts": {},
            "raw_reward_sums": {},
        }
        for _ in range(self.n_steps):
            observations = self.env.policy_observations_torch(full_obs)
            local_actions, records = {}, {}
            for name in self.names:
                model, obs = self.models[name], observations[name]
                model.policy.set_training_mode(False)
                with torch.no_grad():
                    action, value, log_prob = model.policy(obs, deterministic=False)
                local_actions[name] = action.detach().float()
                records[name] = (obs.detach(), action.detach().float(),
                                 value.detach().flatten().float(), log_prob.detach().flatten().float())
            physical_action = self.env.compose_actions_torch(local_actions)
            next_full_obs, _, dones, metadata = self.env.torch_step(physical_action)
            rewards = self.env.separate_rewards_torch(metadata, local_actions)
            next_observations = self.env.policy_observations_torch(next_full_obs)
            for name in self.names:
                obs, action, value, log_prob = records[name]
                next_value = torch.zeros(self.num_envs, device=self.torch_device)
                continuing = (~dones).nonzero(as_tuple=False).flatten()
                if continuing.numel():
                    with torch.no_grad():
                        predicted = self.models[name].policy.predict_values(
                            next_observations[name].index_select(0, continuing)).flatten()
                    next_value.index_copy_(0, continuing, predicted)
                rollouts[name].add(StageStep(
                    self.global_env_steps, ids, obs, action, rewards[name], value,
                    log_prob, next_value, dones))
                self.samples[name] += self.num_envs
                self.models[name].num_timesteps += self.num_envs
                reward_sums[name] += rewards[name].sum()
                reward_sq_sums[name] += rewards[name].square().sum()
                action_abs_sums[name] += local_actions[name].abs().sum()
                self.episode_returns[name] += rewards[name]

            self.episode_lengths += 1
            active_stages = metadata["stage_after"].long().clamp(
                0, self.num_stages - 1)
            current_counts = torch.bincount(
                active_stages, minlength=self.num_stages)[:self.num_stages]
            metrics["stage_current"] = current_counts
            metrics["stage_occupancy"] += current_counts

            completed = metadata["completed_stage"].long()
            valid = completed[(completed >= 0) & (completed < self.num_stages)]
            if valid.numel():
                metrics["stage_completed"] += torch.bincount(
                    valid, minlength=self.num_stages)[:self.num_stages]

            physical_done = metadata["physical_done"].bool()
            success = metadata["task_success"].bool()
            timeout = metadata["timeout"].bool() & physical_done
            failure = physical_done & ~success & ~timeout
            metrics["episode_ends"] += physical_done.sum()
            metrics["successes"] += success.sum()
            metrics["timeouts"] += timeout.sum()
            metrics["failures"] += failure.sum()
            if physical_done.any():
                metrics["episode_length_sum"] += (
                    self.episode_lengths[physical_done].sum())
                for name in self.names:
                    completed_return_sums[name] += (
                        self.episode_returns[name][physical_done].sum())
                    self.episode_returns[name][physical_done] = 0.0
                self.episode_lengths[physical_done] = 0

            for reason, mask in metadata.get("failure_masks", {}).items():
                count = mask.to(self.torch_device, dtype=torch.bool).sum()
                metrics["failure_counts"][reason] = (
                    metrics["failure_counts"].get(reason, 0) + count)
            for term, value in metadata.get("raw_reward_terms", {}).items():
                term_sum = value.to(self.torch_device).float().sum()
                metrics["raw_reward_sums"][term] = (
                    metrics["raw_reward_sums"].get(term, 0.0) + term_sum)

            full_obs = next_full_obs
            self.global_env_steps += 1
            self.global_timesteps += self.num_envs
            metrics["steps"] += 1
            if self.global_timesteps >= self.total_timesteps: break
        sample_count = max(1, self.num_envs * metrics["steps"])
        episode_count = int(metrics["episode_ends"].item())
        for name in self.names:
            mean = float((reward_sums[name] / sample_count).item())
            variance = max(
                0.0, float((reward_sq_sums[name] / sample_count).item()) - mean * mean)
            self.writer.add_scalar(f"{name}/rollout/reward_mean",
                mean, self.global_timesteps)
            self.writer.add_scalar(f"{name}/rollout/reward_std",
                variance ** 0.5, self.global_timesteps)
            action_dim = len(self.env.policy_action_indices[name])
            self.writer.add_scalar(
                f"{name}/rollout/action_abs_mean",
                float((action_abs_sums[name] / (sample_count * action_dim)).item()),
                self.global_timesteps)
            if episode_count:
                self.writer.add_scalar(
                    f"{name}/rollout/episode_return_mean",
                    float((completed_return_sums[name] / episode_count).item()),
                    self.global_timesteps)

        return (full_obs, {name: rollouts[name].finish() for name in self.names},
                metrics)

    def _log_rollout_metrics(self, metrics, transitions, collect_s, update_s, iteration_s):
        self.cumulative_stage_completions += metrics["stage_completed"]

        steps = max(1, metrics["steps"])
        current = metrics["stage_current"].detach().cpu().tolist()
        mean_counts = (metrics["stage_occupancy"] / steps
                       ).detach().cpu().tolist()
        completed = metrics["stage_completed"].detach().cpu().tolist()
        cumulative = self.cumulative_stage_completions.detach().cpu().tolist()

        self.writer.add_scalar("performance/fps_total", transitions / max(iteration_s, 1e-9), self.global_timesteps)
        self.writer.add_scalar("performance/fps_collection", transitions / max(collect_s, 1e-9), self.global_timesteps)
        self.writer.add_scalar("performance/collection_seconds", collect_s, self.global_timesteps)
        self.writer.add_scalar("performance/update_seconds", update_s, self.global_timesteps)
        self.writer.add_scalar("performance/iteration_seconds", iteration_s, self.global_timesteps)

        for stage in range(self.num_stages):
            suffix = f"stage_{stage}"
            self.writer.add_scalar(f"stages/active_count/{suffix}", current[stage], self.global_timesteps)
            self.writer.add_scalar(f"stages/active_fraction/{suffix}", current[stage] / self.num_envs, self.global_timesteps)
            self.writer.add_scalar(f"stages/mean_active_count/{suffix}", mean_counts[stage], self.global_timesteps)
            self.writer.add_scalar(f"stages/completed_rollout/{suffix}", completed[stage], self.global_timesteps)
            self.writer.add_scalar(f"stages/completed_total/{suffix}", cumulative[stage], self.global_timesteps)

        ended = int(metrics["episode_ends"].item())
        successes = int(metrics["successes"].item())
        timeouts = int(metrics["timeouts"].item())
        failures = int(metrics["failures"].item())
        self.writer.add_scalar("episodes/ended_rollout", ended, self.global_timesteps)
        self.writer.add_scalar("episodes/successes_rollout", successes, self.global_timesteps)
        self.writer.add_scalar("episodes/timeouts_rollout", timeouts, self.global_timesteps)
        self.writer.add_scalar("episodes/failures_rollout", failures, self.global_timesteps)
        if ended:
            self.writer.add_scalar("episodes/success_rate",
                successes / ended, self.global_timesteps)
            self.writer.add_scalar("episodes/length_mean",
                float(metrics["episode_length_sum"].item()) / ended, self.global_timesteps)

        sample_count = max(1, self.num_envs * metrics["steps"])
        for term, value in metrics["raw_reward_sums"].items():
            self.writer.add_scalar(f"reward_components/{term}",
                float(value.item()) / sample_count, self.global_timesteps)
        for reason, value in metrics["failure_counts"].items():
            self.writer.add_scalar(f"failures/{reason}",
                int(value.item()), self.global_timesteps)

        active_text = ", ".join(f"S{i}={int(v)}" for i, v in enumerate(current))
        complete_text = ", ".join(f"S{i}={int(v)}" for i, v in enumerate(completed))
        log.info(
            "Stages active [{}], completed [{}], episodes={} (success={}, timeout={}, failure={})",
            active_text, complete_text, ended, successes, timeouts, failures)

    def _update(self, name, batch: FlatBatch):
        if batch is None or len(batch) < 2: return
        model, policy = self.models[name], self.models[name].policy
        policy.set_training_mode(True)
        lr = float(model.lr_schedule(self._lr_progress(name)))
        update_learning_rate(policy.optimizer, lr)
        clip = float(model.clip_range(self.progress_remaining))
        losses, kls = [], []
        for _ in range(int(_policy_value(self.config, name, "n_epochs", self.n_epochs))):
            order = torch.randperm(len(batch), device=self.torch_device)
            for start in range(0, len(batch), self.batch_size):
                idx = order[start:start + self.batch_size]
                advantages = batch.advantages[idx]
                if model.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                values, log_prob, entropy = policy.evaluate_actions(batch.observations[idx], batch.actions[idx])
                ratio = torch.exp(log_prob - batch.old_log_prob[idx])
                policy_loss = -torch.min(advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                value_loss = F.mse_loss(batch.returns[idx], values.flatten())
                entropy_loss = -log_prob.mean() if entropy is None else -entropy.mean()
                loss = policy_loss + model.vf_coef * value_loss + model.ent_coef * entropy_loss
                with torch.no_grad():
                    log_ratio = log_prob - batch.old_log_prob[idx]
                    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                if model.target_kl is not None and float(kl) > 1.5 * model.target_kl: break
                policy.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
                policy.optimizer.step(); losses.append(loss.detach()); kls.append(kl.detach())
            model._n_updates += 1
        policy.set_training_mode(False)
        self.updates[name] += 1; self.lr_samples[name] += len(batch)
        self.writer.add_scalar(f"{name}/train/loss", float(torch.stack(losses).mean()) if losses else 0, self.global_timesteps)
        self.writer.add_scalar(f"{name}/train/approx_kl", float(torch.stack(kls).mean()) if kls else 0, self.global_timesteps)
        self.writer.add_scalar(f"{name}/train/learning_rate", lr, self.global_timesteps)

    def _manifest(self, paths):
        return {"format_version": 1, "trainer": "simultaneous_separate_ppo",
                "policy_names": list(self.names), "global_timesteps": self.global_timesteps,
                "global_env_steps": self.global_env_steps,
                "policies": {name: {"model": paths[name], "samples": self.samples[name],
                    "updates": self.updates[name], "lr_trained_samples": self.lr_samples[name],
                    "observation_groups": self.env.policy_specs[name].get("observation_groups"),
                    "observation_indices": list(self.env.policy_observation_indices[name]),
                    "observation_dim": self.env.policy_observation_spaces[name].shape[0],
                    "uses_action_history": self.env.policy_uses_action_history[name],
                    "action_names": [self.env.action_names[i] for i in self.env.policy_action_indices[name]]}
                    for name in self.names}}

    def save(self, label):
        paths = {}
        for name, model in self.models.items():
            directory = self.run_dir / name; directory.mkdir(parents=True, exist_ok=True)
            model.save(str(directory / f"model_{label}"))
            paths[name] = f"{name}/model_{label}.zip"
        (self.run_dir / MANIFEST_NAME).write_text(json.dumps(self._manifest(paths), indent=2), encoding="utf-8")
        log.info("Saved separate-policy bundle to {}", self.run_dir)

    def learn(self):
        full_obs = self.env.torch_reset()
        save_freq = int(self.config.get("model_save_freq", 1_000_000))
        log.info("Training six simultaneous policies {} on {} envs", self.names, self.num_envs)
        try:
            while self.global_timesteps < self.total_timesteps:
                self._synchronize_device()
                started = time.perf_counter()
                timesteps_before = self.global_timesteps
                full_obs, batches, metrics = self._collect(full_obs)
                self._synchronize_device()
                collected_at = time.perf_counter()
                for name in self.names: self._update(name, batches[name])
                self._synchronize_device()
                updated_at = time.perf_counter()
                transitions = self.global_timesteps - timesteps_before
                collect_s = collected_at - started
                update_s = updated_at - collected_at
                elapsed = updated_at - started
                self._log_rollout_metrics(
                    metrics, transitions, collect_s, update_s, elapsed)
                log.info("Separate PPO: {}/{} transitions, {:.0f} FPS, updates {}",
                         self.global_timesteps, self.total_timesteps,
                         transitions / max(elapsed, 1e-9), self.updates)
                if save_freq > 0 and self.global_timesteps - self.last_save >= save_freq:
                    self.save(str(self.global_timesteps)); self.last_save = self.global_timesteps
            self.save("final"); return self.run_dir
        finally:
            self.writer.close()

def resolve_policy_bundle(bundle_path, checkpoint=None):
    root = Path(bundle_path).expanduser()
    if root.is_file(): root = root.parent
    if root.is_dir() and not (root / MANIFEST_NAME).is_file():
        runs = sorted((p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()), key=lambda p: p.stat().st_mtime)
        if runs: root = runs[-1]
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file(): raise FileNotFoundError(f"Missing {MANIFEST_NAME} below {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("policy_names", ())) != POLICY_NAMES:
        raise ValueError(f"Bundle policy names differ from required {POLICY_NAMES}")
    paths = {}
    for name in POLICY_NAMES:
        relative = (f"{name}/model_{checkpoint}.zip" if checkpoint not in (None, "")
                    else manifest["policies"][name]["model"])
        path = Path(relative); path = path if path.is_absolute() else root / path
        if not path.is_file(): raise FileNotFoundError(f"Missing {name} model: {path}")
        paths[name] = str(path)
    return paths, manifest

def validate_bundle_layout(env, manifest):
    """Reject a checkpoint when YAML changes a saved policy interface."""
    for name in POLICY_NAMES:
        saved = manifest.get("policies", {}).get(name, {})
        actions = [env.action_names[i] for i in env.policy_action_indices[name]]
        if saved.get("action_names") != actions:
            raise ValueError(f"Action layout for {name} differs from the saved bundle")
        indices = saved.get("observation_indices")
        if indices is not None and tuple(indices) != tuple(env.policy_observation_indices[name]):
            raise ValueError(f"Observation layout for {name} differs from the saved bundle")
        dimension = saved.get("observation_dim")
        current_dimension = env.policy_observation_spaces[name].shape[0]
        if dimension is not None and int(dimension) != current_dimension:
            raise ValueError(
                f"Observation dimension for {name} differs from the saved bundle")

class SeparatePolicyRouter:
    def __init__(self, env, models): self.env, self.models = env, models
    def predict_torch(self, full_observation, deterministic=True):
        observations = self.env.policy_observations_torch(full_observation)
        actions = {}
        with torch.no_grad():
            for name in POLICY_NAMES:
                actions[name], _, _ = self.models[name].policy(observations[name], deterministic=deterministic)
        return self.env.compose_actions_torch(actions)
    def predict(self, full_observation, deterministic=True):
        obs = torch.as_tensor(full_observation, dtype=torch.float32, device=self.env.torch_device)
        return self.predict_torch(obs, deterministic).cpu().numpy()

def load_policy_router(env, bundle_path, device=None, checkpoint=None):
    paths, manifest = resolve_policy_bundle(bundle_path, checkpoint)
    validate_bundle_layout(env, manifest)
    device = device or str(env.torch_device)
    models = {name: _load_model(paths[name], env, name, device) for name in POLICY_NAMES}
    return SeparatePolicyRouter(env, models), manifest
