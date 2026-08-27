"""Concurrent on-policy PPO trainer for staged vector environments.

Stable-Baselines3 assumes that one policy acts in every vector-environment
slot. ChairMan needs a different policy in each stage and different env rows
can be in different stages. This module keeps SB3's ActorCriticPolicy,
optimizer, distributions, serialization and PPO loss, while replacing rollout
collection with a stage-routed ragged collector.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from rich import box
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.utils import update_learning_rate
from torch.utils.tensorboard import SummaryWriter


NUM_STAGE_POLICIES = 6
MANIFEST_NAME = "multi_policy_manifest.json"
STAGE_NAMES = (
    "approach chair",
    "reach handles",
    "grasp chair",
    "pull chair",
    "stop chair",
    "lower arms",
)


def _mean_or_zero(values: deque) -> float:
    return float(np.mean(values)) if values else 0.0


def _human_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _stage_value(config: dict, name: str, stage: int, default):
    value = config.get(name, default)
    if isinstance(value, dict):
        return value.get(stage, value.get(str(stage), default))
    if isinstance(value, (list, tuple)):
        if len(value) != NUM_STAGE_POLICIES:
            raise ValueError(
                f"{name} must contain {NUM_STAGE_POLICIES} values, got {len(value)}"
            )
        return value[stage]
    return value


def _learning_rate(config: dict):
    initial = float(config.get("learning_rate", 3e-4))
    if config.get("learning_schedule", "constant") != "linear":
        return initial
    final = float(config.get("final_learning_rate", 0.0))
    return lambda progress_remaining: final + (initial - final) * progress_remaining


def _policy_kwargs(config: dict) -> dict:
    if config.get("net_arch_pivf", False):
        net_arch: Any = {
            "pi": config.get("net_arch_pi", [128, 128, 128]),
            "vf": config.get("net_arch_vf", [128, 128, 128]),
        }
    else:
        net_arch = config.get("net_arch", [128, 128, 128])
    return {
        "net_arch": net_arch,
        "log_std_init": float(config.get("log_std_init", 0.0)),
    }


def _new_stage_model(env, config: dict, stage: int, device: str) -> PPO:
    # The standard SB3 rollout buffer is deliberately tiny: collection is
    # performed by RaggedStageRollout below. The PPO object is retained for its
    # tested policy implementation, hyperparameters and save/load format.
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=_learning_rate(config),
        n_steps=2,
        batch_size=2,
        n_epochs=int(config.get("n_epochs", 4)),
        gamma=float(config.get("gamma", 0.99)),
        gae_lambda=float(config.get("gae_lambda", 0.95)),
        clip_range=float(config.get("clip_range", 0.2)),
        clip_range_vf=config.get("clip_range_vf"),
        normalize_advantage=bool(config.get("normalize_advantage", True)),
        ent_coef=float(config.get("ent_coef", 0.0)),
        vf_coef=float(config.get("vf_coef", 0.5)),
        max_grad_norm=float(config.get("max_grad_norm", 0.5)),
        target_kl=config.get("target_kl"),
        tensorboard_log=None,
        policy_kwargs=_policy_kwargs(config),
        device=device,
        seed=int(config.get("seed", 0)) + stage,
    )


@dataclass
class StageStep:
    global_step: int
    env_ids: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    old_log_prob: torch.Tensor
    next_values: torch.Tensor
    terminals: torch.Tensor


@dataclass
class FlatBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return int(self.observations.shape[0])


class RaggedStageRollout:
    """One global rollout containing only transitions owned by one policy."""

    def __init__(self, num_envs: int, gamma: float, gae_lambda: float):
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.steps: list[StageStep] = []

    def add(self, step: StageStep) -> None:
        if len(step.env_ids):
            self.steps.append(step)

    def finish(self) -> FlatBatch | None:
        if not self.steps:
            return None

        device = self.steps[0].observations.device
        last_gae = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        last_seen_step = torch.full(
            (self.num_envs,), -2, dtype=torch.long, device=device
        )
        advantages_by_record: list[torch.Tensor | None] = [None] * len(self.steps)

        for record_index in range(len(self.steps) - 1, -1, -1):
            record = self.steps[record_index]
            ids = record.env_ids
            nonterminal = ~record.terminals
            contiguous = last_seen_step[ids] == record.global_step + 1
            delta = (
                record.rewards
                + self.gamma * record.next_values * nonterminal.float()
                - record.values
            )
            advantage = delta + (
                self.gamma
                * self.gae_lambda
                * nonterminal.float()
                * contiguous.float()
                * last_gae[ids]
            )
            advantages_by_record[record_index] = advantage
            last_gae[ids] = advantage
            last_seen_step[ids] = record.global_step

        observations = torch.cat([record.observations for record in self.steps])
        actions = torch.cat([record.actions for record in self.steps])
        values = torch.cat([record.values for record in self.steps])
        old_log_prob = torch.cat([record.old_log_prob for record in self.steps])
        advantages = torch.cat(
            [advantage for advantage in advantages_by_record if advantage is not None]
        )
        return FlatBatch(
            observations=observations,
            actions=actions,
            old_values=values,
            old_log_prob=old_log_prob,
            advantages=advantages,
            returns=advantages + values,
        )


class PendingStageBatches:
    """Accumulate rollouts while a downstream policy has too few samples."""

    def __init__(self):
        self.batches: list[FlatBatch] = []
        self.num_samples = 0

    def append(self, batch: FlatBatch | None) -> None:
        if batch is not None:
            self.batches.append(batch)
            self.num_samples += len(batch)

    def pop_all(self) -> FlatBatch:
        if not self.batches:
            raise RuntimeError("Cannot pop an empty stage buffer")
        merged = FlatBatch(
            observations=torch.cat([batch.observations for batch in self.batches]),
            actions=torch.cat([batch.actions for batch in self.batches]),
            old_values=torch.cat([batch.old_values for batch in self.batches]),
            old_log_prob=torch.cat([batch.old_log_prob for batch in self.batches]),
            advantages=torch.cat([batch.advantages for batch in self.batches]),
            returns=torch.cat([batch.returns for batch in self.batches]),
        )
        self.batches.clear()
        self.num_samples = 0
        return merged

    def clear(self) -> None:
        self.batches.clear()
        self.num_samples = 0


class MultiPPOTrainer:
    """Train six independent PPO policies in one uninterrupted ChairMan env."""

    def __init__(self, env, config: dict, *, resume_path: str | None = None):
        if not hasattr(env, "get_current_stages"):
            raise TypeError(
                "MultiPPOTrainer requires config_run.SB3_chairman_multi_env"
            )
        self.env = env
        self.config = config
        self.num_envs = int(env.num_envs)
        env_device = getattr(env, "torch_device", None)
        self.device = str(
            env_device
            if env_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_device = torch.device(self.device)
        self.torch_rollouts = bool(
            hasattr(env, "torch_reset") and hasattr(env, "torch_step")
        )
        self.rollout_steps = int(config.get("n_steps", 128))
        self.batch_size = int(config.get("batch_size", 256))
        self.n_epochs = int(config.get("n_epochs", 4))
        if self.rollout_steps <= 0 or self.batch_size <= 1:
            raise ValueError("n_steps must be positive and batch_size must be greater than 1")

        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.total_timesteps = int(config.get("total_timesteps", 1_000_000))
        self.global_timesteps = 0
        self.global_env_steps = 0
        self.last_save_timestep = 0

        root = Path(config.get("model_save_path", "./output/ppo_models_multi"))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = root / f"run_{timestamp}_{config.get('task')}_concurrent"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir = (
            Path(config.get("tensorboard_log", root / "tensorboard"))
            / self.run_dir.name
        )
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = tensorboard_dir.resolve()
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        self.console = Console()

        if resume_path:
            paths, manifest = resolve_policy_bundle(resume_path)
            self.models = {
                stage: PPO.load(paths[stage], env=env, device=self.device)
                for stage in range(NUM_STAGE_POLICIES)
            }
            self.global_timesteps = int(manifest.get("global_timesteps", 0))
            self.global_env_steps = int(manifest.get("global_env_steps", 0))
        else:
            manifest = {}
            self.models = {
                stage: _new_stage_model(env, config, stage, self.device)
                for stage in range(NUM_STAGE_POLICIES)
            }

        self.pending = {
            stage: PendingStageBatches() for stage in range(NUM_STAGE_POLICIES)
        }
        window = max(1, int(config.get("stage_success_window", 1000)))
        self.recent_outcomes = {
            stage: deque(maxlen=window) for stage in range(NUM_STAGE_POLICIES)
        }
        self.attempts = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        self.successes = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        self.samples = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        self.updates = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        self.frozen = np.zeros(NUM_STAGE_POLICIES, dtype=bool)

        stage_manifest = manifest.get("stages", {})
        for stage in range(NUM_STAGE_POLICIES):
            saved = stage_manifest.get(str(stage), {})
            self.attempts[stage] = int(saved.get("attempts", 0))
            self.successes[stage] = int(saved.get("successes", 0))
            self.samples[stage] = int(saved.get("samples", 0))
            self.updates[stage] = int(saved.get("updates", 0))
            self.frozen[stage] = bool(saved.get("frozen", False))

        episode_window = max(
            1, int(config.get("stage_episode_metrics_window", window))
        )
        self.recent_returns = {
            stage: deque(maxlen=episode_window)
            for stage in range(NUM_STAGE_POLICIES)
        }
        self.recent_lengths = {
            stage: deque(maxlen=episode_window)
            for stage in range(NUM_STAGE_POLICIES)
        }
        self._episode_returns = torch.zeros(
            (NUM_STAGE_POLICIES, self.num_envs),
            dtype=torch.float32,
            device=self.torch_device,
        )
        self._episode_lengths = torch.zeros(
            (NUM_STAGE_POLICIES, self.num_envs),
            dtype=torch.long,
            device=self.torch_device,
        )
        self._action_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.torch_device
        )
        self._action_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.torch_device
        )
        self.last_train_metrics: dict[int, dict[str, float]] = {}
        self.last_train_timestep = np.full(
            NUM_STAGE_POLICIES, -1, dtype=np.int64
        )
        self._last_rollout_stats: dict[int, dict[str, float]] = {}
        self._last_logged_samples = self.samples.copy()
        self._last_collect_seconds = 0.0
        self._last_train_seconds = 0.0
        self._start_time = 0.0
        self._last_log_time = 0.0
        self._last_log_timesteps = self.global_timesteps
        self._last_log_env_steps = self.global_env_steps
        self._terminal_log_iteration = 0

    def _target_samples(self, stage: int) -> int:
        default = max(
            self.batch_size,
            self.rollout_steps * max(1, self.num_envs // NUM_STAGE_POLICIES),
        )
        return max(
            2,
            int(_stage_value(self.config, "stage_rollout_samples", stage, default)),
        )

    @property
    def progress_remaining(self) -> float:
        return max(0.0, 1.0 - self.global_timesteps / max(1, self.total_timesteps))

    def _env_reset_torch(self) -> torch.Tensor:
        if self.torch_rollouts:
            return self.env.torch_reset()
        return torch.as_tensor(
            self.env.reset(), dtype=torch.float32, device=self.torch_device
        )

    def _current_stages_torch(self) -> torch.Tensor:
        if self.torch_rollouts:
            return self.env.get_current_stages_torch()
        return torch.as_tensor(
            self.env.get_current_stages(),
            dtype=torch.long,
            device=self.torch_device,
        )

    def _env_step_torch(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.torch_rollouts:
            return self.env.torch_step(actions)
        obs, rewards, dones, infos = self.env.step(
            actions.detach().cpu().numpy()
        )
        completed = torch.as_tensor(
            [info.get("completed_stage", -1) for info in infos],
            dtype=torch.long,
            device=self.torch_device,
        )
        stage_after = torch.as_tensor(
            [info.get("stage_after", -1) for info in infos],
            dtype=torch.long,
            device=self.torch_device,
        )
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self.torch_device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.torch_device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.torch_device),
            {
                "completed_stage": completed,
                "stage_after_event": stage_after,
            },
        )

    def _policy_actions_torch(
        self, observations: torch.Tensor, stages: torch.Tensor
    ):
        """Route GPU observation rows through the corresponding PPO policy."""
        action_dim = self.env.action_space.shape[0]
        raw_actions = torch.zeros(
            (self.num_envs, action_dim),
            dtype=torch.float32,
            device=self.torch_device,
        )
        records: dict[int, tuple] = {}

        for stage, model in self.models.items():
            env_ids = (stages == stage).nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                continue
            obs_device = observations.index_select(0, env_ids)
            model.policy.set_training_mode(False)
            deterministic = bool(
                self.frozen[stage]
                and self.config.get("frozen_policy_deterministic", True)
            )
            with torch.no_grad():
                actions_t, values_t, log_prob_t = model.policy(
                    obs_device, deterministic=deterministic
                )
            actions_t = actions_t.detach().float()
            raw_actions.index_copy_(0, env_ids, actions_t)
            if not self.frozen[stage]:
                records[stage] = (
                    env_ids,
                    obs_device.detach(),
                    actions_t,
                    values_t.detach().flatten().float(),
                    log_prob_t.detach().flatten().float(),
                )

        env_actions = torch.maximum(
            torch.minimum(raw_actions, self._action_high), self._action_low
        )
        return env_actions, raw_actions, records

    def _next_values_torch(
        self,
        stage: int,
        next_observations: torch.Tensor,
        env_ids: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.zeros(
            env_ids.numel(), dtype=torch.float32, device=self.torch_device
        )
        continuing = (~terminals).nonzero(as_tuple=False).flatten()
        if continuing.numel():
            continuing_env_ids = env_ids.index_select(0, continuing)
            obs_t = next_observations.index_select(0, continuing_env_ids)
            with torch.no_grad():
                predicted = self.models[stage].policy.predict_values(obs_t)
            values.index_copy_(0, continuing, predicted.detach().flatten().float())
        return values

    def _next_values(
        self,
        stage: int,
        next_observations,
        env_ids,
        terminals,
    ) -> torch.Tensor:
        """Compatibility adapter for older CPU-only trainer tests/tools."""
        observations_t = torch.as_tensor(
            next_observations, dtype=torch.float32, device=self.torch_device
        )
        env_ids_t = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.torch_device
        )
        terminals_t = torch.as_tensor(
            terminals, dtype=torch.bool, device=self.torch_device
        )
        result = self._next_values_torch(
            stage, observations_t, env_ids_t, terminals_t
        )
        return result.cpu() if isinstance(next_observations, np.ndarray) else result

    def _update_outcomes(
        self,
        stage: int,
        env_ids: np.ndarray,
        local_terminals: np.ndarray,
        infos: list[dict],
    ) -> None:
        for local_index in np.flatnonzero(local_terminals):
            env_id = int(env_ids[local_index])
            success = int(infos[env_id].get("completed_stage", -1)) == stage
            self.attempts[stage] += 1
            self.successes[stage] += int(success)
            self.recent_outcomes[stage].append(int(success))

    def _collect_rollout(self, observations: torch.Tensor):
        rollouts = {
            stage: RaggedStageRollout(self.num_envs, self.gamma, self.gae_lambda)
            for stage in range(NUM_STAGE_POLICIES)
        }
        transitions = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        reward_sum = torch.zeros(
            NUM_STAGE_POLICIES, dtype=torch.float32, device=self.torch_device
        )
        reward_sq_sum = torch.zeros_like(reward_sum)
        action_sum = torch.zeros_like(reward_sum)
        action_sq_sum = torch.zeros_like(reward_sum)
        action_abs_sum = torch.zeros_like(reward_sum)
        action_elements = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        clipped_elements = torch.zeros(
            NUM_STAGE_POLICIES, dtype=torch.long, device=self.torch_device
        )
        command_abs_sum = torch.zeros_like(reward_sum)
        command_elements = np.zeros(NUM_STAGE_POLICIES, dtype=np.int64)
        terminals_count = torch.zeros_like(clipped_elements)
        successes_count = torch.zeros_like(clipped_elements)

        for _ in range(self.rollout_steps):
            stages_before = self._current_stages_torch()
            env_actions, raw_actions, policy_records = self._policy_actions_torch(
                observations, stages_before
            )
            next_obs, rewards, dones, metadata = self._env_step_torch(env_actions)
            stages_after_event = metadata["stage_after_event"]
            completed = metadata["completed_stage"]

            for stage in range(NUM_STAGE_POLICIES):
                env_ids = (stages_before == stage).nonzero(as_tuple=False).flatten()
                if env_ids.numel() == 0:
                    continue
                local_dones = dones.index_select(0, env_ids)
                local_completed = completed.index_select(0, env_ids)
                local_stage_after = stages_after_event.index_select(0, env_ids)
                local_terminals = (
                    local_dones
                    | (local_completed == stage)
                    | (local_stage_after != stage)
                )
                stage_rewards = rewards.index_select(0, env_ids)
                stage_actions = raw_actions.index_select(0, env_ids)
                clipped_actions = env_actions.index_select(0, env_ids)
                count = env_ids.numel()
                transitions[stage] += count
                reward_sum[stage] += stage_rewards.sum()
                reward_sq_sum[stage] += stage_rewards.square().sum()
                action_sum[stage] += stage_actions.sum()
                action_sq_sum[stage] += stage_actions.square().sum()
                action_abs_sum[stage] += stage_actions.abs().sum()
                action_elements[stage] += stage_actions.numel()
                clipped_elements[stage] += (~torch.isclose(
                    stage_actions, clipped_actions
                )).sum()
                commands = stage_actions[:, -3:]
                command_abs_sum[stage] += commands.abs().sum()
                command_elements[stage] += commands.numel()
                terminals_count[stage] += local_terminals.sum()
                successes_count[stage] += (local_completed == stage).sum()

                self._episode_returns[stage].index_add_(0, env_ids, stage_rewards)
                self._episode_lengths[stage].index_add_(
                    0, env_ids, torch.ones_like(env_ids)
                )
                terminal_local_ids = local_terminals.nonzero(
                    as_tuple=False
                ).flatten()
                if terminal_local_ids.numel():
                    terminal_env_ids = env_ids.index_select(0, terminal_local_ids)
                    terminal_returns = self._episode_returns[
                        stage
                    ].index_select(0, terminal_env_ids).detach().cpu().tolist()
                    terminal_lengths = self._episode_lengths[
                        stage
                    ].index_select(0, terminal_env_ids).detach().cpu().tolist()
                    terminal_successes = (
                        local_completed.index_select(0, terminal_local_ids) == stage
                    ).detach().cpu().tolist()
                    for episode_return, episode_length, success in zip(
                        terminal_returns, terminal_lengths, terminal_successes
                    ):
                        self.recent_returns[stage].append(float(episode_return))
                        self.recent_lengths[stage].append(int(episode_length))
                        self.recent_outcomes[stage].append(int(success))
                    num_terminals = terminal_env_ids.numel()
                    num_successes = int(sum(terminal_successes))
                    self.attempts[stage] += num_terminals
                    self.successes[stage] += num_successes
                    self._episode_returns[stage].index_fill_(
                        0, terminal_env_ids, 0.0
                    )
                    self._episode_lengths[stage].index_fill_(
                        0, terminal_env_ids, 0
                    )

                if stage not in policy_records:
                    continue
                (
                    recorded_ids,
                    recorded_obs,
                    recorded_actions,
                    recorded_values,
                    recorded_log_prob,
                ) = policy_records[stage]
                next_values = self._next_values_torch(
                    stage, next_obs, env_ids, local_terminals
                )
                rollouts[stage].add(
                    StageStep(
                        global_step=self.global_env_steps,
                        env_ids=recorded_ids,
                        observations=recorded_obs,
                        actions=recorded_actions,
                        rewards=stage_rewards,
                        values=recorded_values,
                        old_log_prob=recorded_log_prob,
                        next_values=next_values,
                        terminals=local_terminals,
                    )
                )
                self.samples[stage] += count
                self.models[stage].num_timesteps += count

            observations = next_obs
            self.global_env_steps += 1
            self.global_timesteps += self.num_envs
            if self.global_timesteps >= self.total_timesteps:
                break

        for stage, rollout in rollouts.items():
            self.pending[stage].append(rollout.finish())
        reduced_stats = torch.stack(
            (
                reward_sum,
                reward_sq_sum,
                action_sum,
                action_sq_sum,
                action_abs_sum,
                clipped_elements.float(),
                command_abs_sum,
                terminals_count.float(),
                successes_count.float(),
            ),
            dim=1,
        ).detach().cpu().numpy()
        self._last_rollout_stats = {}
        for stage in range(NUM_STAGE_POLICIES):
            transition_count = int(transitions[stage])
            element_count = int(action_elements[stage])
            reward_mean = (
                reduced_stats[stage, 0] / transition_count
                if transition_count
                else 0.0
            )
            reward_variance = (
                reduced_stats[stage, 1] / transition_count - reward_mean**2
                if transition_count
                else 0.0
            )
            action_mean = (
                reduced_stats[stage, 2] / element_count if element_count else 0.0
            )
            action_variance = (
                reduced_stats[stage, 3] / element_count - action_mean**2
                if element_count
                else 0.0
            )
            self._last_rollout_stats[stage] = {
                "transitions": float(transition_count),
                "reward_mean": float(reward_mean),
                "reward_std": float(np.sqrt(max(0.0, reward_variance))),
                "action_mean": float(action_mean),
                "action_std": float(np.sqrt(max(0.0, action_variance))),
                "action_abs_mean": float(
                    reduced_stats[stage, 4] / element_count
                    if element_count
                    else 0.0
                ),
                "action_clip_fraction": float(
                    reduced_stats[stage, 5] / element_count
                    if element_count
                    else 0.0
                ),
                "walk_command_abs_mean": float(
                    reduced_stats[stage, 6] / command_elements[stage]
                    if command_elements[stage]
                    else 0.0
                ),
                "terminals": float(reduced_stats[stage, 7]),
                "successes": float(reduced_stats[stage, 8]),
            }
        return observations

    def _ppo_update(self, stage: int, batch: FlatBatch) -> dict[str, float]:
        update_started = time.perf_counter()
        model = self.models[stage]
        policy = model.policy
        policy.set_training_mode(True)
        learning_rate = float(model.lr_schedule(self.progress_remaining))
        update_learning_rate(policy.optimizer, learning_rate)
        clip_range = float(model.clip_range(self.progress_remaining))
        clip_range_vf = (
            None
            if model.clip_range_vf is None
            else float(model.clip_range_vf(self.progress_remaining))
        )

        losses: list[torch.Tensor] = []
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        entropy_losses: list[torch.Tensor] = []
        approx_kls: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []
        gradient_norms: list[torch.Tensor] = []
        stop_early = False
        num_samples = len(batch)
        epochs_completed = 0
        minibatches = 0

        for _epoch in range(self.n_epochs):
            permutation = torch.randperm(num_samples, device=batch.observations.device)
            for start in range(0, num_samples, self.batch_size):
                index = permutation[start : start + self.batch_size]
                obs = batch.observations[index]
                actions = batch.actions[index]
                old_values = batch.old_values[index]
                old_log_prob = batch.old_log_prob[index]
                advantages = batch.advantages[index]
                returns = batch.returns[index]

                if model.normalize_advantage and len(advantages) > 1:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                values, log_prob, entropy = policy.evaluate_actions(obs, actions)
                values = values.flatten()
                ratio = torch.exp(log_prob - old_log_prob)
                unclipped = advantages * ratio
                clipped = advantages * torch.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = -torch.min(unclipped, clipped).mean()

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(returns, values_pred)
                entropy_loss = (
                    -log_prob.mean() if entropy is None else -entropy.mean()
                )
                loss = (
                    policy_loss
                    + model.ent_coef * entropy_loss
                    + model.vf_coef * value_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl = (
                        (torch.exp(log_ratio) - 1.0) - log_ratio
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > clip_range
                    ).float().mean()

                if model.target_kl is not None and (
                    float(approx_kl.detach().cpu()) > 1.5 * model.target_kl
                ):
                    stop_early = True
                    break

                policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), model.max_grad_norm
                )
                policy.optimizer.step()

                losses.append(loss.detach())
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy_loss.detach())
                approx_kls.append(approx_kl.detach())
                clip_fractions.append(clip_fraction.detach())
                gradient_norms.append(gradient_norm.detach())
                minibatches += 1
            model._n_updates += 1
            epochs_completed += 1
            if stop_early:
                break

        self.updates[stage] += 1
        policy.set_training_mode(False)
        zero = torch.zeros((), dtype=torch.float32, device=self.torch_device)

        def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values).mean() if values else zero

        returns_variance_t = torch.var(batch.returns, correction=0)
        explained_variance_t = torch.where(
            returns_variance_t > 1e-12,
            1.0
            - torch.var(
                batch.returns - batch.old_values, correction=0
            )
            / returns_variance_t,
            torch.zeros_like(returns_variance_t),
        )
        log_std = getattr(policy, "log_std", None)
        policy_std_t = (
            torch.exp(log_std).mean()
            if log_std is not None
            else torch.full_like(zero, torch.nan)
        )
        summary = torch.stack(
            (
                mean_or_zero(losses),
                mean_or_zero(policy_losses),
                mean_or_zero(value_losses),
                mean_or_zero(entropy_losses),
                mean_or_zero(approx_kls),
                mean_or_zero(clip_fractions),
                explained_variance_t,
                policy_std_t,
                mean_or_zero(gradient_norms),
                batch.advantages.mean(),
                batch.advantages.std(),
                batch.returns.mean(),
                batch.old_values.mean(),
            )
        ).detach().cpu().tolist()
        (
            loss_mean,
            policy_loss_mean,
            value_loss_mean,
            entropy_loss_mean,
            approx_kl_mean,
            clip_fraction_mean,
            explained_variance,
            policy_std,
            gradient_norm_mean,
            advantage_mean,
            advantage_std,
            return_mean,
            value_mean,
        ) = summary
        update_seconds = time.perf_counter() - update_started
        metrics = {
            "loss": loss_mean,
            "policy_loss": policy_loss_mean,
            "value_loss": value_loss_mean,
            "entropy_loss": entropy_loss_mean,
            "entropy": -entropy_loss_mean,
            "approx_kl": approx_kl_mean,
            "clip_fraction": clip_fraction_mean,
            "clip_range": clip_range,
            "explained_variance": explained_variance,
            "policy_std": policy_std,
            "gradient_norm": gradient_norm_mean,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "return_mean": return_mean,
            "value_mean": value_mean,
            "learning_rate": learning_rate,
            "samples": float(num_samples),
            "epochs_completed": float(epochs_completed),
            "minibatches": float(minibatches),
            "early_stop": float(stop_early),
            "update_seconds": update_seconds,
            "update_samples_per_second": num_samples / max(update_seconds, 1e-9),
            "n_updates": float(model._n_updates),
        }
        for name, value in metrics.items():
            if np.isfinite(value):
                self.writer.add_scalar(
                    f"stage_{stage}/train/{name}", value, self.global_timesteps
                )
        return metrics

    def _update_ready_policies(self, *, final: bool = False) -> None:
        for stage in range(NUM_STAGE_POLICIES):
            if self.frozen[stage]:
                self.pending[stage].clear()
                continue
            target = self._target_samples(stage)
            enough = self.pending[stage].num_samples >= target
            if final:
                enough = self.pending[stage].num_samples >= 2
            if not enough:
                continue
            batch = self.pending[stage].pop_all()
            metrics = self._ppo_update(stage, batch)
            self.last_train_metrics[stage] = metrics
            self.last_train_timestep[stage] = self.global_timesteps

    def _success_rate(self, stage: int) -> float:
        outcomes = self.recent_outcomes[stage]
        return float(np.mean(outcomes)) if outcomes else 0.0

    def _freeze_reliable_policies(self) -> None:
        if not self.config.get("freeze_learned_policies", False):
            return
        threshold = float(
            self.config.get(
                "stage_freeze_success_rate",
                self.config.get("stage_advance_success_rate", 0.9),
            )
        )
        minimum = int(
            self.config.get(
                "stage_freeze_min_attempts",
                self.config.get("stage_min_episodes", 1000),
            )
        )
        minimum = min(minimum, self.recent_outcomes[0].maxlen or minimum)
        for stage in range(NUM_STAGE_POLICIES):
            if self.frozen[stage] or len(self.recent_outcomes[stage]) < minimum:
                continue
            # Freeze in curriculum order so a downstream policy is not locked
            # against an input-state distribution that its predecessors are
            # still changing.
            if stage > 0 and not bool(np.all(self.frozen[:stage])):
                continue
            rate = self._success_rate(stage)
            if rate >= threshold:
                self.frozen[stage] = True
                self.pending[stage].clear()
                log.info(
                    f"Stage {stage} policy frozen at rolling success {rate:.3f}"
                )

    @staticmethod
    def _snapshot_counts() -> dict[int, int]:
        """Read current in-memory curriculum occupancy without owning it."""
        try:
            from metasim.cfg.checkers import stages_chairman

            return {
                stage: len(stages_chairman.RAM_SNAPSHOT_BUFFER.get(stage, []))
                for stage in range(1, NUM_STAGE_POLICIES)
            }
        except (ImportError, AttributeError, TypeError):
            return {stage: 0 for stage in range(1, NUM_STAGE_POLICIES)}

    def _policy_status(
        self,
        stage: int,
        current_envs: int,
        samples_this_rollout: int,
    ) -> str:
        if self.frozen[stage]:
            return "FROZEN / INFERENCE" if current_envs else "FROZEN"
        if current_envs:
            return "LEARNING"
        if self.last_train_timestep[stage] == self.global_timesteps:
            return "UPDATED"
        if samples_this_rollout:
            return "COLLECTED"
        if self.pending[stage].num_samples:
            return "BUFFERED"
        return "WAITING"

    def _print_policy_tables(
        self,
        *,
        occupancy: np.ndarray,
        sample_delta: np.ndarray,
        snapshots: dict[int, int],
        fps: float,
        sim_steps_per_second: float,
        elapsed: float,
        eta: float,
        progress_percent: float,
    ) -> None:
        if not self.config.get("terminal_tables", True):
            return
        interval = max(1, int(self.config.get("terminal_log_interval", 1)))
        self._terminal_log_iteration += 1
        if self._terminal_log_iteration % interval:
            return

        active_stages = []
        for stage in range(NUM_STAGE_POLICIES):
            current = int(np.sum(occupancy == stage))
            updated_now = self.last_train_timestep[stage] == self.global_timesteps
            if (
                current
                or sample_delta[stage]
                or self.pending[stage].num_samples
                or updated_now
            ):
                active_stages.append(stage)

        for stage in active_stages:
            current = int(np.sum(occupancy == stage))
            rollout = self._last_rollout_stats.get(stage, {})
            metrics = self.last_train_metrics.get(stage)
            status = self._policy_status(
                stage, current, int(sample_delta[stage])
            )
            title = (
                f"PPO POLICY {stage} · {STAGE_NAMES[stage]} · {status}"
            )
            table = Table(
                title=title,
                title_style="bold cyan" if not self.frozen[stage] else "bold green",
                show_header=True,
                header_style="bold",
                border_style="cyan" if not self.frozen[stage] else "green",
                box=box.ROUNDED,
                pad_edge=False,
            )
            table.add_column("group", style="dim", no_wrap=True)
            table.add_column("metric", no_wrap=True)
            table.add_column("value", justify="right", no_wrap=True)

            rows = [
                ("time", "fps (transitions/s)", f"{fps:,.0f}"),
                ("time", "sim steps/s", f"{sim_steps_per_second:,.2f}"),
                ("time", "elapsed / ETA", f"{_human_duration(elapsed)} / {_human_duration(eta)}"),
                ("time", "progress", f"{progress_percent:.3f}%"),
                ("time", "total timesteps", f"{self.global_timesteps:,}"),
                ("rollout", "active envs", f"{current:,} ({current / max(1, self.num_envs):.1%})"),
                ("rollout", "samples this rollout", f"{int(sample_delta[stage]):,}"),
                ("rollout", "pending / target", f"{self.pending[stage].num_samples:,} / {self._target_samples(stage):,}"),
                ("rollout", "total policy samples", f"{int(self.samples[stage]):,}"),
                ("rollout", "mean reward / step", f"{rollout.get('reward_mean', 0.0):.4f}"),
                ("episode", "mean return", f"{_mean_or_zero(self.recent_returns[stage]):.3f}"),
                ("episode", "mean length", f"{_mean_or_zero(self.recent_lengths[stage]):.1f}"),
                ("episode", "rolling success", f"{self._success_rate(stage):.1%} ({len(self.recent_outcomes[stage])})"),
                ("episode", "success / attempts", f"{int(self.successes[stage]):,} / {int(self.attempts[stage]):,}"),
                ("action", "mean |action|", f"{rollout.get('action_abs_mean', 0.0):.4f}"),
                ("action", "clipped fraction", f"{rollout.get('action_clip_fraction', 0.0):.2%}"),
                ("action", "mean |walk command|", f"{rollout.get('walk_command_abs_mean', 0.0):.4f}"),
                ("curriculum", "restart snapshots", f"{snapshots.get(stage, 0):,}" if stage else "procedural"),
            ]
            if metrics is not None:
                rows.extend(
                    [
                        ("train", "updates", f"{int(self.updates[stage]):,}"),
                        ("train", "learning rate", f"{metrics['learning_rate']:.3e}"),
                        ("train", "loss", f"{metrics['loss']:.5g}"),
                        ("train", "policy loss", f"{metrics['policy_loss']:.5g}"),
                        ("train", "value loss", f"{metrics['value_loss']:.5g}"),
                        ("train", "entropy", f"{metrics['entropy']:.4f}"),
                        ("train", "approx KL", f"{metrics['approx_kl']:.6f}"),
                        ("train", "clip fraction", f"{metrics['clip_fraction']:.2%}"),
                        ("train", "explained variance", f"{metrics['explained_variance']:.4f}"),
                        ("train", "policy std", f"{metrics['policy_std']:.4f}"),
                        ("train", "gradient norm", f"{metrics['gradient_norm']:.4f}"),
                        ("train", "update time", f"{metrics['update_seconds']:.2f}s"),
                    ]
                )

            previous_group = None
            for group, metric, value in rows:
                if previous_group is not None and group != previous_group:
                    table.add_section()
                table.add_row(group if group != previous_group else "", metric, value)
                previous_group = group
            self.console.print(table)

    def _log_rollout(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._start_time
        interval_seconds = max(now - self._last_log_time, 1e-9)
        timestep_delta = self.global_timesteps - self._last_log_timesteps
        env_step_delta = self.global_env_steps - self._last_log_env_steps
        fps = timestep_delta / interval_seconds
        average_fps = (
            (self.global_timesteps - self._initial_timesteps) / max(elapsed, 1e-9)
        )
        sim_steps_per_second = env_step_delta / interval_seconds
        progress_percent = 100.0 * self.global_timesteps / max(1, self.total_timesteps)
        remaining_timesteps = max(0, self.total_timesteps - self.global_timesteps)
        eta = remaining_timesteps / max(average_fps, 1e-9)
        occupancy = self._current_stages_torch().detach().cpu().numpy()
        sample_delta = self.samples - self._last_logged_samples
        snapshots = self._snapshot_counts()
        max_snapshot_stage = max(
            (stage for stage, count in snapshots.items() if count), default=0
        )

        global_scalars = {
            "time/fps": fps,
            "time/average_fps": average_fps,
            "time/sim_steps_per_second": sim_steps_per_second,
            "time/elapsed_seconds": elapsed,
            "time/eta_seconds": eta,
            "time/total_timesteps": self.global_timesteps,
            "time/progress_percent": progress_percent,
            "time/rollout_collection_seconds": self._last_collect_seconds,
            "time/ppo_update_seconds": self._last_train_seconds,
            "curriculum/max_snapshot_stage": max_snapshot_stage,
            "curriculum/total_snapshots": sum(snapshots.values()),
            "curriculum/frozen_policies": int(self.frozen.sum()),
        }
        if torch.cuda.is_available():
            global_scalars.update(
                {
                    "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 2**30,
                    "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                    "system/gpu_max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        for name, value in global_scalars.items():
            self.writer.add_scalar(name, value, self.global_timesteps)

        for stage in range(NUM_STAGE_POLICIES):
            rate = self._success_rate(stage)
            current = int(np.sum(occupancy == stage))
            attempts = int(self.attempts[stage])
            rollout = self._last_rollout_stats.get(stage, {})
            stage_scalars = {
                "rollout/current_envs": current,
                "rollout/env_fraction": current / max(1, self.num_envs),
                "rollout/samples_this_rollout": int(sample_delta[stage]),
                "rollout/total_samples": int(self.samples[stage]),
                "rollout/pending_samples": self.pending[stage].num_samples,
                "rollout/target_samples": self._target_samples(stage),
                "rollout/mean_reward": rollout.get("reward_mean", 0.0),
                "rollout/reward_std": rollout.get("reward_std", 0.0),
                "rollout/terminals": rollout.get("terminals", 0.0),
                "rollout/successes": rollout.get("successes", 0.0),
                "episode/mean_return": _mean_or_zero(self.recent_returns[stage]),
                "episode/mean_length": _mean_or_zero(self.recent_lengths[stage]),
                "episode/rolling_success_rate": rate,
                "episode/lifetime_success_rate": self.successes[stage] / max(1, attempts),
                "episode/attempts": attempts,
                "episode/successes": int(self.successes[stage]),
                "action/mean": rollout.get("action_mean", 0.0),
                "action/std": rollout.get("action_std", 0.0),
                "action/abs_mean": rollout.get("action_abs_mean", 0.0),
                "action/clip_fraction": rollout.get("action_clip_fraction", 0.0),
                "action/walk_command_abs_mean": rollout.get("walk_command_abs_mean", 0.0),
                "state/frozen": int(self.frozen[stage]),
                "state/updates": int(self.updates[stage]),
                "curriculum/snapshots": snapshots.get(stage, 0),
            }
            for name, value in stage_scalars.items():
                self.writer.add_scalar(
                    f"stage_{stage}/{name}", value, self.global_timesteps
                )

        self._print_policy_tables(
            occupancy=occupancy,
            sample_delta=sample_delta,
            snapshots=snapshots,
            fps=fps,
            sim_steps_per_second=sim_steps_per_second,
            elapsed=elapsed,
            eta=eta,
            progress_percent=progress_percent,
        )
        self.writer.flush()
        self._last_log_time = now
        self._last_log_timesteps = self.global_timesteps
        self._last_log_env_steps = self.global_env_steps
        self._last_logged_samples = self.samples.copy()

    def _manifest(self, model_paths: dict[int, str]) -> dict:
        return {
            "format_version": 2,
            "trainer": "concurrent_ragged_multi_ppo",
            "task": self.config.get("task"),
            "num_stage_policies": NUM_STAGE_POLICIES,
            "global_timesteps": self.global_timesteps,
            "global_env_steps": self.global_env_steps,
            "stages": {
                str(stage): {
                    "model": model_paths[stage],
                    "samples": int(self.samples[stage]),
                    "updates": int(self.updates[stage]),
                    "attempts": int(self.attempts[stage]),
                    "successes": int(self.successes[stage]),
                    "rolling_success_rate": self._success_rate(stage),
                    "frozen": bool(self.frozen[stage]),
                }
                for stage in range(NUM_STAGE_POLICIES)
            },
        }

    def save(self, label: str) -> None:
        model_paths: dict[int, str] = {}
        for stage, model in self.models.items():
            stage_dir = self.run_dir / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            model_path = stage_dir / f"model_{label}"
            model.save(str(model_path))
            model_paths[stage] = str(
                (Path(f"stage_{stage}") / f"model_{label}.zip").as_posix()
            )
        manifest_path = self.run_dir / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(self._manifest(model_paths), indent=2), encoding="utf-8"
        )
        log.info(f"Saved concurrent multi-policy bundle to {self.run_dir}")

    def learn(self) -> Path:
        observations = self._env_reset_torch()
        save_freq = int(self.config.get("model_save_freq", 1_000_000))
        self._initial_timesteps = self.global_timesteps
        self._start_time = time.perf_counter()
        self._last_log_time = self._start_time
        self._last_log_timesteps = self.global_timesteps
        self._last_log_env_steps = self.global_env_steps
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.writer.add_text(
            "run/config",
            "```json\n" + json.dumps(self.config, indent=2, default=str) + "\n```",
            self.global_timesteps,
        )
        log.info(
            f"Starting concurrent ChairMan Multi-PPO with {NUM_STAGE_POLICIES} "
            f"policies, {self.num_envs} envs and {self.total_timesteps} timesteps"
        )
        log.info(
            "Torch-native rollout path enabled on {}; observations, stage routing, "
            "walking inference, actions and rollout buffers stay on this device.",
            self.device,
        )
        log.info(
            "TensorBoard log: {}\nRun: tensorboard --logdir {}",
            self.tensorboard_dir,
            self.tensorboard_dir.parent,
        )
        try:
            while self.global_timesteps < self.total_timesteps:
                collect_started = time.perf_counter()
                observations = self._collect_rollout(observations)
                self._last_collect_seconds = time.perf_counter() - collect_started
                train_started = time.perf_counter()
                self._update_ready_policies()
                self._freeze_reliable_policies()
                self._last_train_seconds = time.perf_counter() - train_started
                self._log_rollout()
                if (
                    save_freq > 0
                    and self.global_timesteps - self.last_save_timestep >= save_freq
                ):
                    self.save(str(self.global_timesteps))
                    self.last_save_timestep = self.global_timesteps
            self._update_ready_policies(final=True)
            self.save("final")
            return self.run_dir
        finally:
            self.writer.close()


def _model_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    zipped = Path(f"{path}.zip")
    return zipped if zipped.is_file() else None


def resolve_policy_bundle(bundle_path: str | Path):
    root = Path(bundle_path).expanduser()
    if root.is_file() and root.name == MANIFEST_NAME:
        root = root.parent
    if root.is_dir() and not (root / MANIFEST_NAME).is_file():
        runs = sorted(
            (child for child in root.iterdir() if (child / MANIFEST_NAME).is_file()),
            key=lambda child: child.stat().st_mtime,
        )
        if runs:
            root = runs[-1]
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Multi-policy manifest not found below {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("num_stage_policies", -1)) != NUM_STAGE_POLICIES:
        raise ValueError("The bundle does not contain six ChairMan policies")

    paths: dict[int, str] = {}
    for stage in range(NUM_STAGE_POLICIES):
        relative = manifest.get("stages", {}).get(str(stage), {}).get("model")
        if not relative:
            raise ValueError(f"Manifest has no model entry for stage {stage}")
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        existing = _model_file(candidate)
        if existing is None:
            raise FileNotFoundError(f"Stage {stage} model does not exist: {candidate}")
        paths[stage] = str(existing)
    return paths, manifest


class MultiPolicyRouter:
    """Inference-only batch router shared by PPO evaluation and DAgger."""

    def __init__(self, models: dict[int, PPO], action_space):
        if set(models) != set(range(NUM_STAGE_POLICIES)):
            raise ValueError("The router requires policies for stages 0 through 5")
        self.models = models
        self.action_space = action_space

    def predict(
        self, observations: np.ndarray, stages: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        observations = np.asarray(observations)
        stages = np.asarray(stages, dtype=np.int64)
        actions = np.zeros(
            (len(observations), self.action_space.shape[0]), dtype=np.float32
        )
        for stage, model in self.models.items():
            mask = stages == stage
            if mask.any():
                predicted, _ = model.predict(
                    observations[mask], deterministic=deterministic
                )
                actions[mask] = predicted
        invalid = (stages < 0) | (stages > NUM_STAGE_POLICIES)
        if invalid.any():
            raise ValueError(f"Unexpected stages {np.unique(stages[invalid])}")
        return np.clip(actions, self.action_space.low, self.action_space.high)


def load_policy_router(env, bundle_path: str | Path, device: str | None = None):
    paths, manifest = resolve_policy_bundle(bundle_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        stage: PPO.load(path, env=env, device=device)
        for stage, path in paths.items()
    }
    return MultiPolicyRouter(models, env.action_space), manifest
