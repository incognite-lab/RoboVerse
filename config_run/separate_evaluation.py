"""Shared checkpoint reports and episode-based stage metrics for ChairMan.

Stage SR is the fraction of evaluated episodes that completed that stage,
including episodes which failed later. Unreached stages therefore have SR 0.
It differs from the conditional SR among attempts starting in that stage.
"""
from pathlib import Path
import csv
import re

import numpy as np
from loguru import logger as log
try:
    from .stage_diagnostics import StageDiagnostics
except ImportError:
    from stage_diagnostics import StageDiagnostics


class StageEpisodeMetrics:
    def __init__(self, num_envs, num_stages=6):
        self.completed = np.zeros((num_envs, num_stages), dtype=bool)
        self.counts = np.zeros(num_stages, dtype=np.int64)
        self.episodes = 0

    def step(self, infos, dones, active):
        for index in np.flatnonzero(active):
            info = infos[index]
            stage = int(info.get("completed_stage", -1))
            if stage < 0 and dones[index] and info.get("is_success", False):
                stage = int(info.get("stage_before", -1))
            if 0 <= stage < self.completed.shape[1]:
                self.completed[index, stage] = True
            if dones[index]:
                self.finish(index)

    def finish(self, index, count=1):
        # At a cap only the first outstanding attempt has actually run.
        self.counts += self.completed[index]
        self.episodes += count
        self.completed[index] = False

    def row(self):
        result = {}
        for stage, count in enumerate(self.counts):
            result[f"stage_{stage}_episode_success_count"] = int(count)
            result[f"stage_{stage}_episode_success_rate"] = (
                float(count / self.episodes) if self.episodes else 0.0)
        return result


def evaluate_rollouts(env, predict, episodes, max_steps, *, num_stages=6,
                      count_timeouts=True, step_callback=None, stage_records=None):
    """Assign episode quotas before rollout so fast failures cannot bias SR."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("Evaluation episodes and max_steps must be positive")
    observation = env.reset()
    quotas = np.full(env.num_envs, episodes // env.num_envs, dtype=np.int64)
    quotas[:episodes % env.num_envs] += 1
    finished = np.zeros(env.num_envs, dtype=np.int64)
    returns = np.zeros(env.num_envs)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    metrics = StageEpisodeMetrics(env.num_envs, num_stages)
    diagnostics = StageDiagnostics(env, stage_records, num_stages) if stage_records is not None else None
    end_counts = np.zeros(num_stages, dtype=np.int64)
    rewards_out, lengths_out, successes = [], [], []
    for _ in range(max_steps):
        actions = np.clip(predict(observation), env.action_space.low, env.action_space.high)
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite evaluation action")
        observation, rewards, dones, infos = env.step(actions)
        active = finished < quotas
        returns[active] += np.asarray(rewards)[active]
        lengths[active] += 1
        metrics.step(infos, dones, active)
        if diagnostics is not None:
            diagnostics.step(infos, dones, active)
        if step_callback is not None:
            step_callback(infos)
        for index in np.flatnonzero(np.asarray(dones) & active):
            rewards_out.append(float(returns[index]))
            lengths_out.append(int(lengths[index]))
            successes.append(int(infos[index].get("is_success", False)))
            stage = int(np.clip(infos[index].get("stage_before", 0), 0, num_stages - 1))
            end_counts[stage] += 1
            finished[index] += 1
            returns[index] = lengths[index] = 0
        if np.all(finished >= quotas):
            break
    capped = int((quotas - finished).sum())
    if diagnostics is not None:
        diagnostics.cap((finished < quotas) & (lengths > 0), count_timeouts)
    if count_timeouts:
        stages = env.get_current_stages()
        for index in np.flatnonzero(finished < quotas):
            remaining = int(quotas[index] - finished[index])
            metrics.finish(index, remaining)
            rewards_out.extend([float(returns[index])] + [0.] * (remaining - 1))
            lengths_out.extend([int(lengths[index])] + [0] * (remaining - 1))
            successes.extend([0] * remaining)
            end_counts[int(np.clip(stages[index], 0, num_stages - 1))] += remaining
    row = dict(episodes_finished=len(successes), requested_episodes=episodes, capped_attempts=capped,
               mean_reward=float(np.mean(rewards_out)) if successes else 0.,
               std_reward=float(np.std(rewards_out)) if successes else 0.,
               success_rate=float(np.mean(successes)) if successes else 0.,
               success_count=sum(successes),
               mean_length=float(np.mean(lengths_out)) if successes else 0.,
               std_length=float(np.std(lengths_out)) if successes else 0.)
    row.update({f"end_stage_{stage}_count": int(count) for stage, count in enumerate(end_counts)})
    row.update(metrics.row())
    return row


def ppo_checkpoints(paths, manifest, checkpoint=None, *, stride=1):
    """Sample complete six-policy bundles in training order and include final."""
    if type(stride) is not int or stride < 1:
        raise ValueError("eval_checkpoint_stride must be a positive integer")
    if checkpoint not in (None, ""):
        step = (int(manifest.get("global_timesteps", 0)) if str(checkpoint) == "final"
                else int(checkpoint))
        return [(step, str(checkpoint))]
    per_policy = []
    for path in paths.values():
        labels = set()
        for candidate in Path(path).parent.glob("model_*.zip"):
            match = re.fullmatch(r"model_(\d+|final)\.zip", candidate.name)
            if match:
                labels.add(match.group(1))
        per_policy.append(labels)
    common = set.intersection(*per_policy)
    all_labels = set.union(*per_policy)
    if all_labels - common:
        log.warning("Skipping incomplete PPO bundles: {}", sorted(all_labels - common))
    result = sorted((int(label), label) for label in common if label != "final")[::stride]
    if "final" in common:
        result.append((int(manifest["global_timesteps"]), "final"))
    if not result:
        raise FileNotFoundError("No complete PPO checkpoint bundles found")
    return sorted(result, key=lambda item: (item[0], item[1] == "final"))


def legacy_checkpoint_step(path, fallback):
    """Use saved training progress for final/single legacy student checkpoints."""
    import torch
    sidecar = Path(f"{path}.training.pth")
    if sidecar.is_file():
        payload = torch.load(sidecar, map_location="cpu", weights_only=True)
        return int(payload["next_step"])
    match = re.fullmatch(r"student_model_step_(\d+)\.pth", Path(path).name)
    if match:
        return int(match.group(1))
    log.warning("No saved training step for {}; using x_step={}", path, fallback)
    return fallback


def save_stage_plots(rows, output, prefix, xlabel, num_stages=6):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    for stage in range(num_stages):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot([row["x_step"] for row in rows],
                  [row[f"stage_{stage}_episode_success_rate"] for row in rows], marker="o")
        axis.set(xlabel=xlabel, ylabel="Episodes completing stage (%)", ylim=(-.02, 1.02),
                 title=f"ChairMan separate — Stage {stage} success rate")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(True, alpha=.3)
        figure.tight_layout()
        figure.savefig(Path(output) / f"{prefix}_stage_{stage}_success_plot.png")
        plt.close(figure)


def save_evaluation(rows, output, prefix, xlabel, num_stages=6):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["x_step"])
    csv_path = output / f"{prefix}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        save_stage_plots(rows, output, prefix, xlabel, num_stages)
        for key, label, suffix in (("mean_reward", "Average reward", "reward"),
                                   ("success_rate", "Success rate", "success"),
                                   ("mean_length", "Average episode length", "length")):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot([row["x_step"] for row in rows], [row[key] for row in rows], marker="o")
            goal_stage = rows[0].get("eval_goal_stage")
            if key == "success_rate" and goal_stage is not None:
                goal = f"reach stage {goal_stage}" if goal_stage < 6 else "complete the whole task"
                label = f"{label} ({goal})"
            axis.set(xlabel=xlabel, ylabel=label, title=f"ChairMan separate — {label}")
            axis.grid(True, alpha=.3)
            figure.tight_layout()
            figure.savefig(output / f"{prefix}_{suffix}_plot.png")
            plt.close(figure)
    except ImportError:
        log.warning("matplotlib is unavailable; CSV was saved without plots")
    log.info("Saved evaluation results to {}", csv_path)
