"""Per-attempt endpoint data and checkpoint comparisons for ChairMan evaluation."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger as log


GEOMETRY_FIELDS = (
    "pelvis_chair_xy_m", "pelvis_chair_lateral_m", "pelvis_chair_back_m", "neck_height_m",
    "walk_goal_error_m", "left_hand_target_error_m", "right_hand_target_error_m",
    "finger_grasp_error_rad", "chair_start_error_m", "chair_pull_goal_error_m",
    "chair_speed_m_s", "pelvis_speed_xy_m_s", "finger_open_error_rad", "arm_rest_error_rad",
)
STAGE_GOALS = (
    ("Approach position error (m)", ("walk_goal_error_m",)),
    ("Worst hand position error (m)", ("left_hand_target_error_m", "right_hand_target_error_m")),
    ("Worst finger grasp error (rad)", ("finger_grasp_error_rad",)),
    ("Chair pull target error (m)", ("chair_pull_goal_error_m",)),
    ("Worst finger opening error (rad)", ("finger_open_error_rad",)),
    ("Worst arm rest error (rad)", ("arm_rest_error_rad",)),
)
STAGE_NAMES = ("Approach", "Reach", "Grasp", "Pull", "Release", "Lower arms")
REASON_LABELS = {
    "fall": "Fall",
    "stage_timeout": "Stage time limit",
    "global_timeout": "Episode time limit",
    "evaluation_cap": "Evaluation step cap",
    "chair_displacement": "Chair displacement",
    "stage2_drift_failure": "Hand drift during grasp",
    "stage3_drift_failure": "Hand drift during pull",
    "stage4_hand_drift_failure": "Hand drift during release",
    "stage4_motion_failure": "Robot or chair moving",
    "stage5_motion_failure": "Robot or chair moving",
    "unknown": "Unknown",
}


class StageDiagnostics:
    def __init__(self, env, records, num_stages):
        self.env = env
        self.records = records
        self.num_stages = num_stages
        self.stage = np.full(env.num_envs, -1, dtype=int)
        self.steps = np.zeros(env.num_envs, dtype=int)
        self.episode = np.zeros(env.num_envs, dtype=int)

    def step(self, infos, dones, active):
        for index in np.flatnonzero(active):
            info = infos[index]
            stage = int(info.get("stage_before", -1))
            if self.stage[index] != stage:
                self.stage[index], self.steps[index] = stage, 0
            self.steps[index] += 1
            geometry = info.get("stage_endpoint_geometry", {})
            success = bool(dones[index] and info.get("is_success", False))
            if dones[index] and not success:
                reasons = info.get("failure_reasons") or (
                    "global_timeout" if info.get("TimeLimit.truncated", False) else "unknown",
                )
                self._record(index, stage, "failed", reasons, geometry, self.steps[index])
            elif int(info.get("completed_stage", -1)) == stage or success:
                self._record(index, stage, "completed", (), geometry, self.steps[index])
                self.stage[index], self.steps[index] = -1, 0
            if dones[index]:
                self.episode[index] += 1
                self.stage[index], self.steps[index] = -1, 0

    def cap(self, active, count_timeouts):
        ids = np.flatnonzero(active)
        if not ids.size:
            return
        stages = self.env.get_current_stages()
        capture = getattr(self.env, "stage_endpoint_geometry", None)
        geometry = capture(ids) if capture is not None else {}
        for index in ids:
            stage = int(stages[index])
            steps = self.steps[index] if self.stage[index] == stage else 0
            outcome = "capped_failure" if count_timeouts else "censored"
            self._record(index, stage, outcome, ("evaluation_cap",), geometry.get(index, {}), steps)

    def _record(self, index, stage, outcome, reasons, geometry, steps):
        if 0 <= stage < self.num_stages:
            self.records.append(dict(
                env_id=int(index), episode=int(self.episode[index]) + 1, stage=stage,
                outcome=outcome, stage_steps=int(steps), reasons="|".join(reasons),
                **{key: geometry.get(key) if np.isfinite(geometry.get(key, np.nan)) else None
                   for key in GEOMETRY_FIELDS},
            ))


def save_stage_diagnostics(report, output, prefix, num_stages=6):
    """Write raw attempt data, summary counts and one diagnostic figure per stage."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = sorted(report, key=lambda item: item[0]["x_step"])
    attempts = []
    summaries = []
    failure_counts = []
    by_stage = [[] for _ in range(num_stages)]
    for checkpoint, events in report:
        identity = {key: checkpoint[key] for key in ("checkpoint", "x_step")}
        attempts.extend({**identity, **event} for event in events)
        for stage in range(num_stages):
            stage_events = [event for event in events if event["stage"] == stage]
            outcomes = Counter(event["outcome"] for event in stage_events)
            reasons = Counter(reason for event in stage_events
                              for reason in event["reasons"].split("|") if reason)
            by_stage[stage].append((stage_events, outcomes, reasons))
            stage_key = dict(**identity, stage=stage)
            summaries.append(dict(
                **stage_key, requested_episodes=checkpoint["requested_episodes"],
                entered=len(stage_events), completed=outcomes["completed"],
                failed=outcomes["failed"], capped_failure=outcomes["capped_failure"],
                censored=outcomes["censored"],
                completion_rate=outcomes["completed"] / len(stage_events) if stage_events else None,
            ))
            for reason, count in sorted(reasons.items()):
                distances = _values(
                    [event for event in stage_events if reason in event["reasons"].split("|")],
                    "pelvis_chair_xy_m")
                failure_counts.append(dict(
                    **stage_key, reason=reason, count=count, measured=len(distances),
                    distance_p25_m=_percentile(distances, 25), distance_median_m=_percentile(distances, 50),
                    distance_p75_m=_percentile(distances, 75),
                ))
    _write_csv(output / f"{prefix}_stage_attempts.csv", attempts,
               ("checkpoint", "x_step", "env_id", "episode", "stage", "outcome", "stage_steps", "reasons",
                *GEOMETRY_FIELDS))
    _write_csv(output / f"{prefix}_stage_summary.csv", summaries, list(summaries[0]))
    _write_csv(output / f"{prefix}_stage_failure_reasons.csv", failure_counts,
               ("checkpoint", "x_step", "stage", "reason", "count", "measured",
                "distance_p25_m", "distance_median_m", "distance_p75_m"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is unavailable; stage diagnostics CSVs were saved")
        return
    for stage, entries in enumerate(by_stage):
        _stage_figure(plt, report, stage, entries, output / f"{prefix}_stage_{stage}_diagnostics.png")
    log.info("Saved stage diagnostics to {} ({} stage attempts)", output, len(attempts))


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _values(events, *fields):
    return [max(row[key] for key in fields) for row in events if all(row[key] is not None for key in fields)]


def _percentile(values, quantile):
    return float(np.percentile(values, quantile)) if values else None


def _stage_figure(plt, report, stage, entries, path):
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    counts, distances, positions, progress, times, goal = axes.ravel()
    x = np.arange(len(entries))
    ticks = np.unique(np.linspace(0, len(x) - 1, min(8, len(x)), dtype=int))
    labels = [f'{row["x_step"]:,}' + (" (final)" if "final" in str(row["checkpoint"]) else "")
              for row, _ in report]
    all_reasons = sorted({reason for _, _, reasons in entries for reason in reasons})
    palette = plt.get_cmap("tab10").colors
    colors = {reason: palette[i % len(palette)] for i, reason in enumerate(all_reasons)}
    if all_reasons:
        matrix = np.array([[reasons[reason] for _, _, reasons in entries] for reason in all_reasons])
        heatmap = counts.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
        figure.colorbar(heatmap, ax=counts, label="Count")
        counts.set_yticks(range(len(all_reasons)), [REASON_LABELS.get(r, r) for r in all_reasons])
        if len(x) <= 15:
            for (i, j), value in np.ndenumerate(matrix):
                counts.text(j, i, str(value), ha="center", va="center",
                            color="white" if value > matrix.max() / 2 else "black", fontsize=8)
    else:
        _empty(counts, "No failures or evaluation caps")
    counts.set_title("Termination reasons (can overlap)")

    failures = [[event for event in events if event["outcome"] != "completed"]
                for events, _, _ in entries]
    for reason_index, reason in enumerate(all_reasons):
        per_checkpoint = [_values([event for event in events if reason in event["reasons"].split("|")],
                                  "pelvis_chair_xy_m") for events in failures]
        # Offset concurrent reasons so their distance distributions remain visible.
        offset = .6 * ((reason_index + .5) / len(all_reasons) - .5)
        points = [(i + offset, value) for i, values in enumerate(per_checkpoint) for value in values]
        if points:
            indices, values = zip(*points)
            distances.scatter(indices, values, color=colors[reason], s=10, alpha=.5)
    distances.set(title="Distance at failure or cap", ylabel="Pelvis to chair base, XY (m)")
    if not any(_values(events, "pelvis_chair_xy_m") for events in failures):
        _empty(distances, "No measured failure distances")
    for reason in all_reasons:
        distances.scatter([], [], color=colors[reason], label=REASON_LABELS.get(reason, reason), s=15)
    if all_reasons:
        distances.legend(fontsize=7)

    # Position relative to the chair at termination, pooled across checkpoints.
    for reason in all_reasons:
        points = [(event["pelvis_chair_lateral_m"], event["pelvis_chair_back_m"])
                  for events in failures for event in events
                  if reason in event["reasons"].split("|") and event["pelvis_chair_lateral_m"] is not None
                  and event["pelvis_chair_back_m"] is not None]
        if points:
            lateral, back = zip(*points)
            positions.scatter(lateral, back, s=10, color=colors[reason], alpha=.45)
    positions.scatter([0], [0], marker="+", color="black", s=60)
    positions.set(title="Failure positions across checkpoints", xlabel="Chair lateral offset (m)",
                  ylabel="Offset toward chair back (m)", aspect="equal")

    requested = [row["requested_episodes"] for row, _ in report]
    progress.plot(x, [len(events) / total for (events, _, _), total in zip(entries, requested)],
                  marker="o", label="Entered / requested")
    progress.plot(x, [outcomes["completed"] / len(events) if events else np.nan
                      for events, outcomes, _ in entries], marker="o", label="Completed / entered")
    progress.plot(x, [sum(outcomes[key] for key in ("failed", "capped_failure", "censored")) / len(events)
                      if events else np.nan for events, outcomes, _ in entries], marker="o",
                  label="Failed or capped / entered")
    progress.set(title="Stage progress across training", ylim=(-.02, 1.02))
    progress.yaxis.set_major_formatter(PercentFormatter(1))
    progress.legend(fontsize=8)

    reached = [i for i, (events, _, _) in enumerate(entries) if events]
    selected = [reached[j] for j in np.unique(np.linspace(0, len(reached) - 1, min(3, len(reached)), dtype=int))]
    horizon = max((event["stage_steps"] for i in selected for event in entries[i][0]), default=0)
    for i in selected:
        events, _, _ = entries[i]
        completed = sorted(event["stage_steps"] for event in events if event["outcome"] == "completed")
        rates = np.arange(1, len(completed) + 1) / len(events)
        times.step([0, *completed, horizon], [0, *rates, len(completed) / len(events)],
                   where="post", label=labels[i])
    times.set(title="Completion vs time in stage", xlabel="Stage control steps", ylabel="% of stage attempts",
              ylim=(-.02, 1.02))
    times.yaxis.set_major_formatter(PercentFormatter(1))
    if reached:
        times.legend(title="Training step", fontsize=8)
    else:
        _empty(times, "No stage attempts")

    label, keys = STAGE_GOALS[stage] if stage < len(STAGE_GOALS) else (
        "Pelvis to chair base, XY (m)", ("pelvis_chair_xy_m",))
    for completed, name in ((True, "Completed"), (False, "Failed or capped")):
        values = [_values([event for event in events if (event["outcome"] == "completed") == completed],
                          *keys) for events, _, _ in entries]
        if any(values):
            q25, median, q75 = np.asarray(
                [[_percentile(v, q) for v in values] for q in (25, 50, 75)], dtype=float)
            line, = goal.plot(x, median, marker="o", label=name)
            goal.fill_between(x, q25, q75, color=line.get_color(), alpha=.15)
    goal.set(title="Goal error at stage exit (median, middle 50%)", ylabel=label)
    if goal.lines:
        goal.legend(fontsize=8)
    else:
        _empty(goal, "No measured stage endpoints")
    for axis in (counts, distances, progress, goal):
        axis.set(xlim=(-.5, len(x) - .5), xlabel="Training checkpoint")
        axis.set_xticks(ticks, [labels[i] for i in ticks], rotation=25, ha="right")
    for axis in (distances, positions, progress, times, goal):
        axis.grid(alpha=.2)
    name = f" · {STAGE_NAMES[stage]}" if stage < len(STAGE_NAMES) else ""
    figure.suptitle(f"ChairMan separate  |  Stage {stage}{name}", fontsize=16, fontweight="bold")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _empty(axis, message):
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes, color="0.35")
