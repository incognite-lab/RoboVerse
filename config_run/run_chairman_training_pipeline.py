#!/usr/bin/env python3
"""Run the complete Chairman teacher/student training pipeline.

The pipeline is intentionally implemented outside ``main.py`` so all existing
config entry points continue to behave exactly as before.  It runs:

    PPO train -> PPO eval -> DAgger train -> DAgger eval
              -> GRPO train -> GRPO eval

Before every dependent step it updates only ``load_model_path`` in the
corresponding YAML file.  Evaluation results are read from the CSV files
created by ``main.py`` and the checkpoint with the highest selected metric is
copied to ``top_model`` in that stage's pipeline artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
CONFIG_ROOT = SCRIPT_DIR / "configs"

CONFIG_NAMES = {
    "train_ppo": "chairman_simple/train_ppo",
    "eval_ppo": "chairman_simple/eval_ppo",
    "train_dagger": "chairman_simple/train_dagger",
    "eval_dagger": "chairman_simple/eval_dagger",
    "train_grpo": "chairman_simple/train_GRPO",
    "eval_grpo": "chairman_simple/eval_GRPO",
}

CHECKPOINT_PATTERNS = {
    "ppo": re.compile(r"model_(\d+)\.zip$"),
    "dagger": re.compile(r"student_model_(?:step_\d+|final)\.pth$"),
    "grpo": re.compile(r"student_grpo_(?:step_\d+|final)\.pth$"),
}

EVALUATION_CSV = {
    "ppo": "eval_ppo_summary.csv",
    "dagger": "eval_dagger_results.csv",
    "grpo": "eval_grpo_results.csv",
}


class PipelineError(RuntimeError):
    """A pipeline step completed without producing its required artifact."""


def config_path(config_name: str) -> Path:
    return CONFIG_ROOT / f"{config_name}.yaml"


def load_yaml(config_name: str) -> dict[str, Any]:
    path = config_path(config_name)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise PipelineError(f"Configuration is not a YAML mapping: {path}")
    return data


def resolve_repo_path(value: str, *, config_name: str, key: str) -> Path:
    if not value:
        raise PipelineError(f"Missing '{key}' in {config_path(config_name)}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def yaml_path_value(path: Path) -> str:
    """Return a stable path suitable for configs loaded from the repo root."""
    path = path.resolve()
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return str(path)
    return f"./{relative.as_posix()}"


def update_load_model_path(config_name: str, model_path: Path) -> None:
    """Update one YAML key without reformatting or discarding YAML comments."""
    path = config_path(config_name)
    original = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(?P<indent>[ \t]*)load_model_path[ \t]*:.*$", re.MULTILINE)
    matches = list(pattern.finditer(original))

    if len(matches) > 1:
        raise PipelineError(
            f"Refusing to update {path}: it contains multiple active load_model_path keys"
        )

    # JSON string syntax is also valid YAML and safely quotes spaces/special chars.
    encoded_path = json.dumps(yaml_path_value(model_path), ensure_ascii=False)
    if matches:
        updated = pattern.sub(
            lambda match: f"{match.group('indent')}load_model_path: {encoded_path}",
            original,
            count=1,
        )
    else:
        separator = "" if original.endswith("\n") else "\n"
        updated = (
            original
            + separator
            + "\n# Managed by run_chairman_training_pipeline.py\n"
            + f"load_model_path: {encoded_path}\n"
        )

    temporary = path.with_name(f".{path.name}.pipeline-{os.getpid()}.tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, path)
    print(f"[pipeline] {config_name}: load_model_path = {yaml_path_value(model_path)}", flush=True)


def checkpoint_snapshot(root: Path, stage: str) -> dict[Path, tuple[int, int]]:
    """Capture size and mtime for matching checkpoints below ``root``."""
    if not root.is_dir():
        return {}
    pattern = CHECKPOINT_PATTERNS[stage]
    snapshot: dict[Path, tuple[int, int]] = {}
    for candidate in root.rglob("*"):
        if candidate.is_file() and pattern.fullmatch(candidate.name):
            stat = candidate.stat()
            snapshot[candidate.resolve()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def changed_checkpoints(
    before: dict[Path, tuple[int, int]], root: Path, stage: str
) -> list[Path]:
    after = checkpoint_snapshot(root, stage)
    changed = sorted(path for path, signature in after.items() if before.get(path) != signature)
    if not changed:
        expected = CHECKPOINT_PATTERNS[stage].pattern
        raise PipelineError(
            f"The {stage.upper()} training command succeeded, but produced no new or changed "
            f"checkpoint matching {expected!r} below {root}"
        )
    return changed


def create_evaluation_view(
    pipeline_dir: Path, stage: str, checkpoints: Iterable[Path]
) -> Path:
    """Expose only checkpoints from this run to an evaluator.

    The training configs currently use reusable output directories.  A small
    per-run view prevents checkpoints left by older runs from participating in
    the current evaluation.  Symlinks avoid duplicating large model files.
    """
    view_dir = pipeline_dir / stage
    view_dir.mkdir(parents=True, exist_ok=False)
    sources: dict[str, str] = {}

    for source in checkpoints:
        destination = view_dir / source.name
        if destination.exists() or destination.is_symlink():
            raise PipelineError(
                f"Two fresh {stage.upper()} checkpoints have the same filename: {source.name}"
            )
        destination.symlink_to(source.resolve())
        sources[source.name] = yaml_path_value(source)

    (view_dir / "checkpoint_sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return view_dir


def run_main(python: str, config_name: str) -> None:
    command = [python, str(MAIN_SCRIPT.relative_to(REPO_ROOT)), config_name]
    print(f"\n[pipeline] Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def select_best_checkpoint(
    stage: str, evaluation_dir: Path, metric: str
) -> tuple[Path, dict[str, str]]:
    csv_path = evaluation_dir / EVALUATION_CSV[stage]
    if not csv_path.is_file():
        raise PipelineError(
            f"{stage.upper()} evaluation did not create the expected file: {csv_path}"
        )

    candidates: list[tuple[float, float, float, dict[str, str]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                primary = float(row[metric])
                mean_reward = float(row["mean_reward"])
                step = float(row.get("x_step", 0))
                checkpoint_name = row["checkpoint"]
            except (KeyError, TypeError, ValueError):
                continue
            if not checkpoint_name or not (
                math.isfinite(primary) and math.isfinite(mean_reward) and math.isfinite(step)
            ):
                continue
            if metric != "mean_reward":
                secondary = mean_reward
            else:
                try:
                    secondary = float(row.get("success_rate", 0) or 0)
                except (TypeError, ValueError):
                    secondary = -math.inf
                if not math.isfinite(secondary):
                    secondary = -math.inf
            candidates.append((primary, secondary, step, row))

    if not candidates:
        raise PipelineError(
            f"No valid '{metric}' values were found in evaluation results: {csv_path}"
        )

    _, _, _, best_row = max(candidates, key=lambda item: item[:3])
    checkpoint_name = Path(best_row["checkpoint"]).name
    checkpoint = evaluation_dir / checkpoint_name
    if not checkpoint.is_file():
        raise PipelineError(
            f"Best checkpoint listed in {csv_path} does not exist: {checkpoint}"
        )
    return checkpoint, best_row


def copy_top_model(
    stage: str, checkpoint: Path, row: dict[str, str], metric: str
) -> Path:
    suffix = ".zip" if stage == "ppo" else ".pth"
    top_model = checkpoint.parent / f"top_model{suffix}"
    shutil.copy2(checkpoint, top_model)
    print(
        f"[pipeline] Best {stage.upper()} checkpoint: {checkpoint.name} "
        f"({metric}={row[metric]}); copied to {top_model}",
        flush=True,
    )
    return top_model


def training_output_dir(config_name: str) -> Path:
    config = load_yaml(config_name)
    return resolve_repo_path(
        str(config.get("model_save_path", "")),
        config_name=config_name,
        key="model_save_path",
    )


def create_pipeline_dir() -> Path:
    root = SCRIPT_DIR / "output" / "training_pipeline"
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate.resolve()


def write_summary(pipeline_dir: Path, summary: dict[str, Any]) -> None:
    (pipeline_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_configuration() -> None:
    if not MAIN_SCRIPT.is_file():
        raise PipelineError(f"Missing entry point: {MAIN_SCRIPT}")
    for config_name in CONFIG_NAMES.values():
        load_yaml(config_name)
    for key in ("train_ppo", "train_dagger", "train_grpo"):
        training_output_dir(CONFIG_NAMES[key])


def run_pipeline(python: str, metric: str) -> Path:
    validate_configuration()
    pipeline_dir = create_pipeline_dir()
    summary: dict[str, Any] = {
        "selection_metric": metric,
        "pipeline_dir": yaml_path_value(pipeline_dir),
        "stages": {},
    }
    write_summary(pipeline_dir, summary)
    print(f"[pipeline] Artifacts: {pipeline_dir}", flush=True)

    # PPO teacher training and evaluation.
    ppo_output = training_output_dir(CONFIG_NAMES["train_ppo"])
    ppo_before = checkpoint_snapshot(ppo_output, "ppo")
    run_main(python, CONFIG_NAMES["train_ppo"])
    ppo_view = create_evaluation_view(
        pipeline_dir,
        "ppo",
        changed_checkpoints(ppo_before, ppo_output, "ppo"),
    )
    update_load_model_path(CONFIG_NAMES["eval_ppo"], ppo_view)
    run_main(python, CONFIG_NAMES["eval_ppo"])
    ppo_checkpoint, ppo_row = select_best_checkpoint("ppo", ppo_view, metric)
    top_ppo = copy_top_model("ppo", ppo_checkpoint, ppo_row, metric)
    summary["stages"]["ppo"] = {
        "best_checkpoint": ppo_checkpoint.name,
        "top_model": yaml_path_value(top_ppo),
        metric: ppo_row[metric],
    }
    write_summary(pipeline_dir, summary)

    # DAgger learns from the selected PPO teacher.
    update_load_model_path(CONFIG_NAMES["train_dagger"], top_ppo)
    dagger_output = training_output_dir(CONFIG_NAMES["train_dagger"])
    dagger_before = checkpoint_snapshot(dagger_output, "dagger")
    run_main(python, CONFIG_NAMES["train_dagger"])
    dagger_view = create_evaluation_view(
        pipeline_dir,
        "dagger",
        changed_checkpoints(dagger_before, dagger_output, "dagger"),
    )
    update_load_model_path(CONFIG_NAMES["eval_dagger"], dagger_view)
    run_main(python, CONFIG_NAMES["eval_dagger"])
    dagger_checkpoint, dagger_row = select_best_checkpoint("dagger", dagger_view, metric)
    top_dagger = copy_top_model("dagger", dagger_checkpoint, dagger_row, metric)
    summary["stages"]["dagger"] = {
        "best_checkpoint": dagger_checkpoint.name,
        "top_model": yaml_path_value(top_dagger),
        metric: dagger_row[metric],
    }
    write_summary(pipeline_dir, summary)

    # GRPO starts from the selected DAgger student.
    update_load_model_path(CONFIG_NAMES["train_grpo"], top_dagger)
    grpo_output = training_output_dir(CONFIG_NAMES["train_grpo"])
    grpo_before = checkpoint_snapshot(grpo_output, "grpo")
    run_main(python, CONFIG_NAMES["train_grpo"])
    grpo_view = create_evaluation_view(
        pipeline_dir,
        "grpo",
        changed_checkpoints(grpo_before, grpo_output, "grpo"),
    )
    update_load_model_path(CONFIG_NAMES["eval_grpo"], grpo_view)
    run_main(python, CONFIG_NAMES["eval_grpo"])
    grpo_checkpoint, grpo_row = select_best_checkpoint("grpo", grpo_view, metric)
    top_grpo = copy_top_model("grpo", grpo_checkpoint, grpo_row, metric)
    summary["stages"]["grpo"] = {
        "best_checkpoint": grpo_checkpoint.name,
        "top_model": yaml_path_value(top_grpo),
        metric: grpo_row[metric],
    }
    write_summary(pipeline_dir, summary)

    print(f"\n[pipeline] Complete. Summary: {pipeline_dir / 'pipeline_summary.json'}", flush=True)
    return pipeline_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO, DAgger and GRPO training/evaluation in sequence."
    )
    parser.add_argument(
        "--metric",
        choices=("mean_reward", "success_rate"),
        default="mean_reward",
        help="Metric used to choose top_model after every evaluation (default: mean_reward).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch config_run/main.py (default: current interpreter).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all required configs without starting training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_configuration()
        if args.check:
            print("[pipeline] Configuration check passed.")
            return 0
        run_pipeline(args.python, args.metric)
        return 0
    except subprocess.CalledProcessError as error:
        print(
            f"[pipeline] Command failed with exit code {error.returncode}: "
            f"{' '.join(error.cmd)}",
            file=sys.stderr,
        )
        return error.returncode or 1
    except (OSError, PipelineError, yaml.YAMLError) as error:
        print(f"[pipeline] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
