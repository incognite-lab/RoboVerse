"""Benchmark actual DAgger training on one allocated GPU, then optionally train.

Sequential isolated processes; fixed resolution, learning/data ratio and replay
byte budget. Search environments, optimizer batches, encoder chunks,
checkpointing and CPU threads. --train-best is opt-in.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_run.dagger_vp.scaling import (
    recurrent_config, recurrent_candidate, replay_ratio, scale_dagger_config,
)


def candidate_key(config):
    return json.dumps(config, sort_keys=True)


def measurement_error(summary, config, max_memory_fraction):
    rate = summary.get('steady_stored_samples_per_second')
    if rate is None or not math.isfinite(rate) or rate <= 0:
        return 'missing/nonfinite steady throughput'
    if not recurrent_config(config):
        return None
    trained = summary.get('steady_trained_samples_per_second')
    if trained is None or not math.isfinite(trained) or trained <= 0 or not summary.get('student_update_calls', 0):
        return 'no measured student learning'
    if not math.isclose(trained / rate, float(replay_ratio(config)), rel_tol=.03):
        return 'measured learning/data ratio differs from the candidate'
    expected_shape = [3, config.get('dagger_image_height', 128), config.get('dagger_image_width', 128)]
    if summary.get('image_shape') != expected_shape:
        return 'profile used different image dimensions'
    loss = summary.get('steady_mse_loss')
    if loss is None or not math.isfinite(loss):
        return 'missing/nonfinite training loss'
    used, total = summary.get('sampled_peak_device_memory_bytes'), summary.get('gpu_total_bytes')
    if used is None or not total:
        return 'missing total-device memory telemetry (nvidia-smi)'
    # Nyx/Genesis memory lies outside the torch allocator. Check both metrics.
    if max(used, summary.get('peak_torch_reserved_bytes', 0)) > max_memory_fraction * total:
        return 'insufficient GPU memory headroom'
    measured = summary.get('effective_config', {})
    for key in ('num_envs', 'dagger_store_per_step', 'dagger_buffer_steps', 'dagger_new_batch_size',
                'dagger_new_visual_batch_size', 'dagger_new_checkpoint_visual', 'dagger_new_buffer_device',
                'dagger_train_every', 'dagger_updates_per_train', 'dagger_torch_threads'):
        if measured.get(key) != config.get(key):
            return f'profile changed production setting {key}'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='chairman_separate/train_dagger_cluster_l40')
    parser.add_argument('--envs', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--sequence-batches', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--visual-batches', type=int, nargs='+', default=[4, 8, 16, 32, 64, 0],
                        help='0 encodes all frames together; keeps the full image resolution')
    parser.add_argument('--threads', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--replay-gib', type=float, default=8., help='FULL replay tensor budget in benchmark and production')
    parser.add_argument('--replay-device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=301, help='Minimum steps; aligned to complete learning/logging intervals')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--max-train-every', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=2, help='Candidates retained at each refinement stage')
    parser.add_argument('--confirm-runs', type=int, default=2)
    parser.add_argument('--max-memory-fraction', type=float, default=.92)
    parser.add_argument('--timeout', type=float, default=3600., help='Maximum seconds per candidate, including setup')
    parser.add_argument('--train-best', action='store_true')
    args = parser.parse_args()
    if (min(args.envs + args.sequence_batches + args.threads + [args.log_every, args.steps,
            args.top_k, args.confirm_runs, args.max_train_every]) <= 0 or min(args.visual_batches) < 0
            or not math.isfinite(args.replay_gib) or args.replay_gib <= 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or not 0 < args.max_memory_fraction < 1):
        parser.error('Counts/budgets must be positive; visual batch >= 0; memory fraction strictly between 0 and 1')
    config_path = ROOT / 'config_run/configs' / f'{args.config}.yaml'
    config = yaml.safe_load(config_path.read_text())
    recurrent = recurrent_config(config)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'config_run/output/dagger_scaling' / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / 'search.json').write_text(json.dumps(vars(args), indent=2))
    results, configurations, seen = [], {}, set()
    if recurrent:
        print(f'Recurrent student: {config.get("dagger_image_width")}x{config.get("dagger_image_height")}; '
              f'{float(replay_ratio(config)):.6g} learned frames/new stored frame; '
              f'full {args.replay_gib} GiB replay on {args.replay_device}', flush=True)

    def ranked(rows):
        return sorted((r for r in rows if r['status'] == 'ok'),
                      key=lambda r: r['steady_stored_samples_per_second'], reverse=True)

    def run_case(candidate, stage, repeat=False):
        key = candidate_key(candidate)
        if not repeat and key in seen:
            return
        seen.add(key)
        name = f'{len(results):03d}_{stage}_env{candidate["num_envs"]}'
        case = output / name
        yaml_path = output / f'{name}.yaml'
        yaml_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        configurations[name] = deepcopy(candidate)
        interval = math.lcm(args.log_every, int(candidate['dagger_train_every']))
        # First complete interval is discarded; the following two include
        # complete update cycles. No burn-in-only window is eligible.
        burn = int(candidate.get('dagger_new_burn_in', 8)) + int(candidate.get('dagger_new_sequence_length', 16)) if recurrent else 1
        interval *= max(1, math.ceil(burn / interval))
        steps = max(3, math.ceil((args.steps - 1) / interval)) * interval + 1
        command = [sys.executable, str(ROOT / 'config_run/profile_dagger.py'),
                   '--config-file', str(yaml_path), '--num-envs', str(candidate['num_envs']),
                   '--steps', str(steps), '--log-every', str(interval), '--output', str(case), '--quiet']
        row = dict(name=name, stage=stage, num_envs=candidate['num_envs'], status='failed',
                   config_file=str(yaml_path), log_file=str(output / f'{name}.log'))
        print(f'{name}: {steps} steps; log {row["log_file"]}', flush=True)
        try:
            with Path(row['log_file']).open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout)
            row['returncode'] = process.returncode
            summary_path = case / 'summary.json'
            if process.returncode or not summary_path.exists():
                row['reason'] = 'worker failed (inspect log for OOM/error)'
            else:
                summary = json.loads(summary_path.read_text())
                reason = measurement_error(summary, candidate, args.max_memory_fraction)
                row.update({key: summary.get(key) for key in (
                    'steady_stored_samples_per_second', 'steady_trained_samples_per_second',
                    'mean_gpu_utilization_percent', 'sampled_peak_device_memory_bytes', 'gpu_total_bytes',
                    'peak_torch_cuda_bytes', 'peak_torch_reserved_bytes', 'steady_mse_loss')})
                row.update(status='rejected' if reason else 'ok', reason=reason)
        except subprocess.TimeoutExpired:
            row['reason'] = f'timeout after {args.timeout}s'
        except (OSError, ValueError, KeyError) as error:
            row['reason'] = str(error)
        results.append(row)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)

    def make(n, batch=None, visual=None, checkpoint=True, threads=None):
        if recurrent:
            candidate = recurrent_candidate(config, n, replay_gib=args.replay_gib, batch_size=batch,
                visual_batch_size=visual if visual is not None else config.get('dagger_new_visual_batch_size', 4),
                checkpoint_visual=checkpoint, threads=threads or args.threads[0], replay_device=args.replay_device)
            if candidate['dagger_train_every'] > args.max_train_every:
                raise ValueError('exact learning ratio would require too long a collection interval')
            return candidate
        return scale_dagger_config(config, n)

    for n in sorted(set(args.envs)):
        try:
            run_case(make(n), 'env')
        except ValueError as error:
            print(f'Skipping env={n}: {error}', flush=True)
    seeds = ranked(results)[:args.top_k]
    if not seeds:
        raise SystemExit(f'No valid measured training candidates. Inspect {output}')
    if recurrent:
        # Refine env count around throughput leaders, rather than assuming
        # that the biggest allocation is necessarily fastest.
        for row in seeds:
            for n in sorted({max(1, row['num_envs'] * 3 // 4), row['num_envs'] * 5 // 4}):
                if n < min(args.envs) or n > max(args.envs):
                    continue
                try:
                    run_case(make(n), 'env_refine')
                except ValueError as error:
                    print(f'Skipping refinement env={n}: {error}', flush=True)
        seeds = ranked(results)[:args.top_k]
        for row in seeds:
            for batch in sorted(set(args.sequence_batches)):
                try:
                    run_case(make(row['num_envs'], batch=batch), 'batch')
                except ValueError as error:
                    print(f'Skipping batch={batch}: {error}', flush=True)
        seeds = ranked(results)[:args.top_k]
        for row in seeds:
            seed = configurations[row['name']]
            for visual in dict.fromkeys(args.visual_batches):
                for checkpoint in (False, True):
                    if visual == 0 and checkpoint:
                        continue  # Zero disables checkpointing in the model.
                    run_case(make(seed['num_envs'], batch=seed['dagger_new_batch_size'],
                                  visual=visual, checkpoint=checkpoint), 'encoder')
        seeds = ranked(results)[:args.top_k]
        for row in seeds:
            seed = configurations[row['name']]
            for threads in sorted(set(args.threads)):
                candidate = deepcopy(seed)
                candidate['dagger_torch_threads'] = threads
                run_case(candidate, 'threads')
        finalists = ranked(results)[:args.top_k]
        confirmed = []
        for row in finalists:
            candidate = configurations[row['name']]
            start = len(results)
            for _ in range(args.confirm_runs):
                run_case(candidate, 'confirm', repeat=True)
            repeats = results[start:]
            if len(repeats) == args.confirm_runs and all(r['status'] == 'ok' for r in repeats):
                confirmed.append(dict(row, steady_stored_samples_per_second=median(
                    r['steady_stored_samples_per_second'] for r in repeats)))
        if not confirmed:
            raise SystemExit(f'No stable confirmed candidate; no training launched. Inspect {output}')
        best = max(confirmed, key=lambda r: r['steady_stored_samples_per_second'])
    else:
        best = ranked(results)[0]
    tuned = deepcopy(configurations[best['name']])
    tuned['model_save_path'] = f'./config_run/output/dagger_separate_l40_tuned_{stamp}'
    tuned['tensorboard_log'] = tuned['model_save_path'] + '/tensorboard'
    config_name = f'chairman_separate/train_dagger_cluster_l40_tuned_{stamp}'
    tuned_path = ROOT / 'config_run/configs' / f'{config_name}.yaml'
    tuned_path.write_text('# Selected by measured training throughput; verify final policy quality.\n'
                          + yaml.safe_dump(tuned, sort_keys=False))
    (output / 'best.json').write_text(json.dumps(dict(best=best, config=str(tuned_path)), indent=2))
    print(f'Best confirmed: {best["num_envs"]} envs; {best["steady_stored_samples_per_second"]:.1f} stored samples/s')
    print(f'Config: {tuned_path}')
    print(f'Run: {sys.executable} config_run/main_separate.py {config_name}', flush=True)
    if args.train_best:
        subprocess.run([sys.executable, str(ROOT / 'config_run/main_separate.py'), config_name], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
