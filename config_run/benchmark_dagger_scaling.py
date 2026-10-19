"""Sequentially benchmark all-env DAgger collection on the allocated GPU.

python config_run/benchmark_dagger_scaling.py --train-best

No parallel GPU processes are launched. Each candidate uses the same replay
capacity for benchmarking, sampling ratio and total production-transition
budget. The chosen config optimizes measured sample throughput; final student
quality still requires evaluation after changing the optimizer batch size.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_run.dagger_vp.scaling import scale_dagger_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='chairman_separate/train_dagger_cluster_l40')
    parser.add_argument('--envs', type=int, nargs='+', default=[50, 100, 200, 400])
    parser.add_argument('--steps', type=int, default=201)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--train-best', action='store_true')
    args = parser.parse_args()
    if args.log_every <= 0 or args.steps < 2 * args.log_every + 1 or min(args.envs) <= 0:
        parser.error('Use positive env/log counts and at least 2 * log-every + 1 steps')
    config_path = ROOT / 'config_run/configs' / f'{args.config}.yaml'
    config = yaml.safe_load(config_path.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'config_run/output/dagger_scaling' / stamp
    output.mkdir(parents=True, exist_ok=False)
    # Equal small replay capacity avoids allocating the production 23 GiB for
    # every short benchmark. It fits a full training batch for each candidate.
    replay_samples = max(20_000, 2 * max(scale_dagger_config(config, n)['dagger_batch_size'] for n in args.envs))
    results = []
    for num_envs in args.envs:
        case = output / f'env_{num_envs}'
        command = [sys.executable, str(ROOT / 'config_run/profile_dagger.py'),
                   '--config', args.config, '--scale-collection', '--num-envs', str(num_envs),
                   '--steps', str(args.steps), '--log-every', str(args.log_every),
                   '--buffer-steps', str(max(1, replay_samples // num_envs)), '--output', str(case)]
        print(f'Benchmarking {num_envs} envs; log: {output / (case.name + ".log")}', flush=True)
        with (output / f'{case.name}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        summary_file = case / 'summary.json'
        if process.returncode or not summary_file.exists():
            results.append(dict(num_envs=num_envs, status='failed', returncode=process.returncode))
        else:
            summary = json.loads(summary_file.read_text())
            results.append(dict(num_envs=num_envs, status='ok', **{
                key: summary[key] for key in ('steady_stored_samples_per_second', 'batch_size',
                                               'peak_torch_cuda_bytes', 'train_seconds_including_setup_and_save')
            }))
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)
    successful = [r for r in results if r['status'] == 'ok' and r['steady_stored_samples_per_second'] is not None]
    if not successful:
        raise SystemExit(f'No complete throughput measurements. Inspect logs under {output}')
    best = max(successful, key=lambda r: r['steady_stored_samples_per_second'])
    tuned = scale_dagger_config(config, best['num_envs'])
    tuned['model_save_path'] = f'./config_run/output/dagger_separate_l40_tuned_{stamp}'
    tuned['tensorboard_log'] = tuned['model_save_path'] + '/tensorboard'
    config_name = f'chairman_separate/train_dagger_cluster_l40_tuned_{stamp}'
    tuned_path = ROOT / 'config_run/configs' / f'{config_name}.yaml'
    tuned_path.write_text('# Selected by measured stored samples/s; evaluate student quality separately.\n'
                          + yaml.safe_dump(tuned, sort_keys=False))
    print(f'Best measured candidate: {best["num_envs"]} envs; {best["steady_stored_samples_per_second"]:.1f} stored samples/s')
    print(f'Config: {tuned_path}')
    print(f'Run: {sys.executable} config_run/main_separate.py {config_name}', flush=True)
    if args.train_best:
        subprocess.run([sys.executable, str(ROOT / 'config_run/main_separate.py'), config_name], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
