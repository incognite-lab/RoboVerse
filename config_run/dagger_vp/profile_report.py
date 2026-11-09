"""Render a synchronized DAgger profile summary as Markdown and CSV.

python -m config_run.dagger_vp.profile_report path/to/summary.json
"""
import csv
import json
from pathlib import Path
import sys

PHASE_NAMES = {
    'camera_and_inputs': 'Obraz a příprava vstupů',
    'inference_and_actions': 'Učitel, inference studenta a výběr akce',
    'buffer_store': 'Zápis do replaye',
    'env_step': 'Krok prostředí včetně fyziky a resetů',
    'student_updates': 'Učení studenta',
}


def write_report(summary_path):
    summary_path = Path(summary_path)
    data = json.loads(summary_path.read_text())
    phases = data['steady_phase_timings_nonoverlapping']
    steps = data['steady_measured_vector_steps']
    if steps <= 0 or not phases:
        raise ValueError('Profile has no complete steady-state timing window')
    total = sum(row['seconds'] for row in phases.values())
    lines = [f'# Měření DAggeru: {data["gpu"]}', '',
             f'{data["num_envs"]} prostředí; RGB {data["image_shape"][2]}×{data["image_shape"][1]}; '
             f'ukládáno {data["store_per_step"]} envů/krok. '
             f'Tabulka: posledních {steps} vektorových kroků po úvodním intervalu.', '',
             '| Část | Celkem [s] | Na vektorový krok [ms] | Podíl měřených částí |',
             '|---|---:|---:|---:|']
    csv_rows = []
    for name, row in sorted(phases.items(), key=lambda item: item[1]['seconds'], reverse=True):
        ms = row['seconds'] * 1000 / steps
        share = row['seconds'] / total * 100
        lines.append(f'| {PHASE_NAMES.get(name, name)} | {row["seconds"]:.3f} | {ms:.3f} | {share:.2f} % |')
        csv_rows.append(dict(group='nonoverlapping_phase', name=name, seconds=row['seconds'],
                             calls=row['calls'], ms_per_vector_step=ms, percent_of_phase_sum=share))
    lines += ['', f'Součet měřených částí: {total:.3f} s, {total * 1000 / steps:.3f} ms/krok.',
              'Části se nepřekrývají. Podíly nezahrnují start scény, načtení modelů, '
              'logování, ukládání checkpointu a ostatní režii mezi bloky.', '',
              '## Podrobnosti uvnitř částí', '',
              '| Operace | Celkem [s] | Počet volání | Průměr na volání [ms] |',
              '|---|---:|---:|---:|']
    for name, row in sorted(data['steady_timers_inclusive_do_not_sum'].items(),
                            key=lambda item: item[1]['seconds'], reverse=True):
        lines.append(f'| {name} | {row["seconds"]:.3f} | {row["calls"]} | {1000*row["seconds"]/row["calls"]:.3f} |')
        csv_rows.append(dict(group='inclusive_suboperation', name=name, seconds=row['seconds'],
                             calls=row['calls'], ms_per_vector_step=row['seconds'] * 1000 / steps,
                             percent_of_phase_sum=''))
    lines += ['', '**Podrobné operace jsou vnořené; nesčítat je mezi sebou ani s hlavní tabulkou.**',
              'Samostatné CUDA hranice mají teacher_predict, student_rollout_forward, '
              'student_training_forward, replay_sample, backward a optimizer_step. Ostatní '
              'podrobné řádky měří dobu volání na hostiteli a nemusí zachytit dokončení '
              'asynchronní GPU práce. Hlavní tabulka má synchronizované celé fáze.', '',
              f'- Sestavení prostředí: {data["build_seconds"]:.3f} s.',
              f'- Trénovací volání včetně načtení učitele a finálního uložení: {data["train_seconds_including_setup_and_save"]:.3f} s.',
              f'- Ukládané vzorky/s po zahřátí: {data["steady_stored_samples_per_second"]:.3f}.',
              f'- Učené snímky/s po zahřátí: {data["steady_trained_samples_per_second"]:.3f}.',
              f'- Průměrné vytížení GPU během celého trénovacího volání: {data["mean_gpu_utilization_percent"]:.1f} %.',
              f'- Vzorkovaná špička celkové GPU paměti: {data["sampled_peak_device_memory_bytes"]/2**30:.3f} GiB.',
              f'- Synchronizované hranice měření CUDA: {data["timing_cuda_synchronized"]}.', '',
              'Synchronizace zlepšuje přiřazení času, ale může snížit běžné překrývání CPU/GPU práce. '
              'Měřen je expertův rollout se skutečným učením studenta; nejde o test jeho úspěšnosti.']
    report = summary_path.with_name('timing_report.md')
    report.write_text('\n'.join(lines) + '\n')
    with summary_path.with_name('timing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return report


if __name__ == '__main__':
    print(write_report(sys.argv[1]))
