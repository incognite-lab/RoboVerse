"""Stage events, curriculum resets and interval throughput accounting."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from config_run.dagger_vp.metrics import DaggerMetrics, HardwareMonitor


class _Clock:
    value = 0.0
    def __call__(self):
        return self.value


class _Writer:
    def __init__(self):
        self.values = {}
    def add_scalar(self, tag, value, step):
        self.values[tag] = value


def _buffer():
    return SimpleNamespace(size=8, max_samples=10, imgs=torch.zeros(10, 3, 2, 2, dtype=torch.uint8),
                           joints=torch.zeros(10, 2), expert_actions=torch.zeros(10, 3))


def _info(before, after, completed=-1, success=False, timeout=False):
    return dict(stage_before=before, stage_after=after, completed_stage=completed,
                is_success=success, **{'TimeLimit.truncated': timeout})


class DaggerMetricsTest(unittest.TestCase):
    def test_snapshot_entry_does_not_imply_completion_and_reset_closes_attempt(self):
        clock = _Clock()
        metrics = DaggerMetrics([0, 3], 6, .05, clock)
        clock.value = 1
        metrics.observe([False, False], [_info(0, 0), _info(3, 4, completed=3)], 2, True)
        clock.value = 2
        metrics.observe([False, True], [_info(0, 1, completed=0), _info(4, 3)], 2, False,
                        {'fall': np.array([False, True])})
        writer = _Writer()
        metrics.write(writer, 2, _buffer(), .5, .0003, end_step=10)
        v = writer.values
        self.assertEqual(v['Stages/stage_0/completed_total'], 1)
        self.assertEqual(v['Stages/stage_3/completed_total'], 1)
        self.assertEqual(v['Stages/stage_1/completed_total'], 0)
        self.assertEqual(v['Stages/stage_2/completed_total'], 0)
        self.assertEqual(v['Stages/stage_3/entered_total'], 2)
        self.assertEqual(v['Stages/stage_4/failed_total'], 1)
        self.assertEqual(v['Stages/stage_0/completion_steps_mean'], 2)
        self.assertAlmostEqual(v['Stages/stage_0/completion_sim_seconds_mean'], .1)
        self.assertEqual(v['Stages/stage_3/completion_wall_seconds_mean'], 1)
        self.assertEqual(v['Failures/fall_total'], 1)
        self.assertEqual(v['Speed/vector_steps_per_second'], 1)
        self.assertEqual(v['Speed/transitions_per_second'], 2)
        self.assertEqual(v['Speed/stored_samples_per_second'], 2)
        self.assertEqual(v['Student/expert_action_fraction'], .5)
        self.assertEqual(v['Episodes/length_steps_recent_mean'], 2)
        self.assertEqual(v['Time/estimated_remaining_seconds'], 8)

    def test_final_success_and_repeated_completions_count_events_not_unique_slots(self):
        clock = _Clock()
        metrics = DaggerMetrics([5], 6, .05, clock)
        for t in (1, 2):
            clock.value = t
            metrics.observe([True], [_info(5, 5, completed=5, success=True)], 1, False)
        writer = _Writer()
        metrics.write(writer, 2, _buffer(), 0, .0003)
        self.assertEqual(writer.values['Stages/stage_5/completed_total'], 2)
        self.assertEqual(writer.values['Stages/stage_5/completed_env_slots_total'], 1)
        self.assertEqual(writer.values['Stages/stage_5/failed_total'], 0)
        self.assertEqual(writer.values['Episodes/success_rate_recent'], 1)
        self.assertEqual(writer.values['Stages/stage_5/completion_steps_mean'], 1)

    def test_windows_reset_but_totals_and_update_sample_counts_persist(self):
        clock = _Clock()
        metrics = DaggerMetrics([0], 6, .05, clock)
        clock.value = 1
        with metrics.measure('student_updates'):
            clock.value = 1.25
        metrics.record_updates([1., .5], 8)
        metrics.record_predictions(np.array([[2., 0.]]), np.array([[1., 0.]]), -1, 1)
        metrics.observe([True], [_info(0, 0, timeout=True)], 1, True)
        writer = _Writer()
        metrics.write(writer, 1, _buffer(), 1, .0003)
        self.assertEqual(writer.values['Progress/trained_samples_this_session'], 16)
        self.assertEqual(writer.values['Student/rollout_action_mse'], .5)
        self.assertEqual(writer.values['Student/action_out_of_bounds_fraction'], .5)
        self.assertEqual(writer.values['Timing/student_updates_ms_per_vector_step'], 250)
        clock.value = 2.25
        metrics.observe([False], [_info(0, 0)], 1, False)
        writer = _Writer()
        metrics.write(writer, 2, _buffer(), 0, .0003)
        self.assertEqual(writer.values['Stages/stage_0/failed_recent'], 0)
        self.assertEqual(writer.values['Stages/stage_0/failed_total'], 1)
        self.assertEqual(writer.values['Speed/gradient_updates_per_second'], 0)
        self.assertEqual(writer.values['Progress/trained_samples_this_session'], 16)
        self.assertNotIn('Student/window_mse_loss', writer.values)
        self.assertNotIn('Episodes/success_rate_recent', writer.values)

    def test_gpu_csv_ignores_unsupported_fields_and_multiple_devices(self):
        values = HardwareMonitor.parse_smi('45, 12, 1024, 48000, [N/A], 52\n')
        self.assertEqual(values['GPU/utilization_percent'], 45)
        self.assertEqual(values['GPU/device_memory_used_mib'], 1024)
        self.assertNotIn('GPU/power_watts', values)
        self.assertEqual(HardwareMonitor.parse_smi('0, 0\n0, 0\n'), {})


if __name__ == '__main__':
    unittest.main()
