"""Candidate comparisons preserve replay/data budgets rather than vector steps."""
from copy import deepcopy
import unittest

from config_run.dagger_vp.scaling import scale_dagger_config
from config_run.dagger_vp.beta_schedule import beta_at_step


class DaggerScalingTest(unittest.TestCase):
    def test_preserves_data_memory_beta_and_sample_replay_ratio(self):
        base = dict(num_envs=50, dagger_store_per_step=50, dagger_buffer_steps=10000,
                    dagger_batch_size=1024, dagger_train_every=20, dagger_updates_per_train=5,
                    total_timesteps=1000000, model_save_freq=5000, beta_schedule='linear', beta_start=1., beta_end=0.)
        original = deepcopy(base)
        for n in [50, 100, 200, 400]:
            c = scale_dagger_config(base, n)
            self.assertEqual(c['dagger_store_per_step'], n)
            self.assertEqual(c['dagger_buffer_steps'] * n, 500000)
            self.assertEqual(c['total_timesteps'] * n, 50000000)
            self.assertEqual(c['model_save_freq'] * n, 250000)
            self.assertEqual(c['dagger_updates_per_train'] * c['dagger_batch_size'] / (c['dagger_train_every'] * n), 5.12)
            for fraction in [0, .2, .5, 1]:
                self.assertAlmostEqual(beta_at_step(c, c['total_timesteps'] * fraction), 1 - fraction)
        self.assertEqual(base, original)

    def test_rejects_discarded_samples_and_invalid_environment_counts(self):
        base = dict(num_envs=100, dagger_store_per_step=50)
        with self.assertRaises(ValueError):
            scale_dagger_config(base, 200)
        base = dict(num_envs=0, dagger_store_per_step=0)
        with self.assertRaises(ValueError):
            scale_dagger_config(base, 200)


class DaggerScalingRunnerTest(unittest.TestCase):
    def test_selects_fastest_samples_not_largest_env_and_launches_selected_config(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import yaml
        import config_run.benchmark_dagger_scaling as runner

        base = dict(num_envs=200, dagger_store_per_step=200, dagger_buffer_steps=2500,
                    dagger_batch_size=4096, dagger_train_every=20, dagger_updates_per_train=5,
                    total_timesteps=250000, model_save_freq=1250, beta_schedule='linear', beta_start=1., beta_end=0.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / 'config_run/configs/chairman_separate'
            configs.mkdir(parents=True)
            (configs / 'train_dagger_cluster_l40.yaml').write_text(yaml.safe_dump(base))
            (root / 'config_run/profile_dagger.py').write_text('''
import json, pathlib, sys
a=sys.argv
n=int(a[a.index('--num-envs')+1]); p=pathlib.Path(a[a.index('--output')+1]); p.mkdir(parents=True)
p.joinpath('summary.json').write_text(json.dumps(dict(steady_stored_samples_per_second={50:100,100:200,200:150}[n], batch_size=round(20.48*n), peak_torch_cuda_bytes=0, train_seconds_including_setup_and_save=1)))
''')
            (root / 'config_run/main_separate.py').write_text('''
import pathlib,sys
pathlib.Path('training_started').write_text(sys.argv[1])
''')
            args = ['benchmark', '--envs', '50', '100', '200', '--train-best']
            with patch.object(runner, 'ROOT', root), patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            selected = (root / 'training_started').read_text()
            tuned = yaml.safe_load((root / 'config_run/configs' / f'{selected}.yaml').read_text())
            self.assertEqual(tuned['num_envs'], 100)
            self.assertEqual(tuned['dagger_store_per_step'], 100)
            self.assertEqual(tuned['dagger_batch_size'], 2048)
            self.assertEqual(tuned['total_timesteps'] * tuned['num_envs'], 50000000)
            results = json.loads(next((root / 'config_run/output/dagger_scaling').glob('*/results.json')).read_text())
            self.assertEqual(len(results), 3)


class SteadyRateTest(unittest.TestCase):
    def test_excludes_first_interval_and_short_tail(self):
        from types import SimpleNamespace
        from config_run.dagger_vp.scaling import steady_sample_rate
        events = [SimpleNamespace(step=s, value=v) for s, v in [(100, 1), (200, 500), (300, 600), (301, 10)]]
        self.assertEqual(steady_sample_rate(events, 100), 550)
        self.assertIsNone(steady_sample_rate(events[:1], 100))


class RecurrentScalingTest(unittest.TestCase):
    def base(self):
        return dict(train_or_eval='train_dagger_new', num_envs=80, dagger_store_per_step=8,
                    dagger_image_width=640, dagger_image_height=360, dagger_buffer_steps=512,
                    dagger_batch_size=8, dagger_new_batch_size=2, dagger_new_sequence_length=16,
                    dagger_new_burn_in=8, dagger_train_every=20, dagger_updates_per_train=10,
                    total_timesteps=1000000, model_save_freq=10000,
                    beta_schedule='linear', beta_start=1., beta_end=0.)

    def test_full_resolution_byte_budget_learning_ratio_and_beta_are_preserved(self):
        from config_run.dagger_vp.scaling import recurrent_candidate, replay_ratio, replay_sample_bytes
        base = self.base()
        original = deepcopy(base)
        for n in (16, 32, 64, 128, 256, 512):
            for batch in (2, 16, 64):
                c = recurrent_candidate(base, n, batch_size=batch, replay_gib=8)
                self.assertEqual(replay_ratio(c), replay_ratio(base))
                self.assertEqual(c['num_envs'], c['dagger_store_per_step'])
                self.assertGreaterEqual(c['dagger_buffer_steps'], 24)
                self.assertLessEqual(c['dagger_buffer_steps'] * n * replay_sample_bytes(c), 8 * 2**30)
                self.assertEqual(c['total_timesteps'] * n, 80000000)
                self.assertLessEqual(abs(c['model_save_freq'] * n - 800000), n / 2)
                self.assertEqual((c['dagger_image_width'], c['dagger_image_height']), (640, 360))
                self.assertAlmostEqual(beta_at_step(c, c['total_timesteps'] // 2), .5)
        self.assertEqual(base, original)
        for kwargs in (dict(num_envs=1000, replay_gib=.01), dict(num_envs=1, batch_size=0)):
            with self.assertRaises(ValueError):
                recurrent_candidate(base, **kwargs)

    def test_replay_estimate_matches_actual_sequence_buffer_tensor_storage(self):
        from config_run.dagger_recurrent.buffer import SequenceBuffer
        from config_run.dagger_vp.scaling import replay_sample_bytes
        buffer = SequenceBuffer(24, 2, (3, 360, 640), 43, (1, 256))
        self.assertEqual(buffer.storage_bytes, 24 * 2 * replay_sample_bytes(self.base()))

    def test_rejects_no_learning_wrong_dimensions_memory_pressure_and_override(self):
        from config_run.benchmark_dagger_scaling import measurement_error
        from config_run.dagger_vp.scaling import recurrent_candidate
        c = recurrent_candidate(self.base(), 64)
        good = dict(steady_stored_samples_per_second=100, steady_trained_samples_per_second=200,
                    student_update_calls=20, image_shape=[3,360,640], steady_mse_loss=.1,
                    sampled_peak_device_memory_bytes=30*2**30, gpu_total_bytes=48*2**30,
                    effective_config=c)
        self.assertIsNone(measurement_error(good, c, .92))
        for updates in (dict(steady_trained_samples_per_second=0), dict(student_update_calls=0),
                        dict(steady_trained_samples_per_second=100), dict(image_shape=[3,128,128]),
                        dict(steady_mse_loss=float('nan')), dict(sampled_peak_device_memory_bytes=47*2**30),
                        dict(sampled_peak_device_memory_bytes=None),
                        dict(effective_config=dict(c, dagger_buffer_steps=24))):
            self.assertIsNotNone(measurement_error(dict(good, **updates), c, .92))

    def test_staged_search_confirms_and_launches_exact_winning_settings(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import yaml
        import config_run.benchmark_dagger_scaling as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / 'config_run/configs/chairman_separate'
            configs.mkdir(parents=True)
            (configs / 'train_dagger_cluster_l40.yaml').write_text(yaml.safe_dump(self.base()))
            (root / 'config_run/profile_dagger.py').write_text('''
import json, pathlib, sys, yaml
a=sys.argv; c=yaml.safe_load(pathlib.Path(a[a.index('--config-file')+1]).read_text())
n=c['num_envs']
if n==8: sys.exit(9)
p=pathlib.Path(a[a.index('--output')+1]); p.mkdir(parents=True)
rate=(200 if n==4 else 100)+20*c['dagger_new_batch_size']
rate+=100*(c['dagger_new_visual_batch_size']==0)+50*(not c['dagger_new_checkpoint_visual'])+c['dagger_torch_threads']
ratio=c['dagger_updates_per_train']*c['dagger_new_batch_size']*16/(c['dagger_train_every']*n)
p.joinpath('summary.json').write_text(json.dumps(dict(steady_stored_samples_per_second=rate,
steady_trained_samples_per_second=rate*ratio if n!=5 else 0, student_update_calls=10,
image_shape=[3,360,640], steady_mse_loss=.1, gpu_total_bytes=48*2**30,
sampled_peak_device_memory_bytes=20*2**30, effective_config=c)))
''')
            (root / 'config_run/main_separate.py').write_text('''
import pathlib,sys
pathlib.Path('training_started').write_text(sys.argv[1])
''')
            args = ['benchmark', '--envs', '2', '4', '8', '--sequence-batches', '1', '2',
                    '--visual-batches', '1', '0', '--threads', '1', '2', '--top-k', '1',
                    '--replay-gib', '.25', '--confirm-runs', '2', '--train-best']
            with patch.object(runner, 'ROOT', root), patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            chosen = (root / 'training_started').read_text()
            c = yaml.safe_load((root / 'config_run/configs' / f'{chosen}.yaml').read_text())
            self.assertEqual(c['num_envs'], 4)
            self.assertEqual(c['dagger_new_batch_size'], 2)
            self.assertEqual(c['dagger_new_visual_batch_size'], 0)
            self.assertFalse(c['dagger_new_checkpoint_visual'])
            self.assertEqual(c['dagger_torch_threads'], 2)
            results = json.loads(next((root / 'config_run/output/dagger_scaling').glob('*/results.json')).read_text())
            self.assertEqual(sum(r['stage']=='confirm' for r in results), 2)
            self.assertTrue(any(r['status']=='failed' for r in results))
            self.assertTrue(any(r['status']=='rejected' for r in results))


if __name__ == '__main__':
    unittest.main()
