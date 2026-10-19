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


if __name__ == '__main__':
    unittest.main()
