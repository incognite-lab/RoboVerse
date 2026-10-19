"""Linear mixing must use absolute steps, including legacy checkpoint resume."""
from pathlib import Path
import tempfile
import unittest

import torch

from config_run.dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
from config_run.dagger_vp.checkpoint import load_dagger_checkpoint, save_dagger_checkpoint


class DaggerBetaScheduleTest(unittest.TestCase):
    def test_linear_million_step_endpoints_and_intermediate_values(self):
        config = dict(total_timesteps=1_000_000, beta_start=1., beta_end=0., beta_decay=.1)
        # Old decay is ignored by the new default linear schedule.
        for step, expected in [(0, 1.), (200000, .8), (500000, .5), (900000, .1), (1000000, 0.), (1100000, 0.)]:
            self.assertAlmostEqual(beta_at_step(config, step), expected)
        self.assertEqual(dagger_target_steps(config, 200000), 1000000)
        with self.assertRaises(ValueError):
            dagger_target_steps(config, 1000000)

    def test_last_executed_step_and_final_checkpoint_beta(self):
        config = dict(total_timesteps=4)
        self.assertEqual([beta_at_step(config, i) for i in range(5)], [1., .75, .5, .25, 0.])

    def test_resuming_old_sidecar_recomputes_beta_and_restores_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'student_model_step_199999.pth')
            save_dagger_checkpoint(path, model, optimizer, 200000, .1)
            restored = torch.nn.Linear(2, 1)
            new_optimizer = torch.optim.Adam(restored.parameters(), lr=.9)
            step, beta = load_dagger_checkpoint(path, restored, new_optimizer, 'cpu', dict(total_timesteps=1000000))
            self.assertEqual(step, 200000)
            self.assertAlmostEqual(beta, .8)
            self.assertEqual(new_optimizer.param_groups[0]['lr'], .003)
            for left, right in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right)

    def test_legacy_plain_weights_use_same_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            path = str(Path(directory) / 'student_model_step_199999.pth')
            torch.save(model.state_dict(), path)
            step, beta = load_dagger_checkpoint(path, model, torch.optim.Adam(model.parameters()), 'cpu',
                                               dict(total_timesteps=1000000))
            self.assertEqual(step, 200000)
            self.assertAlmostEqual(beta, .8)

    def test_explicit_legacy_exponential_and_validation(self):
        self.assertAlmostEqual(beta_at_step(dict(total_timesteps=100, beta_schedule='exponential', beta_decay=.5), 2), .25)
        for config in [dict(total_timesteps=0), dict(beta_start=2), dict(beta_end=float('nan')),
                       dict(beta_start=.1, beta_end=.2), dict(beta_schedule='wrong')]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                beta_at_step(config, 0)


class DaggerBetaTrainingLoopTest(unittest.TestCase):
    def _run(self, root, resume_step=0):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from config_run import separate_dagger as dagger

        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 4)
                torch.nn.init.zeros_(self.head.weight)
                torch.nn.init.zeros_(self.head.bias)
            def forward(self, images, joints):
                return self.head(joints)

        class Env:
            num_envs = 2
            NUM_POLICY_STAGES = 6
            action_space = SimpleNamespace(shape=(4,), low=-np.ones(4), high=np.ones(4))
            def __init__(self):
                self.actions = []
            def reset(self):
                return np.zeros((2, 4), dtype=np.float32)
            def get_current_stages(self):
                return np.zeros(2, dtype=np.int64)
            def step(self, actions):
                self.actions.append(actions.copy())
                infos = [dict(stage_before=0, stage_after=0, completed_stage=-1) for _ in range(2)]
                return self.reset(), np.zeros(2), np.zeros(2, dtype=bool), infos

        config = dict(total_timesteps=4, expert_model_path='fake', model_save_path=str(root / 'output'),
                      tensorboard_log=str(root / 'tb'), dagger_buffer_steps=4, dagger_store_per_step=2,
                      dagger_buffer_device='cpu', dagger_train_every=100, dagger_log_every=2,
                      model_save_freq=0, beta_schedule='linear', beta_start=1., beta_end=0.)
        if resume_step:
            checkpoint = str(root / 'old.pth')
            student = Student()
            save_dagger_checkpoint(checkpoint, student, torch.optim.Adam(student.parameters()), resume_step, .01)
            config.update(train_or_eval='load_and_train_dagger', load_student_path=checkpoint)
        env = Env()
        handler = SimpleNamespace(scene_inst=SimpleNamespace(dt=.01), scenario=SimpleNamespace(decimation=5),
                                  task=SimpleNamespace(failure_masks={}))
        base = SimpleNamespace(env=SimpleNamespace(handler=handler))
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2, 4), .8, dtype=np.float32))
        inputs = (torch.zeros(2, 3, 2, 2, dtype=torch.uint8), torch.zeros(2, 3, 2, 2), torch.zeros(2, 1))
        with patch.object(dagger, '_device', return_value='cpu'), \
             patch.object(dagger, 'load_policy_router', return_value=(teacher, {})), \
             patch.object(dagger, '_new_student', side_effect=lambda *args: (Student(), 4, 1)), \
             patch.object(dagger, '_student_inputs', return_value=inputs), \
             patch.object(dagger.random, 'random', return_value=.55):
            dagger.train_dagger(config, base, env)
        final = torch.load(root / 'output/student_model_final.pth.training.pth', weights_only=True)
        self.assertEqual(final['next_step'], 4)
        self.assertEqual(final['beta'], 0.)
        return env.actions

    def test_new_run_uses_linear_probabilities_and_stops_at_target(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            actions = self._run(Path(directory))
        self.assertEqual(len(actions), 4)
        np.testing.assert_allclose(actions[:2], .8)
        np.testing.assert_allclose(actions[2:], 0.)

    def test_resume_runs_only_remaining_steps_and_ignores_old_exponential_beta(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            actions = self._run(Path(directory), resume_step=2)
        self.assertEqual(len(actions), 2)
        np.testing.assert_allclose(actions, 0.)
