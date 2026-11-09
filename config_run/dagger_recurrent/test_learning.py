"""Regressions for the measured collapse, lost demonstrations and evaluation bias."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .buffer import SequenceBuffer
from .evaluation import EvaluationBeta, evaluate_policy
from .student_net import RecurrentVisionStudent


class LearningRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_encoder_learns_visual_and_proprio_tasks_without_saturation(self):
        torch.manual_seed(4)
        model = RecurrentVisionStudent(hidden_size=32)
        model.collect_diagnostics = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        # Independent factors: color requires vision, angle requires proprio.
        images = torch.zeros(8, 3, 32, 32)
        images[::2, 0] = 1
        images[1::2, 2] = 1
        q = torch.zeros(8, 43)
        q[:, 0] = torch.tensor([-1., -1, 1, 1, -1, -1, 1, 1])
        values = (images, q, torch.zeros(8, 43), torch.zeros(8, 6), torch.zeros(8, 34))
        targets = torch.zeros(8, 34)
        targets[:, 0] = torch.tensor([1., -1] * 4) * .4
        targets[:, 1] = q[:, 0] * .4
        losses = []
        before = {k: p.detach().clone() for k, p in model.named_parameters()
                  if k in ('visual_encoder.conv1.weight', 'proprio_encoder.0.weight', 'gru.weight_ih_l0')}
        for _ in range(60):
            prediction, _ = model(*values)
            loss = (prediction-targets).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(np.mean(losses[-5:]), losses[0] * .1)
        self.assertLess(float(model.last_diagnostics['gru_input_saturation_fraction']), .1)
        for key, old in before.items():
            self.assertGreater(float((dict(model.named_parameters())[key].detach()-old).abs().max()), 1e-5)
        with torch.no_grad():
            predictions, _ = model(*values)
        self.assertGreater(float((predictions[0, 0]-predictions[1, 0]).abs()), .4)
        self.assertGreater(float((predictions[0, 1]-predictions[2, 1]).abs()), .4)

    def test_teacher_windows_survive_online_overwrite_and_archive_restore(self):
        torch.manual_seed(6)
        buffer = SequenceBuffer(4, 2, (3, 1, 1), 43, (1, 3),
                                demonstrations_per_stage=2, demonstration_length=2, demonstration_fraction=1.)
        def add(marker, stage, teacher=True):
            x = torch.full((2, 1), float(marker))
            values = (torch.full((2, 3, 1, 1), marker, dtype=torch.uint8),
                      x.expand(-1, 43), x.expand(-1, 43), x.expand(-1, 6), x.expand(-1, 34))
            buffer.add(values, x.expand(-1, 34), x.view(1, 2, 1).expand(-1, -1, 3), torch.tensor([False, True]))
            buffer.record_demonstrations(torch.full((2,), stage), torch.full((2,), teacher))
        for stage in range(6):
            add(stage*10, stage)
            add(stage*10+1, stage)
        for i in range(20):
            add(200+i, 0, teacher=False)
        restored = SequenceBuffer(4, 2, (3, 1, 1), 43, (1, 3),
                                 demonstrations_per_stage=2, demonstration_length=2, demonstration_fraction=1.)
        restored.load_archive(buffer.archive_state())
        self.assertEqual(sorted(restored.demonstrations), list(range(6)))
        self.assertEqual(buffer.demonstrations_seen[0], 1)
        # Clear invalidates only live trajectories, retaining demonstrators.
        buffer.clear()
        add(240, 0, teacher=False)
        add(241, 0, teacher=False)
        values, targets, hidden, resets = buffer.sample(512, 2)
        markers = values[1][..., 0]
        self.assertEqual(sorted(set((markers[:, -1] // 10).tolist())), list(range(6)))
        torch.testing.assert_close(markers[:, 1]-markers[:, 0], torch.ones(512))
        torch.testing.assert_close(targets[..., 0], markers)
        torch.testing.assert_close(hidden[0, :, 0], markers[:, 0])
        self.assertTrue(((resets[:, 0] == resets[:, 1])).all())

    def test_beta_waits_for_warmup_and_consecutive_independent_success(self):
        config = dict(dagger_teacher_warmup_steps=100, dagger_eval_required_passes=2,
                      dagger_eval_success_threshold=.8, dagger_beta_after_warmup=1.)
        gate = EvaluationBeta(config)
        passed = dict(episodes_finished=20, requested_episodes=20, success_rate=.9)
        failed = dict(passed, success_rate=0.)
        self.assertEqual(gate.observe(passed, 99), 1.)
        self.assertEqual(gate.observe(passed, 100), 1.)
        self.assertEqual(gate.observe(failed, 101), 1.)
        self.assertEqual(gate.observe(passed, 102), 1.)
        self.assertAlmostEqual(gate.observe(passed, 103), .9)
        resumed = EvaluationBeta(config, gate.state_dict(), start_step=500)
        self.assertEqual(resumed.observe(passed, 501), .9)
        self.assertEqual(resumed.observe(passed, 599), .9)
        self.assertEqual(gate.observe(failed, 104), .9)
        self.assertEqual(gate.observe(failed, 105), 1.)

    def test_archive_keeps_brief_stages_between_ticks_and_ignores_burn_in_only_stages(self):
        for teacher in (True, False):
            with self.subTest(teacher=teacher):
                buffer = SequenceBuffer(4, 1, (3,1,1), 43, (1,3), demonstrations_per_stage=1,
                                        demonstration_length=4, demonstration_burn_in=2)
                for step, stage in enumerate((0, 0, 4, 5)):
                    x = torch.full((1,1), float(step))
                    values = (torch.full((1,3,1,1), step, dtype=torch.uint8),
                              x.expand(-1,43), x.expand(-1,43), x.expand(-1,6), x.expand(-1,34))
                    buffer.add(values, x.expand(-1,34), torch.zeros(1,1,3), torch.tensor([False]))
                    buffer.record_demonstrations(torch.tensor([stage]), torch.tensor([teacher or step != 2]))
                self.assertEqual(sorted(buffer.demonstrations), [4,5] if teacher else [])

    def test_evaluation_waits_for_slow_success_instead_of_counting_fast_failures_twice(self):
        class Model:
            num_actions = 34
            def eval(self):
                return self
            def initial_hidden(self, n):
                return torch.zeros(1, n, 1)
            def __call__(self, *args, **kwargs):
                return torch.zeros(2, 34), kwargs['hidden']
        class Environment:
            num_envs = 2
            action_space = SimpleNamespace(low=-np.ones(34), high=np.ones(34))
            def reset(self):
                self.steps = 0
                return np.zeros((2, 1))
            def get_current_stages(self):
                return np.zeros(2, dtype=np.int64)
            def step(self, actions):
                self.steps += 1
                return (np.zeros((2, 1)), np.zeros(2), np.array([True, self.steps == 6]),
                        [dict(is_success=False, stage_before=0), dict(is_success=True, stage_before=5)])
        adapter = SimpleNamespace(read=lambda previous: (torch.zeros(2, 3, 32, 32, dtype=torch.uint8),))
        env = Environment()
        result = evaluate_policy(Model(), adapter, env, 2, 8)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(env.steps, 6)
        result = evaluate_policy(Model(), adapter, env, 2, 3)
        self.assertEqual(result['capped_attempts'], 1)
        self.assertEqual(result['episodes_finished'], 2)
        self.assertEqual(result['success_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
