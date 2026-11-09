"""Partial-teacher controller and independent stage probe regression tests."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from .stage_beta import StageSawtoothBeta, stage_measurement
from .evaluation import evaluate_policy, evaluation_resets
from .test_pipeline import FakeEnvironment, TinyStudent
from . import runner
from .inputs import StudentInputs
from . import stage_probe


def _fake_probe_server(connection, config, metadata):
    """Real spawned process; replace only physics/rendering in the IPC test."""
    import io
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            if request['start_stage'] == 1:
                connection.send(dict(missing_snapshot='no snapshot is available'))
            elif request['start_stage'] == 2:
                connection.send(dict(error='simulator failed'))
            else:
                state = (torch.load(io.BytesIO(request['weights']),weights_only=True)
                         if request['weights'] is not None else {})
                connection.send(dict(row=dict(proprio_sum=float(state['proprio_mean'].sum()) if state else None)))
    finally:
        connection.close()


def measurements(rate=1., **changes):
    return {s: dict(success_rate=rate, attempts=32, censored=0, capped_attempts=0, **changes)
            for s in range(6)}


class StageBetaTest(unittest.TestCase):
    def controller(self, **config):
        c = StageSawtoothBeta(config)
        c.set_reference(measurements(), 5000)
        return c

    def test_warmup_and_exact_nominal_decay(self):
        c = StageSawtoothBeta({})
        self.assertEqual(c.beta_at_step(4999), 1)
        with self.assertRaises(ValueError):
            c.set_reference(measurements(), 4999)
        c.set_reference(measurements(), 5000)
        self.assertAlmostEqual(c.beta_at_step(5500), .5)
        self.assertEqual(c.beta_at_step(6000), 0)
        self.assertEqual(c.beta_at_step(7000), 0)
        with self.assertRaises(ValueError):
            c.set_reference(measurements(), 7000)

    def test_single_correction_pause_and_fresh_measurement(self):
        c = self.controller()
        self.assertEqual(c.observe(measurements(.7), 5500), list(range(5)))
        self.assertAlmostEqual(c.beta, .6)  # one correction, not five
        self.assertEqual(c.observe(measurements(.7), 5500), [])
        self.assertAlmostEqual(c.beta_at_step(6000), .6)
        self.assertAlmostEqual(c.beta_at_step(6100), .5)
        self.assertEqual(c.observe(measurements(.9), 6100), [])  # exactly threshold
        self.assertEqual(c.observe({5: measurements(0)[5]}, 6200), [])

    def test_unknown_small_and_capped_measurements_do_not_penalize(self):
        for key, value in [('attempts', 29), ('censored', 1), ('capped_attempts', 1),
                           ('success_rate', None)]:
            with self.subTest(key=key):
                c = self.controller()
                sample = measurements(.0)
                for row in sample.values():
                    row[key] = value
                self.assertEqual(c.observe(sample, 5500), [])
        c = self.controller()
        self.assertEqual(c.observe({}, 5500), [])

    def test_upper_bound_cannot_lock_warmup_and_reference_is_copied(self):
        c = StageSawtoothBeta({})
        reference = measurements()
        c.set_reference(reference, 5000)
        reference[0]['success_rate'] = 0
        self.assertEqual(c.reference[0]['success_rate'], 1)
        c.observe(measurements(0), 5000)
        self.assertEqual(c.pause_until, 0)
        self.assertAlmostEqual(c.beta_at_step(5100), .9)

    def test_resume_exact_curve_in_warmup_decay_and_pause(self):
        for step in (100, 5200, 5600):
            c = StageSawtoothBeta({})
            if step >= 5000:
                c.set_reference(measurements(), 5000)
            if step == 5600:
                c.observe(measurements(.5), 5500)
            c.beta_at_step(step)
            resumed = StageSawtoothBeta({}, saved=c.state_dict(), start_step=step)
            for at in (step, step+200, step+800):
                self.assertAlmostEqual(c.beta_at_step(at), resumed.beta_at_step(at))
        with self.assertRaisesRegex(ValueError, 'steps differ'):
            StageSawtoothBeta({}, saved=c.state_dict(), start_step=0)

    def test_invalid_settings(self):
        for config in ({'dagger_beta_decay_steps': 0}, {'dagger_beta_stage_tolerance': float('nan')},
                       {'dagger_target_stage': 6}, {'dagger_beta_min_stage_attempts': 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                StageSawtoothBeta(config)

    def test_evaluation_context_restores_partial_goal_even_on_error(self):
        env = FakeEnvironment()
        env.env = env.base
        env.handler.task = SimpleNamespace(dagger_target_stage=4, curriculum_max_stage=4)
        with self.assertRaises(RuntimeError):
            with evaluation_resets(env, start_stage=2, stop_stage=2):
                self.assertEqual(env.handler.task.dagger_target_stage, 2)
                raise RuntimeError('test')
        self.assertEqual(env.handler.task.dagger_target_stage, 4)
        self.assertEqual(env.handler.task.curriculum_max_stage, 4)
        self.assertFalse(hasattr(env.handler.task, 'eval_start_stage'))

    def test_stage4_finishes_and_original_full_goal_remains_stage5(self):
        from metasim.test.test_chairman_single_adapted_rewards import make_env
        from metasim.cfg.checkers.checkers import _ChairManChecker
        for target in (None, 4):
            _, handler, _ = make_env()
            handler.task.dagger_target_stage = target
            with ExitStack() as stack:
                for stage in range(6):
                    stack.enter_context(patch(
                        f'metasim.cfg.checkers.stages_chairman.stege{stage}_chacker',
                        side_effect=lambda states, handler, mask: (torch.zeros_like(mask), mask.clone())))
                _ChairManChecker().check(handler)
            self.assertEqual(handler.task.just_finished.nonzero().flatten().tolist(),
                             [5 if target is None else 4])

    def test_curriculum_never_selects_unused_stage5(self):
        from metasim.cfg.checkers.stages_chairman import _curriculum_limit
        handler = SimpleNamespace(task=SimpleNamespace(dagger_target_stage=4, curriculum_max_stage=None))
        self.assertEqual(_curriculum_limit(handler, 5), 4)
        handler.task.curriculum_max_stage = 2
        self.assertEqual(_curriculum_limit(handler, 5), 2)

    def test_cluster_configs_use_identical_controller_settings(self):
        import yaml
        directory = Path(__file__).resolve().parents[1]/'configs/chairman_separate'
        names = ('train_dagger', 'train_dagger_l40_400', 'train_dagger_cluster_l40',
                 'train_dagger_rtx3060_profiled')
        configs = [yaml.safe_load((directory/(name+'.yaml')).read_text()) for name in names]
        keys = ('beta_schedule', 'beta_start', 'beta_end', 'dagger_target_stage',
                'dagger_teacher_warmup_steps', 'dagger_eval_every', 'dagger_eval_episodes',
                'dagger_eval_max_steps', 'dagger_stage_eval_num_envs', 'dagger_prefix_eval_every')
        for key in keys + tuple(k for k in configs[0] if k.startswith('dagger_beta_')):
            self.assertTrue(all(c[key] == configs[0][key] for c in configs), key)
        self.assertEqual(configs[0]['beta_schedule'], 'stage_sawtooth')


class StageProbeEnvironment(FakeEnvironment):
    def __init__(self):
        super().__init__()
        self.env = self.base
        self.handler.task = SimpleNamespace(dagger_target_stage=4, curriculum_max_stage=4)
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1
        return super().reset()

    def get_current_stages(self):
        return np.full(2, getattr(self.handler.task, 'eval_start_stage', 0), dtype=np.int64)

    def step(self, actions):
        self.steps += 1
        stage = int(self.get_current_stages()[0])
        success = bool(np.allclose(actions, .2))
        infos = [dict(stage_before=stage, stage_after=stage, completed_stage=stage if success else -1,
                      is_success=success) for _ in range(2)]
        return np.zeros((2,43), np.float32), np.zeros(2), np.ones(2, dtype=bool), infos

    def close(self):
        self.closed = True


class StageRunnerTest(unittest.TestCase):
    def test_spawned_process_receives_current_weights_reports_errors_and_closes(self):
        student = TinyStudent(hidden_size=8)
        with patch.object(stage_probe, '_serve', _fake_probe_server):
            worker = stage_probe.StageProbeProcess({}, {})
        try:
            arguments = dict(start_stage=0,stop_stage=0,episodes=2,max_steps=2,expert=False)
            self.assertEqual(worker.evaluate(student,**arguments)['proprio_sum'],0)
            student.proprio_mean.fill_(1)
            self.assertEqual(worker.evaluate(student,**arguments)['proprio_sum'],student.proprio_dim)
            self.assertIsNone(worker.evaluate(student,**dict(arguments,expert=True))['proprio_sum'])
            with self.assertRaisesRegex(RuntimeError,'no snapshot is available'):
                worker.evaluate(student,**dict(arguments,start_stage=1))
            with self.assertRaisesRegex(RuntimeError,'simulator failed'):
                worker.evaluate(student,**dict(arguments,start_stage=2))
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_reference_probes_do_not_reset_training_and_resume_keeps_reference(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        env, probe = StageProbeEnvironment(), StageProbeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2,34), .2, np.float32))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(runner, '_device', return_value='cpu'))
            stack.enter_context(patch.object(runner, 'load_policy_router', return_value=(teacher, {})))
            stack.enter_context(patch.object(runner, 'RecurrentVisionStudent', TinyStudent))
            stack.enter_context(patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent', TinyStudent))
            class InlineProbe:
                # Exercise the same protocol without starting a real simulator.
                def __init__(self, config, metadata):
                    self.adapter = StudentInputs(probe.base,probe,'cpu',metadata['camera_name'],
                                                 metadata['joint_names'],image_size=metadata['image_shape'][-2:])
                def evaluate(self, student, *, start_stage, stop_stage, episodes, max_steps, expert):
                    with evaluation_resets(probe,start_stage,stop_stage):
                        return evaluate_policy(student,self.adapter,probe,episodes,max_steps,
                                               teacher=teacher if expert else None)
                def close(self):
                    probe.close()
            factory = stack.enter_context(patch.object(runner, 'StageProbeProcess', side_effect=InlineProbe))
            config = dict(train_or_eval='train_dagger_new', expert_model_path='fake',
                model_save_path=directory, tensorboard_log=directory+'/logs', total_timesteps=6,
                dagger_image_width=32, dagger_image_height=32, dagger_new_hidden_size=8,
                dagger_buffer_steps=8, dagger_store_per_step=2, dagger_train_every=2,
                dagger_updates_per_train=1, dagger_log_every=2, dagger_new_sequence_length=2,
                dagger_new_burn_in=0, dagger_new_batch_size=1, beta_schedule='stage_sawtooth',
                dagger_target_stage=4, dagger_teacher_warmup_steps=2, dagger_beta_decay_steps=10,
                dagger_beta_recovery_pause_steps=2, dagger_eval_every=2, dagger_eval_episodes=2,
                dagger_eval_max_steps=2, dagger_beta_min_stage_attempts=2, dagger_prefix_eval_every=4)
            runner.run_dagger(config, env.base, env)
            self.assertEqual(env.reset_count, 1)
            self.assertEqual(env.steps, 6)
            factory.assert_called_once()
            self.assertTrue(probe.closed)
            path = Path(directory)/'recurrent/student_model_final.pth'
            saved = torch.load(path, weights_only=True)['metadata']['beta_controller']
            self.assertEqual(set(saved['reference']), set(range(5)))
            self.assertTrue(all(r['success_rate'] == 1 for r in saved['reference'].values()))
            self.assertLessEqual(saved['beta'], 1)
            self.assertGreater(saved['beta'], .6)  # corrections slowed nominal decay
            self.assertTrue((path.parent/'teacher_stage_reference.json').is_file())
            factory.reset_mock()
            config.update(train_or_eval='load_and_train_dagger_new', load_student_path=str(path), total_timesteps=8)
            runner.run_dagger(config, env.base, env)
            restored = torch.load(path, weights_only=True)['metadata']['beta_controller']
            self.assertEqual(restored['reference'], saved['reference'])
            self.assertEqual(restored['warmup_end'], 2)
            self.assertEqual(env.reset_count, 2)
            class FailingProbe(InlineProbe):
                def evaluate(self, *args, **kwargs):
                    raise RuntimeError('evaluation interrupted after training step')
            factory.side_effect = FailingProbe
            config['total_timesteps'] = 10
            with self.assertRaisesRegex(RuntimeError,'evaluation interrupted'):
                runner.run_dagger(config, env.base, env)
            recovery = path.parent/'student_model_interrupted_step_10.pth'
            state = torch.load(recovery,weights_only=True)['metadata']['beta_controller']
            self.assertEqual(state['last_step'],10)
            StageSawtoothBeta(config,saved=state,start_step=10)

    def test_capped_attempts_are_unknown_not_zero_success(self):
        env = FakeEnvironment()
        student = TinyStudent(hidden_size=8)
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.zeros((2,34), np.float32))
        row = evaluate_policy(student, None, env, 2, 1, teacher=teacher)
        measurement = stage_measurement(row, 0)
        self.assertEqual(measurement['attempts'], 0)
        self.assertIsNone(measurement['success_rate'])
        self.assertEqual(measurement['censored'], 2)
        self.assertFalse(StageSawtoothBeta({}).reliable(measurement))


if __name__ == '__main__':
    unittest.main()
