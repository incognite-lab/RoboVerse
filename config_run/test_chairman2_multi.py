"""Chairman2 task wiring and multi-policy curriculum regressions (CPU)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from config_run import test_SB3_chairman2_env as env_fixture
from metasim.cfg.checkers import _ChairMan2Checker
from metasim.cfg.checkers import stages_chairman2 as stages
from metasim.cfg.tasks.humanoidbench.ChairMan_2 import Chairman2Cfg


class Chairman2MultiTest(unittest.TestCase):
    def test_five_policy_train_save_load_resume_and_reject_old_schema(self):
        import json
        import tempfile
        from config_run.test_multi_ppo_trainer import FakeStageVecEnv
        from config_run.multi_ppo_trainer import MultiPPOTrainer, load_policy_router, resolve_policy_bundle
        from metasim.utils.chairman2_geometry import TASK_VERSION
        env = FakeStageVecEnv()
        env.NUM_POLICY_STAGES = 5
        env.env = SimpleNamespace(scenario=SimpleNamespace(task=SimpleNamespace(task_version=TASK_VERSION)))
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(task='chairman2', total_timesteps=40, n_steps=2, batch_size=4,
                       n_epochs=1, stage_rollout_samples=4, model_save_path=directory,
                       tensorboard_log=directory, net_arch=[8], terminal_tables=False,
                       freeze_learned_policies=False, model_save_freq=0)
            trainer = MultiPPOTrainer(env, cfg)
            run = trainer.learn()
            manifest = json.loads((run / 'multi_policy_manifest.json').read_text())
            self.assertEqual(manifest['num_stage_policies'], 5)
            self.assertEqual(manifest['task_version'], TASK_VERSION)
            self.assertEqual(set(manifest['stages']), set(map(str, range(5))))
            self.assertTrue(all(row['updates'] > 0 for row in manifest['stages'].values()))
            router, _ = load_policy_router(env, run)
            self.assertEqual(len(router.models), 5)
            router.predict(env.reset(), env.get_current_stages())
            resumed = MultiPPOTrainer(env, cfg, resume_path=str(run))
            self.assertEqual(resumed.global_timesteps, trainer.global_timesteps)
            resumed.writer.close()
            with self.assertRaises(ValueError):
                resolve_policy_bundle(run, expected_num_stages=6)
            with self.assertRaises(ValueError):
                resolve_policy_bundle(run, expected_task_version='old_schema')

    def test_main_routes_training_and_resume_to_chairman2_multi_trainer(self):
        source = ast.parse(Path(__file__).with_name('main_multi.py').read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        for mode in ('train', 'load_and_train'):
            config = dict(task='chairman2', robots=['g1_without_hands'],
                          train_or_eval=mode, load_model_path='bundle')
            scenario = SimpleNamespace(robots=[SimpleNamespace(name='g1_without_hands')], task=SimpleNamespace())
            wrapper, trainer = Mock(), Mock()
            namespace = dict(sys=SimpleNamespace(argv=['main_multi.py', 'test']),
                             load_config_from_yaml=lambda _: config, log=Mock(),
                             ScenarioCfg=Mock(return_value=scenario), MetaSimVecEnv=Mock(),
                             get_sensors_from_config=Mock(), get_cameras_from_config=Mock())
            exec(compile(ast.Module(body=[main], type_ignores=[]), '<main_multi>', 'exec'), namespace)
            with patch.dict(sys.modules, {
                'SB3_chairman2_env': SimpleNamespace(StableBaseline3VecEnv=wrapper),
                'multi_ppo_trainer': SimpleNamespace(MultiPPOTrainer=trainer, single_training_stage=lambda _: None),
            }):
                namespace['main']()
            self.assertIs(trainer.call_args.args[0], wrapper.return_value)
            self.assertIs(trainer.call_args.args[1], config)
            if mode == 'load_and_train':
                self.assertEqual(trainer.call_args.kwargs['resume_path'], 'bundle')
            trainer.return_value.learn.assert_called_once()
            wrapper.return_value.close.assert_called_once()

    def test_task_checker_robot_reset_and_timeouts(self):
        wrapper = env_fixture.Chairman2Test().make_env()
        task = Chairman2Cfg()
        self.assertIsInstance(task.checker, _ChairMan2Checker)
        from config_run.test_chairman2_checkers import scene
        states, handler = scene()
        for reward in task.reward_functions:
            reward.actual_stage = torch.zeros(2, dtype=torch.long)
            reward.completed_stages = torch.zeros(2, dtype=torch.long)
            if hasattr(reward, 'reset'):
                reward.reset(torch.arange(2), states)
            value = reward(states, handler.robot.name)
            self.assertTrue(torch.isfinite(value).all(), type(reward).__name__)
        wrapper.env.env.handler.scenario = SimpleNamespace(
            sim_params=SimpleNamespace(dt=0.002), decimation=5)
        with patch.dict(stages.STAGE_TIMEOUTS, {0: 123}):
            scenario = wrapper.env.env.handler.scenario
            dt = (scenario.sim_params.dt or 0.002) * scenario.decimation
            import math
            self.assertEqual(wrapper.stage_confirmation_steps[0],
                             math.ceil(123 * stages.STAGE_TIMEOUT_REFERENCE_DT / dt) + 1)

    def test_curriculum_cap_and_explicit_stage_on_legacy_reset(self):
        handler = SimpleNamespace(num_envs=2, device='cpu', robot=SimpleNamespace(name='g1_without_hands'),
                                  task=SimpleNamespace(curriculum_max_stage=0), set_states=Mock(), get_states=Mock())
        current = torch.zeros(2, dtype=torch.long)
        buffer = {i: [{'stage': i}] for i in range(1, 6)}
        with patch.object(stages, 'RAM_SNAPSHOT_BUFFER', buffer), patch.object(stages, 'stage0_init', return_value={}), \
             patch.object(stages, 'load_snapshot_chairman', side_effect=lambda i: buffer[i][0]):
            stages._reset_chairman_legacy(handler, torch.arange(2), current, current.clone(), False, None)
            self.assertEqual(current.tolist(), [0, 0])
            stages._reset_chairman_legacy(handler, torch.arange(2), current, current.clone(), False, 3)
            self.assertEqual(current.tolist(), [3, 3])


if __name__ == '__main__':
    unittest.main()
