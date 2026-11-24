"""Recurrent DAgger tests without running Genesis or loading a PPO bundle."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from metasim.cfg.robots.g1_cfg_with_hands import G1WithHandsCfg
from . import runner
from .buffer import SequenceBuffer
from .checkpoint import load_student
from .inputs import IMU_BODY_NAME, RolloutState, StudentInputs, world_to_body
from .student_net import ACTION_GROUP_SIZES, RecurrentVisionStudent


class TinyStudent(RecurrentVisionStudent):
    """Keep the real recurrent model, replace only the costly image encoder."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Conv2d(3, 512, 1), torch.nn.Flatten(),
        )


class FakeEnvironment:
    num_envs = 2
    NUM_POLICY_STAGES = 6
    LOCOMOTION_COMMAND_NAMES = ("walk_vx", "walk_vy", "walk_yaw_rate")
    robot_name = "g1_with_hands"

    def __init__(self):
        self.robot_joint_names = tuple(G1WithHandsCfg().joint_limits)
        self.action_names = tuple(name for name in self.robot_joint_names
                                  if not any(part in name for part in ("hip_", "knee_", "ankle_")))
        self.action_names += self.LOCOMOTION_COMMAND_NAMES
        self.action_groups = {}
        cursor = 0
        for name, size in ACTION_GROUP_SIZES:
            self.action_groups[name] = tuple(range(cursor, cursor + size))
            cursor += size
        self.action_space = SimpleNamespace(shape=(34,), low=np.full(34, -.5, np.float32),
                                            high=np.full(34, .5, np.float32))
        self.robot = SimpleNamespace(
            joint_names=list(reversed(self.robot_joint_names)),
            joint_pos=torch.arange(86, dtype=torch.float32).reshape(2, 43),
            joint_vel=torch.arange(86, dtype=torch.float32).reshape(2, 43) + 100,
            body_names=["pelvis", "imu_in_pelvis"], body_state=torch.zeros(2, 2, 13),
        )
        self.robot.body_state[:, :, 3] = 1  # identity WXYZ
        self.states = SimpleNamespace(robots={self.robot_name: self.robot},
                                      cameras={"camera0": SimpleNamespace(rgb=torch.zeros(2, 32, 32, 4, dtype=torch.uint8))})
        self.handler = SimpleNamespace(
            scenario=SimpleNamespace(defer_camera_rendering=False, decimation=5),
            scene_inst=SimpleNamespace(dt=.004), get_states=lambda: self.states,
            get_camera_states=lambda: self.states.cameras,
        )
        self.base = SimpleNamespace(env=SimpleNamespace(handler=self.handler))
        self.steps = 0
        self.actions = []

    def reset(self):
        self.steps = 0
        return np.zeros((2, 43), dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        self.actions.append(actions.copy())
        done = np.array([self.steps % 2 == 0, self.steps % 3 == 0])
        infos = [dict(stage_before=0, stage_after=0, completed_stage=-1,
                      is_success=bool(value)) for value in done]
        return np.zeros((2, 43), np.float32), np.ones(2), done, infos

    def get_current_stages(self):
        return np.zeros(2, dtype=np.int64)

    def close(self):
        pass


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_adapter_maps_all_joint_names_and_rotates_imu(self):
        env = FakeEnvironment()
        previous = torch.ones(2, 34)
        adapter = StudentInputs(env.base, env, "cpu")
        values = adapter.read(previous)
        torch.testing.assert_close(values[1], env.robot.joint_pos.flip(1))
        torch.testing.assert_close(values[2], env.robot.joint_vel.flip(1))
        torch.testing.assert_close(values[3][:, 3:], torch.tensor([[0., 0., -1.]]).expand(2, -1))
        self.assertIs(values[4], previous)
        half = 2 ** -.5
        rotated = world_to_body(torch.tensor([[0., 1., 0.]]), torch.tensor([[half, 0., 0., half]]))
        torch.testing.assert_close(rotated, torch.tensor([[1., 0., 0.]]), atol=1e-6, rtol=1e-6)
        env.handler.scenario.defer_camera_rendering = True
        self.assertEqual(adapter.read(previous)[0].shape, (2, 3, 128, 128))

    def test_partial_done_clears_only_finished_memory_and_executed_action(self):
        state = RolloutState(TinyStudent(hidden_size=8), 2)
        hidden = torch.ones(1, 2, 8)
        state.after_step(hidden, np.full((2, 34), .5), [True, False])
        self.assertEqual(state.hidden[0, 0].sum().item(), 0)
        self.assertEqual(state.hidden[0, 1].sum().item(), 8)
        self.assertEqual(state.previous_actions[0].sum().item(), 0)
        torch.testing.assert_close(state.previous_actions[1], torch.full((34,), .5))
        self.assertEqual(state.reset_mask.tolist(), [True, False])

    def test_imu_uses_sensor_link_orientation_and_velocity_instead_of_pelvis(self):
        env = FakeEnvironment()
        half = 2 ** -.5
        env.robot.body_state[:, 0, 10:13] = torch.tensor([9., 8., 7.])
        env.robot.body_state[:, 1, 3:7] = torch.tensor([half, half, 0., 0.])
        env.robot.body_state[:, 1, 10:13] = torch.tensor([0., 0., 2.])
        adapter = StudentInputs(env.base, env, "cpu")
        imu = adapter.read(torch.zeros(2, 34))[3]
        expected = torch.tensor([[0., 2., 0., 0., -1., 0.]]).expand(2, -1)
        torch.testing.assert_close(imu, expected, atol=3e-7, rtol=3e-7)

    def test_missing_sensor_link_does_not_silently_fall_back_to_pelvis(self):
        env = FakeEnvironment()
        env.robot.body_names = ["pelvis"]
        with self.assertRaisesRegex(ValueError, "requires IMU body 'imu_in_pelvis'"):
            StudentInputs(env.base, env, "cpu")

    def test_imu_rotation_matches_independent_scipy_reference(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(123)
        rotations = Rotation.random(1000, random_state=rng)
        # Non-unit quaternions also test normalization of the incoming WXYZ.
        quaternions = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]] * 2.5, dtype=torch.float32)
        vectors = rng.normal(size=(1000, 3))
        result = world_to_body(torch.tensor(vectors, dtype=torch.float32), quaternions)
        expected = torch.tensor(rotations.inv().apply(vectors), dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(result, world_to_body(torch.tensor(vectors, dtype=torch.float32), -quaternions))

    def test_projected_gravity_preserves_direction_and_unit_length(self):
        from scipy.spatial.transform import Rotation
        rotations = Rotation.from_euler("xyz", [[0, 0, 0], [0, 0, 90], [90, 0, 0], [0, 90, 0]], degrees=True)
        q = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32)
        gravity = world_to_body(torch.tensor([[0., 0., -1.]]).expand(4, -1), q)
        expected = torch.tensor([[0., 0., -1.], [0., 0., -1.], [0., -1., 0.], [1., 0., 0.]])
        torch.testing.assert_close(gravity, expected, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(gravity.norm(dim=1), torch.ones(4))

    def test_ring_samples_contiguous_environment_tracks_and_resets(self):
        buffer = SequenceBuffer(5, 2, (3, 32, 32), 43, (1, 8))
        for step in range(8):
            labels = torch.tensor([[step * 10.], [step * 10. + 1]])
            values = (torch.zeros(2, 3, 32, 32, dtype=torch.uint8), labels.expand(2, 43),
                      labels.expand(2, 43), labels.expand(2, 6), labels.expand(2, 34))
            buffer.add(values, labels.expand(2, 34), labels.view(1, 2, 1).expand(1, 2, 8),
                       torch.tensor([step % 2 == 0, step % 2 == 0]))
        values, targets, hidden, resets = buffer.sample(128, 2, 1)
        markers = values[1][..., 0]
        self.assertTrue((markers[:, 1:] - markers[:, :-1] == 10).all())
        self.assertTrue((markers >= 30).all() and (markers <= 71).all())
        torch.testing.assert_close(hidden[0, :, 0], markers[:, 0])
        torch.testing.assert_close(targets[..., 0], markers)
        self.assertTrue(torch.equal(resets, (markers.long() // 10) % 2 == 0))
        buffer.clear()
        self.assertEqual(buffer.size, 0)
        with self.assertRaises(ValueError):
            buffer.sample(1, 2)

    def test_train_resume_eval_video_and_checkpoint_validation(self):
        env = FakeEnvironment()
        env.states.cameras["camera0"].rgb = torch.zeros(2, 360, 640, 3, dtype=torch.uint8)
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.ones((2, 34), np.float32))
        stored = []
        class InspectBuffer(SequenceBuffer):
            def add(self, values, *args):
                stored.append((values[4].clone(), args[-1].clone()))
                return super().add(values, *args)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, "_device", return_value="cpu"), \
                patch.object(runner, "load_policy_router", return_value=(teacher, {})), \
                patch.object(runner, "RecurrentVisionStudent", TinyStudent), \
                patch("config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent", TinyStudent), \
                patch.object(runner, "SequenceBuffer", InspectBuffer):
            root = Path(temporary)
            config = dict(train_or_eval="train_dagger_new", expert_model_path="fake_teacher",
                          dagger_image_width=640, dagger_image_height=360,
                          model_save_path=str(root), tensorboard_log=str(root / "logs"),
                          total_timesteps=4, dagger_buffer_steps=4, dagger_store_per_step=50,
                          dagger_train_every=3, dagger_updates_per_train=1, dagger_log_every=2,
                          dagger_new_sequence_length=2, dagger_new_burn_in=1, dagger_new_batch_size=1,
                          dagger_new_hidden_size=8, model_save_freq=3,
                          beta_start=1., beta_end=1., eval_episodes=2, eval_total_step_cap=4)
            runner.run_dagger(config, env.base, env)
            path = root / "recurrent/student_model_final.pth"
            self.assertTrue(path.is_file())
            self.assertFalse((root / "student_model_final.pth").exists())
            # Stored previous actions are the clipped teacher actions, never
            # the randomly initialized student's proposal.
            self.assertTrue((stored[0][0] == 0).all())
            self.assertTrue((stored[1][0] == .5).all())
            masks = stored[2][1]
            self.assertEqual(sorted(masks.tolist()), [False, True])
            self.assertTrue((stored[2][0][masks] == 0).all())
            self.assertTrue((stored[2][0][~masks] == .5).all())
            config.update(train_or_eval="load_and_train_dagger_new", load_student_path=str(path),
                          total_timesteps=6)
            runner.run_dagger(config, env.base, env)
            student, metadata, next_step = load_student(path, env, env.base, "cpu")
            self.assertEqual(next_step, 6)
            self.assertEqual(metadata["imu_body_name"], "imu_in_pelvis")
            self.assertEqual(metadata["image_shape"], [3, 360, 640])
            payload = torch.load(path, weights_only=True)
            payload["metadata"].pop("imu_body_name")
            previous_checkpoint = root / "previous.pth"
            torch.save(payload, previous_checkpoint)
            _, migrated, _ = load_student(previous_checkpoint, env, env.base, "cpu")
            self.assertEqual(migrated["imu_body_name"], "imu_in_pelvis")
            env.robot.body_state[:, 1, 3:7] = torch.tensor([0., 1., 0., 0.])
            with self.assertRaisesRegex(ValueError, "Legacy checkpoint used pelvis IMU"):
                load_student(previous_checkpoint, env, env.base, "cpu")
            env.robot.body_state[:, 1, 3:7] = torch.tensor([1., 0., 0., 0.])
            config.update(train_or_eval="eval_dagger_new", load_model_path=str(root))
            rows = runner.run_dagger(config, env.base, env)
            self.assertGreater(len(rows), 0)
            self.assertEqual(rows[-1]["success_rate"], 1.)
            self.assertTrue((root / "recurrent/eval_dagger_new_results.csv").is_file())
            fake_writer = SimpleNamespace(isOpened=lambda: True, write=lambda frame: None,
                                          release=lambda: None)
            with patch("cv2.VideoWriter", return_value=fake_writer):
                config.update(train_or_eval="eval_dagger_new_video", eval_max_steps=4)
                runner.run_dagger(config, env.base, env)
            env.handler.scenario.decimation = 10
            with self.assertRaisesRegex(ValueError, "control period"):
                load_student(path, env, env.base, "cpu")
            env.handler.scenario.decimation = 5
            env.action_names = tuple(reversed(env.action_names))
            with self.assertRaisesRegex(ValueError, "action names"):
                load_student(path, env, env.base, "cpu")
            legacy = root / "legacy.pth"
            torch.save(student.state_dict(), legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_student(legacy, env, env.base, "cpu")

    def test_invalid_replay_settings_report_the_specific_parameter(self):
        env = FakeEnvironment()
        for config, message in (
            (dict(dagger_buffer_steps=23), "dagger_buffer_steps=23.*24"),
            (dict(dagger_store_per_step=0), "dagger_store_per_step must be positive, got 0"),
        ):
            with self.subTest(config=config), patch.object(runner, "load_policy_router") as teacher:
                with self.assertRaisesRegex(ValueError, message):
                    runner.train_dagger(config, env.base, env)
                teacher.assert_not_called()

    def test_training_evaluates_without_teacher_and_resumes_archived_demonstrations(self):
        env = FakeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.ones((2, 34), np.float32)*.25)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_device', return_value='cpu'), \
                patch.object(runner, 'load_policy_router', return_value=(teacher, {})), \
                patch.object(runner, 'RecurrentVisionStudent', TinyStudent), \
                patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent', TinyStudent):
            config = dict(train_or_eval='train_dagger_new', expert_model_path='fake',
                          total_timesteps=6, model_save_path=directory, tensorboard_log=directory+'/logs',
                          model_save_freq=0, dagger_buffer_steps=4, dagger_store_per_step=2,
                          dagger_image_width=32, dagger_image_height=32,
                          dagger_new_sequence_length=2, dagger_new_burn_in=1, dagger_new_batch_size=1,
                          dagger_new_hidden_size=8, dagger_train_every=3, dagger_updates_per_train=1,
                          dagger_log_every=3, beta_schedule='evaluation', dagger_eval_every=3,
                          dagger_beta_after_warmup=1.,
                          dagger_eval_episodes=2, dagger_eval_max_steps=5, dagger_eval_required_passes=1,
                          dagger_teacher_warmup_steps=0, dagger_demonstrations_per_stage=2,
                          dagger_teacher_anchor_fraction=.5)
            runner.train_dagger(config, env.base, env)
            path = Path(directory)/'recurrent/student_model_final.pth'
            payload = torch.load(path, weights_only=True)
            self.assertAlmostEqual(payload['metadata']['beta_controller']['beta'], .8)
            archive = torch.load(path.parent/'teacher_demonstrations.pth', weights_only=True)
            self.assertTrue(archive['replay']['windows'][0])
            self.assertEqual(len((path.parent/'training_evaluations.jsonl').read_text().splitlines()), 2)
            config.update(train_or_eval='load_and_train_dagger_new', load_student_path=str(path), total_timesteps=9)
            runner.train_dagger(config, env.base, env)
            self.assertTrue(torch.load(path.parent/'teacher_demonstrations.pth', weights_only=True)['replay']['windows'][0])

    def test_old_recurrent_checkpoint_remains_evaluable_but_cannot_resume_training(self):
        from .checkpoint import LEGACY_FORMAT
        env = FakeEnvironment()
        model = TinyStudent(hidden_size=8, architecture_version=1)
        metadata = dict(model_kwargs=dict(num_joints=43, hidden_size=8),
                        action_names=list(env.action_names), joint_names=list(env.robot_joint_names),
                        imu_layout='body_angular_velocity_xyz_projected_gravity_xyz', imu_body_name=IMU_BODY_NAME,
                        camera_name='camera0', control_dt=.02, image_shape=[3,32,32])
        with tempfile.TemporaryDirectory() as directory, \
                patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent', TinyStudent), \
                patch.object(runner, 'load_policy_router', return_value=(None, {})), \
                patch.object(runner, '_device', return_value='cpu'):
            path = Path(directory)/'old.pth'
            torch.save(dict(format=LEGACY_FORMAT, metadata=metadata, next_step=42,
                            student_state_dict=model.state_dict()), path)
            restored, _, step = load_student(path, env, env.base, 'cpu')
            self.assertEqual(restored.architecture_version, 1)
            self.assertEqual(step, 42)
            with self.assertRaisesRegex(ValueError, 'fresh v2 model'):
                runner.train_dagger(dict(train_or_eval='load_and_train_dagger_new', expert_model_path='fake',
                                         load_student_path=str(path)), env.base, env)

    def test_teacher_baseline_failure_stops_before_training_and_preserves_existing_files(self):
        env = FakeEnvironment()
        failed = dict(episodes_finished=2, requested_episodes=2, success_rate=0., success_count=0, capped_attempts=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_device', return_value='cpu'), \
                patch.object(runner, 'load_policy_router', return_value=(None, {})), \
                patch.object(runner, 'RecurrentVisionStudent', TinyStudent), \
                patch.object(runner, 'evaluate_policy', return_value=failed):
            checkpoint_dir = Path(directory)/'recurrent'
            checkpoint_dir.mkdir()
            existing = ['student_model_final.pth','student_model_final.pth.training.pth','teacher_demonstrations.pth']
            for name in existing:
                (checkpoint_dir/name).write_bytes(b'previous successful training')
            with self.assertRaisesRegex(RuntimeError, 'Teacher baseline failed'):
                runner.train_dagger(dict(expert_model_path='fake', model_save_path=directory,
                                         tensorboard_log=directory+'/logs', total_timesteps=10,
                                         dagger_image_width=32, dagger_image_height=32,
                                         dagger_buffer_steps=24, dagger_new_hidden_size=8,
                                         dagger_teacher_baseline=True), env.base, env)
            self.assertEqual(env.actions, [])
            for name in existing:
                self.assertEqual((checkpoint_dir/name).read_bytes(),b'previous successful training')
            self.assertEqual(list(checkpoint_dir.glob('student_model_interrupted*')),[])

    def test_teacher_anchor_stays_expert_controlled_when_beta_is_zero(self):
        env = FakeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2,34), .25, np.float32))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_device', return_value='cpu'), \
                patch.object(runner, 'load_policy_router', return_value=(teacher, {})), \
                patch.object(runner, 'RecurrentVisionStudent', TinyStudent):
            runner.train_dagger(dict(expert_model_path='fake', model_save_path=directory,
                                     tensorboard_log=directory+'/logs', total_timesteps=4,
                                     dagger_image_width=32, dagger_image_height=32,
                                     dagger_buffer_steps=24, dagger_store_per_step=2,
                                     dagger_new_hidden_size=8, dagger_train_every=100,
                                     beta_schedule='linear', beta_start=0., beta_end=0.,
                                     dagger_teacher_anchor_fraction=.5), env.base, env)
            first = np.all(np.isclose(env.actions[0], .25), axis=1)
            self.assertEqual(int(first.sum()), 1)
            for actions in env.actions:
                np.testing.assert_array_equal(np.all(np.isclose(actions, .25), axis=1), first)

    def test_camera_geometry_is_validated_and_evaluation_restores_curriculum_on_error(self):
        from .evaluation import evaluation_resets
        config = dict(cameras={'camera1': dict(params=dict(mount_pos=[.08,0,.35], fov_vertical=58.7))})
        metadata = dict(camera_config=runner.camera_config(config))
        runner.validate_camera_config(metadata, config)
        config['cameras']['camera1']['params']['mount_pos'] = [0.,0.,.5]
        with self.assertRaisesRegex(ValueError, 'Camera calibration'):
            runner.validate_camera_config(metadata, config)
        with patch.object(runner.log,'warning') as warning:
            runner.validate_camera_config(metadata,config,strict=False)
            warning.assert_called_once()
        env = FakeEnvironment()
        env.env = env.base
        env.handler.task = SimpleNamespace(reset_to_stage0=False, use_snapshot_curriculum=True,
                                           snapshot_save_probability=.1, eval_start_stage=None)
        before = vars(env.handler.task).copy()
        with self.assertRaisesRegex(RuntimeError, 'evaluation error'):
            with evaluation_resets(env):
                self.assertTrue(env.handler.task.reset_to_stage0)
                self.assertFalse(env.handler.task.use_snapshot_curriculum)
                raise RuntimeError('evaluation error')
        self.assertEqual(vars(env.handler.task), before)

    def test_evaluation_and_video_continue_after_scene_calibration_change(self):
        env = FakeEnvironment()
        model = TinyStudent(hidden_size=8)
        previous = dict(cameras={'camera1':dict(params=dict(gaussian_scale=[3.,3.,3.]))})
        metadata = dict(camera_config=runner.camera_config(previous),camera_name='camera0',
                        joint_names=list(env.robot_joint_names),image_shape=[3,32,32])
        for video in (False,True):
            with self.subTest(video=video),tempfile.TemporaryDirectory() as directory:
                config = dict(cameras={'camera1':dict(params=dict(gaussian_scale=[3.634]*3))},
                              eval_episodes=2,eval_total_step_cap=4,eval_max_steps=4,
                              video_save_path=str(Path(directory)/'eval.mp4'))
                writer = SimpleNamespace(isOpened=lambda:True,write=lambda frame:None,release=lambda:None)
                with patch.object(runner,'load_student',return_value=(model,metadata,42)), \
                     patch.object(runner.log,'warning') as warning, \
                     patch('cv2.VideoWriter',return_value=writer):
                    row = runner.evaluate_checkpoint(config,env.base,env,'student.pth','cpu',video=video)
                warning.assert_called_once()
                self.assertEqual(row['episodes_finished'],2)
                self.assertEqual(row['success_count'],2)
                self.assertEqual(metadata['camera_config'],runner.camera_config(previous))

    def test_main_dispatches_new_and_existing_modes(self):
        # Match the import path of `python config_run/main_separate.py`;
        # main_multi's existing helpers use absolute sibling imports.
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
            from config_run import main_separate
        for mode in ("train_dagger_new", "load_and_train_dagger_new", "eval_dagger_new", "eval_dagger_new_video",
                     "train_dagger", "eval_dagger", "eval_dagger_video"):
            env = FakeEnvironment()
            config = dict(train_or_eval=mode)
            with self.subTest(mode=mode), \
                    patch.object(main_separate.sys, "argv", ["main_separate.py", "test"]), \
                    patch.object(main_separate, "load_config_from_yaml", return_value=config), \
                    patch.object(main_separate, "build_env", return_value=(env.base, env)), \
                    patch("config_run.dagger_recurrent.runner.run_dagger") as new, \
                    patch("config_run.separate_dagger.run_dagger") as old:
                main_separate.main()
                (new if "_new" in mode else old).assert_called_once_with(config, env.base, env)
                (old if "_new" in mode else new).assert_not_called()

    def test_environment_sets_dagger_flags_for_new_modes(self):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
            from config_run import main_separate
        for mode, expected in (("train_dagger_new", 1), ("load_and_train_dagger_new", 1),
                               ("eval_dagger_new", 2), ("eval_dagger_new_video", 2)):
            scenario = SimpleNamespace(task=SimpleNamespace(), robots=[SimpleNamespace(name="g1_with_hands")])
            config = dict(task="chairmanseparate", robots=["g1_with_hands"], train_or_eval=mode,
                          sim="genesis", fix_base_link=False,render_ground=False)
            with self.subTest(mode=mode), \
                    patch.object(main_separate, "ScenarioCfg", return_value=scenario) as make_scenario, \
                    patch.object(main_separate, "MetaSimVecEnv"), \
                    patch.object(main_separate, "StableBaseline3VecEnv"):
                main_separate.build_env(config)
                self.assertIs(make_scenario.call_args.kwargs['render_ground'],False)
                self.assertEqual(scenario.dagger, expected)
                if expected == 1:
                    self.assertTrue(scenario.defer_camera_rendering)


if __name__ == "__main__":
    unittest.main()
