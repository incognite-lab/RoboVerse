"""Policy playback freezes physics and recurrent state while inspection continues."""
from contextlib import ExitStack
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from config_run.nyx_preview import preview_nyx
from config_run.nyx_preview_policy import PreviewPolicy, PreviewPolicySettings
from config_run.dagger_recurrent.test_pipeline import FakeEnvironment, TinyStudent
from metasim.cfg.sensors import NyxGaussianSplatCameraCfg


class PreviewPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def settings(self, **kwargs):
        return PreviewPolicySettings(enabled=True,path='fixture.pth',**kwargs)

    def test_configuration_rejects_invalid_settings_before_scene_creation(self):
        for raw in (True, {'enabled':'false'}, {'start_paused':'false'},
                    {'type':'ppo'}, {'unknown':True}, {'enabled':True},
                    {'enabled':True,'path':'x','type':'student','checkpoint':'final'}):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                PreviewPolicySettings.from_config(dict(nyx_preview_policy=raw,nyx_preview_controls=True))
        with self.assertRaisesRegex(ValueError,'controls'):
            PreviewPolicySettings.from_config(dict(nyx_preview_policy={'enabled':True,'path':'x'}))
        self.assertFalse(PreviewPolicySettings.from_config({}).enabled)

    def test_student_uses_robot_image_restores_free_camera_and_resets_only_done_memory(self):
        env = FakeEnvironment()
        env.torch_device = 'cpu'
        env.handler.task = SimpleNamespace()
        env.handler.scenario.defer_camera_rendering = True
        view = {'robot':False}
        frames = []
        def render():
            self.assertTrue(view['robot'])
            self.assertIsNone(env.handler._camera_states_cache)
            env.states.cameras['camera0'].rgb.fill_(64)
            return env.states.cameras
        env.handler.get_camera_states = render
        class Student(TinyStudent):
            def forward(self,*values,**kwargs):
                frames.append(values[0].clone())
                return super().forward(*values,**kwargs)
        student = Student(hidden_size=8).eval()
        meta = dict(camera_name='camera0',joint_names=env.robot_joint_names,image_shape=[3,32,32],target_stage=4)
        controller = SimpleNamespace(apply=lambda **kwargs:view.update(robot=kwargs.get('robot_view',False)))
        with patch('config_run.dagger_recurrent.checkpoint.load_student',return_value=(student,meta,123)):
            playback = PreviewPolicy(self.settings(),env.base,env)
        self.assertEqual(env.handler.task.dagger_target_stage,4)
        playback.step(controller)
        self.assertFalse(view['robot'])
        self.assertIsNone(env.handler._camera_states_cache)
        torch.testing.assert_close(frames[-1],torch.full_like(frames[-1],64/255))
        self.assertTrue(playback.state.hidden.abs().sum()>0)
        playback.step(controller)  # Fake env 0 finishes on its second step.
        self.assertTrue((playback.state.hidden[:,0]==0).all())
        self.assertTrue((playback.state.previous_actions[0]==0).all())
        self.assertTrue(playback.state.hidden[:,1].abs().sum()>0)
        torch.testing.assert_close(playback.state.previous_actions[1],torch.from_numpy(env.actions[-1][1]))
        self.assertEqual(playback.episodes,1)
        self.assertEqual(playback.steps,2)

    def test_student_camera_is_restored_when_sensor_read_fails(self):
        playback = object.__new__(PreviewPolicy)
        playback.student = object()
        playback.handler = SimpleNamespace(_camera_states_cache=object())
        playback.state = SimpleNamespace(previous_actions=None)
        playback.adapter = SimpleNamespace(read=Mock(side_effect=RuntimeError('sensor failed')))
        controller = SimpleNamespace(apply=Mock())
        with self.assertRaisesRegex(RuntimeError,'sensor failed'):
            playback.step(controller)
        self.assertEqual(controller.apply.call_args_list[0].kwargs,{'robot_view':True})
        self.assertEqual(controller.apply.call_args_list[1].kwargs,{})
        self.assertIsNone(playback.handler._camera_states_cache)

    def test_teacher_uses_existing_router_clips_actions_and_advances_observation(self):
        env = FakeEnvironment()
        env.torch_device = 'cpu'
        router = SimpleNamespace(predict=Mock(return_value=np.full((2,34),3.,np.float32)))
        # Existing trainer imports its SB3 sibling as a top-level module.
        with patch.object(sys,'path',[str(Path(__file__).resolve().parents[2]/'config_run'),*sys.path]), \
             patch('config_run.separate_ppo_trainer.load_policy_router',return_value=(router,{})) as loader:
            playback = PreviewPolicy(self.settings(type='teacher',checkpoint='final'),env.base,env)
        self.assertEqual(loader.call_args.kwargs['checkpoint'],'final')
        before = playback.observation
        playback.step(None)
        self.assertIs(router.predict.call_args.args[0],before)
        self.assertTrue((env.actions[-1]==.5).all())
        self.assertEqual(playback.steps,1)
        router.predict.return_value = np.full((2,34),np.nan)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            playback.step(None)
        self.assertEqual(env.steps,1)

    def test_space_freezes_physics_and_policy_but_keeps_camera_and_render_active(self):
        for start_paused in (False,True):
            with self.subTest(start_paused=start_paused),ExitStack() as stack:
                camera = NyxGaussianSplatCameraCfg(gaussian_splat_path='fixture.ply')
                handler = SimpleNamespace(cameras=[camera],_camera_states_cache=None,
                    get_camera_states=Mock(return_value={'camera0':SimpleNamespace(rgb=object())}))
                base = SimpleNamespace(env=SimpleNamespace(handler=handler),reset=Mock())
                env = SimpleNamespace(step=Mock())
                playback = SimpleNamespace(adapter=None,step=Mock(),caption=lambda paused:'PAUSED' if paused else 'RUNNING')
                controller = SimpleNamespace(update=Mock(),apply=Mock(),mode='FREE')
                window = SimpleNamespace(closed=False,show=Mock(),close=Mock(),
                    poll=Mock(side_effect=[({'w'},set()),({'w'},{'space'}),({'w'},set()),({'w'},{'space'})]))
                factory = stack.enter_context(patch('config_run.nyx_preview.PreviewPolicy',return_value=playback))
                stack.enter_context(patch('config_run.nyx_preview_controls.PreviewWindow',return_value=window))
                stack.enter_context(patch('config_run.nyx_preview_controls.PreviewCamera',return_value=controller))
                stack.enter_context(patch('config_run.nyx_preview.robot_camera_from_config',return_value=object()))
                stack.enter_context(patch('config_run.nyx_preview.time.sleep'))
                config = dict(nyx_preview_controls=True,nyx_preview_max_frames=4,
                    nyx_preview_policy=dict(enabled=True,path='fixture.pth',start_paused=start_paused))
                self.assertEqual(preview_nyx(config,base,env),4)
                factory.assert_called_once()
                base.reset.assert_not_called()  # Playback owns the single wrapped reset.
                self.assertEqual(playback.step.call_count,2)
                self.assertEqual(controller.update.call_count,4)
                self.assertEqual(window.show.call_count,4)
                self.assertEqual(handler.get_camera_states.call_count,4)
                captions = [call.args[1] for call in window.show.call_args_list]
                self.assertEqual(['PAUSED' in value for value in captions],
                                 [start_paused,not start_paused,not start_paused,start_paused])
                window.close.assert_called_once()

    def test_build_env_only_loads_locomotion_wrapper_for_policy_playback(self):
        with patch.object(sys,'path',[str(Path(__file__).resolve().parents[2]/'config_run'),*sys.path]):
            from config_run import main_separate
        for enabled in (False,True):
            with self.subTest(enabled=enabled):
                scenario = SimpleNamespace(task=SimpleNamespace(),robots=[SimpleNamespace(name='g1_with_hands')])
                config = dict(task='chairmanseparate',robots=['g1_with_hands'],train_or_eval='preview_nyx',
                    eval_start_stage=4,
                    nyx_preview_controls=True,nyx_preview_policy=dict(enabled=enabled,path='fixture.pth'))
                with patch.object(main_separate,'ScenarioCfg',return_value=scenario), \
                     patch.object(main_separate,'MetaSimVecEnv') as make_base, \
                     patch.object(main_separate,'StableBaseline3VecEnv') as wrapper, \
                     patch.object(main_separate,'load_config_from_yaml',return_value={'policies':{'waist':{}}}):
                    base,env = main_separate.build_env(config)
                self.assertIs(base,make_base.return_value)
                self.assertIs(env,wrapper.return_value if enabled else base)
                self.assertEqual(wrapper.call_count,int(enabled))
                self.assertTrue(scenario.defer_camera_rendering)
                self.assertEqual(scenario.task.eval_start_stage,4)
                self.assertEqual(scenario.task.separate_policy_specs,{'waist':{}} if enabled else {})


if __name__ == '__main__':
    unittest.main()
