"""Calibration refreshes the native window without stepping/resetting physics."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
import torch

from config_run.nyx_preview import preview_nyx
from metasim.cfg.sensors import NyxGaussianSplatCameraCfg


class NyxPreviewTest(unittest.TestCase):
    def make_base(self, **camera_kwargs):
        camera = NyxGaussianSplatCameraCfg(gaussian_splat_path='fixture.ply',**camera_kwargs)
        handler = SimpleNamespace(cameras=[camera],_camera_states_cache=object(),step=Mock())
        def render():
            self.assertIsNone(handler._camera_states_cache)
            handler._camera_states_cache = object()
        handler.get_camera_states = Mock(side_effect=render)
        base = SimpleNamespace(env=SimpleNamespace(handler=handler),reset=Mock(),step=Mock())
        return base,handler

    def test_static_scene_renders_every_frame_even_with_unchanged_physics_time(self):
        base,handler = self.make_base(open_nyx_window=True)
        with patch('config_run.nyx_preview.time.sleep'):
            count = preview_nyx(dict(nyx_preview_max_frames=3),base)
        self.assertEqual(count,3)
        base.reset.assert_called_once_with()
        self.assertEqual(handler.get_camera_states.call_count,3)
        base.step.assert_not_called()
        handler.step.assert_not_called()

    def test_ctrl_c_stops_preview(self):
        base,handler = self.make_base(open_nyx_window=True)
        handler.get_camera_states.side_effect = KeyboardInterrupt
        self.assertEqual(preview_nyx({},base),0)

    def test_controlled_window_dispatches_input_and_closes_after_render_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                base,handler = self.make_base()
                handler.get_camera_states.side_effect = (RuntimeError('render failed') if fail else None)
                handler.get_camera_states.return_value = {'camera0':SimpleNamespace(rgb=object())}
                window = SimpleNamespace(poll=Mock(return_value=({'w'},{'c'})),closed=False,
                                         show=Mock(),close=Mock())
                controller = SimpleNamespace(update=Mock(),mode='ROBOT')
                with patch('config_run.nyx_preview_controls.PreviewCamera',return_value=controller), \
                     patch('config_run.nyx_preview_controls.PreviewWindow',return_value=window), \
                     patch('config_run.nyx_preview.robot_camera_from_config',return_value=object()), \
                     patch('config_run.nyx_preview.time.sleep'):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'render failed'):
                            preview_nyx(dict(nyx_preview_controls=True,nyx_preview_max_frames=1),base)
                    else:
                        self.assertEqual(preview_nyx(dict(nyx_preview_controls=True,nyx_preview_max_frames=1),base),1)
                        controller.update.assert_called_once()
                        window.show.assert_called_once()
                window.close.assert_called_once()

    def test_invisible_and_invalid_config_fail_before_reset(self):
        for config in ({},dict(nyx_preview_fps=0),dict(nyx_preview_fps=float('nan')),
                       dict(nyx_preview_max_frames=-1)):
            base,_ = self.make_base()
            with self.subTest(config=config),self.assertRaises(ValueError):
                preview_nyx(config,base)
            base.reset.assert_not_called()

    def test_g_resamples_and_v_restores_clean_renderer_image(self):
        from config_run.visual_augmentation import VisualAugmentation
        base,handler = self.make_base()
        raw = torch.full((1,64,64,3),100,dtype=torch.uint8)
        handler.get_camera_states.side_effect = None
        handler.get_camera_states.return_value = {'camera0':SimpleNamespace(rgb=raw)}
        displayed,samples = [],[]
        class RecordingAugmentation(VisualAugmentation):
            def sample(self,*args,**kwargs):
                result = super().sample(*args,**kwargs)
                samples.append(result.describe())
                return result
        window = SimpleNamespace(poll=Mock(side_effect=[(set(),{'g'}),(set(),{'v'}),(set(),{'g'})]),
            closed=False,show=lambda rgb,caption:displayed.append((rgb.clone(),caption)),close=Mock())
        with patch('config_run.nyx_preview_controls.PreviewCamera',return_value=SimpleNamespace(update=Mock(),mode='FREE')), \
             patch('config_run.nyx_preview_controls.PreviewWindow',return_value=window), \
             patch('config_run.nyx_preview.robot_camera_from_config',return_value=object()), \
             patch('config_run.nyx_preview.VisualAugmentation',RecordingAugmentation), \
             patch('config_run.nyx_preview.time.sleep'):
            count = preview_nyx(dict(nyx_preview_controls=True,nyx_preview_augmentation=True,
                                nyx_preview_augmentation_size=[32,32],nyx_preview_max_frames=3),base)
        self.assertEqual(count,3)
        self.assertEqual(len(samples),2)
        self.assertNotEqual(samples[0],samples[1])
        self.assertEqual(displayed[0][0].shape,(1,32,32,3))
        self.assertIn('Aug ON',displayed[0][1])
        self.assertIn('Aug OFF',displayed[1][1])
        self.assertIn('Aug ON',displayed[2][1])
        torch.testing.assert_close(displayed[1][0],raw,rtol=0,atol=0)
        torch.testing.assert_close(raw,torch.full_like(raw,100),rtol=0,atol=0)
        window.close.assert_called_once()

    def test_calibration_config_uses_native_window_and_actual_sim_geometry(self):
        root = Path(__file__).resolve().parents[2]
        config = yaml.safe_load((root/'config_run/configs/chairman_separate/calibrate_nyx.yaml').read_text())
        camera = NyxGaussianSplatCameraCfg(**config['cameras']['camera1']['params'])
        self.assertEqual(config['train_or_eval'],'preview_nyx')
        self.assertEqual(config['num_envs'],1)
        self.assertTrue(config['nyx_preview_controls'])
        self.assertFalse(camera.open_nyx_window)
        self.assertTrue(camera.render_sim_geometry)
        self.assertIsNone(camera.mount_to)
        self.assertFalse(camera.debug_calibrate)
        self.assertEqual(camera.gaussian_splat_path,'./config_run/splat/splat_2.ply')


if __name__ == '__main__':
    unittest.main()
