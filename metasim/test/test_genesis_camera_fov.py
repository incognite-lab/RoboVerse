"""Optional real Nyx test: RUN_GENESIS_CAMERA_TESTS=1 python -m unittest ..."""
import copy
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(os.environ.get("RUN_GENESIS_CAMERA_TESTS") == "1", "requires Genesis/Nyx GPU and local assets")
class GenesisCameraFovTest(unittest.TestCase):
    def test_independent_axes_reach_renderer_and_output_keeps_requested_size(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config_run"))
        from main_separate import build_env, load_config_from_yaml

        config = load_config_from_yaml("chairman_separate/train_dagger")
        config.update(num_envs=1, headless=True, train_or_eval="train_dagger_new",
                      snapshot_save_probability=0., log_termination_reasons=False)
        template = next(iter(config["cameras"].values()))
        config["cameras"] = {}
        for name, horizontal, vertical in (("wide", 90., 60.), ("narrow", 40., 75.),
                                           ("legacy", None, 65.), ("student", 90., 58.71550708558255)):
            item = copy.deepcopy(template)
            params = item["params"]
            for field in ("fov", "fov_horizontal", "fov_vertical"):
                params.pop(field, None)
            params.update(name=name, width=128, height=128, spp=1, debug_show=False,
                          debug_show_all_envs=False, open_nyx_window=False)
            if name == "student":
                params.update(width=640, height=360)
            if horizontal is None:
                params["fov"] = vertical
            else:
                params.update(fov_horizontal=horizontal, fov_vertical=vertical)
            config["cameras"][name] = item
        base, env = build_env(config)
        try:
            env.reset()
            handler = base.env.handler
            states = handler.get_camera_states()
            for cfg in handler.cameras:
                with self.subTest(camera=cfg.name):
                    sensor = handler.camera_inst_dict[cfg.name]
                    projection = cfg.render_projection
                    self.assertEqual(sensor._options.res, (projection.render_width, projection.render_height))
                    self.assertAlmostEqual(sensor._options.fov, cfg.vertical_fov)
                    self.assertEqual(tuple(states[cfg.name].rgb.shape), (1, cfg.height, cfg.width, 3))
                    self.assertEqual(states[cfg.name].rgb.dtype, torch.uint8)
                    if cfg.name == "wide":
                        self.assertGreater(sensor._options.res[0], 128)
                    elif cfg.name == "narrow":
                        self.assertGreater(sensor._options.res[1], 128)
                    elif cfg.name == "student":
                        self.assertEqual(sensor._options.res, (640, 360))
                    else:
                        self.assertEqual(sensor._options.res, (128, 128))
            # The converted frames must also be served by the deferred cache.
            self.assertIs(handler.get_camera_states(), states)
            # Real rendered frame -> full-size student -> ResNet gradients.
            from config_run.dagger_recurrent.inputs import StudentInputs
            from config_run.dagger_recurrent.student_net import RecurrentVisionStudent
            adapter = StudentInputs(base, env, "cuda", "student", image_size=(360, 640))
            values = adapter.read(torch.zeros(1, 34, device="cuda"))
            self.assertEqual(tuple(values[0].shape), (1, 3, 360, 640))
            torch.testing.assert_close(values[0], states["student"].rgb.permute(0, 3, 1, 2))
            student = RecurrentVisionStudent(hidden_size=32, visual_batch_size=1).cuda().train()
            actions, _ = student(values[0].float() / 255, *values[1:])
            self.assertEqual(tuple(actions.shape), (1, 34))
            actions.square().mean().backward()
            self.assertGreater(student.visual_encoder.conv1.weight.grad.abs().sum().item(), 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
