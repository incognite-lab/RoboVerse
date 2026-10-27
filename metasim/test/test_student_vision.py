"""Student image geometry, deploy parity and resolution compatibility."""
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from config_run.student_vision import image_size_from_config, preprocess_student_rgb


class StudentVisionTest(unittest.TestCase):
    def test_640_by_360_reaches_student_unchanged_without_cropping_or_resizing(self):
        rgb = torch.randint(256, (2, 360, 640, 4), dtype=torch.uint8)
        with patch("config_run.student_vision._resize_axis", side_effect=AssertionError("Unexpected resize")), \
                patch("config_run.student_vision.center_crop_bounds", side_effect=AssertionError("Unexpected crop")):
            actual = preprocess_student_rgb(rgb, output_size=(360, 640))
        torch.testing.assert_close(actual, rgb[..., :3].permute(0, 3, 1, 2))
        self.assertEqual(actual.shape, (2, 3, 360, 640))
        self.assertTrue(actual.is_contiguous())

    def test_real_camera_preprocessing_matches_deploy(self):
        from config_run.deploy_dagger_student_real import preprocess_frame
        rng = np.random.default_rng(4)
        bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        for height, width in ((360, 640), (128, 128)):
            with self.subTest(size=(height, width)):
                actual = preprocess_student_rgb(rgb, output_size=(height, width))
                expected = preprocess_frame(bgr, width, height, "cpu") * 255
                self.assertLessEqual((actual.float() - expected).abs().max().item(), 1.001)

    def test_existing_128_square_pipeline_still_works(self):
        image = torch.randint(256, (2, 128, 128, 3), dtype=torch.uint8)
        torch.testing.assert_close(preprocess_student_rgb(image), image.permute(0, 3, 1, 2))
        self.assertEqual(image_size_from_config({}), (128, 128))

    def test_standard_configs_match_target_and_replay_budget(self):
        from metasim.cfg.sensors.cameras import NyxGaussianSplatCameraCfg
        root = Path(__file__).resolve().parents[2] / "config_run/configs/chairman_separate"
        for name in ("train_dagger", "eval_dagger", "eval_dagger_video"):
            with self.subTest(config=name):
                cfg = yaml.safe_load((root / f"{name}.yaml").read_text())
                self.assertEqual(image_size_from_config(cfg), (360, 640))
                params = next(iter(cfg["cameras"].values()))["params"]
                projection = NyxGaussianSplatCameraCfg(**params).render_projection
                self.assertEqual((projection.render_width, projection.render_height), (640, 360))
                self.assertFalse(projection.needs_resampling)
                if name == "train_dagger":
                    image_bytes = cfg["dagger_buffer_steps"] * cfg["dagger_store_per_step"] * 3 * 360 * 640
                    self.assertLess(image_bytes, 3 * 1024**3)
                    self.assertEqual(cfg["dagger_new_buffer_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
