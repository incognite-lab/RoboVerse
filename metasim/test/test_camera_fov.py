"""Camera FOV resolution and exact output rays without initializing Genesis."""
import math
import unittest

import torch

from metasim.cfg.sensors.cameras import NyxGaussianSplatCameraCfg, PinholeCameraCfg
from metasim.utils.camera_projection import make_camera_projection


def camera(**kwargs):
    return NyxGaussianSplatCameraCfg(gaussian_splat_path="test.ply", **kwargs)


class CameraFovTest(unittest.TestCase):
    def test_legacy_vertical_angle_and_derived_horizontal_remain_unchanged(self):
        cfg = camera(width=1920, height=1080, fov=65)
        self.assertAlmostEqual(cfg.vertical_fov, 65)
        self.assertAlmostEqual(cfg.horizontal_fov, 97.11433166531434)
        self.assertEqual((cfg.render_projection.render_width, cfg.render_projection.render_height), (1920, 1080))
        self.assertFalse(cfg.render_projection.needs_resampling)

    def test_default_lens_matches_existing_pinhole_camera(self):
        old = PinholeCameraCfg(width=320, height=180)
        new = camera(width=320, height=180)
        self.assertAlmostEqual(new.horizontal_fov, old.horizontal_fov)
        self.assertAlmostEqual(new.vertical_fov, old.vertical_fov)
        torch.testing.assert_close(torch.tensor(new.intrinsics), torch.tensor(old.intrinsics))

    def test_either_explicit_axis_derives_the_other(self):
        vertical = camera(width=1920, height=1080, fov_vertical=65)
        horizontal = camera(width=1920, height=1080, fov_horizontal=65)
        self.assertAlmostEqual(vertical.horizontal_fov, 97.11433166531434)
        self.assertAlmostEqual(horizontal.vertical_fov, 39.43048686077504)
        square = camera(width=128, height=128, fov_horizontal=65)
        self.assertAlmostEqual(square.vertical_fov, 65)

    def test_independent_angles_define_distinct_focal_lengths(self):
        cfg = camera(width=128, height=128, fov_horizontal=90, fov_vertical=60)
        self.assertAlmostEqual(cfg.horizontal_fov, 90)
        self.assertAlmostEqual(cfg.vertical_fov, 60)
        self.assertAlmostEqual(cfg.intrinsics[0][0], 64)
        self.assertAlmostEqual(cfg.intrinsics[1][1], 64 / math.tan(math.radians(30)))
        self.assertTrue(cfg.render_projection.needs_resampling)
        # Configclass copying/serialization must retain the explicit inputs.
        self.assertEqual(cfg.to_dict()["fov_horizontal"], 90)
        self.assertEqual(cfg.copy().vertical_fov, 60)

    def test_rendered_ray_coordinates_match_both_requested_angles(self):
        for output_width, output_height, horizontal, vertical in (
            (128, 128, 97.1143316653, 65),  # wide field, square output
            (128, 128, 40, 75),           # narrow field, square output
            (160, 90, 80, 55),            # non-square output, fractional crop
        ):
            with self.subTest(horizontal=horizontal, vertical=vertical):
                plan = make_camera_projection(output_width, output_height, horizontal, vertical)
                tan_y = math.tan(math.radians(vertical / 2))
                tan_x = math.tan(math.radians(horizontal / 2))
                # Synthetic renderer image encodes x/z and y/z per native
                # pixel. Linear interpolation of these coordinates is exact.
                xx = (2 * (torch.arange(plan.render_width) + .5) / plan.render_width - 1)
                yy = (2 * (torch.arange(plan.render_height) + .5) / plan.render_height - 1)
                yy, xx = torch.meshgrid(yy, xx, indexing="ij")
                native = torch.stack((xx * tan_y * plan.render_width / plan.render_height,
                                      yy * tan_y, torch.ones_like(xx)), dim=-1)
                result = plan.resample_rgb(native.unsqueeze(0).expand(2, -1, -1, -1))
                expected_x = (2 * (torch.arange(output_width) + .5) / output_width - 1) * tan_x
                expected_y = (2 * (torch.arange(output_height) + .5) / output_height - 1) * tan_y
                self.assertEqual(result.shape, (2, output_height, output_width, 3))
                torch.testing.assert_close(result[0, :, :, 0], expected_x.expand(output_height, -1),
                                           atol=5e-7, rtol=2e-6)
                torch.testing.assert_close(result[0, :, :, 1], expected_y[:, None].expand(-1, output_width),
                                           atol=5e-7, rtol=2e-6)
                self.assertGreaterEqual(plan.render_width, output_width)
                self.assertGreaterEqual(plan.render_height, output_height)

    def test_rgb_output_preserves_dtype_and_legacy_returns_same_tensor(self):
        plan = make_camera_projection(128, 128, 90, 60)
        native = torch.full((2, plan.render_height, plan.render_width, 3), 123, dtype=torch.uint8)
        result = plan.resample_rgb(native)
        self.assertEqual(result.dtype, torch.uint8)
        self.assertTrue((result == 123).all())
        self.assertEqual(plan.resample_rgb(native[0]).shape, (128, 128, 3))
        legacy = camera(width=128, height=128, fov=65).render_projection
        image = torch.zeros(2, 128, 128, 3, dtype=torch.uint8)
        self.assertIs(legacy.resample_rgb(image), image)

    def test_full_hd_center_crop_and_direct_square_have_same_intrinsics(self):
        full = camera(width=1920, height=1080, fov_vertical=65)
        crop_intrinsics = torch.tensor(full.intrinsics, dtype=torch.float64)
        crop_intrinsics[0, 2] -= 420
        crop_intrinsics[:2] *= 128 / 1080
        direct = camera(width=128, height=128, fov_horizontal=65, fov_vertical=65)
        torch.testing.assert_close(crop_intrinsics, torch.tensor(direct.intrinsics, dtype=torch.float64))

    def test_invalid_fovs_and_conflicting_legacy_alias_are_rejected(self):
        for kwargs in (dict(fov_vertical=0), dict(fov_horizontal=180), dict(fov=-1),
                       dict(fov_vertical=float("nan")), dict(fov_horizontal=float("inf")),
                       dict(fov=65, fov_vertical=60), dict(width=0), dict(height=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                camera(**kwargs)
        self.assertEqual(camera(fov=65, fov_vertical=65).vertical_fov, 65)


if __name__ == "__main__":
    unittest.main()
