"""Opt-in real Genesis/Nyx reset freshness check; requires local robot/splat assets.

RUN_GENESIS_CAMERA_TESTS=1 python -m unittest metasim.test.test_genesis_deferred_camera
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('RUN_GENESIS_CAMERA_TESTS') == '1', 'requires Genesis/Nyx GPU and scene assets')
class GenesisDeferredCameraTest(unittest.TestCase):
    def test_resets_do_not_render_and_same_timestep_geometry_is_fresh(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'config_run'))
        from main_separate import build_env, load_config_from_yaml

        config = load_config_from_yaml('chairman_separate/train_dagger')
        config.update(num_envs=4, snapshot_save_probability=0.0, log_termination_reasons=False)
        base, env = build_env(config)
        handler = base.env.handler
        camera = handler.camera_inst_dict['camera0']
        try:
            self.assertTrue(handler.scenario.defer_camera_rendering)
            with patch.object(camera, 'read', wraps=camera.read) as read:
                env.reset()
                base.reset(env_ids=[0])
                base.reset(env_ids=[3])
                self.assertEqual(handler.get_states().cameras, {})
                read.assert_not_called()

                first = handler.get_camera_states()
                self.assertEqual(tuple(first['camera0'].rgb.shape), (4, 128, 128, 3))
                self.assertEqual(read.call_count, 1)
                self.assertIs(handler.get_camera_states(), first)
                self.assertEqual(read.call_count, 1)

                renderer = camera._shared_metadata.renderer
                robot = handler.object_inst_dict[handler.robot.name]
                start, count = robot._vgeom_start, robot.n_vgeoms
                geometry_before = renderer._geom_pos_tensor_cuda[start:start + count].clone()
                camera_before = camera._attached_pos[3].clone()
                timestep = handler.scene_inst.t
                delta = torch.tensor([0.25, 0., 0.], device=handler.device)
                robot.set_pos(robot.get_pos(envs_idx=[3]) + delta, envs_idx=[3])
                handler._state_cache_expire = True
                handler.get_states()
                self.assertEqual(read.call_count, 1)
                second = handler.get_camera_states()
                self.assertIsNot(second, first)
                self.assertEqual(read.call_count, 2)
                self.assertEqual(handler.scene_inst.t, timestep)
                torch.testing.assert_close(camera._attached_pos[3], camera_before + delta)
                geometry_after = renderer._geom_pos_tensor_cuda[start:start + count].clone()
                torch.testing.assert_close(geometry_after, geometry_before + delta, atol=2e-5, rtol=1e-5)

                # Compare actual Nyx input buffers against its original scalar
                # pose path on the exact same, non-advancing scene state.
                rotations = renderer._cam_rot_tensor_cuda.clone()
                positions = renderer._cam_pos_tensor_cuda.clone()
                type(camera)._metasim_original_render_current_state(camera)
                torch.testing.assert_close(renderer._cam_rot_tensor_cuda, rotations)
                torch.testing.assert_close(renderer._cam_pos_tensor_cuda, positions)
                torch.testing.assert_close(renderer._geom_pos_tensor_cuda[start:start + count], geometry_after)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
