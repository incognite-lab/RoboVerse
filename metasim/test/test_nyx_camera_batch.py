"""Batched poses must match Nyx's per-environment transforms and buffer order."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from metasim.test.test_nyx_solver_compat import _adapter


class NyxCameraBatchTest(unittest.TestCase):
    def _exercise(self, batch):
        n = max(batch, 1)
        device = 'cpu'
        pos = torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3) / 10
        lookat = pos + torch.tensor([1., 0.3, -0.2], device=device)
        up = torch.tensor([0., 0., 1.], device=device).expand(n, 3)
        sensors = []
        for idx in range(2):
            sensors.append(SimpleNamespace(
                _idx=idx, _camera_idx=idx, _link=object() if idx == 0 else None,
                _attached_pos=pos if idx == 0 else None,
                _attached_lookat=lookat, _attached_up=up,
                _options=SimpleNamespace(pos=(2., 3., 4.), lookat=(0., 0., 0.), up=(0., 0., 1.)),
                move_to_attach=Mock(), _stale=True,
            ))
        metadata = SimpleNamespace(
            sensors=sensors,
            image_cache={i: torch.empty((n, 2, 2, 3), dtype=torch.uint8, device=device) for i in range(2)},
        )
        events = []
        frame = torch.empty((2, 2, 3), dtype=torch.uint8, device=device)
        renderer = SimpleNamespace(
            _cam_pos_tensor_cuda=torch.empty((2, 3), device=device),
            _cam_rot_tensor_cuda=torch.empty((2, 4), device=device),
            update=lambda: events.append(('update',)),
        )
        def update_scene(env_idx):
            events.append(('scene', env_idx))
            renderer.env_idx = env_idx
        def render(camera_index):
            events.append(('render', renderer.env_idx, camera_index))
            camera = sensors[camera_index]
            values = (pos[renderer.env_idx], lookat[renderer.env_idx], up[renderer.env_idx]) if camera_index == 0 else (
                torch.tensor(camera._options.pos, device=device),
                torch.tensor(camera._options.lookat, device=device),
                torch.tensor(camera._options.up, device=device),
            )
            transform = _adapter.gu.pos_lookat_up_to_T(*values)
            expected_quat = _adapter.gu.R_to_quat(transform[:3, :3])
            torch.testing.assert_close(renderer._cam_pos_tensor_cuda[camera_index], values[0])
            torch.testing.assert_close(renderer._cam_rot_tensor_cuda[camera_index], expected_quat)
            # Reuse one output buffer, as Nyx does. Copies must precede reuse.
            return frame.fill_(renderer.env_idx * 2 + camera_index)
        renderer.update_scene = update_scene
        renderer.render = render
        metadata.renderer = renderer
        scene = SimpleNamespace(t=17, visualizer=SimpleNamespace(update_visual_states=Mock()))
        for camera in sensors:
            camera._shared_metadata = metadata
            camera._manager = SimpleNamespace(_sim=SimpleNamespace(_B=batch, scene=scene))
        with patch.object(_adapter.gs, 'EPS', 1e-12, create=True):
            _adapter._render_nyx_camera_batch(sensors[0])
        expected_events = [('update',)]
        for env in range(n):
            expected_events += [('scene', env), ('render', env, 0), ('render', env, 1)]
            for idx in range(2):
                self.assertTrue(bool((metadata.image_cache[idx][env] == env * 2 + idx).all()))
        self.assertEqual(events, expected_events)
        self.assertEqual(metadata.last_render_timestep, scene.t)
        sensors[0].move_to_attach.assert_called_once_with()
        sensors[1].move_to_attach.assert_not_called()
        self.assertTrue(all(not s._stale for s in sensors))

    def test_attached_and_static_cameras_single_and_multiple_environments(self):
        for batch in (0, 1, 4):
            with self.subTest(batch=batch):
                self._exercise(batch)


if __name__ == '__main__':
    unittest.main()
