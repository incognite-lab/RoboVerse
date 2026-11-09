"""Marker-based checks of temporal alignment, including collection and ring wrap.

Run: python -m unittest config_run.dagger_recurrent.test_sequence_integrity
Uses synthetic simulator states; does not require Genesis or a GPU.
"""
import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import runner
from .buffer import SequenceBuffer, train_sequence_step
from .student_net import RecurrentVisionStudent
from .test_pipeline import FakeEnvironment, TinyStudent


class SequenceIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_every_window_and_environment_before_and_after_multiple_wraps(self):
        capacity, env_count, length, burn = 29, 4, 16, 8
        replay = SequenceBuffer(capacity, env_count, (3, 1, 1), 43, (2, 3))
        for step in range(97):
            envs = torch.arange(env_count)
            marker = step * 10 + envs.float()
            image = torch.stack((torch.full_like(envs, step), envs, (step + envs) % 256), 1)
            values = (image.to(torch.uint8).reshape(env_count, 3, 1, 1),
                      marker[:, None].expand(-1, 43), (marker + 1000)[:, None].expand(-1, 43),
                      (marker + 2000)[:, None].expand(-1, 6),
                      (marker - 10)[:, None].expand(-1, 34))
            hidden = torch.stack((marker, marker + 3000)).unsqueeze(-1).expand(-1, -1, 3)
            resets = (step + envs) % 11 == 0
            replay.add(values, (marker + 4000)[:, None].expand(-1, 34), hidden, resets)
            # Mutating producer memory must not change already stored samples.
            values[0].zero_()
            if replay.filled_steps < length + burn:
                continue
            starts = torch.arange(replay.filled_steps - length - burn + 1).repeat_interleave(env_count)
            chosen_envs = envs.repeat(replay.filled_steps - length - burn + 1)
            with patch.object(torch, "randint", side_effect=[starts, chosen_envs]):
                sampled, target, initial, mask = replay.sample(len(starts), length, burn)
            times = step + 1 - replay.filled_steps + starts[:, None] + torch.arange(length + burn)
            expected = times * 10 + chosen_envs[:, None]
            torch.testing.assert_close(sampled[0][:, :, 0, 0, 0] * 255, times.float())
            torch.testing.assert_close(sampled[0][:, :, 1, 0, 0] * 255,
                                       chosen_envs[:, None].expand_as(times).float())
            torch.testing.assert_close(sampled[0][:, :, 2, 0, 0] * 255,
                                       ((times + chosen_envs[:, None]) % 256).float())
            for value, offset in zip(sampled[1:], (0, 1000, 2000, -10)):
                torch.testing.assert_close(value, (expected + offset).float().unsqueeze(-1).expand_as(value))
            torch.testing.assert_close(target, (expected + 4000).float().unsqueeze(-1).expand_as(target))
            for layer in range(2):
                torch.testing.assert_close(initial[layer],
                                           (expected[:, :1] + layer * 3000).float().expand(-1, 3))
            self.assertTrue(torch.equal(mask, (times + chosen_envs[:, None]) % 11 == 0))

    def test_burn_in_and_visual_chunks_match_step_by_step_rollout(self):
        torch.manual_seed(91)
        model = RecurrentVisionStudent(hidden_size=12, visual_batch_size=5, checkpoint_visual=False).train()
        batch, steps, burn = 3, 24, 8
        values = (torch.rand(batch, steps, 3, 32, 32), torch.randn(batch, steps, 43),
                  torch.randn(batch, steps, 43), torch.randn(batch, steps, 6),
                  torch.randn(batch, steps, 34))
        resets = torch.zeros(batch, steps, dtype=torch.bool)
        resets[0, [0, 9, 20]] = True
        resets[1, [7, 8, 19]] = True  # immediately before and after the burn-in boundary
        resets[2, [3, 15]] = True
        initial = torch.randn(1, batch, 12)
        with torch.no_grad():
            streamed, hidden = [], initial.clone()
            for step in range(steps):
                action, hidden = model(*(value[:, step] for value in values),
                                       hidden=hidden, reset_mask=resets[:, step])
                streamed.append(action)
            expected = torch.stack(streamed, 1)
            whole, final = model(*values, hidden=initial, reset_mask=resets)
            torch.testing.assert_close(whole, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(final, hidden, atol=1e-6, rtol=1e-5)
            # Reordering independent sequences must only reorder their outputs.
            order = torch.tensor([2, 0, 1])
            reordered, _ = model(*(value[order] for value in values), hidden=initial[:, order],
                                 reset_mask=resets[order])
            torch.testing.assert_close(reordered, expected[order], atol=1e-6, rtol=1e-5)
        # Exercise the actual learner: exact chronological labels imply zero loss
        # after burn-in as well. Detects any shifted target or discarded burn-in state.
        replay = SimpleNamespace(sample=lambda *args: (values, expected, initial, resets))
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        loss = train_sequence_step(model, optimizer, replay, batch, steps - burn, burn)
        self.assertLess(loss, 1e-12)

    def test_nyx_batch_copies_each_environment_frame_before_renderer_reuses_output(self):
        # Importing metasim.sim.genesis initializes a GPU at module scope.
        # Compile the production batching function without that package side
        # effect; geometry conversion and the Nyx backend are outside this test.
        source = Path(__file__).resolve().parents[2] / "metasim/sim/genesis/gaussian_splat.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_render_nyx_camera_batch")
        namespace = dict(torch=torch, gu=SimpleNamespace(
            pos_lookat_up_to_T=lambda *args: torch.eye(4).expand(4, 4, 4),
            R_to_quat=lambda matrix: torch.tensor([1., 0., 0., 0.]).expand(4, 4)))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        render_batch = namespace[function.name]

        count = 4

        class Renderer:
            _cam_pos_tensor_cuda = torch.zeros(2, 3)
            _cam_rot_tensor_cuda = torch.zeros(2, 4)
            output = torch.zeros(2, 2, 3, dtype=torch.uint8)

            def update(self):
                pass

            def update_scene(renderer, env_idx):
                renderer.env_idx = env_idx
                for camera_idx in range(2):
                    torch.testing.assert_close(renderer._cam_pos_tensor_cuda[camera_idx],
                                               torch.tensor([float(env_idx), float(camera_idx), 0.]))

            def render(renderer, camera_index):
                # Nyx can reuse one output allocation: replay rows must be copies.
                renderer.output.fill_(renderer.env_idx * 10 + camera_index)
                return renderer.output

        scene = SimpleNamespace(t=123, visualizer=SimpleNamespace(update_visual_states=lambda: None))
        manager = SimpleNamespace(_sim=SimpleNamespace(_B=count, scene=scene))
        metadata = SimpleNamespace(renderer=Renderer(), sensors=[],
                                   image_cache={i: torch.empty(count, 2, 2, 3, dtype=torch.uint8)
                                                for i in range(2)})
        for index in range(2):
            metadata.sensors.append(SimpleNamespace(
                _idx=index, _camera_idx=index, _link=None, _stale=True,
                _attached_pos=torch.tensor([[float(env), float(index), 0.] for env in range(count)]),
                _attached_lookat=torch.tensor([[float(env), float(index), 1.] for env in range(count)]),
                _attached_up=torch.tensor([0., 1., 0.]), _manager=manager, _shared_metadata=metadata))
        render_batch(metadata.sensors[0])
        metadata.renderer.output.zero_()
        for camera in metadata.sensors:
            self.assertFalse(camera._stale)
            for env in range(count):
                torch.testing.assert_close(metadata.image_cache[camera._idx][env],
                                           torch.full((2, 2, 3), env * 10 + camera._idx, dtype=torch.uint8))
        self.assertEqual(metadata.last_render_timestep, scene.t)

    def test_collector_aligns_camera_teacher_previous_action_and_pre_step_memory(self):
        class MarkedEnvironment(FakeEnvironment):
            def mark(self):
                markers = torch.tensor([self.steps * 10, self.steps * 10 + 1], dtype=torch.float32)
                self.robot.joint_pos[:] = markers[:, None]
                self.robot.joint_vel[:] = (markers + 100)[:, None]
                self.robot.body_state[:, 1, 10] = markers + 200
                self.states.cameras["camera0"].rgb[:] = markers.to(torch.uint8)[:, None, None, None]
                return markers[:, None].expand(-1, 43).numpy().copy()

            def reset(self):
                super().reset()
                return self.mark()

            def step(self, actions):
                _, rewards, dones, infos = super().step(actions)
                return self.mark(), rewards, dones, infos

        # Both teacher-controlled and student-controlled collection are audited.
        for beta in (0., 1.):
            with self.subTest(beta=beta), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                env = MarkedEnvironment()
                collected, inferred = [], []

                class InspectStudent(TinyStudent):
                    def forward(self, *values, hidden=None, reset_mask=None):
                        action, next_hidden = super().forward(*values, hidden=hidden, reset_mask=reset_mask)
                        inferred.append((hidden.clone(), next_hidden.clone(), reset_mask.clone()))
                        return action, next_hidden

                class InspectReplay(SequenceBuffer):
                    def add(self, values, expert, hidden, reset):
                        collected.append((tuple(value.clone() for value in values), expert.clone(),
                                          hidden.clone(), reset.clone()))
                        return super().add(values, expert, hidden, reset)

                teacher = SimpleNamespace(predict=lambda obs, deterministic: np.repeat(obs[:, :1], 34, axis=1))
                stack.enter_context(patch.object(runner, "_device", return_value="cpu"))
                stack.enter_context(patch.object(runner, "load_policy_router", return_value=(teacher, {})))
                stack.enter_context(patch.object(runner, "RecurrentVisionStudent", InspectStudent))
                stack.enter_context(patch.object(runner, "SequenceBuffer", InspectReplay))
                # Nontrivial, persistent selection: slot zero corresponds to env one.
                selection = stack.enter_context(patch.object(torch, "randperm", return_value=torch.tensor([1, 0])))
                runner.train_dagger(dict(expert_model_path="fake", model_save_path=temporary,
                                         tensorboard_log=str(Path(temporary) / "logs"), total_timesteps=12,
                                         dagger_image_width=32, dagger_image_height=32,
                                         dagger_buffer_steps=5, dagger_store_per_step=1,
                                         dagger_new_sequence_length=2, dagger_new_burn_in=1,
                                         dagger_new_hidden_size=8, dagger_train_every=100,
                                         dagger_log_every=100, beta_start=beta, beta_end=beta), env.base, env)
                self.assertEqual(selection.call_count, 1)
                self.assertEqual(len(collected), 12)
                for step, (values, expert, hidden, reset) in enumerate(collected):
                    marker = float(step * 10 + 1)
                    torch.testing.assert_close(values[0], torch.full_like(values[0], int(marker)))
                    torch.testing.assert_close(values[1], torch.full_like(values[1], marker))
                    torch.testing.assert_close(values[2], torch.full_like(values[2], marker + 100))
                    self.assertEqual(values[3][0, 0].item(), marker + 200)
                    torch.testing.assert_close(expert, torch.full_like(expert, marker))
                    done_before = step == 0 or step % 3 == 0  # selected env one
                    self.assertEqual(reset.tolist(), [done_before])
                    expected_hidden = torch.zeros_like(hidden) if done_before else inferred[step - 1][1][:, 1:2]
                    torch.testing.assert_close(hidden, expected_hidden)
                    torch.testing.assert_close(hidden, inferred[step][0][:, 1:2])
                    previous = np.zeros((1, 34), np.float32) if done_before else env.actions[step - 1][1:2]
                    torch.testing.assert_close(values[4], torch.as_tensor(previous))


if __name__ == "__main__":
    unittest.main()
