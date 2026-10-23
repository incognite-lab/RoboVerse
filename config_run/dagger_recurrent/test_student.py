"""CPU tests: python -m unittest config_run.dagger_recurrent.test_student."""

import io
import unittest

import torch

from .student_net import ACTION_GROUP_SIZES, RecurrentVisionStudent


class RecurrentStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = RecurrentVisionStudent(hidden_size=32)

    def inputs(self, batch=2, steps=3):
        return (
            torch.rand(batch, steps, 3, 32, 32),
            torch.randn(batch, steps, 43),
            torch.randn(batch, steps, 43),
            torch.randn(batch, steps, 6),
            torch.randn(batch, steps, 34),
        )

    def test_sequence_matches_streaming_with_partial_episode_resets(self):
        # Also checks that visual batch statistics cannot leak future frames
        # during training. Memory remains explicit and can span caller steps.
        self.model.train()
        values = self.inputs()
        resets = torch.tensor([[True, False, True], [False, True, False]])
        initial = torch.randn(1, 2, 32)
        initial_copy = initial.clone()
        with torch.no_grad():
            sequence, final = self.model(*values, hidden=initial, reset_mask=resets)
            hidden = initial.clone()
            outputs = []
            for step in range(3):
                actions, hidden = self.model(
                    *(value[:, step] for value in values),
                    hidden=hidden, reset_mask=resets[:, step],
                )
                outputs.append(actions)
            torch.testing.assert_close(sequence, torch.stack(outputs, dim=1), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(final, hidden, atol=1e-5, rtol=1e-5)
            # A reset environment must give the same output as a fresh policy.
            fresh, _ = self.model(*(value[0:1, 2] for value in values))
            torch.testing.assert_close(sequence[0:1, 2], fresh, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(initial, initial_copy)

    def test_head_concatenation_matches_environment_action_order(self):
        for index, (name, _) in enumerate(ACTION_GROUP_SIZES):
            last_layer = self.model.action_heads[name][-1]
            torch.nn.init.zeros_(last_layer.weight)
            torch.nn.init.constant_(last_layer.bias, index + 1)
        with torch.no_grad():
            actions, hidden = self.model(*(value[:, 0] for value in self.inputs()))
        self.assertEqual(actions.shape, (2, 34))
        self.assertEqual(hidden.shape, (1, 2, 32))
        self.assertEqual(self.model.proprio_dim, 126)
        expected = torch.tensor([i + 1 for i, (_, size) in enumerate(ACTION_GROUP_SIZES) for _ in range(size)])
        torch.testing.assert_close(actions, expected.float().expand(2, -1))
        self.assertEqual(self.model.action_slices["direction"], slice(31, 34))

    def test_sequence_training_reaches_both_encoders_gru_and_all_heads(self):
        self.model.train()
        actions, _ = self.model(*self.inputs(), reset_mask=torch.zeros(2, 3, dtype=torch.bool))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        before = self.model.gru.weight_ih_l0.detach().clone()
        loss = (actions - torch.randn_like(actions)).square().mean()
        loss.backward()
        for parameter in (
            self.model.visual_encoder.conv1.weight,
            self.model.proprio_encoder[0].weight,
            self.model.gru.weight_ih_l0,
            *(head[-1].weight for head in self.model.action_heads.values()),
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.gru.weight_ih_l0))

    def test_checkpoint_round_trip_preserves_normalization_and_predictions(self):
        self.model.set_proprio_normalization(torch.randn(126), torch.rand(126) + 0.1)
        self.model.eval()
        values = self.inputs(batch=1, steps=1)
        with torch.no_grad():
            expected, _ = self.model(*values)
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = RecurrentVisionStudent(hidden_size=32).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            actual, _ = restored(*values)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(restored.proprio_mean, self.model.proprio_mean)

    def test_rejects_incompatible_sensor_layout_and_mask(self):
        values = self.inputs()
        for index, size in ((1, 31), (2, 31), (3, 3), (4, 17)):
            bad = list(values)
            bad[index] = torch.zeros(2, 3, size)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.model(*bad)
        with self.assertRaises(ValueError):
            self.model(*values, reset_mask=torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            self.model.set_proprio_normalization(torch.zeros(126), torch.zeros(126))


if __name__ == "__main__":
    unittest.main()
