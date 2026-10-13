import unittest
from types import SimpleNamespace

import torch

from config_run.SB3_chairman_multi_absolute_env import (
    StableBaseline3VecEnv as AbsoluteObservationEnv,
)
from config_run.SB3_chairman_multi_relative_env import (
    StableBaseline3VecEnv as RelativeObservationEnv,
)
from metasim.cfg.checkers.stages_chairman import chairman_random_generator


class ChairmanMultiObservationVariantsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pelvis_pos = torch.tensor([[1.0, 2.0, 0.8]])
        self.pelvis_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.pelvis_velocity = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        self.chair_pos = torch.tensor([[3.0, 5.0, 0.1]])
        # Deliberately use the equivalent negative quaternion to verify that
        # observations use one continuous/canonical representation.
        self.chair_quat = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        self.args = (
            self.pelvis_pos,
            self.pelvis_quat,
            self.pelvis_velocity,
            self.chair_pos,
            self.chair_quat,
        )

    def test_relative_layout(self) -> None:
        env = RelativeObservationEnv.__new__(RelativeObservationEnv)
        observation = env._task_observation_torch(*self.args)

        self.assertEqual(tuple(observation.shape), (1, 13))
        self.assertTrue(
            torch.allclose(
                observation[0, 6:9], torch.tensor([2.0, 3.0, -0.7])
            )
        )
        self.assertTrue(
            torch.allclose(
                observation[0, 9:13], torch.tensor([1.0, 0.0, 0.0, 0.0])
            )
        )

    def test_absolute_layout(self) -> None:
        env = AbsoluteObservationEnv.__new__(AbsoluteObservationEnv)
        observation = env._task_observation_torch(*self.args)

        self.assertEqual(tuple(observation.shape), (1, 20))
        self.assertTrue(torch.equal(observation[0, :3], self.pelvis_pos[0]))
        self.assertTrue(torch.equal(observation[0, 13:16], self.chair_pos[0]))
        self.assertTrue(
            torch.equal(
                observation[0, 16:20], torch.tensor([1.0, 0.0, 0.0, 0.0])
            )
        )

    def test_environment_randomization_rng_is_isolated_from_policy_rng(self) -> None:
        first_handler = SimpleNamespace(
            device="cpu", task=SimpleNamespace(randomization_seed=42)
        )
        second_handler = SimpleNamespace(
            device="cpu", task=SimpleNamespace(randomization_seed=42)
        )
        first = torch.rand(8, generator=chairman_random_generator(first_handler))

        # Simulate arbitrary policy initialization/training RNG consumption.
        torch.manual_seed(9876)
        torch.rand(10_000)

        second = torch.rand(8, generator=chairman_random_generator(second_handler))
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
