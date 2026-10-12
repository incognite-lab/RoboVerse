import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.checkers.stages_chairman import (
    _cached_stage0_batch,
    _randomize_stage0_batch,
)


class ChairmanStage0ResetRandomizationTest(unittest.TestCase):
    @staticmethod
    def _handler():
        joints = [
            SimpleNamespace(name="root_joint"),
            SimpleNamespace(name="floor_slide_x"),
            SimpleNamespace(name="floor_slide_y"),
            SimpleNamespace(name="floor_rotate_z"),
        ]
        return SimpleNamespace(
            robot=SimpleNamespace(name="g1_with_hands"),
            object_inst_dict={"chair": SimpleNamespace(joints=joints)},
        )

    @staticmethod
    def _batch(count=128):
        return {
            "g1_with_hands": {
                "pos": torch.zeros(count, 3),
                "qpos": torch.zeros(count, 41),
            },
            "chair": {
                "joint_pos": torch.zeros(count, 3),
                "qpos": torch.zeros(count, 3),
            },
        }

    def test_randomizes_each_row_in_stage0_ranges(self):
        batch = self._batch()
        _randomize_stage0_batch(self._handler(), batch)

        robot = batch["g1_with_hands"]
        chair = batch["chair"]
        self.assertTrue(
            torch.all((-4.0 <= robot["pos"][:, 0]) & (robot["pos"][:, 0] <= -2.0))
        )
        self.assertTrue(
            torch.all((-1.2 <= robot["pos"][:, 1]) & (robot["pos"][:, 1] <= 1.2))
        )
        self.assertTrue(torch.equal(robot["pos"][:, :2], robot["qpos"][:, :2]))
        self.assertTrue(
            torch.all(
                (0.5 <= chair["joint_pos"][:, 0])
                & (chair["joint_pos"][:, 0] <= 1.0)
            )
        )
        self.assertTrue(
            torch.all(
                (-0.2 <= chair["joint_pos"][:, 1])
                & (chair["joint_pos"][:, 1] <= 0.2)
            )
        )
        self.assertTrue(
            torch.all(
                (1.27 <= chair["joint_pos"][:, 2])
                & (chair["joint_pos"][:, 2] <= 1.87)
            )
        )
        self.assertTrue(torch.equal(chair["joint_pos"], chair["qpos"]))
        self.assertGreater(torch.unique(robot["pos"][:, 0]).numel(), 1)
        self.assertGreater(torch.unique(chair["joint_pos"][:, 0]).numel(), 1)

    def test_cached_template_draws_new_values_on_every_reset(self):
        handler = self._handler()
        handler.device = torch.device("cpu")
        handler.task = SimpleNamespace()
        handler.pack_state_batch = lambda _states: self._batch(count=1)
        env_ids = torch.arange(128)

        first = _cached_stage0_batch(handler, env_ids)
        second = _cached_stage0_batch(handler, env_ids)

        self.assertFalse(
            torch.equal(
                first["g1_with_hands"]["pos"], second["g1_with_hands"]["pos"]
            )
        )
        self.assertFalse(
            torch.equal(
                first["chair"]["joint_pos"], second["chair"]["joint_pos"]
            )
        )


if __name__ == "__main__":
    unittest.main()
