import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_2 import Stage0ArmPos
from metasim.utils import chairman2_geometry as geometry


def make_states(arm_offset):
    names = list(geometry.STAGE0_JOINT_TARGETS)
    targets = torch.tensor(list(geometry.STAGE0_JOINT_TARGETS.values()))
    positions = targets.unsqueeze(0).clone()
    arm_mask = torch.tensor([not name.startswith("waist_") for name in names])
    offset = torch.as_tensor(arm_offset, dtype=positions.dtype)
    positions[:, arm_mask] += offset
    robot = SimpleNamespace(joint_names=names, joint_pos=positions)
    return SimpleNamespace(robots={"g1_without_hands": robot})


class Stage0ArmPosRewardTest(unittest.TestCase):
    def evaluate_first_step(self, offset):
        reward = Stage0ArmPos()
        reward.actual_stage = torch.zeros(1, dtype=torch.long)
        return reward(make_states(offset), "g1_without_hands").item()

    def test_pose_bonus_orders_quality_and_is_positive_at_target(self):
        exact = self.evaluate_first_step(0.0)
        near = self.evaluate_first_step(geometry.JOINT_TOLERANCE)
        far = self.evaluate_first_step(0.8)

        self.assertGreater(exact, near)
        self.assertGreater(near, far)
        self.assertAlmostEqual(exact, 1.0, places=6)
        self.assertAlmostEqual(near, torch.exp(torch.tensor(-3.0)).item(), places=6)
        self.assertEqual(far, 0.0)

    def test_regress_is_twenty_five_percent_stronger_than_progress(self):
        toward = Stage0ArmPos()
        toward.actual_stage = torch.zeros(1, dtype=torch.long)
        toward(make_states(0.45), "g1_without_hands")
        positive = toward(make_states(0.30), "g1_without_hands").item()

        away = Stage0ArmPos()
        away.actual_stage = torch.zeros(1, dtype=torch.long)
        away(make_states(0.30), "g1_without_hands")
        negative = away(make_states(0.45), "g1_without_hands").item()

        self.assertAlmostEqual(positive, 1.0, places=6)
        self.assertAlmostEqual(negative, -1.25, places=6)

    def test_each_joint_contributes_its_own_pose_bonus(self):
        offsets = torch.full((10,), 0.8)
        offsets[0] = 0.0

        one_exact = self.evaluate_first_step(offsets)

        self.assertAlmostEqual(one_exact, 0.1, places=6)

    def test_waist_is_not_assigned_to_arm_policy(self):
        reward = Stage0ArmPos()

        self.assertEqual(len(reward.joint_targets), 10)
        self.assertFalse(any(name.startswith("waist_") for name in reward.joint_targets))

    def test_reward_is_zero_outside_stage_zero(self):
        reward = Stage0ArmPos()
        reward.actual_stage = torch.ones(1, dtype=torch.long)

        value = reward(make_states(0.8), "g1_without_hands")

        self.assertTrue(torch.equal(value, torch.zeros_like(value)))


if __name__ == "__main__":
    unittest.main()
