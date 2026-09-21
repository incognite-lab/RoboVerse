import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg,
    Stage0ArmPos,
    Stage2UpperBodyPoseRetentionReward,
    UpperBodyCenterOfMassPenalty,
)
from metasim.utils import chairman2_geometry as geometry


class ChairmanMultiCenterOfMassRewardTest(unittest.TestCase):
    def test_penalty_uses_horizontal_upper_com_error(self) -> None:
        inertial = geometry.upper_body_inertials(str(geometry.G1_WITH_HANDS_URDF))
        torso_offset = torch.tensor(inertial["torso_link"][1])
        desired_error = torch.tensor([0.0, 0.05, 0.20])
        body_state = torch.zeros((3, 2, 13))
        body_state[..., 3] = 1.0
        body_state[:, 1, 0] = desired_error - torso_offset[0]
        body_state[:, 1, 1] = -torso_offset[1]
        robot = SimpleNamespace(
            body_names=["pelvis", "torso_link"],
            body_state=body_state,
        )
        states = SimpleNamespace(robots={"g1_with_hands": robot})

        values = UpperBodyCenterOfMassPenalty()(states, "g1_with_hands")

        self.assertTrue(torch.allclose(values[:2], torch.zeros(2), atol=1e-6))
        self.assertGreater(values[2].item(), 0.0)
        self.assertLessEqual(values[2].item(), 1.0)

    def test_com_penalty_is_global_and_waist_shaping_is_absent(self) -> None:
        cfg = ChairmanmultiCfg()
        names = [type(reward).__name__ for reward in cfg.reward_functions]
        index = names.index("UpperBodyCenterOfMassPenalty")

        self.assertEqual(cfg.reward_weights[index], -0.2)
        self.assertNotIn("UprightPenaltyCfg", names)
        self.assertNotIn("WaistStraightReward", names)
        for overrides in cfg.stage_reward_weights.values():
            self.assertNotIn("UprightPenaltyCfg", overrides)
            self.assertNotIn("WaistStraightReward", overrides)
        self.assertFalse(any(name.startswith("waist_") for name in Stage0ArmPos().required_pos))
        self.assertFalse(
            any(name.startswith("waist_") for name in Stage2UpperBodyPoseRetentionReward().joint_names)
        )


if __name__ == "__main__":
    unittest.main()
