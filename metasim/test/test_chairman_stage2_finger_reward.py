import math
import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg,
    Stage2FingerJointPositionReward,
)
from metasim.cfg.tasks.humanoidbench.ChairMan_separate import (
    ChairmanseparateCfg,
    LeftStage2FingerJointPositionReward,
    RightStage2FingerJointPositionReward,
)


ROBOT_NAME = "g1_with_hands"
JOINT_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
TARGETS = torch.tensor(
    [0.0, -0.14, 0.0, 1.16, 0.0, 1.16, 0.0,
     0.0, 0.14, 0.0, -1.16, 0.0, -1.16, 0.0]
)


def _states(positions: torch.Tensor):
    robot = SimpleNamespace(joint_names=JOINT_NAMES, joint_pos=positions)
    return SimpleNamespace(robots={ROBOT_NAME: robot})


class ChairmanStage2FingerRewardTest(unittest.TestCase):
    def test_multi_reward_is_exponential_in_stages_two_and_three(self):
        positions = torch.stack((TARGETS, TARGETS + 0.25, TARGETS))
        reward = Stage2FingerJointPositionReward(error_scale=0.25)
        reward.actual_stage = torch.tensor([2, 2, 3])

        values = reward(_states(positions), ROBOT_NAME)

        expected = torch.tensor([1.0, math.exp(-1.0), 1.0])
        self.assertTrue(torch.allclose(values, expected, atol=1e-6))
        self.assertTrue(torch.all((values >= 0.0) & (values <= 1.0)))

    def test_separate_rewards_score_each_hand_independently(self):
        positions = TARGETS.unsqueeze(0).repeat(2, 1)
        positions[0, 7:] += 0.50
        right_reward = RightStage2FingerJointPositionReward(error_scale=0.25)
        left_reward = LeftStage2FingerJointPositionReward(error_scale=0.25)
        stages = torch.tensor([2, 1])
        right_reward.actual_stage = stages
        left_reward.actual_stage = stages

        states = _states(positions)
        right_values = right_reward(states, ROBOT_NAME)
        left_values = left_reward(states, ROBOT_NAME)

        self.assertTrue(torch.allclose(right_values, torch.tensor([1.0, 0.0])))
        self.assertTrue(
            torch.allclose(
                left_values, torch.tensor([math.exp(-2.0), 0.0]), atol=1e-6
            )
        )

    def test_task_configs_replace_close_and_force_rewards(self):
        multi_cfg = ChairmanmultiCfg()
        multi_names = [type(reward).__name__ for reward in multi_cfg.reward_functions]
        self.assertEqual(len(multi_cfg.reward_weights), len(multi_names))
        self.assertIn("Stage2FingerJointPositionReward", multi_names)
        self.assertNotIn("CloseGraspReward", multi_names)
        self.assertNotIn("GraspForceReward", multi_names)

        separate_cfg = ChairmanseparateCfg()
        separate_names = [
            type(reward).__name__ for reward in separate_cfg.reward_functions
        ]
        self.assertEqual(len(separate_cfg.reward_weights), len(separate_names))
        self.assertIn("LeftStage2FingerJointPositionReward", separate_names)
        self.assertIn("RightStage2FingerJointPositionReward", separate_names)
        self.assertNotIn("LeftCloseGraspReward", separate_names)
        self.assertNotIn("RightCloseGraspReward", separate_names)
        self.assertNotIn("LeftGraspForceReward", separate_names)
        self.assertNotIn("RightGraspForceReward", separate_names)


if __name__ == "__main__":
    unittest.main()
