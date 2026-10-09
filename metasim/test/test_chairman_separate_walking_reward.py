import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_separate import (
    ApproachAndStandReward,
    Stage0ReferenceVelocityReward,
)
from metasim.utils.chair_navigation import CHAIR_FINAL_DISTANCE


ROBOT_NAME = "g1_with_hands"


def _states(num_envs: int):
    robot_body = torch.zeros((num_envs, 1, 13), dtype=torch.float32)
    chair_body = torch.zeros((num_envs, 1, 13), dtype=torch.float32)
    robot_body[:, :, 3] = 1.0
    chair_body[:, :, 3] = 1.0
    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((num_envs, 1), dtype=torch.float32),
    )
    chair = SimpleNamespace(
        body_names=["base_link"],
        body_state=chair_body,
    )
    return SimpleNamespace(robots={ROBOT_NAME: robot}, objects={"chair": chair})


class ChairmanSeparateWalkingRewardTest(unittest.TestCase):
    def test_motion_toward_target_is_rewarded_and_away_is_penalized(self):
        states = _states(3)
        pelvis = states.robots[ROBOT_NAME].body_state[:, 0]

        # Identity chair rotation puts the target on the positive Y axis.
        pelvis[:, 1] = 0.0
        pelvis[:, 7:9] = torch.tensor(
            [[0.0, 0.5], [0.0, 0.0], [0.0, -0.5]]
        )

        reward = Stage0ReferenceVelocityReward()
        reward.actual_stage = torch.zeros(3, dtype=torch.long)
        values = reward(states, ROBOT_NAME)

        self.assertAlmostEqual(values[0].item(), 1.0, places=6)
        self.assertGreater(values[1].item(), 0.0)
        self.assertLess(values[2].item(), 0.0)

    def test_standing_at_target_receives_maximum_reward(self):
        states = _states(1)
        pelvis = states.robots[ROBOT_NAME].body_state[:, 0]
        pelvis[:, 1] = CHAIR_FINAL_DISTANCE
        pelvis[:, 7:9] = 0.0

        reward = Stage0ReferenceVelocityReward()
        reward.actual_stage = torch.zeros(1, dtype=torch.long)

        self.assertAlmostEqual(reward(states, ROBOT_NAME).item(), 1.0, places=6)

    def test_approach_reward_uses_same_signal_in_stages_zero_to_two(self):
        states = _states(4)
        pelvis = states.robots[ROBOT_NAME].body_state[:, 0]
        pelvis[:, 7:9] = torch.tensor([[0.0, -0.5]] * 4)

        reward = ApproachAndStandReward()
        reward.actual_stage = torch.tensor([0, 1, 2, 3])
        values = reward(states, ROBOT_NAME)

        self.assertTrue(torch.all(values[:3] < 0.0))
        self.assertEqual(values[3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
