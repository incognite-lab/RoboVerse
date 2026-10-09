import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_separate import (
    ChairmanseparateCfg,
    LeftReleaseFingersReward,
    LeftStage3HandDriftPenalty,
    PulledChairStillnessReward,
)


ROBOT_NAME = "g1_with_hands"
LEFT_FINGER_TARGETS = {
    "left_hand_thumb_0_joint": 0.396,
    "left_hand_thumb_1_joint": 0.700,
    "left_hand_thumb_2_joint": 1.000,
    "left_hand_middle_0_joint": -1.500,
    "left_hand_middle_1_joint": -1.700,
    "left_hand_index_0_joint": -1.500,
    "left_hand_index_1_joint": -1.700,
}


def _states(num_envs=3):
    robot_body = torch.zeros((num_envs, 3, 13), dtype=torch.float32)
    chair_body = torch.zeros((num_envs, 3, 13), dtype=torch.float32)
    robot_body[:, :, 3] = 1.0
    chair_body[:, :, 3] = 1.0
    joint_names = list(LEFT_FINGER_TARGETS)
    positions = torch.tensor(
        list(LEFT_FINGER_TARGETS.values()), dtype=torch.float32
    ).repeat(num_envs, 1)
    robot = SimpleNamespace(
        body_names=["pelvis", "left_endeffector", "endeffector"],
        body_state=robot_body,
        joint_names=joint_names,
        joint_pos=positions,
        joint_vel=torch.zeros_like(positions),
    )
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=chair_body,
    )
    return SimpleNamespace(robots={ROBOT_NAME: robot}, objects={"chair": chair})


class ChairmanSeparateStage34RewardsTest(unittest.TestCase):
    def test_stage3_hand_drift_is_side_specific_and_stage_masked(self):
        states = _states()
        states.robots[ROBOT_NAME].body_state[:, 1, 0] = 0.11
        reward = LeftStage3HandDriftPenalty()
        reward.actual_stage = torch.tensor([2, 3, 4])

        values = reward(states, ROBOT_NAME)

        self.assertTrue(torch.equal(values, torch.tensor([0.0, 1.0, 0.0])))

    def test_stage4_release_updates_only_its_own_hand(self):
        states = _states()
        reward = LeftReleaseFingersReward()
        reward.actual_stage = torch.tensor([3, 4, 5])
        reward(states, ROBOT_NAME)  # Initialize per-environment history.
        states.robots[ROBOT_NAME].joint_pos.zero_()

        values = reward(states, ROBOT_NAME)

        self.assertEqual(values[0].item(), 0.0)
        self.assertGreater(values[1].item(), 0.7)
        self.assertEqual(values[2].item(), 0.0)

    def test_chair_stillness_penalty_is_active_only_after_pull(self):
        states = _states()
        chair = states.objects["chair"].body_state[:, 0]
        chair[:, :3] = torch.tensor([0.15, 0.0, 0.1])
        reward = PulledChairStillnessReward()
        reward.actual_stage = torch.tensor([3, 4, 5])

        values = reward(states, ROBOT_NAME)

        self.assertEqual(values[0].item(), 0.0)
        self.assertGreater(values[1].item(), 0.9)
        self.assertGreater(values[2].item(), 0.9)

    def test_task_registers_stage3_and_stage4_raw_terms(self):
        cfg = ChairmanseparateCfg()
        names = {type(reward).__name__ for reward in cfg.reward_functions}
        expected = {
            "PullChairReward",
            "PulledChairStillnessReward",
            "LeftStage3HandDriftPenalty",
            "RightStage3HandDriftPenalty",
            "LeftReleaseFingersReward",
            "RightReleaseFingersReward",
        }
        self.assertTrue(expected.issubset(names))
        self.assertEqual(len(cfg.reward_functions), len(cfg.reward_weights))


if __name__ == "__main__":
    unittest.main()
