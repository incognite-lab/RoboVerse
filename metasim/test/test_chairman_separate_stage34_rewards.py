import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from metasim.cfg.checkers import stages_chairman as checker
from metasim.cfg.tasks.humanoidbench.ChairMan_separate import (
    ArmStageProgressReward,
    ChairmanseparateCfg,
    DirectionStageProgressReward,
    FingerStageProgressReward,
    LeftReleaseFingersReward,
    LeftStage3HandDriftPenalty,
    PullChairReward,
    PulledChairStillnessReward,
    Stage3ReverseVelocityReward,
    STAGE3_PULL_MARGIN,
    STAGE3_REWARD_TARGET_X,
    WaistStageProgressReward,
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
    def test_stage3_stationary_rewards_peak_beyond_checker_boundary(self):
        states = _states(num_envs=4)
        boundary_x = 0.75 - checker.CHAIR_PULL_DISTANCE_THRESHOLD
        chair = states.objects["chair"].body_state[:, 0]
        chair[:, 0] = torch.tensor([
            boundary_x + 0.001, boundary_x,
            STAGE3_REWARD_TARGET_X, STAGE3_REWARD_TARGET_X - 0.05,
        ])
        chair[:, 2] = 0.1

        for reward_type in (Stage3ReverseVelocityReward, PullChairReward):
            reward = reward_type()
            reward.actual_stage = torch.full((4,), 3)
            values = reward(states, ROBOT_NAME)
            self.assertAlmostEqual(values[2].item(), 1.0, places=6)
            self.assertTrue(torch.all(values[[0, 1, 3]] < values[2]))
        self.assertAlmostEqual(boundary_x - STAGE3_REWARD_TARGET_X, STAGE3_PULL_MARGIN)

        # At the checker boundary the robot should still move backward.
        reverse = Stage3ReverseVelocityReward()
        reverse.actual_stage = torch.tensor([2, 3, 3, 4])
        states.robots[ROBOT_NAME].body_state[1, 0, 7] = (
            -reverse.target_speed * STAGE3_PULL_MARGIN / reverse.slowdown_distance
        )
        values = reverse(states, ROBOT_NAME)
        torch.testing.assert_close(values, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_checker_accepts_reward_target_with_position_noise(self):
        states = _states(num_envs=4)
        chair = states.objects["chair"].body_state[:, 0]
        boundary_x = 0.75 - checker.CHAIR_PULL_DISTANCE_THRESHOLD
        chair[:, 0] = torch.tensor([
            boundary_x + 0.001, boundary_x,
            STAGE3_REWARD_TARGET_X + 0.005, STAGE3_REWARD_TARGET_X,
        ])
        chair[:, 2] = 0.1
        handler = SimpleNamespace(
            device=torch.device("cpu"), num_envs=4,
            robot=SimpleNamespace(name=ROBOT_NAME),
            task=SimpleNamespace(failure_masks={}),
            scenario=SimpleNamespace(sim_params=SimpleNamespace(dt=0.02), decimation=1),
        )
        pose = {name: torch.ones(4, dtype=torch.bool) for name in (
            "hands_near", "orientations_correct", "fingers_correct",
            "hands_in_recovery_envelope",
        )}
        mask = torch.ones(4, dtype=torch.bool)
        # Isolate distance/stillness checks from falls and grasp geometry.
        with patch.object(checker, "common_chairman_checker", return_value=~mask), \
             patch.object(checker, "stage2_grasp_pose_status", return_value=pose):
            for _ in range(checker.STAGE3_HOLD_STEPS):
                _, success = checker.stege3_chacker(states, handler, mask)
            self.assertTrue(torch.equal(success, torch.tensor([False, True, True, True])))
            terminated, _ = checker.stege4_chacker(states, handler, mask)
            self.assertFalse(torch.any(terminated))
            states.robots[ROBOT_NAME].joint_pos.zero_()
            for _ in range(5):  # Stage 4 requires 0.10 s with open fingers.
                _, success = checker.stege4_chacker(states, handler, mask)
            self.assertTrue(torch.all(success))

    def test_progress_bonus_changes_only_at_policy_task_milestones(self):
        states = _states(num_envs=6)
        stages = torch.arange(6)
        expected = {
            WaistStageProgressReward: [0, 0, 0, 0, 1, 1],
            ArmStageProgressReward: [0, 1, 1, 1, 1, 1],
            FingerStageProgressReward: [0, 0, 1, 1, 2, 2],
            DirectionStageProgressReward: [0, 0, 0, 1, 2, 2],
        }

        for reward_type, values in expected.items():
            reward = reward_type()
            reward.actual_stage = stages
            self.assertTrue(
                torch.equal(
                    reward(states, ROBOT_NAME),
                    torch.tensor(values, dtype=torch.float32),
                ),
                reward_type.__name__,
            )

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
