import math
import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg,
    Stage0ReferenceVelocityReward,
    Stage1HandDistanceReward,
    Stage1HandOrientationReward,
    Stage1JointVelocityPenalty,
)


ROBOT_NAME = "g1_with_hands"


def _states(num_envs: int):
    robot_body = torch.zeros((num_envs, 3, 13), dtype=torch.float32)
    chair_body = torch.zeros((num_envs, 3, 13), dtype=torch.float32)
    robot_body[:, :, 3] = 1.0
    chair_body[:, :, 3] = 1.0
    robot = SimpleNamespace(
        body_names=["pelvis", "left_endeffector", "endeffector"],
        body_state=robot_body,
        joint_names=[
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_hip_pitch_joint",
            "left_hand_thumb_0_joint",
        ],
        joint_pos=torch.zeros((num_envs, 4), dtype=torch.float32),
        joint_vel=torch.zeros((num_envs, 4), dtype=torch.float32),
    )
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=chair_body,
    )
    return SimpleNamespace(
        robots={ROBOT_NAME: robot}, objects={"chair": chair}
    )


class ChairmanMultiSimpleRewardsTest(unittest.TestCase):
    def test_stage0_tracks_reference_velocity_and_slows_in_last_ten_cm(self):
        states = _states(4)
        pelvis = states.robots[ROBOT_NAME].body_state[:, 0]

        # With an identity chair quaternion the target is at y=0.77 m.
        pelvis[:, 1] = torch.tensor([0.0, 0.72, 0.77, 0.0])
        pelvis[:, 7:9] = torch.tensor(
            [[0.0, 0.5], [0.0, 0.25], [0.0, 0.0], [0.0, 0.0]]
        )

        reward = Stage0ReferenceVelocityReward()
        reward.actual_stage = torch.zeros(4, dtype=torch.long)
        values = reward(states, ROBOT_NAME)

        self.assertTrue(torch.allclose(values[:3], torch.ones(3), atol=1e-6))
        self.assertLess(values[3].item(), 0.01)

    def test_stage1_joint_speed_has_a_free_threshold(self):
        states = _states(4)
        states.robots[ROBOT_NAME].joint_vel[:] = torch.tensor(
            [
                [1.5, 1.0, 100.0, 100.0],
                [2.25, 1.0, 0.0, 0.0],
                [3.5, 0.0, 0.0, 0.0],
                [3.5, 0.0, 0.0, 0.0],
            ]
        )
        penalty = Stage1JointVelocityPenalty()
        penalty.actual_stage = torch.tensor([1, 1, 1, 0])

        values = penalty(states, ROBOT_NAME)

        self.assertTrue(
            torch.allclose(values, torch.tensor([0.0, 0.5, 1.0, 0.0]))
        )

    def test_stage1_distance_uses_farther_hand_exponentially(self):
        states = _states(3)
        states.robots[ROBOT_NAME].body_state[1, 1, 0] = 0.20
        reward = Stage1HandDistanceReward(distance_scale=0.20)
        reward.actual_stage = torch.tensor([1, 1, 0])

        values = reward(states, ROBOT_NAME)

        expected = torch.tensor([1.0, math.exp(-1.0), 0.0])
        self.assertTrue(torch.allclose(values, expected, atol=1e-6))

    def test_stage1_orientation_is_pure_exponential_state_reward(self):
        states = _states(3)
        # Orthogonal unit quaternions produce checker error 1 - |dot| = 1.
        states.robots[ROBOT_NAME].body_state[1, 1, 3:7] = torch.tensor(
            [0.0, 1.0, 0.0, 0.0]
        )
        reward = Stage1HandOrientationReward(error_scale=0.05)
        reward.actual_stage = torch.tensor([1, 1, 0])

        first = reward(states, ROBOT_NAME)
        second = reward(states, ROBOT_NAME)

        expected = torch.tensor([1.0, math.exp(-20.0), 0.0])
        self.assertTrue(torch.allclose(first, expected, atol=1e-7))
        self.assertTrue(torch.equal(first, second))

    def test_old_stage0_and_stage1_shaping_is_not_active(self):
        cfg = ChairmanmultiCfg()
        names = {type(reward).__name__ for reward in cfg.reward_functions}
        self.assertFalse(
            names
            & {
                "WalkToChairProgressReward",
                "KeepChairStillPenalty",
                "OpenGraspReward",
                "Stage1ArmJointVelocityPenalty",
                "ReachChairProgressReward",
                "HandOrientationProgressReward",
                "HandTargetStillnessReward",
                "PreciseHandTargetReward",
            }
        )


if __name__ == "__main__":
    unittest.main()
