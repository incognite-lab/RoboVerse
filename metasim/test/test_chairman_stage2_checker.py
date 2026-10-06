import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.checkers.stages_chairman import (
    DISTANCE_TO_CHAIR_HANDLE_THRESHOLD,
    STAGE2_FINGER_JOINT_TOLERANCE,
    stage2_grasp_pose_status,
)
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    Stage2FingerJointPositionReward,
)
from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS


ROBOT_NAME = "g1_with_hands"


def _states():
    num_envs = 4
    joint_names = list(STAGE2_FINGER_JOINT_TARGETS)
    joint_targets = torch.tensor(
        [STAGE2_FINGER_JOINT_TARGETS[name] for name in joint_names]
    )
    joint_pos = joint_targets.unsqueeze(0).repeat(num_envs, 1)
    joint_pos[3, 0] += STAGE2_FINGER_JOINT_TOLERANCE + 0.01

    robot_body = torch.zeros((num_envs, 2, 13))
    chair_body = torch.zeros((num_envs, 2, 13))
    robot_body[:, :, 3] = 1.0
    chair_body[:, :, 3] = 1.0
    robot_body[1, 0, 0] = DISTANCE_TO_CHAIR_HANDLE_THRESHOLD + 0.01
    robot_body[2, 1, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    robot = SimpleNamespace(
        body_names=["endeffector", "left_endeffector"],
        body_state=robot_body,
        joint_names=joint_names,
        joint_pos=joint_pos,
    )
    chair = SimpleNamespace(
        body_names=["target_hand_right", "target_hand_left"],
        body_state=chair_body,
    )
    return SimpleNamespace(robots={ROBOT_NAME: robot}, objects={"chair": chair})


class ChairmanStage2CheckerTest(unittest.TestCase):
    def test_pose_conditions_cover_position_orientation_and_all_finger_joints(self):
        status = stage2_grasp_pose_status(
            _states(), ROBOT_NAME, torch.arange(4, dtype=torch.long)
        )

        self.assertTrue(
            torch.equal(
                status["hands_near"],
                torch.tensor([True, False, True, True]),
            )
        )
        self.assertTrue(
            torch.equal(
                status["orientations_correct"],
                torch.tensor([True, True, False, True]),
            )
        )
        self.assertTrue(
            torch.equal(
                status["fingers_correct"],
                torch.tensor([True, True, True, False]),
            )
        )

    def test_checker_and_reward_share_exact_finger_targets(self):
        reward = Stage2FingerJointPositionReward()
        self.assertEqual(reward.finger_targets, STAGE2_FINGER_JOINT_TARGETS)


if __name__ == "__main__":
    unittest.main()
