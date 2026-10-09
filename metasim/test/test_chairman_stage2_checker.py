import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from metasim.cfg.checkers.stages_chairman import (
    DISTANCE_TO_CHAIR_HANDLE_THRESHOLD,
    STAGE2_FINGER_JOINT_TOLERANCE,
    stage2_grasp_pose_status,
    stege3_chacker,
    stege4_chacker,
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

    robot_body = torch.zeros((num_envs, 3, 13))
    chair_body = torch.zeros((num_envs, 3, 13))
    robot_body[:, :, 3] = 1.0
    chair_body[:, :, 3] = 1.0
    robot_body[1, 0, 0] = DISTANCE_TO_CHAIR_HANDLE_THRESHOLD + 0.01
    robot_body[2, 1, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    chair_body[:, 2, :3] = torch.tensor([-0.25, 0.0, 0.1])

    robot = SimpleNamespace(
        body_names=["endeffector", "left_endeffector", "pelvis"],
        body_state=robot_body,
        joint_names=joint_names,
        joint_pos=joint_pos,
    )
    chair = SimpleNamespace(
        body_names=["target_hand_right", "target_hand_left", "base_link"],
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

    def test_stage3_reuses_pose_goal_without_requiring_contact(self):
        states = _states()
        handler = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=4,
            robot=SimpleNamespace(name=ROBOT_NAME),
            task=SimpleNamespace(failure_masks={}),
            scenario=SimpleNamespace(
                sim_params=SimpleNamespace(dt=0.01), decimation=5
            ),
        )
        mask = torch.ones(4, dtype=torch.bool)

        # Five consecutive valid steps are required. The mock states contain
        # no contact data at all, so this also guards against restoring the old
        # contact-based Stage-3 reset.
        with patch(
            "metasim.cfg.checkers.stages_chairman.common_chairman_checker",
            return_value=torch.zeros(4, dtype=torch.bool),
        ):
            for _ in range(4):
                terminated, success = stege3_chacker(states, handler, mask)
                self.assertFalse(torch.any(terminated))
                self.assertFalse(torch.any(success))
            terminated, success = stege3_chacker(states, handler, mask)

        expected = torch.tensor([True, False, False, False])
        self.assertTrue(torch.equal(success, expected))
        self.assertTrue(torch.equal(terminated, expected))

    def test_stage4_requires_retained_hands_while_opening_fingers(self):
        states = _states()
        states.robots[ROBOT_NAME].joint_pos.zero_()
        states.robots[ROBOT_NAME].joint_pos[3, 0] = 0.16
        handler = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=4,
            robot=SimpleNamespace(name=ROBOT_NAME),
            task=SimpleNamespace(failure_masks={}),
            scenario=SimpleNamespace(
                sim_params=SimpleNamespace(dt=0.01), decimation=5
            ),
        )
        mask = torch.ones(4, dtype=torch.bool)

        with patch(
            "metasim.cfg.checkers.stages_chairman.common_chairman_checker",
            return_value=torch.zeros(4, dtype=torch.bool),
        ):
            terminated, success = stege4_chacker(states, handler, mask)
            self.assertFalse(torch.any(terminated))
            self.assertFalse(torch.any(success))
            terminated, success = stege4_chacker(states, handler, mask)

        expected = torch.tensor([True, False, False, False])
        self.assertTrue(torch.equal(success, expected))
        self.assertTrue(torch.equal(terminated, expected))


if __name__ == "__main__":
    unittest.main()
