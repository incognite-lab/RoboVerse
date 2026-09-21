import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.checkers.stages_chairman2 import stage1_success, stage2_success
from metasim.cfg.tasks.humanoidbench.ChairMan_2 import (
    ExtendArmsReward,
    MoveHandsBehindBackrestReward,
    UpperBodyCenterOfMassReward,
)
from metasim.utils import chairman2_geometry as geometry


def _common_metrics(num_envs: int = 2) -> dict[str, torch.Tensor]:
    scalar = torch.zeros(num_envs)
    return {
        "robot_drift": scalar.clone(),
        "chair_drift": scalar.clone(),
        "robot_speed": scalar.clone(),
        "robot_yaw_speed": scalar.clone(),
        "chair_speed": scalar.clone(),
        "chair_yaw_speed": scalar.clone(),
        "heading": scalar.clone(),
        "chair_yaw": scalar.clone(),
        "arm_speed": scalar.clone(),
    }


class Chairman2TaskSpaceStagesTest(unittest.TestCase):
    def test_upper_body_inertials_are_the_torso_subtree_only(self) -> None:
        inertials = geometry.upper_body_inertials()
        self.assertIn("torso_link", inertials)
        self.assertIn("head_link", inertials)
        self.assertIn("left_hand_index_1_link", inertials)
        self.assertIn("right_hand_thumb_2_link", inertials)
        self.assertNotIn("pelvis", inertials)
        self.assertNotIn("left_hip_pitch_link", inertials)
        self.assertGreater(sum(mass for mass, _ in inertials.values()), 0.0)

    def test_upper_body_com_reward_prefers_projection_above_pelvis(self) -> None:
        robot = SimpleNamespace(joint_pos=torch.zeros((3, 1)))
        states = SimpleNamespace(robots={"g1_without_hands": robot})
        reward = UpperBodyCenterOfMassReward()
        reward.control_dt = 0.02
        reward.metrics = {
            "upper_body_com_horizontal_error": torch.tensor([0.0, 0.05, 0.20])
        }

        values = reward(states, "g1_without_hands")

        self.assertEqual(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 0.0)
        self.assertGreater(values[1].item(), values[2].item())

    def test_chair_relative_depth_points_toward_the_seat(self) -> None:
        torso = torch.tensor([[0.0, 0.70, 0.85, 2**-0.5, 0.0, 0.0, -(2**-0.5)]])
        targets = torch.tensor([[[0.15, 0.29, 0.96], [-0.15, 0.29, 0.96]]])
        end_effectors = torch.tensor([[[0.15, 0.19, 0.95], [-0.15, 0.19, 0.95]]])
        chair_direction_to_robot = torch.tensor([[0.0, 1.0]])

        metrics = geometry.hand_task_space_metrics(
            torso, end_effectors, targets, chair_direction_to_robot
        )

        self.assertTrue(
            torch.allclose(metrics["hand_forward_reach"], torch.full((1, 2), 0.51))
        )
        self.assertTrue(
            torch.allclose(metrics["hand_behind_backrest"], torch.full((1, 2), 0.10))
        )
        self.assertTrue(
            torch.allclose(metrics["hand_below_target"], torch.full((1, 2), 0.01))
        )

    def test_stage1_requires_both_hands_forward_and_above_backrest(self) -> None:
        metrics = _common_metrics()
        metrics["hand_forward_reach"] = torch.full((2, 2), geometry.HAND_FORWARD_REACH_MIN)
        metrics["hand_height_above_backrest"] = torch.zeros((2, 2))
        self.assertTrue(stage1_success(metrics).all())

        metrics["hand_forward_reach"][0, 0] -= 0.001
        metrics["hand_height_above_backrest"][1, 1] -= 0.001
        self.assertFalse(stage1_success(metrics).any())

    def test_stage2_is_geometric_and_does_not_require_contact(self) -> None:
        metrics = _common_metrics()
        metrics["hand_behind_backrest"] = torch.full(
            (2, 2), geometry.HAND_BEHIND_BACKREST_MIN
        )
        metrics["hand_below_target"] = torch.zeros((2, 2))
        # Deliberately no contact/contact_force entry in the metrics.
        self.assertTrue(stage2_success(metrics).all())

        metrics["hand_behind_backrest"][0, 0] -= 0.001
        metrics["hand_below_target"][1, 1] -= 0.001
        self.assertFalse(stage2_success(metrics).any())

    def test_dense_rewards_improve_toward_the_new_geometry(self) -> None:
        robot = SimpleNamespace(joint_pos=torch.zeros((2, 1)))
        states = SimpleNamespace(robots={"g1_without_hands": robot})

        cases = (
            (
                ExtendArmsReward(),
                {
                    "hand_forward_reach": torch.full((2, 2), 0.20),
                    "hand_height_above_backrest": torch.full((2, 2), -0.20),
                },
                {
                    "hand_forward_reach": torch.full((2, 2), 0.49),
                    "hand_height_above_backrest": torch.full((2, 2), -0.01),
                },
            ),
            (
                MoveHandsBehindBackrestReward(),
                {
                    "hand_behind_backrest": torch.full((2, 2), -0.10),
                    "hand_below_target": torch.full((2, 2), -0.20),
                },
                {
                    "hand_behind_backrest": torch.full((2, 2), 0.09),
                    "hand_below_target": torch.full((2, 2), -0.01),
                },
            ),
        )
        for reward, far, near in cases:
            reward.actual_stage = torch.full((2,), reward.stage, dtype=torch.long)
            reward.metrics = {**_common_metrics(), **far}
            far_reward = reward(states, "g1_without_hands")
            reward.metrics = {**_common_metrics(), **near}
            near_reward = reward(states, "g1_without_hands")
            self.assertTrue((near_reward > far_reward).all())


if __name__ == "__main__":
    unittest.main()
