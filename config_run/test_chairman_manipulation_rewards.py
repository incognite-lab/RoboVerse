import math
from types import SimpleNamespace

import numpy as np
import torch

from metasim.cfg.robots.g1_cfg_with_hands import G1WithHandsCfg
from metasim.cfg.tasks.humanoidbench.ChairMan import (
    CloseGraspReward,
    GraspForceReward,
    HandOrientationProgressReward,
    HandTargetStillnessReward,
    ReachChairProgressReward,
)


def _yaw_quaternion(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _states(
    *,
    left_position=(0.0, 0.0, 0.0),
    right_position=(0.0, 0.0, 0.0),
    left_quaternion=None,
    right_quaternion=None,
    hand_velocity=0.0,
    closure_fraction=0.0,
    contact_forces=None,
):
    joint_names = list(G1WithHandsCfg().joint_limits)
    joint_pos = torch.zeros((1, len(joint_names)), dtype=torch.float32)
    close_targets = CloseGraspReward().finger_targets_dict
    for name, target in close_targets.items():
        joint_pos[0, joint_names.index(name)] = closure_fraction * target

    robot_body_names = ["pelvis", "left_endeffector", "endeffector"]
    robot_body_state = torch.zeros((1, 3, 13), dtype=torch.float32)
    robot_body_state[:, :, 3] = 1.0
    robot_body_state[0, 1, :3] = torch.tensor(left_position)
    robot_body_state[0, 2, :3] = torch.tensor(right_position)
    robot_body_state[0, 1, 3:7] = (
        left_quaternion if left_quaternion is not None else _yaw_quaternion(0.0)
    )
    robot_body_state[0, 2, 3:7] = (
        right_quaternion if right_quaternion is not None else _yaw_quaternion(0.0)
    )
    robot_body_state[0, 1:3, 7] = hand_velocity

    contact = None
    extras = {}
    if contact_forces is not None:
        forces = torch.as_tensor(contact_forces, dtype=torch.float32)
        contact = {
            "link_a": torch.arange(6, dtype=torch.long).unsqueeze(0),
            "link_b": torch.full((1, 6), 6, dtype=torch.long),
            "valid_mask": torch.ones((1, 6), dtype=torch.bool),
            "force_b": torch.stack(
                (forces, torch.zeros_like(forces), torch.zeros_like(forces)), dim=-1
            ).unsqueeze(0),
        }
        tip_names = [
            "left_hand_thumb_2_link",
            "left_hand_index_1_link",
            "left_hand_middle_1_link",
            "right_hand_thumb_2_link",
            "right_hand_index_1_link",
            "right_hand_middle_1_link",
        ]
        global_map = {index: ("g1_with_hands", name) for index, name in enumerate(tip_names)}
        global_map[6] = ("chair", "base_link")
        extras = {"global_link_map": global_map, "num_bodies_per_env": 10}

    robot = SimpleNamespace(
        joint_names=np.asarray(joint_names),
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        body_names=robot_body_names,
        body_state=robot_body_state,
        contact=contact,
    )

    chair_body_names = ["base_link", "target_hand_left", "target_hand_right"]
    chair_body_state = torch.zeros((1, 3, 13), dtype=torch.float32)
    chair_body_state[:, :, 3] = 1.0
    chair_body_state[0, 1, :3] = torch.tensor([0.5, 0.2, 1.0])
    chair_body_state[0, 2, :3] = torch.tensor([0.5, -0.2, 1.0])
    chair = SimpleNamespace(body_names=chair_body_names, body_state=chair_body_state)
    return SimpleNamespace(
        robots={"g1_with_hands": robot},
        objects={"chair": chair},
        extras=extras,
    )


def _evaluate(reward, states, stage):
    reward.actual_stage = torch.tensor([stage], dtype=torch.long)
    return reward(states, "g1_with_hands").item()


def test_stage1_reach_requires_both_hands():
    both_far = _states(left_position=(0.0, 0.2, 1.0), right_position=(0.0, -0.2, 1.0))
    one_close = _states(left_position=(0.5, 0.2, 1.0), right_position=(0.0, -0.2, 1.0))
    both_close = _states(left_position=(0.49, 0.2, 1.0), right_position=(0.49, -0.2, 1.0))

    assert _evaluate(ReachChairProgressReward(), both_close, 1) > _evaluate(
        ReachChairProgressReward(), one_close, 1
    )
    assert _evaluate(ReachChairProgressReward(), one_close, 1) > _evaluate(
        ReachChairProgressReward(), both_far, 1
    )


def test_stage1_orientation_and_stillness_match_checker_goal():
    aligned = _states(
        left_position=(0.5, 0.2, 1.0), right_position=(0.5, -0.2, 1.0)
    )
    misaligned = _states(
        left_position=(0.5, 0.2, 1.0),
        right_position=(0.5, -0.2, 1.0),
        left_quaternion=_yaw_quaternion(math.pi),
        right_quaternion=_yaw_quaternion(math.pi),
    )
    moving = _states(
        left_position=(0.5, 0.2, 1.0),
        right_position=(0.5, -0.2, 1.0),
        hand_velocity=0.5,
    )

    assert _evaluate(HandOrientationProgressReward(), aligned, 1) > _evaluate(
        HandOrientationProgressReward(), misaligned, 1
    )
    assert _evaluate(HandTargetStillnessReward(), aligned, 1) > _evaluate(
        HandTargetStillnessReward(), moving, 1
    )


def test_stage2_dense_closure_and_contact_rewards_are_monotonic():
    near_kwargs = {
        "left_position": (0.5, 0.2, 1.0),
        "right_position": (0.5, -0.2, 1.0),
    }
    open_hands = _states(**near_kwargs, closure_fraction=0.0)
    half_closed = _states(**near_kwargs, closure_fraction=0.5)
    closed = _states(**near_kwargs, closure_fraction=1.0)
    assert _evaluate(CloseGraspReward(), closed, 2) > _evaluate(
        CloseGraspReward(), half_closed, 2
    )
    assert _evaluate(CloseGraspReward(), half_closed, 2) > _evaluate(
        CloseGraspReward(), open_hands, 2
    )

    no_contact = _states(**near_kwargs, contact_forces=[0, 0, 0, 0, 0, 0])
    partial_contact = _states(**near_kwargs, contact_forces=[2, 0, 0, 2, 0, 0])
    complete_contact = _states(**near_kwargs, contact_forces=[2, 2, 2, 2, 2, 2])
    assert _evaluate(GraspForceReward(), complete_contact, 2) > _evaluate(
        GraspForceReward(), partial_contact, 2
    )
    assert _evaluate(GraspForceReward(), partial_contact, 2) > _evaluate(
        GraspForceReward(), no_contact, 2
    )
