"""Behavioral tests for the synchronized two-point ChairMan reach reward."""

import math
from types import SimpleNamespace

import pytest
import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import ReachChairProgressReward


def scene(num_envs=1, yaw=math.pi / 2):
    robot = SimpleNamespace(
        joint_pos=torch.zeros(num_envs, 5),
        body_names=["left_endeffector", "endeffector"],
        body_state=torch.zeros(num_envs, 2, 13),
    )
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=torch.zeros(num_envs, 3, 13),
    )
    chair.body_state[:, 0, 3:7] = torch.tensor(
        [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    )
    lateral = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
    back = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0])
    center = torch.tensor([0.75, 0.0, 0.96])
    chair.body_state[:, 1, :3] = center + 0.15 * lateral
    chair.body_state[:, 2, :3] = center - 0.15 * lateral
    states = SimpleNamespace(
        robots={"g1_with_hands": robot}, objects={"chair": chair}
    )

    targets = chair.body_state[:, 1:3, :3]
    up = torch.tensor([0.0, 0.0, 0.15])
    points = torch.stack(
        (
            targets + 0.15 * back + up,
            targets,
        ),
        dim=2,
    )
    return states, points


def evaluate(reward, states, positions):
    states.robots["g1_with_hands"].body_state[:, :, :3] = positions
    return reward(states, "g1_with_hands")


def stationary_value(reward, states, positions):
    evaluate(reward, states, positions)
    return evaluate(reward, states, positions)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -0.7])
def test_approach_points_are_above_and_in_front_of_targets(yaw):
    states, points = scene(yaw=yaw)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    torch.testing.assert_close(reward.path_points_from_states(states), points)

    expected_offset = torch.tensor(
        [-0.15 * math.sin(yaw), 0.15 * math.cos(yaw), 0.15]
    )
    torch.testing.assert_close(
        points[:, :, 0] - points[:, :, 1],
        expected_offset.expand_as(points[:, :, 0]),
    )

    evaluate(reward, states, points[:, :, 0] + torch.tensor([0.0, 0.0, -0.30]))
    approach_value = stationary_value(reward, states, points[:, :, 0]).item()
    assert (reward.waypoint_index == 1).all()
    target_value = stationary_value(reward, states, points[:, :, 1]).item()

    assert approach_value == pytest.approx(1.0 / 3.0)
    assert target_value == pytest.approx(1.0)


def test_piecewise_linear_slopes_are_weighted_one_then_two():
    states, points = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)

    first_start = points[:, :, 0] + torch.tensor([0.0, 0.0, -0.40])
    evaluate(reward, states, first_start)
    first_half = points[:, :, 0] + torch.tensor([0.0, 0.0, -0.20])
    approach_half_value = stationary_value(reward, states, first_half).item()

    stationary_value(reward, states, points[:, :, 0])
    descent_half = 0.5 * (points[:, :, 0] + points[:, :, 1])
    descent_half_value = stationary_value(reward, states, descent_half).item()

    assert approach_half_value == pytest.approx(0.5 / 3.0, abs=1.0e-6)
    assert descent_half_value == pytest.approx(2.0 / 3.0, abs=1.0e-6)


def test_both_hands_must_reach_approach_points_before_switch():
    states, points = scene(num_envs=2)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(2, dtype=torch.long)
    positions = points[:, :, 0].clone()
    positions[0, 0, 2] -= 0.049
    positions[0, 1, 2] -= 0.051
    positions[1, 0, 2] -= 0.049
    positions[1, 1, 2] -= 0.049

    evaluate(reward, states, positions)

    torch.testing.assert_close(
        reward.waypoint_index, torch.tensor([0, 1])
    )


def test_closer_state_and_positive_progress_both_increase_reward():
    states, points = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    far = points[:, :, 0] + torch.tensor([0.0, 0.0, -0.40])
    closer = points[:, :, 0] + torch.tensor([0.0, 0.0, -0.20])

    far_value = stationary_value(reward, states, far)
    moving_closer = evaluate(reward, states, closer)
    closer_value = evaluate(reward, states, closer)

    assert closer_value.item() > far_value.item()
    assert moving_closer.item() > closer_value.item()


def test_direct_final_target_does_not_skip_approach_points():
    states, points = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)

    evaluate(reward, states, points[:, :, 1])

    assert (reward.waypoint_index == 0).all()


def test_reward_uses_position_but_not_hand_velocity_or_orientation():
    results = []
    for altered_state in (False, True):
        states, points = scene()
        reward = ReachChairProgressReward()
        reward.actual_stage = torch.ones(1, dtype=torch.long)
        start = points[:, :, 0] + torch.tensor([0.0, 0.0, -0.30])
        evaluate(reward, states, start)
        if altered_state:
            body = states.robots["g1_with_hands"].body_state
            body[:, :, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
            body[:, :, 7:13] = 100.0
        results.append(
            evaluate(reward, states, start + torch.tensor([0, 0, 0.01]))
        )

    torch.testing.assert_close(results[0], results[1])


def test_env_histories_are_independent_and_partial_reset_is_local():
    states, points = scene(num_envs=3)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.tensor([1, 1, 2])
    positions = points[:, :, 0].clone()
    positions[0, 1, 2] -= 0.10

    result = evaluate(reward, states, positions)

    torch.testing.assert_close(
        reward.waypoint_index, torch.tensor([0, 1, 0])
    )
    assert result[2] == 0
    reward.reset(torch.tensor([0]), states)
    torch.testing.assert_close(
        reward.waypoint_index, torch.tensor([0, 1, 0])
    )
    assert torch.isnan(reward.initial_distances[0]).all()
    assert torch.isnan(reward.previous_potential[0]).all()


def test_reward_is_zero_outside_stage_one():
    states, points = scene(num_envs=2)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.tensor([0, 2])

    torch.testing.assert_close(
        evaluate(reward, states, points[:, :, 0]), torch.zeros(2)
    )


@pytest.mark.parametrize("offset", [0, -0.1, float("nan"), float("inf")])
def test_invalid_path_offset_rejected(offset):
    with pytest.raises(ValueError):
        ReachChairProgressReward(path_offset=offset)
