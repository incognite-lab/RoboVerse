"""Behavioral tests for the exponential ChairMan hand-target reward."""

from types import SimpleNamespace

import pytest
import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import ReachChairProgressReward


def scene(num_envs=1):
    robot = SimpleNamespace(
        body_names=["left_endeffector", "endeffector"],
        body_state=torch.zeros(num_envs, 2, 13),
    )
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=torch.zeros(num_envs, 3, 13),
    )
    chair.body_state[:, 1, :3] = torch.tensor([0.60, 0.20, 1.00])
    chair.body_state[:, 2, :3] = torch.tensor([0.60, -0.20, 1.00])
    states = SimpleNamespace(
        robots={"g1_with_hands": robot}, objects={"chair": chair}
    )
    return states, chair.body_state[:, 1:3, :3]


def evaluate(reward, states, positions):
    states.robots["g1_with_hands"].body_state[:, :, :3] = positions
    return reward(states, "g1_with_hands")


def test_reward_is_one_when_both_end_effectors_are_on_targets():
    states, targets = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)

    torch.testing.assert_close(evaluate(reward, states, targets), torch.ones(1))


def test_reward_uses_exponential_of_mean_hand_distance():
    states, targets = scene()
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    positions = targets.clone()
    positions[:, 0, 0] -= 0.10
    positions[:, 1, 0] -= 0.30

    result = evaluate(reward, states, positions)

    # Mean distance is (0.10 + 0.30) / 2 = 0.20 m.
    expected = torch.exp(torch.tensor([-1.0]))
    torch.testing.assert_close(result, expected)


def test_distances_are_averaged_before_exponential_mapping():
    states, targets = scene(num_envs=2)
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(2, dtype=torch.long)
    positions = targets.clone()
    # Both environments have the same mean distance of 0.20 m.
    positions[0, 0, 0] -= 0.40
    positions[1, :, 0] -= 0.20

    result = evaluate(reward, states, positions)

    torch.testing.assert_close(result[0], result[1])


def test_reward_increases_monotonically_and_exponentially_near_target():
    states, targets = scene(num_envs=3)
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(3, dtype=torch.long)
    positions = targets.clone()
    positions[0, :, 0] -= 0.30
    positions[1, :, 0] -= 0.20
    positions[2, :, 0] -= 0.10

    result = evaluate(reward, states, positions)

    assert result[0] < result[1] < result[2]
    torch.testing.assert_close(result[1] / result[0], result[2] / result[1])


def test_reward_ignores_orientation_velocity_and_other_body_state():
    states, targets = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    positions = targets - torch.tensor([0.20, 0.0, 0.0])
    baseline = evaluate(reward, states, positions)

    body_state = states.robots["g1_with_hands"].body_state
    body_state[:, :, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    body_state[:, :, 7:13] = 100.0

    torch.testing.assert_close(evaluate(reward, states, positions), baseline)


def test_reward_is_zero_outside_stage_one():
    states, targets = scene(num_envs=3)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.tensor([0, 1, 2])

    result = evaluate(reward, states, targets)

    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 0.0]))


def test_visualization_points_are_exactly_the_reward_targets():
    states, targets = scene(num_envs=2)
    reward = ReachChairProgressReward()

    points = reward.path_points_from_states(states)

    assert points.shape == (2, 2, 1, 3)
    torch.testing.assert_close(points.squeeze(2), targets)


@pytest.mark.parametrize("scale", [0, -0.1, float("nan"), float("inf")])
def test_invalid_distance_scale_rejected(scale):
    with pytest.raises(ValueError):
        ReachChairProgressReward(distance_scale=scale)
