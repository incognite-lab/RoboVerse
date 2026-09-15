"""Behavioral tests for the two-phase exponential ChairMan reach reward."""

import math
from types import SimpleNamespace

import pytest
import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import ReachChairProgressReward


def scene(num_envs=1, yaw=0.0):
    robot = SimpleNamespace(
        body_names=["left_endeffector", "endeffector"],
        body_state=torch.zeros(num_envs, 2, 13),
    )
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=torch.zeros(num_envs, 3, 13),
    )
    chair.body_state[:, 0, 3:7] = torch.tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    )
    chair.body_state[:, 1, :3] = torch.tensor([0.60, 0.20, 1.00])
    chair.body_state[:, 2, :3] = torch.tensor([0.60, -0.20, 1.00])
    states = SimpleNamespace(
        robots={"g1_with_hands": robot}, objects={"chair": chair}
    )
    targets = chair.body_state[:, 1:3, :3]
    front = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0])
    approach = targets + 0.07 * front
    return states, torch.stack((approach, targets), dim=2)


def evaluate(reward, states, positions):
    states.robots["g1_with_hands"].body_state[:, :, :3] = positions
    return reward(states, "g1_with_hands")


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -0.7])
def test_first_points_are_seven_centimetres_in_front_of_targets(yaw):
    states, expected_points = scene(yaw=yaw)
    reward = ReachChairProgressReward()

    points = reward.path_points_from_states(states)

    torch.testing.assert_close(points, expected_points)
    distances = torch.linalg.vector_norm(
        points[:, :, 0] - points[:, :, 1], dim=-1
    )
    torch.testing.assert_close(distances, torch.full_like(distances, 0.07))


def test_approach_phase_uses_exponential_of_mean_hand_distance():
    states, points = scene()
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    positions = points[:, :, 0].clone()
    positions[:, 0, 0] -= 0.10
    positions[:, 1, 0] -= 0.30

    result = evaluate(reward, states, positions)

    # Phase 1 occupies reward interval 0..0.5 and mean distance is 0.20 m.
    expected = 0.5 * torch.exp(torch.tensor([-1.0]))
    torch.testing.assert_close(result, expected)
    assert not reward.target_phase.any()


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

    torch.testing.assert_close(reward.target_phase, torch.tensor([False, True]))


def test_switch_is_permanent_and_final_target_reward_is_one():
    states, points = scene()
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    far = points[:, :, 0] - torch.tensor([0.30, 0.0, 0.0])
    approach_reward = evaluate(reward, states, far)
    switched_reward = evaluate(reward, states, points[:, :, 0])
    target_reward = evaluate(reward, states, points[:, :, 1])

    assert reward.target_phase.all()
    assert switched_reward.item() > approach_reward.item()
    torch.testing.assert_close(target_reward, torch.ones(1))

    evaluate(reward, states, far)
    assert reward.target_phase.all()


def test_target_phase_uses_exponential_of_mean_hand_distance():
    states, points = scene()
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    evaluate(reward, states, points[:, :, 0])
    positions = points[:, :, 1].clone()
    positions[:, 0, 0] -= 0.10
    positions[:, 1, 0] -= 0.30

    result = evaluate(reward, states, positions)

    expected = 0.5 + 0.5 * torch.exp(torch.tensor([-1.0]))
    torch.testing.assert_close(result, expected)


def test_going_directly_to_targets_does_not_skip_approach_phase():
    states, points = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)

    result = evaluate(reward, states, points[:, :, 1])

    assert not reward.target_phase.any()
    expected = 0.5 * torch.exp(torch.tensor([-0.07 / 0.20]))
    torch.testing.assert_close(result, expected)


def test_hand_distances_are_averaged_before_exponential_mapping():
    states, points = scene(num_envs=2)
    reward = ReachChairProgressReward(distance_scale=0.20)
    reward.actual_stage = torch.ones(2, dtype=torch.long)
    positions = points[:, :, 0].clone()
    positions[0, 0, 0] -= 0.40
    positions[1, :, 0] -= 0.20

    result = evaluate(reward, states, positions)

    torch.testing.assert_close(result[0], result[1])


def test_reward_ignores_orientation_velocity_and_other_body_state():
    states, points = scene()
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(1, dtype=torch.long)
    positions = points[:, :, 0] - torch.tensor([0.20, 0.0, 0.0])
    baseline = evaluate(reward, states, positions)

    body_state = states.robots["g1_with_hands"].body_state
    body_state[:, :, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    body_state[:, :, 7:13] = 100.0

    torch.testing.assert_close(evaluate(reward, states, positions), baseline)


def test_partial_reset_only_resets_selected_environment_phase():
    states, points = scene(num_envs=2)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.ones(2, dtype=torch.long)
    evaluate(reward, states, points[:, :, 0])
    assert reward.target_phase.all()

    reward.reset(torch.tensor([0]), states)

    torch.testing.assert_close(reward.target_phase, torch.tensor([False, True]))


def test_reward_is_zero_outside_stage_one():
    states, points = scene(num_envs=3)
    reward = ReachChairProgressReward()
    reward.actual_stage = torch.tensor([0, 1, 2])

    result = evaluate(reward, states, points[:, :, 0])

    assert result[0] == 0
    assert result[2] == 0
    torch.testing.assert_close(reward.target_phase, torch.tensor([False, True, False]))


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("distance_scale", 0.0),
        ("approach_offset", -0.1),
        ("switch_tolerance", float("nan")),
    ],
)
def test_invalid_parameters_are_rejected(argument, value):
    with pytest.raises(ValueError):
        ReachChairProgressReward(**{argument: value})
