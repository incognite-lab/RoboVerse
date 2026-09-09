import math
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan import (
    STAY_NEAR_ANCHOR_REWARD_WEIGHT,
    StayNearAnchorReward,
    WalkToChairProgressReward,
)
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    STAY_NEAR_ANCHOR_REWARD_WEIGHT as MULTI_STAY_NEAR_ANCHOR_REWARD_WEIGHT,
    StayNearAnchorReward as MultiStayNearAnchorReward,
)
from metasim.utils.chair_navigation import chair_back_direction_xy


def _yaw_quaternion(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _reward_at(
    *,
    robot_xy: tuple[float, float],
    robot_yaw: float,
    robot_velocity_xy: tuple[float, float],
    stage: int = 0,
) -> float:
    robot_body = torch.zeros((1, 1, 13), dtype=torch.float32)
    robot_body[0, 0, :2] = torch.tensor(robot_xy)
    robot_body[0, 0, 3:7] = _yaw_quaternion(robot_yaw)
    robot_body[0, 0, 7:9] = torch.tensor(robot_velocity_xy)

    chair_body = torch.zeros((1, 1, 13), dtype=torch.float32)
    chair_body[0, 0, :2] = torch.tensor([0.75, 0.0])
    chair_body[0, 0, 3:7] = _yaw_quaternion(math.pi / 2.0)

    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((1, 1)),
    )
    chair = SimpleNamespace(body_names=["base_link"], body_state=chair_body)
    states = SimpleNamespace(
        robots={"g1_with_hands": robot},
        objects={"chair": chair},
    )

    reward = WalkToChairProgressReward()
    reward.actual_stage = torch.tensor([stage])
    return reward(states, "g1_with_hands").item()


def test_chair_back_direction_follows_chair_orientation():
    direction = chair_back_direction_xy(_yaw_quaternion(math.pi / 2.0).unsqueeze(0))
    torch.testing.assert_close(direction, torch.tensor([[-1.0, 0.0]]), atol=1.0e-6, rtol=0.0)


def test_navigation_reward_prefers_the_complete_staged_path():
    far_walking = _reward_at(
        robot_xy=(-2.5, 0.0), robot_yaw=0.0, robot_velocity_xy=(0.8, 0.0)
    )
    far_stopped = _reward_at(
        robot_xy=(-2.5, 0.0), robot_yaw=0.0, robot_velocity_xy=(0.0, 0.0)
    )
    staging_aligned_and_moving = _reward_at(
        robot_xy=(-0.75, 0.0), robot_yaw=0.0, robot_velocity_xy=(0.4, 0.0)
    )
    final_facing_and_stopped = _reward_at(
        robot_xy=(0.0, 0.0), robot_yaw=0.0, robot_velocity_xy=(0.0, 0.0)
    )
    final_facing_away = _reward_at(
        robot_xy=(0.0, 0.0), robot_yaw=math.pi, robot_velocity_xy=(0.0, 0.0)
    )
    overshot = _reward_at(
        robot_xy=(0.2, 0.0), robot_yaw=0.0, robot_velocity_xy=(0.0, 0.0)
    )

    assert far_walking > far_stopped
    assert staging_aligned_and_moving > far_walking
    assert final_facing_and_stopped > staging_aligned_and_moving
    assert final_facing_and_stopped > final_facing_away
    assert overshot == 0.0


def test_navigation_reward_is_zero_outside_stage_zero():
    reward = _reward_at(
        robot_xy=(-2.5, 0.0),
        robot_yaw=0.0,
        robot_velocity_xy=(0.8, 0.0),
        stage=1,
    )
    assert reward == 0.0


def test_stay_near_anchor_is_a_penalty_for_xy_movement():
    robot_body = torch.zeros((1, 1, 13), dtype=torch.float32)
    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((1, 1)),
    )
    states = SimpleNamespace(robots={"g1_with_hands": robot})

    penalty = StayNearAnchorReward()
    penalty.actual_stage = torch.tensor([1])

    assert penalty(states, "g1_with_hands").item() == 0.0

    robot_body[0, 0, 0] = penalty.max_xy_drift / 2.0
    assert math.isclose(
        penalty(states, "g1_with_hands").item(), 0.5, rel_tol=0.0, abs_tol=1.0e-6
    )

    robot_body[0, 0, 0] = penalty.max_xy_drift
    assert penalty(states, "g1_with_hands").item() == 1.0
    assert STAY_NEAR_ANCHOR_REWARD_WEIGHT < 0.0


def test_stay_near_anchor_penalty_is_inactive_outside_manipulation_stages():
    robot_body = torch.zeros((1, 1, 13), dtype=torch.float32)
    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((1, 1)),
    )
    states = SimpleNamespace(robots={"g1_with_hands": robot})

    penalty = StayNearAnchorReward()
    penalty.actual_stage = torch.tensor([0])
    robot_body[0, 0, 0] = 1.0

    assert penalty(states, "g1_with_hands").item() == 0.0


def test_multi_stay_near_anchor_is_a_positive_reward():
    robot_body = torch.zeros((1, 1, 13), dtype=torch.float32)
    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((1, 1)),
    )
    states = SimpleNamespace(robots={"g1_with_hands": robot})

    reward = MultiStayNearAnchorReward()
    reward.actual_stage = torch.tensor([1])

    assert reward(states, "g1_with_hands").item() == 1.0

    robot_body[0, 0, 0] = reward.max_xy_drift / 2.0
    assert math.isclose(
        reward(states, "g1_with_hands").item(), 0.5, rel_tol=0.0, abs_tol=1.0e-6
    )

    robot_body[0, 0, 0] = reward.max_xy_drift
    assert reward(states, "g1_with_hands").item() == 0.0
    assert MULTI_STAY_NEAR_ANCHOR_REWARD_WEIGHT > 0.0


def _multi_anchor_states(xy_positions):
    xy = torch.as_tensor(xy_positions, dtype=torch.float32)
    robot_body = torch.zeros((xy.shape[0], 1, 13), dtype=torch.float32)
    robot_body[:, 0, :2] = xy
    robot = SimpleNamespace(
        body_names=["pelvis"],
        body_state=robot_body,
        joint_pos=torch.zeros((xy.shape[0], 1), dtype=torch.float32),
    )
    return SimpleNamespace(robots={"g1_with_hands": robot})


def test_multi_stay_near_anchor_is_bounded_monotonic_and_xy_isotropic():
    distances = torch.tensor([0.00, 0.03, 0.06, 0.09, 0.12, 0.18])
    states = _multi_anchor_states(torch.zeros((distances.numel(), 2)))
    reward = MultiStayNearAnchorReward()
    reward.actual_stage = torch.ones(distances.numel(), dtype=torch.long)
    reward.reset(torch.arange(distances.numel()), states)

    # Alternate X/Y displacement to verify that only radial XY drift matters.
    states.robots["g1_with_hands"].body_state[::2, 0, 0] = distances[::2]
    states.robots["g1_with_hands"].body_state[1::2, 0, 1] = distances[1::2]
    values = reward(states, "g1_with_hands")

    expected = torch.tensor([1.00, 0.75, 0.50, 0.25, 0.00, 0.00])
    torch.testing.assert_close(values, expected, atol=1.0e-6, rtol=0.0)
    assert torch.all(values[:-1] >= values[1:])
    assert torch.all((values >= 0.0) & (values <= 1.0))


def test_multi_stay_near_anchor_partial_reset_does_not_move_other_anchors():
    states = _multi_anchor_states([[0.0, 0.0], [1.0, 0.0]])
    reward = MultiStayNearAnchorReward()
    reward.actual_stage = torch.tensor([1, 1], dtype=torch.long)
    reward.reset(torch.tensor([0, 1]), states)

    robot_body = states.robots["g1_with_hands"].body_state
    robot_body[0, 0, 0] = 0.06
    robot_body[1, 0, 0] = 1.06
    torch.testing.assert_close(
        reward(states, "g1_with_hands"), torch.tensor([0.5, 0.5])
    )

    # Reset only env 0 at its new position. Env 1 must keep its old anchor.
    reward.reset(torch.tensor([0]), states)
    torch.testing.assert_close(
        reward(states, "g1_with_hands"), torch.tensor([1.0, 0.5])
    )


def test_multi_stay_near_anchor_reanchors_on_stage_transition():
    states = _multi_anchor_states([[0.0, 0.0]])
    reward = MultiStayNearAnchorReward()
    reward.actual_stage = torch.tensor([1], dtype=torch.long)
    reward.reset(torch.tensor([0]), states)

    states.robots["g1_with_hands"].body_state[0, 0, 0] = 0.06
    assert math.isclose(
        reward(states, "g1_with_hands").item(), 0.5, abs_tol=1.0e-6
    )

    # Stage 2 starts from the achieved stage-1 pose, so that pose becomes the
    # new anchor instead of inheriting a stale stage-1 reference.
    reward.actual_stage[0] = 2
    assert reward(states, "g1_with_hands").item() == 1.0


def test_multi_stay_near_anchor_is_zero_in_inactive_stages():
    states = _multi_anchor_states([[0.0, 0.0], [0.0, 0.0]])
    reward = MultiStayNearAnchorReward()
    reward.actual_stage = torch.tensor([0, 3], dtype=torch.long)
    reward.reset(torch.tensor([0, 1]), states)

    torch.testing.assert_close(
        reward(states, "g1_with_hands"), torch.tensor([0.0, 0.0])
    )
