import math
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan import WalkToChairProgressReward
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
