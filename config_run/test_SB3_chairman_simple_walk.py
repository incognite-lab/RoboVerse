"""Walking adapter contracts without launching a simulator."""
import numpy as np
import pytest
import torch

import config_run.SB3_chairman_env as walking_module
from config_run.SB3_chairman_simple_walk import StableBaseline3VecEnv
from config_run.test_SB3_chairman_env import FakeMotionPolicy, make_fake_metasim_env
from metasim.cfg.checkers.stages_chairman_simple import stage0_init


def make_env(monkeypatch):
    monkeypatch.setattr(walking_module, "G1MotionPolicy", FakeMotionPolicy)
    env = make_fake_metasim_env()
    env.scenario.task = env.env.handler.task
    return StableBaseline3VecEnv(env)


def test_absolute_goal_rotates_to_body_and_holds_uncontrolled_joints(monkeypatch):
    env = make_env(monkeypatch)
    robot = env.env.env.handler.get_states().robots[env.robot_name]
    robot.body_state[:, env._pelvis_index, 0] = -1.0
    robot.body_state[:, env._pelvis_index, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    actions = np.array([[0.0, -0.5, np.pi / 2, -1.2, -1.4]] * 2, dtype=np.float32)
    targets = env._compose_robot_targets(actions)
    np.testing.assert_allclose(env.last_requested_locomotion_command, [[0, -0.5, 0]] * 2, atol=1e-6)
    for name in env.upper_body_joint_names:
        expected = actions[:, 3] if name == env.action_names[3] else (
            actions[:, 4] if name == env.action_names[4] else
            env.env.scenario.robots[0].default_joint_positions[name]
        )
        np.testing.assert_allclose(targets[:, env.sim_joint_names.index(name)], expected)
    np.testing.assert_allclose(targets[:, env._leg_state_indices],
                               np.tile(FakeMotionPolicy.DEFAULT_ANGLES + 0.01, (2, 1)))


def test_observation_layout_and_stop_at_goal(monkeypatch):
    env = make_env(monkeypatch)
    robot = env.env.env.handler.get_states().robots[env.robot_name]
    robot.body_state[:, env._pelvis_index, 0] = -0.7
    robot.body_state[:, env._pelvis_index, 1] = 0.2
    actions = np.array([[0.2, -0.7, 0, -1.2, -1.4]] * 2, dtype=np.float32)
    env._compose_robot_targets(actions)
    np.testing.assert_array_equal(env.last_locomotion_command, 0)
    obs = env.add_extra_to_obs(np.zeros((2, 43)))
    assert obs.shape == (2, 51)
    np.testing.assert_allclose(obs[:, :3], actions[:, :3])
    assert np.isfinite(obs).all()
    assert env.action_space.shape == (5,)


def test_yaw_uses_shortest_rotation_and_rejects_invalid_actions(monkeypatch):
    env = make_env(monkeypatch)
    robot = env.env.env.handler.get_states().robots[env.robot_name]
    yaw = -3.0
    robot.body_state[:, env._pelvis_index, 3:7] = torch.tensor([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    env._compose_robot_targets(np.array([[0, 0, 2.6, 0, 0]] * 2))
    np.testing.assert_allclose(env.last_requested_locomotion_command[:, 2], 5.6 - 2*np.pi, atol=1e-6)
    with pytest.raises(ValueError):
        env._compose_robot_targets(np.full((2, 5), np.nan))
    env._reset_motion_state([1])
    assert env.motion_policy.reset_calls[-1] == [1]
    np.testing.assert_array_equal(env.last_locomotion_command[1], 0)


def test_floating_reset_has_all_physical_joints():
    state = stage0_init("g1_with_hands")
    robot = state["robots"]["g1_with_hands"]
    assert len(robot["dof_pos"]) == 43
    assert not any(name.startswith("base") for name in robot["dof_pos"])
    np.testing.assert_allclose(robot["pos"], [-1.5, 0, 0.8])
    np.testing.assert_allclose(robot["rot"], [np.cos(0.15), 0, 0, np.sin(0.15)], atol=1e-6)
