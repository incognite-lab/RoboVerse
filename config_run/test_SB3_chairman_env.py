from types import SimpleNamespace

import numpy as np
import torch

import config_run.SB3_chairman_env as chairman_module
from config_run.policy import G1MotionPolicy as RealG1MotionPolicy
from metasim.cfg.robots.g1_cfg_with_hands import G1WithHandsCfg


class FakeMotionPolicy:
    JOINT_NAMES = RealG1MotionPolicy.JOINT_NAMES
    NUM_ACTIONS = RealG1MotionPolicy.NUM_ACTIONS
    DEFAULT_ANGLES = RealG1MotionPolicy.DEFAULT_ANGLES
    MAX_COMMAND = RealG1MotionPolicy.MAX_COMMAND
    CONTROL_DT = RealG1MotionPolicy.CONTROL_DT

    def __init__(self, *, device, control_dt):
        self.device = device
        self.control_dt = control_dt
        self.calls = []
        self.reset_calls = []
        self.last_action = np.zeros((1, self.NUM_ACTIONS), dtype=np.float32)

    def predict_joint_positions(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["joint_positions"].shape[0]
        self.last_action = np.zeros((batch_size, self.NUM_ACTIONS), dtype=np.float32)
        return np.tile((self.DEFAULT_ANGLES + 0.01)[None, :], (batch_size, 1))

    def reset(self, env_ids=None):
        self.reset_calls.append(env_ids)


def make_fake_metasim_env(num_envs=2):
    robot_cfg = G1WithHandsCfg()
    robot_cfg.fix_base_link = False
    joint_names = tuple(robot_cfg.joint_limits)

    body_names = list(chairman_module.StableBaseline3VecEnv.MAIN_ROBOT_LINK_NAMES)
    if "pelvis" not in body_names:
        body_names.append("pelvis")
    body_state = torch.zeros((num_envs, len(body_names), 13), dtype=torch.float32)
    body_state[:, :, 3] = 1.0

    joint_pos = torch.zeros((num_envs, len(joint_names)), dtype=torch.float32)
    joint_vel = torch.zeros_like(joint_pos)
    for index, name in enumerate(joint_names):
        if name in RealG1MotionPolicy.JOINT_NAMES:
            policy_index = RealG1MotionPolicy.JOINT_NAMES.index(name)
            joint_pos[:, index] = float(RealG1MotionPolicy.DEFAULT_ANGLES[policy_index])

    robot_state = SimpleNamespace(
        joint_names=np.asarray(joint_names),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_names=body_names,
        body_state=body_state,
    )
    states = SimpleNamespace(robots={robot_cfg.name: robot_state})
    handler = SimpleNamespace(device=torch.device("cpu"), get_states=lambda: states)
    scenario = SimpleNamespace(
        robots=[robot_cfg],
        sim="genesis",
        sim_params=SimpleNamespace(dt=0.002),
        decimation=5,
    )
    return SimpleNamespace(
        scenario=scenario,
        num_envs=num_envs,
        env=SimpleNamespace(handler=handler),
    )


def test_chairman_action_is_upper_body_plus_walking_command(monkeypatch):
    monkeypatch.setattr(chairman_module, "G1MotionPolicy", FakeMotionPolicy)
    wrapped = chairman_module.StableBaseline3VecEnv(make_fake_metasim_env())

    assert len(wrapped.upper_body_joint_names) == 31
    assert wrapped.action_space.shape == (34,)
    assert wrapped.observation_space.shape == (292,)
    assert wrapped.action_names[-3:] == ("walk_vx", "walk_vy", "walk_yaw_rate")
    np.testing.assert_allclose(wrapped.action_space.low[-3:], -RealG1MotionPolicy.MAX_COMMAND)
    np.testing.assert_allclose(wrapped.action_space.high[-3:], RealG1MotionPolicy.MAX_COMMAND)
    assert wrapped._motion_decimation == 2


def test_chairman_composes_upper_and_motion_policy_targets(monkeypatch):
    monkeypatch.setattr(chairman_module, "G1MotionPolicy", FakeMotionPolicy)
    wrapped = chairman_module.StableBaseline3VecEnv(make_fake_metasim_env())

    upper_targets = np.zeros((wrapped.num_envs, len(wrapped.upper_body_joint_names)), dtype=np.float32)
    upper_targets[:, 0] = 0.2
    commands = np.tile(np.array([[0.4, -0.1, 0.3]], dtype=np.float32), (wrapped.num_envs, 1))
    actions = np.concatenate([upper_targets, commands], axis=1)

    full_targets = wrapped._compose_robot_targets(actions)
    assert full_targets.shape == (wrapped.num_envs, len(wrapped.sim_joint_names))
    np.testing.assert_allclose(
        full_targets[:, wrapped._leg_state_indices],
        np.tile(
            (RealG1MotionPolicy.DEFAULT_ANGLES + 0.01)[None, :],
            (wrapped.num_envs, 1),
        ),
    )
    np.testing.assert_allclose(full_targets[:, wrapped._upper_state_indices], upper_targets)
    np.testing.assert_allclose(
        wrapped.motion_policy.calls[0]["command"],
        [[0.5, 0.0, 0.0], [0.1, 0.0, 0.0]],
    )
    assert wrapped.motion_policy.calls[0]["angular_velocity_frame"] == "world"

    # motion.pt runs at 50 Hz while this environment acts at 100 Hz.
    wrapped._compose_robot_targets(actions)
    assert len(wrapped.motion_policy.calls) == 1
    wrapped._compose_robot_targets(actions)
    assert len(wrapped.motion_policy.calls) == 2


def test_chairman_partial_reset_resets_walking_memory(monkeypatch):
    monkeypatch.setattr(chairman_module, "G1MotionPolicy", FakeMotionPolicy)
    wrapped = chairman_module.StableBaseline3VecEnv(make_fake_metasim_env())
    wrapped._cached_leg_targets[:] = 123.0
    wrapped.last_locomotion_command[:] = 1.0

    wrapped._reset_motion_state([1])

    assert wrapped.motion_policy.reset_calls[-1] == [1]
    np.testing.assert_allclose(wrapped._cached_leg_targets[1], RealG1MotionPolicy.DEFAULT_ANGLES)
    np.testing.assert_allclose(wrapped.last_locomotion_command[1], 0.0)
    np.testing.assert_allclose(wrapped._cached_leg_targets[0], 123.0)


def test_full_g1_config_matches_pretrained_motion_controller():
    robot_cfg = G1WithHandsCfg()

    assert robot_cfg.num_joints == 43
    assert robot_cfg.urdf_path.endswith("g1_mygym.urdf")
    expected_kp = [100, 100, 100, 150, 40, 40] * 2
    expected_kd = [2, 2, 2, 4, 2, 2] * 2
    for index, name in enumerate(RealG1MotionPolicy.JOINT_NAMES):
        actuator = robot_cfg.actuators[name]
        assert actuator.stiffness == expected_kp[index]
        assert actuator.damping == expected_kd[index]
        assert np.isclose(
            robot_cfg.default_joint_positions[name],
            RealG1MotionPolicy.DEFAULT_ANGLES[index],
        )
