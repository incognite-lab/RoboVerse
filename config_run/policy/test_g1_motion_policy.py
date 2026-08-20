from pathlib import Path

import numpy as np

from config_run.policy.g1_motion_policy import G1MotionPolicy


POLICY_PATH = Path(__file__).with_name("motion.pt")


def make_policy(**kwargs):
    return G1MotionPolicy(POLICY_PATH, **kwargs)


def standing_inputs(batch_size=None):
    q = G1MotionPolicy.DEFAULT_ANGLES.copy()
    dq = np.zeros(12, dtype=np.float32)
    omega = np.zeros(3, dtype=np.float32)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    if batch_size is not None:
        q = np.tile(q, (batch_size, 1))
        dq = np.tile(dq, (batch_size, 1))
    return {
        "joint_positions": q,
        "joint_velocities": dq,
        "angular_velocity": omega,
        "base_quaternion_wxyz": quaternion,
        "command": command,
    }


def test_observation_layout_and_scaling():
    policy = make_policy(clip_commands=False)
    q = G1MotionPolicy.DEFAULT_ANGLES + 0.1
    dq = np.full(12, 2.0, dtype=np.float32)
    previous = np.full(12, 0.3, dtype=np.float32)
    obs = policy.build_observation(
        joint_positions=q,
        joint_velocities=dq,
        angular_velocity=[1.0, 2.0, 3.0],
        base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        command=[0.5, -0.25, 1.0],
        previous_action=previous,
        time_seconds=0.2,
    )[0]

    np.testing.assert_allclose(obs[0:3], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(obs[6:9], [1.0, -0.5, 0.25])
    np.testing.assert_allclose(obs[9:21], 0.1, atol=1e-6)
    np.testing.assert_allclose(obs[21:33], 0.1)
    np.testing.assert_allclose(obs[33:45], 0.3)
    np.testing.assert_allclose(obs[45:47], [1.0, 0.0], atol=1e-6)


def test_policy_returns_joint_positions_and_reset_is_deterministic():
    policy = make_policy()
    inputs = standing_inputs()

    first = policy.predict_joint_positions(**inputs)
    second = policy.predict_joint_positions(**inputs)
    assert first.shape == (12,)
    assert np.all(np.isfinite(first))
    assert np.all(first >= G1MotionPolicy.JOINT_LOWER_LIMITS)
    assert np.all(first <= G1MotionPolicy.JOINT_UPPER_LIMITS)
    assert not np.allclose(first, second)

    policy.reset()
    first_after_reset = policy.predict_joint_positions(**inputs)
    np.testing.assert_allclose(first, first_after_reset)


def test_vectorized_policy_and_partial_reset():
    policy = make_policy()
    inputs = standing_inputs(batch_size=3)
    targets = policy.predict_joint_positions(**inputs)
    assert targets.shape == (3, 12)
    assert policy.batch_size == 3

    hidden_before = policy._policy.hidden_state.detach().clone()
    policy.reset([1])
    hidden_after = policy._policy.hidden_state.detach()
    assert np.any(hidden_after[:, 0, :].cpu().numpy() != 0.0)
    assert np.all(hidden_after[:, 1, :].cpu().numpy() == 0.0)
    np.testing.assert_array_equal(hidden_before[:, 2, :].cpu(), hidden_after[:, 2, :].cpu())


def test_mapping_api_uses_named_joint_order():
    policy = make_policy()
    q = dict(zip(G1MotionPolicy.JOINT_NAMES, G1MotionPolicy.DEFAULT_ANGLES.tolist()))
    dq = dict.fromkeys(G1MotionPolicy.JOINT_NAMES, 0.0)
    targets = policy.predict_joint_dict(
        joint_positions=q,
        joint_velocities=dq,
        angular_velocity=[0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        command=[0.0, 0.0, 0.0],
    )
    assert tuple(targets) == G1MotionPolicy.JOINT_NAMES
    assert all(np.isfinite(value) for value in targets.values())
