import math
from types import SimpleNamespace

import numpy as np
import torch

from metasim.cfg.robots.g1_cfg_with_hands import G1WithHandsCfg
from metasim.cfg.tasks.humanoidbench.ChairMan import (
    ArmDownReward,
    ChairmanCfg,
    CloseGraspReward,
    DeltaActionRateCfg,
    DoFVelocityAccelerationCfg,
    GraspForceReward,
    HandOrientationProgressReward,
    HandTargetStillnessReward,
    KeepFingersOpenPenalty,
    MaintainAnyGraspReward,
    PulledChairStillnessReward,
    PullChairReward,
    ReachChairProgressReward,
    ReleaseFingersReward,
    Stage3HandDriftPenalty,
    StageProgressCfg,
)
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg,
    GraspForceReward as MultiGraspForceReward,
    HandOrientationProgressReward as MultiHandOrientationProgressReward,
    HandTargetStillnessReward as MultiHandTargetStillnessReward,
    MultiPolicyStageCompletionReward,
    ReachChairProgressReward as MultiReachChairProgressReward,
    Stage1ArmJointVelocityPenalty,
    WaistStraightReward,
)
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


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
    chair_position=(0.75, 0.0, 0.1),
    chair_velocity=(0.0, 0.0, 0.0),
    robot_velocity=(0.0, 0.0, 0.0),
    arm_fraction=0.0,
    waist_position=(0.0, 0.0, 0.0),
):
    joint_names = list(G1WithHandsCfg().joint_limits)
    joint_pos = torch.zeros((1, len(joint_names)), dtype=torch.float32)
    close_targets = CloseGraspReward().finger_targets_dict
    for name, target in close_targets.items():
        joint_pos[0, joint_names.index(name)] = closure_fraction * target
    for name in ArmDownReward().arm_joint_scales:
        joint_pos[0, joint_names.index(name)] = arm_fraction
    for name, value in zip(
        ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
        waist_position,
    ):
        joint_pos[0, joint_names.index(name)] = value

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
    robot_body_state[0, 0, 7:10] = torch.tensor(robot_velocity)

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
    chair_body_state[0, 0, :3] = torch.tensor(chair_position)
    chair_body_state[0, 0, 7:10] = torch.tensor(chair_velocity)
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


def test_delta_action_rate_reinitializes_when_genesis_dof_layout_changes():
    reward = DeltaActionRateCfg()
    reset_states = SimpleNamespace(
        robots={
            "g1_with_hands": SimpleNamespace(
                joint_pos_target=torch.zeros((2, 49), dtype=torch.float32)
            )
        }
    )
    reward.reset(torch.tensor([0, 1]), reset_states)

    step_robot = SimpleNamespace(
        joint_pos_target=torch.zeros((2, 43), dtype=torch.float32),
        joint_names=np.asarray(list(G1WithHandsCfg().joint_limits)),
    )
    step_states = SimpleNamespace(robots={"g1_with_hands": step_robot})
    first_penalty = reward(step_states, "g1_with_hands")
    torch.testing.assert_close(first_penalty, torch.zeros(2))
    assert reward.prev_actions.shape == (2, 43)

    step_robot.joint_pos_target = torch.full((2, 43), 0.35)
    second_penalty = reward(step_states, "g1_with_hands")
    torch.testing.assert_close(second_penalty, torch.ones(2))


def test_dof_velocity_reward_reinitializes_when_genesis_dof_layout_changes():
    reward = DoFVelocityAccelerationCfg()
    reset_robot = SimpleNamespace(joint_vel=torch.zeros((2, 49), dtype=torch.float32))
    reward.reset(
        torch.tensor([0, 1]),
        SimpleNamespace(robots={"g1_with_hands": reset_robot}),
    )

    joint_names = np.asarray(list(G1WithHandsCfg().joint_limits))
    step_robot = SimpleNamespace(
        joint_names=joint_names,
        joint_vel=torch.zeros((2, 43), dtype=torch.float32),
    )
    states = SimpleNamespace(robots={"g1_with_hands": step_robot})
    penalty = reward(states, "g1_with_hands")
    torch.testing.assert_close(penalty, torch.zeros(2))
    assert reward.prev_joint_vel.shape == (2, 43)


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


def test_stage1_arm_joint_velocity_penalty_is_thresholded_and_exponential():
    reward = Stage1ArmJointVelocityPenalty()
    states = _states()
    shoulder_idx = list(states.robots["g1_with_hands"].joint_names).index(
        "left_shoulder_pitch_joint"
    )

    def penalty_at(speed, stage=1):
        states.robots["g1_with_hands"].joint_vel.zero_()
        states.robots["g1_with_hands"].joint_vel[0, shoulder_idx] = speed
        return _evaluate(reward, states, stage)

    assert penalty_at(1.5) == 0.0
    low_excess = penalty_at(2.0)
    medium_excess = penalty_at(2.5)
    full_excess = penalty_at(3.0)
    assert 0.0 < low_excess < medium_excess < full_excess <= 1.0
    assert (medium_excess - low_excess) > low_excess
    assert penalty_at(3.0, stage=0) == 0.0

    config = ChairmanmultiCfg()
    reward_names = [type(item).__name__ for item in config.reward_functions]
    reward_index = reward_names.index("Stage1ArmJointVelocityPenalty")
    assert config.reward_weights[reward_index] == -4.0


def test_waist_straight_reward_is_active_only_in_stages_1_and_2():
    reward = WaistStraightReward()
    straight = _states(waist_position=(0.0, 0.0, 0.0))
    slightly_bent = _states(waist_position=(0.10, 0.10, 0.10))
    bent = _states(waist_position=(0.30, 0.30, 0.30))

    assert math.isclose(_evaluate(reward, straight, 1), 1.0, abs_tol=1e-6)
    assert math.isclose(_evaluate(reward, straight, 2), 1.0, abs_tol=1e-6)
    assert _evaluate(reward, straight, 1) > _evaluate(reward, slightly_bent, 1)
    assert _evaluate(reward, slightly_bent, 1) > _evaluate(reward, bent, 1)
    assert math.isclose(_evaluate(reward, straight, 0), 0.0, abs_tol=1e-6)
    assert math.isclose(_evaluate(reward, straight, 3), 0.0, abs_tol=1e-6)

    config = ChairmanmultiCfg()
    reward_names = [type(item).__name__ for item in config.reward_functions]
    reward_index = reward_names.index("WaistStraightReward")
    assert config.reward_weights[reward_index] == 2.0


def test_multi_stage1_rewards_match_full_task_reward_functions():
    far = _states(
        left_position=(0.0, 0.2, 1.0),
        right_position=(0.0, -0.2, 1.0),
        left_quaternion=_yaw_quaternion(math.pi),
        right_quaternion=_yaw_quaternion(math.pi),
        hand_velocity=0.30,
    )
    closer = _states(
        left_position=(0.45, 0.2, 1.0),
        right_position=(0.45, -0.2, 1.0),
        left_quaternion=_yaw_quaternion(math.pi / 2.0),
        right_quaternion=_yaw_quaternion(math.pi / 2.0),
        hand_velocity=0.10,
    )
    farther_again = _states(
        left_position=(0.40, 0.2, 1.0),
        right_position=(0.40, -0.2, 1.0),
        left_quaternion=_yaw_quaternion(math.pi),
        right_quaternion=_yaw_quaternion(math.pi),
        hand_velocity=0.30,
    )

    reward_pairs = (
        (ReachChairProgressReward(), MultiReachChairProgressReward()),
        (HandOrientationProgressReward(), MultiHandOrientationProgressReward()),
        (HandTargetStillnessReward(), MultiHandTargetStillnessReward()),
    )
    for full_reward, multi_reward in reward_pairs:
        full_reward.actual_stage = torch.tensor([1], dtype=torch.long)
        multi_reward.actual_stage = torch.tensor([1], dtype=torch.long)
        for state in (far, closer, closer, farther_again):
            torch.testing.assert_close(
                multi_reward(state, "g1_with_hands"),
                full_reward(state, "g1_with_hands"),
            )


def test_multi_grasp_force_matches_two_of_three_checker_rule_per_hand():
    near_kwargs = {
        "left_position": (0.5, 0.2, 1.0),
        "right_position": (0.5, -0.2, 1.0),
    }
    one_tip_each = _states(
        **near_kwargs, contact_forces=[2, 0, 0, 2, 0, 0]
    )
    two_tips_each = _states(
        **near_kwargs, contact_forces=[2, 2, 0, 2, 2, 0]
    )
    three_tips_each = _states(
        **near_kwargs, contact_forces=[2, 2, 2, 2, 2, 2]
    )

    reward = MultiGraspForceReward()
    one_score = _evaluate(reward, one_tip_each, 2)
    two_score = _evaluate(reward, two_tips_each, 2)
    three_score = _evaluate(reward, three_tips_each, 2)
    assert two_score > one_score
    assert three_score >= two_score


def test_stage3_grasp_and_hand_drift_terms_are_zero_at_the_safe_goal():
    target_kwargs = {
        "left_position": (0.5, 0.2, 1.0),
        "right_position": (0.5, -0.2, 1.0),
    }
    robust_grasp = _states(**target_kwargs, contact_forces=[1, 1, 1, 1, 1, 1])
    lost_grasp = _states(**target_kwargs, contact_forces=[0, 0, 0, 0, 0, 0])
    assert math.isclose(
        _evaluate(MaintainAnyGraspReward(), robust_grasp, 3), 0.0, abs_tol=1e-6
    )
    assert _evaluate(MaintainAnyGraspReward(), lost_grasp, 3) < -0.9

    safe = _states(**target_kwargs)
    drifting = _states(
        left_position=(0.58, 0.2, 1.0),
        right_position=(0.5, -0.2, 1.0),
    )
    assert math.isclose(
        _evaluate(Stage3HandDriftPenalty(), safe, 3), 0.0, abs_tol=1e-6
    )
    assert _evaluate(Stage3HandDriftPenalty(), drifting, 3) > 0.5


def test_stage3_pull_prefers_continuous_motion_and_penalizes_a_pause():
    reward = PullChairReward()
    moving = _states(
        chair_position=(0.25, 0.0, 0.1),
        chair_velocity=(-0.35, 0.0, 0.0),
    )
    paused = _states(chair_position=(0.2465, 0.0, 0.1))
    reversed_motion = _states(
        chair_position=(0.2515, 0.0, 0.1),
        chair_velocity=(0.20, 0.0, 0.0),
    )

    initial = _evaluate(reward, moving, 3)
    forward_progress = _evaluate(
        reward,
        _states(
            chair_position=(0.2465, 0.0, 0.1),
            chair_velocity=(-0.35, 0.0, 0.0),
        ),
        3,
    )
    pause_reward = _evaluate(reward, paused, 3)
    reverse_reward = _evaluate(reward, reversed_motion, 3)

    assert initial > 0.0
    assert forward_progress > initial
    assert pause_reward < 0.0
    assert reverse_reward < pause_reward


def test_stage4_stability_is_a_penalty_and_finger_release_has_signed_progress():
    stable = _states(chair_position=(-0.25, 0.0, 0.1))
    moving = _states(
        chair_position=(-0.25, 0.0, 0.1),
        chair_velocity=(0.10, 0.0, 0.0),
    )
    assert math.isclose(
        _evaluate(PulledChairStillnessReward(), stable, 4), 0.0, abs_tol=1e-6
    )
    assert _evaluate(PulledChairStillnessReward(), moving, 4) > 0.0

    reward = ReleaseFingersReward()
    closed = _states(closure_fraction=1.0, chair_position=(-0.25, 0.0, 0.1))
    half_open = _states(closure_fraction=0.5, chair_position=(-0.25, 0.0, 0.1))
    fully_open = _states(closure_fraction=0.0, chair_position=(-0.25, 0.0, 0.1))
    _evaluate(reward, closed, 4)
    opening_reward = _evaluate(reward, half_open, 4)
    final_reward = _evaluate(reward, fully_open, 4)
    reclosing_reward = _evaluate(reward, half_open, 4)
    assert opening_reward > 0.0
    assert final_reward > opening_reward
    assert reclosing_reward < 0.0


def test_stage5_arm_lowering_has_signed_progress_and_fingers_stay_open():
    reward = ArmDownReward()
    arms_up = _states(arm_fraction=1.0, chair_position=(-0.25, 0.0, 0.1))
    arms_halfway = _states(arm_fraction=0.5, chair_position=(-0.25, 0.0, 0.1))
    arms_down = _states(arm_fraction=0.0, chair_position=(-0.25, 0.0, 0.1))
    _evaluate(reward, arms_up, 5)
    lowering_reward = _evaluate(reward, arms_halfway, 5)
    final_reward = _evaluate(reward, arms_down, 5)
    raising_reward = _evaluate(reward, arms_halfway, 5)
    assert lowering_reward > 0.0
    assert final_reward > lowering_reward
    assert raising_reward < 0.0

    open_fingers = _states(closure_fraction=0.0)
    closing_fingers = _states(closure_fraction=0.5)
    assert math.isclose(
        _evaluate(KeepFingersOpenPenalty(), open_fingers, 5), 0.0, abs_tol=1e-6
    )
    assert _evaluate(KeepFingersOpenPenalty(), closing_fingers, 5) > 0.9


def test_full_and_multi_tasks_activate_balanced_stage3_to_stage5_rewards():
    for config_cls, completion_cls in (
        (ChairmanCfg, StageProgressCfg),
        (ChairmanmultiCfg, MultiPolicyStageCompletionReward),
    ):
        config = config_cls()
        assert len(config.reward_functions) == len(config.reward_weights)
        weights = {
            type(reward).__name__: weight
            for reward, weight in zip(config.reward_functions, config.reward_weights)
        }
        assert weights["MaintainAnyGraspReward"] == 6.0
        assert weights["Stage3HandDriftPenalty"] == -6.0
        assert weights["PullChairReward"] == 16.0
        assert weights["PulledChairStillnessReward"] == -8.0
        assert weights["ReleaseFingersReward"] == 14.0
        assert weights["ArmDownReward"] == 14.0
        assert weights["KeepFingersOpenPenalty"] == -3.0
        assert isinstance(config.reward_functions[-1], completion_cls)


def test_multi_completion_bonus_uses_previous_stage_and_has_single_consumer():
    config = ChairmanmultiCfg()
    reward_names = [type(reward).__name__ for reward in config.reward_functions]
    assert "StageProgressCfg" not in reward_names
    assert reward_names.count("MultiPolicyStageCompletionReward") == 1
    completion_index = reward_names.index("MultiPolicyStageCompletionReward")
    assert config.reward_weights[completion_index] == 100.0

    current_stage = torch.tensor([1], dtype=torch.long)
    reward_stage = torch.tensor([0], dtype=torch.long)

    class CapturingDenseReward:
        def __init__(self):
            self.actual_stage = current_stage
            self.seen_stage = None

        def __call__(self, states, robot_name):
            self.seen_stage = self.actual_stage.clone()
            return (self.actual_stage == 0).float()

    dense = CapturingDenseReward()
    completion = MultiPolicyStageCompletionReward()
    completion.actual_stage = current_stage
    completion.completed_stages = torch.ones(1, dtype=torch.long)
    states = SimpleNamespace(
        robots={
            "g1_with_hands": SimpleNamespace(joint_pos=torch.zeros((1, 1)))
        }
    )
    task = SimpleNamespace(
        reward_functions=[dense, completion],
        reward_weights=[1.0, 100.0],
        reward_stage=reward_stage,
    )
    fake_vec_env = SimpleNamespace(
        num_envs=1,
        env=SimpleNamespace(
            handler=SimpleNamespace(
                device=torch.device("cpu"),
                get_states=lambda: states,
            )
        ),
        scenario=SimpleNamespace(
            task=task,
            robots=[SimpleNamespace(name="g1_with_hands")],
        ),
    )

    total = MetaSimVecEnv._calculate_rewards(fake_vec_env)
    torch.testing.assert_close(total, torch.tensor([101.0]))
    torch.testing.assert_close(dense.seen_stage, reward_stage)
    assert dense.actual_stage is current_stage
    assert completion.actual_stage is current_stage
    torch.testing.assert_close(completion.completed_stages, torch.zeros(1, dtype=torch.long))
