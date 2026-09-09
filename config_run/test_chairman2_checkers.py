"""CPU checks for stage transitions, failures and per-environment anchors."""
from types import SimpleNamespace

import unittest
import torch

from metasim.cfg.checkers import stages_chairman2 as c


def scene():
    names = list(c.STAGE0_JOINT_TARGETS)
    bodies = ["pelvis", "left_shoulder_roll_link", "right_shoulder_roll_link",
              "left_hand_palm_link", "right_hand_palm_link"]
    robot = SimpleNamespace(joint_names=names, joint_pos=torch.zeros(2, len(names)),
                            body_names=bodies, body_state=torch.zeros(2, len(bodies), 13))
    robot.body_state[:, :, 2] = 1.0
    robot.body_state[:, :, 3] = 1.0
    chair = SimpleNamespace(body_names=["base_link", "target_hand_left", "target_hand_right"],
                            body_state=torch.zeros(2, 3, 13))
    chair.body_state[:, :, 3] = 1.0
    chair.body_state[:, 1:, 2] = 0.8
    states = SimpleNamespace(robots={"robot": robot}, objects={"chair": chair})
    handler = SimpleNamespace(robot=SimpleNamespace(name="robot"), task=SimpleNamespace(),
                              num_envs=2, scenario=SimpleNamespace(
                                  sim_params=SimpleNamespace(dt=0.002), decimation=10))
    return states, handler, torch.tensor([True, True])


def set_pose(states, targets):
    states.robots["robot"].joint_pos[:] = torch.tensor(list(targets.values()))


def check_every_joint_and_robot_drift(stage, targets):
    states, handler, mask = scene()
    checker = getattr(c, f"stege{stage}_chacker")
    set_pose(states, targets)
    states.robots["robot"].joint_pos[0, -1] += 0.3
    terminated, success = checker(states, handler, mask)
    assert success.tolist() == [False, True]
    assert terminated.tolist() == [False, True]
    states.robots["robot"].body_state[0, 0, 0] += 0.51
    set_pose(states, targets)
    terminated, success = checker(states, handler, mask)
    assert terminated[0] and not success[0]


def test_stage2_chair_drift_and_fall_override_pose():
    states, handler, mask = scene()
    c.stege2_chacker(states, handler, mask)
    set_pose(states, c.STAGE2_JOINT_TARGETS)
    states.objects["chair"].body_state[0, 0, 0] += 0.06
    states.robots["robot"].body_state[1, 1:3, 2] = 0.2
    terminated, success = c.stege2_chacker(states, handler, mask)
    assert terminated.all() and not success.any()


def test_stage_entry_reanchors_only_active_environments():
    states, handler, mask = scene()
    c.stege0_chacker(states, handler, mask)
    states.robots["robot"].body_state[0, 0, 0] = 3.0
    set_pose(states, c.STAGE2_JOINT_TARGETS)
    _, success = c.stege2_chacker(states, handler, torch.tensor([True, False]))
    assert success.tolist() == [True, False]
    assert handler.task.chairman_robot_anchor[:, 0].tolist() == [3.0, 0.0]
    # reset_chairman invalidates recorded_stage, including same-stage resets.
    handler.task.recorded_stage[0] = -1
    states.robots["robot"].body_state[0, 0, 0] = -3.0
    _, success = c.stege2_chacker(states, handler, torch.tensor([True, False]))
    assert success[0]


def test_palms_use_chair_frame_and_both_hands():
    states, handler, mask = scene()
    # local +Y is world -X after a 90 degree yaw.
    states.objects["chair"].body_state[:, 0, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    palms = states.robots["robot"].body_state[:, 3:5, :3]
    palms[:] = states.objects["chair"].body_state[:, 1:, :3]
    palms[:, :, 0] -= c.PALM_FRONT_OFFSET
    palms[1, 1, 0] += 0.2
    _, success = c.stege3_chacker(states, handler, mask)
    assert success.tolist() == [True, False]


def test_pull_requires_direction_distance_and_both_speeds():
    states, handler, mask = scene()
    c.stege4_chacker(states, handler, mask)
    chair = states.objects["chair"].body_state
    chair[:, 0, 1] = 1.0
    chair[0, 0, 7] = 0.3
    states.robots["robot"].body_state[1, 0, 7] = 0.3
    for _ in range(c.STAGE4_HOLD_STEPS):
        assert not c.stege4_chacker(states, handler, mask)[1].any()
    chair[:, 0, 7] = 0
    states.robots["robot"].body_state[:, 0, 7] = 0
    chair[1, 0, 1] = -1.0
    for _ in range(c.STAGE4_HOLD_STEPS):
        _, success = c.stege4_chacker(states, handler, mask)
    assert success.tolist() == [True, False]


def test_lift_is_above_chair_not_arms_at_sides():
    states, handler, mask = scene()
    palms = states.robots["robot"].body_state[:, 3:5, :3]
    palms[:] = states.objects["chair"].body_state[:, 1:, :3]
    palms[:, :, 2] += c.PALM_LIFT_HEIGHT + 0.01
    palms[1, 1, 2] = 0.5
    _, success = c.stege5_chacker(states, handler, mask)
    assert success.tolist() == [True, False]


def test_walk_stops_facing_chair_and_timeout():
    states, handler, mask = scene()
    robot = states.robots["robot"].body_state
    robot[:, 0, 1] = c.CHAIR_FINAL_DISTANCE
    robot[:, 0, 3:7] = torch.tensor([2**-0.5, 0, 0, -2**-0.5])
    robot[1, 0, 7] = 0.3
    for _ in range(c.STAGE1_HOLD_STEPS):
        _, success = c.stege1_chacker(states, handler, mask)
    assert success.tolist() == [True, False]
    handler.task.stage_steps[:] = 10000
    terminated, success = c.stege1_chacker(states, handler, mask)
    assert terminated.all() and not success.any()


def test_empty_mask_does_not_initialize_stage_state():
    states, handler, mask = scene()
    for stage in range(6):
        terminated, success = getattr(c, f"stege{stage}_chacker")(states, handler, ~mask)
        assert not terminated.any() and not success.any()
    assert not hasattr(handler.task, "recorded_stage")


def test_stage0_joint_pose():
    check_every_joint_and_robot_drift(0, c.STAGE0_JOINT_TARGETS)


def test_stage2_joint_pose():
    check_every_joint_and_robot_drift(2, c.STAGE2_JOINT_TARGETS)


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith("test_")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
