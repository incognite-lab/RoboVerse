"""Behavioral checks of the complete weighted stage-1 reward on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config_run.test_chairman_manipulation_rewards import _states
from config_run.test_chairman_stage2_rewards import reward_env
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg, STAGE1_REWARD_WEIGHTS,
    MULTI_POLICY_STAGE_COMPLETION_WEIGHT,
)
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


def scene(distance=0.0, speed=0.0):
    states = _states(
        left_position=(0.5 - distance, 0.2, 1.0),
        right_position=(0.5 - distance, -0.2, 1.0),
        hand_velocity=speed,
    )
    robot = states.robots["g1_with_hands"]
    robot.joint_pos_target = robot.joint_pos.clone()
    torso = torch.zeros(1, 1, 13)
    torso[:, :, 3] = 1.0
    robot.body_names.append("torso_link")
    robot.body_state = torch.cat((robot.body_state, torso), dim=1)
    arm_ids = [i for i, name in enumerate(robot.joint_names)
               if any(part in name for part in ("shoulder", "elbow", "wrist"))]
    robot.joint_vel[:, arm_ids] = speed
    return states


def task_for(states):
    task = ChairmanmultiCfg()
    stages = torch.ones(1, dtype=torch.long)
    completed = torch.zeros(1, dtype=torch.long)
    for reward in task.reward_functions:
        reward.actual_stage = stages
        reward.completed_stages = completed
        if hasattr(reward, "termination_events"):
            reward.termination_events = torch.zeros(1, dtype=torch.bool)
        if hasattr(reward, "set_control_context"):
            reward.set_control_context(torch.zeros(1, 3), torch.zeros(1, 3))
    return task


def total(task, states):
    return MetaSimVecEnv._calculate_rewards(reward_env(task, states, count=1)).item()


class Stage1BalanceTest(unittest.TestCase):
    def test_calm_approach_and_goal_have_positive_reward(self):
        values = []
        for distance, speed in ((0.30, 0.10), (0.15, 0.10), (0.05, 0.05), (0.0, 0.0)):
            with self.subTest(distance=distance, speed=speed):
                states = scene(distance, speed)
                task = task_for(states)
                total(task, states)  # initialize histories
                # A small real target change, plus progress towards the goal.
                states.robots["g1_with_hands"].joint_pos_target += 0.01
                if distance:
                    states.robots["g1_with_hands"].body_state[:, 1:3, 0] += 0.005
                value = total(task, states)
                self.assertGreater(value, 0.0)
                values.append(value)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])), values)

    def test_departure_and_violent_motion_are_worse_than_holding(self):
        states = scene()
        task = task_for(states)
        holding = total(task, states)
        robot = states.robots["g1_with_hands"]
        robot.body_state[:, 1:3, 0] -= 0.40
        robot.body_state[:, 1:3, 7] = 2.0
        robot.joint_pos_target += 1.0
        robot.joint_vel.fill_(6.0)
        departing = total(task, states)
        self.assertLess(departing, holding)
        # Reward ordering matters: correct waist/orientation can still add
        # positive terms during an otherwise undesirable transition.
        fast = scene(distance=0.15, speed=6.0)
        calm = scene(distance=0.15, speed=0.0)
        self.assertLess(total(task_for(fast), fast), total(task_for(calm), calm))

    def test_terminal_bonus_and_failure_penalty_survive_reweighting(self):
        states = scene()
        for successful, event_reward in ((True, 500.0), (False, -250.0)):
            with self.subTest(successful=successful):
                task = task_for(states)
                dense = total(task, states)
                if successful:
                    task.reward_stage = torch.ones(1, dtype=torch.long)
                    task.reward_functions[0].actual_stage.fill_(2)
                    task.reward_functions[0].completed_stages.fill_(1)
                else:
                    task.reward_functions[0].termination_events.fill_(True)
                self.assertAlmostEqual(total(task, states) - dense, event_reward, places=4)

    def test_overrides_leave_other_stages_unchanged(self):
        task = ChairmanmultiCfg()
        for reward, weight in zip(task.reward_functions, task.reward_weights):
            name = type(reward).__name__
            if name not in STAGE1_REWARD_WEIGHTS:
                continue
            reward.actual_stage = torch.arange(6)
            isolated = SimpleNamespace(
                reward_functions=[reward], reward_weights=[weight],
                stage_reward_weights=task.stage_reward_weights,
            )
            with patch.object(type(reward), "__call__", return_value=torch.ones(6)):
                updated = MetaSimVecEnv._calculate_rewards(reward_env(isolated, None))
                isolated.stage_reward_weights = {stage: weights for stage, weights in task.stage_reward_weights.items() if stage != 1}
                original = MetaSimVecEnv._calculate_rewards(reward_env(isolated, None))
            mask = torch.arange(6) != 1
            torch.testing.assert_close(updated[mask], original[mask])
            self.assertAlmostEqual(updated[1].item(), STAGE1_REWARD_WEIGHTS[name], places=6)

    def test_positive_shaping_budget_below_completion_discount_cost(self):
        maxima = {"HandOrientationProgressReward": 1.375, "PreciseHandTargetReward": 1.5}
        upper_bound = sum(weight * maxima.get(name, 1.0)
                          for name, weight in STAGE1_REWARD_WEIGHTS.items() if weight > 0)
        self.assertLess(upper_bound, (1 - 0.995) * MULTI_POLICY_STAGE_COMPLETION_WEIGHT)


if __name__ == "__main__":
    unittest.main()
