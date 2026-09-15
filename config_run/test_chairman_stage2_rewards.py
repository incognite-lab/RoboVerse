"""CPU behavioral checks for stage-2 retention and isolated reward weights."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg, Stage2HandRetentionReward, WaistStraightReward,
    STAGE2_REWARD_WEIGHTS,
)
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


def scene():
    robot = SimpleNamespace(
        body_names=["left_endeffector", "endeffector"],
        body_state=torch.zeros(6, 2, 13),
    )
    chair = SimpleNamespace(
        body_names=["target_hand_left", "target_hand_right"],
        body_state=torch.zeros(6, 2, 13),
    )
    return SimpleNamespace(robots={"g1_with_hands": robot}, objects={"chair": chair})


def reward_env(task, states, count=6):
    return SimpleNamespace(
        num_envs=count,
        scenario=SimpleNamespace(task=task, robots=[SimpleNamespace(name="g1_with_hands")]),
        env=SimpleNamespace(handler=SimpleNamespace(device=torch.device("cpu"), get_states=lambda: states)),
    )


class Stage2RewardsTest(unittest.TestCase):
    def test_retention_is_continuous_and_uses_worse_hand(self):
        states = scene()
        reward = Stage2HandRetentionReward()
        reward.actual_stage = torch.full((6,), 2)
        distances = torch.tensor([0.0, 0.049, 0.051, 0.07, 0.10, 0.15])
        states.robots["g1_with_hands"].body_state[:, 1, 0] = distances
        result = reward(states)
        torch.testing.assert_close(result, torch.exp(-(distances / 0.10).square()))
        self.assertTrue(torch.all(result[:-1] > result[1:]))
        self.assertGreater(result[3].item(), 0.6)  # valid stage-1 entry at 7 cm
        self.assertGreater(result[-1].item(), 0.0)  # no hard cutoff
        # Improving the already perfect hand cannot hide the worse one.
        states.robots["g1_with_hands"].body_state[:, 0, 0] = distances
        torch.testing.assert_close(reward(states), result)
        # Moving chair and robot together leaves distances unchanged.
        for obj in (states.robots["g1_with_hands"], states.objects["chair"]):
            obj.body_state[:, :, :3] += torch.tensor([2.0, -3.0, 1.0])
        torch.testing.assert_close(reward(states), result)

    def test_retention_only_rewards_stage2_and_validates_scale(self):
        states = scene()
        reward = Stage2HandRetentionReward()
        torch.testing.assert_close(reward(states), torch.zeros(6))
        reward.actual_stage = torch.arange(6)
        torch.testing.assert_close(reward(states), torch.tensor([0., 0., 1., 0., 0., 0.]))
        for scale in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Stage2HandRetentionReward(distance_scale=scale)

    def test_shared_weights_change_only_stage2(self):
        task = ChairmanmultiCfg()
        self.assertEqual(len(task.reward_functions), len(task.reward_weights))
        tested = set()
        for reward, weight in zip(task.reward_functions, task.reward_weights):
            name = type(reward).__name__
            if name not in STAGE2_REWARD_WEIGHTS:
                continue
            tested.add(name)
            reward.actual_stage = torch.arange(6)
            isolated = SimpleNamespace(
                reward_functions=[reward], reward_weights=[weight],
                stage_reward_weights={2: task.stage_reward_weights[2]},
            )
            with patch.object(type(reward), "__call__", return_value=torch.ones(6)):
                result = MetaSimVecEnv._calculate_rewards(reward_env(isolated, None))
            expected = torch.full((6,), weight)
            expected[2] = STAGE2_REWARD_WEIGHTS[name]
            torch.testing.assert_close(result, expected)
        self.assertEqual(tested, set(STAGE2_REWARD_WEIGHTS))

    def test_successful_exit_uses_stage2_weight_and_restores_routing(self):
        reward = WaistStraightReward()
        routing = torch.tensor([2, 3, 3])
        reward.actual_stage = routing
        states = SimpleNamespace(robots={"g1_with_hands": SimpleNamespace(
            joint_names=list(reward.joint_names), joint_pos=torch.zeros(3, 3),
        )})
        task = SimpleNamespace(
            reward_functions=[reward], reward_weights=[0.01],
            reward_stage=torch.tensor([1, 2, 3]),
            stage_reward_weights={2: STAGE2_REWARD_WEIGHTS},
        )
        result = MetaSimVecEnv._calculate_rewards(reward_env(task, states, count=3))
        torch.testing.assert_close(result, torch.tensor([0.01, STAGE2_REWARD_WEIGHTS["WaistStraightReward"], 0.0]))
        self.assertIs(reward.actual_stage, routing)
        # Existing tasks with no overrides still use scalar weights.
        del task.stage_reward_weights
        result = MetaSimVecEnv._calculate_rewards(reward_env(task, states, count=3))
        torch.testing.assert_close(result, torch.tensor([0.01, 0.01, 0.0]))


if __name__ == "__main__":
    unittest.main()
