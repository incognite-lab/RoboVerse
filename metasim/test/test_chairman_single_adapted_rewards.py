"""CPU regression checks for a single policy across the full ChairMan task."""
import copy
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.tasks.humanoidbench import ChairMan as single
from metasim.cfg.tasks.humanoidbench import ChairMan_multi as multi
from metasim.test.test_chairman_multi_adapted_rewards import ROBOT, states_for_stages
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


def make_env():
    states = states_for_stages()
    cfg = single.ChairmanCfg()
    cfg.reward_functions = copy.deepcopy(cfg.reward_functions)
    cfg.snapshot_save_probability = 0.0
    stages = torch.arange(6)
    completed = torch.zeros(6, dtype=torch.long)
    for reward in cfg.reward_functions:
        reward.actual_stage = stages
        reward.completed_stages = completed
    handler = SimpleNamespace(
        device=torch.device("cpu"), num_envs=6, task=cfg,
        robot=SimpleNamespace(name=ROBOT), get_states=lambda: states,
    )
    env = object.__new__(MetaSimVecEnv)
    env.num_envs = 6
    env.env = SimpleNamespace(handler=handler, close=lambda: None)
    env.scenario = SimpleNamespace(task=cfg, robots=[handler.robot])
    return env, handler, states


class ChairmanSingleAdaptedRewardsTest(unittest.TestCase):
    def test_body_calculations_match_multi_with_independent_histories(self):
        states = states_for_stages()
        for configured in single.ChairmanCfg().reward_functions:
            name = type(configured).__name__
            if name == "StageCompletionReward":
                continue
            with self.subTest(reward=name):
                a, b = type(configured)(), getattr(multi, name)()
                a.actual_stage, b.actual_stage = torch.arange(6), torch.arange(6)
                if name == "TerminationCfg":
                    a.termination_events = b.termination_events = torch.arange(6) == 3
                for frame in range(3):
                    current = copy.deepcopy(states)
                    current.robots[ROBOT].joint_pos += 0.02 * frame
                    current.robots[ROBOT].joint_pos_target += 0.03 * frame
                    current.robots[ROBOT].body_state[:, 1, 0] += 0.05 * frame
                    current.objects["chair"].body_state[:, 0, 0] -= 0.02 * frame
                    left, right = a(current, ROBOT), b(current, ROBOT)
                    self.assertTrue(torch.isfinite(left).all())
                    torch.testing.assert_close(left, right)
                    if frame == 1 and hasattr(a, "reset"):
                        a.reset(torch.tensor([2, 4]), current)
                        b.reset(torch.tensor([2, 4]), current)

    def test_all_weights_are_registered_and_dense_scale_is_bounded(self):
        cfg = single.ChairmanCfg()
        names = [type(r).__name__ for r in cfg.reward_functions]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(cfg.reward_weights, [0.0] * len(names))
        used = set()
        for stage, weights in cfg.stage_reward_weights.items():
            used.update(weights)
            self.assertFalse(set(weights) - set(names))
            self.assertEqual(weights["StageCompletionReward"], 500.0)
            adapted = {n.replace("MultiPolicyStageCompletionReward", "StageCompletionReward"): w
                       for n, w in multi.ChairmanmultiCfg().stage_reward_weights[stage].items()}
            self.assertEqual(weights, adapted)
            positive_dense = sum(max(w, 0.0) for n, w in weights.items() if n != "StageCompletionReward")
            self.assertLess(positive_dense, 2.5)
        self.assertEqual(set(names), used)
        self.assertNotIn("ContinuousStageReward", names)
        self.assertNotIn("StageProgressCfg", names)

    def test_checker_transitions_and_rewards_continue_until_full_success(self):
        env, handler, states = make_env()
        task = handler.task
        with ExitStack() as stack:
            for stage in range(6):
                stack.enter_context(patch(
                    f"metasim.cfg.checkers.stages_chairman.stege{stage}_chacker",
                    side_effect=lambda states, handler, mask: (torch.zeros_like(mask), mask.clone()),
                ))
            terminated = _ChairManChecker().check(handler)
        self.assertFalse(terminated.any())
        torch.testing.assert_close(task.just_finished, torch.tensor([False] * 5 + [True]))
        torch.testing.assert_close(task.reward_stage, torch.arange(6))
        torch.testing.assert_close(task.reward_functions[0].actual_stage, torch.arange(1, 7))
        total = env._calculate_rewards()
        expected = torch.zeros(6)
        for name, raw in task.last_raw_reward_terms.items():
            weights = torch.tensor([task.stage_reward_weights[s].get(name, 0.0) for s in range(6)])
            contribution = weights * raw
            torch.testing.assert_close(task.last_reward_terms[name], contribution)
            expected += contribution
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(task.last_reward_terms["StageCompletionReward"], torch.full((6,), 500.0))
        torch.testing.assert_close(task.completed_stage_events, torch.arange(6))
        # Re-evaluation does not pay the same stage completion twice.
        env._calculate_rewards()
        self.assertFalse(task.last_reward_terms["StageCompletionReward"].any())

    def test_physical_failure_gets_penalty_without_completion(self):
        env, handler, states = make_env()
        with ExitStack() as stack:
            for stage in range(6):
                stack.enter_context(patch(
                    f"metasim.cfg.checkers.stages_chairman.stege{stage}_chacker",
                    side_effect=lambda states, handler, mask: (mask.clone(), torch.zeros_like(mask)),
                ))
            terminated = _ChairManChecker().check(handler)
        self.assertTrue(terminated.all())
        env._calculate_rewards()
        torch.testing.assert_close(handler.task.last_reward_terms["TerminationCfg"],
                                   torch.tensor([-10.0] * 3 + [-250.0] * 3))
        self.assertFalse(handler.task.last_reward_terms["StageCompletionReward"].any())

    def test_completion_handles_uninitialized_state_and_consumes_flag_once(self):
        states = states_for_stages()
        reward = single.StageCompletionReward()
        self.assertFalse(reward(states, ROBOT).any())
        reward.actual_stage = torch.arange(6)  # Stage 0 must receive its bonus too.
        reward.completed_stages = torch.tensor([1, 0, 1, 0, 1, 1])
        torch.testing.assert_close(reward(states, ROBOT), torch.tensor([1., 0., 1., 0., 1., 1.]))
        self.assertFalse(reward(states, ROBOT).any())


if __name__ == "__main__":
    unittest.main()
