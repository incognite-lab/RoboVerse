"""Behavioral regressions for multi-stage reward incentives and curriculum."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config_run.test_chairman_stage1_balance import scene, task_for, total
from config_run.test_multi_ppo_actor_inheritance import config, actor_state
from config_run.test_multi_ppo_trainer import FakeStageVecEnv
from config_run.multi_ppo_trainer import MultiPPOTrainer
from metasim.cfg.checkers import stages_chairman as checkers
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    ChairmanmultiCfg, ReleaseFingersReward, ArmDownReward, PullChairReward,
    STAGE1_REWARD_WEIGHTS, STAGE2_REWARD_WEIGHTS, LATE_STAGE_REWARD_WEIGHTS,
    PULL_CHAIR_REWARD_WEIGHT,
)


def handler():
    return SimpleNamespace(num_envs=1, device=torch.device("cpu"),
        robot=SimpleNamespace(name="g1_with_hands"), task=SimpleNamespace(),
        scenario=SimpleNamespace(sim_params=SimpleNamespace(dt=0.002), decimation=5))


def stable_reward(stage, angle):
    states = scene()
    robot = states.robots["g1_with_hands"]
    states.objects["chair"].body_state[:, 0, :3] = torch.tensor([-.25, 0, .1])
    names = ReleaseFingersReward().finger_targets_dict if stage == 4 else ArmDownReward().arm_joint_scales
    for name in names:
        robot.joint_pos[:, list(robot.joint_names).index(name)] = angle
    robot.joint_pos_target = robot.joint_pos.clone()
    task = task_for(states)
    task.reward_functions[0].actual_stage.fill_(stage)
    total(task, states)
    return total(task, states)


class TrainingSafeguardsTest(unittest.TestCase):
    def test_waiting_near_late_goals_is_worse_than_completing(self):
        for stage, near, goal in ((4, .151, .149), (5, .351, .349)):
            waiting = stable_reward(stage, near)
            terminal = 500 + stable_reward(stage, goal)
            self.assertLess(waiting + .995 * terminal, terminal)

    def test_pull_rewards_progress_without_large_starting_cost(self):
        states = scene()
        reward = PullChairReward()
        reward.actual_stage = torch.tensor([3])
        reward(states, "g1_with_hands")
        waiting = reward(states, "g1_with_hands").item() * PULL_CHAIR_REWARD_WEIGHT
        self.assertGreaterEqual(waiting, 0.0)
        self.assertLess(waiting, .01)
        chair = states.objects["chair"].body_state
        chair[:, 0, 0] -= .004
        chair[:, 0, 7] = -.35
        pulling = reward(states, "g1_with_hands").item() * PULL_CHAIR_REWARD_WEIGHT
        self.assertGreater(pulling, waiting + .5)
        chair[:, 0, 0] += .004
        chair[:, 0, 7] = .35
        self.assertLess(reward(states, "g1_with_hands").item(), 0.0)

    def test_dense_budgets_do_not_pay_for_delaying_success_or_rushing_failure(self):
        task = ChairmanmultiCfg()
        active = {
            1: set(STAGE1_REWARD_WEIGHTS),
            2: set(STAGE2_REWARD_WEIGHTS) | {"CloseGraspReward", "GraspForceReward", "Stage2HandRetentionReward", "PreciseHandTargetReward"},
            3: set(LATE_STAGE_REWARD_WEIGHTS) | {"PullChairReward", "MaintainAnyGraspReward", "Stage3HandDriftPenalty"},
            4: set(LATE_STAGE_REWARD_WEIGHTS) | {"ReleaseFingersReward", "PulledChairStillnessReward"},
            5: set(LATE_STAGE_REWARD_WEIGHTS) | {"ArmDownReward", "PulledChairStillnessReward", "KeepFingersOpenPenalty"},
        }
        ranges = {"HandOrientationProgressReward": (-1, 1.375),
                  "HandTargetStillnessReward": (-1, 1), "CloseGraspReward": (-.2, 1),
                  "PullChairReward": (-1, 1), "MaintainAnyGraspReward": (-1, 0),
                  "ReleaseFingersReward": (-1, 1), "ArmDownReward": (-1, 1)}
        for stage in range(1, 6):
            lower = upper = 0.0
            for fn, default in zip(task.reward_functions, task.reward_weights):
                name = type(fn).__name__
                if name not in active[stage] or name == "TerminationCfg":
                    continue
                weight = task.stage_reward_weights[stage].get(name, default)
                lo, hi = ranges.get(name, (0, 1))
                if name == "PreciseHandTargetReward":
                    hi = 1.5 if stage == 1 else 1.0
                lower += min(weight * lo, weight * hi)
                upper += max(weight * lo, weight * hi)
            self.assertLess(upper, .005 * (500 + lower), (stage, upper))
            self.assertLess(-lower, .005 * (250 - upper), (stage, lower))
        # Stage 0 still uses the preexisting scalar weights.
        self.assertNotIn(0, task.stage_reward_weights)

    def test_old_snapshots_are_capped_until_actor_is_initialized(self):
        class Env(FakeStageVecEnv):
            def set_curriculum_max_stage(self, stage):
                self.cap = stage
        with tempfile.TemporaryDirectory() as tmp:
            env = Env()
            trainer = MultiPPOTrainer(env, config(tmp))
            try:
                self.assertEqual(env.cap, 0)
                h = handler()
                h.task.curriculum_max_stage = env.cap
                self.assertEqual(checkers._curriculum_limit(h, 5), 0)
                trainer.samples[0] = 4
                trainer._initialize_stage_actor(1)
                self.assertEqual(env.cap, 1)
                h.task.curriculum_max_stage = env.cap
                self.assertEqual(checkers._curriculum_limit(h, 5), 1)
                self.assertEqual(trainer.actor_initialization[1]["kind"], "previous_stage")
            finally:
                trainer.writer.close()

    def test_legacy_reset_respects_cap_but_explicit_stage_is_allowed(self):
        h = handler()
        h.num_envs = 4
        h.task.curriculum_max_stage = 0
        h.set_states = lambda **kwargs: None
        h.get_states = lambda: None
        stages = torch.zeros(4, dtype=torch.long)
        completed = torch.zeros_like(stages)
        with patch.object(checkers, "RAM_SNAPSHOT_BUFFER", {i: [{}] for i in range(1, 6)}), \
             patch.object(checkers, "stage0_init", return_value={}), \
             patch.object(checkers, "load_snapshot_chairman", return_value={}), \
             patch.object(checkers.random, "randint", side_effect=lambda low, high: high):
            checkers._reset_chairman_legacy(h, torch.arange(4), stages, completed, False, None)
            torch.testing.assert_close(stages, torch.zeros(4, dtype=torch.long))
            h.task.curriculum_max_stage = 1
            checkers._reset_chairman_legacy(h, torch.arange(4), stages, completed, False, None)
            torch.testing.assert_close(stages, torch.ones(4, dtype=torch.long))
            checkers._reset_chairman_legacy(h, torch.arange(4), stages, completed, False, 2)
            torch.testing.assert_close(stages, torch.full((4,), 2, dtype=torch.long))

    def test_logged_components_sum_to_reward_and_transition_resets_history(self):
        states = scene()
        task = task_for(states)
        task.reward_stage = torch.ones(1, dtype=torch.long)
        task.reward_functions[0].actual_stage.fill_(2)
        task.reward_functions[0].completed_stages.fill_(1)
        task.completed_stage_events = torch.ones(1, dtype=torch.long)
        closure = next(fn for fn in task.reward_functions if type(fn).__name__ == "CloseGraspReward")
        closure.prev_closure_per_joint = torch.ones(1, 14)
        result = total(task, states)
        summed = sum(value.item() for value in task.last_reward_terms.values())
        self.assertAlmostEqual(result, summed, places=4)
        self.assertEqual(task.last_reward_terms["MultiPolicyStageCompletionReward"].item(), 500)
        self.assertTrue(torch.isnan(closure.prev_closure_per_joint).all())
        self.assertEqual(task.completed_stage_events.item(), 1)

    def test_transferred_actor_has_exploration_floor(self):
        class Env(FakeStageVecEnv):
            finger_action_indices = (1,)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(Env(), config(tmp))
            try:
                trainer.samples[1] = 4
                trainer.models[1].policy.log_std.data.fill_(-10)
                trainer._initialize_stage_actor(2)
                torch.testing.assert_close(trainer.models[2].policy.log_std.exp(), torch.tensor([.1, .2]))
            finally:
                trainer.writer.close()

    def test_freeze_confirms_same_actor_before_any_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(FakeStageVecEnv(), config(tmp, freeze_learned_policies=True,
                stage_min_episodes=4, stage_success_window=4))
            try:
                trainer.updates[0] = 1
                trainer._freeze_outcomes[0].extend([1] * 4)
                trainer._freeze_reliable_policies()
                self.assertFalse(trainer.frozen[0])
                self.assertEqual(trainer._freeze_streak[0], 1)
                before = actor_state(trainer.models[0])
                trainer._update_ready_policies()
                for key, value in actor_state(trainer.models[0]).items():
                    torch.testing.assert_close(value, before[key])
                trainer._freeze_outcomes[0].extend([1] * 4)
                trainer._freeze_reliable_policies()
                self.assertTrue(trainer.frozen[0])
            finally:
                trainer.writer.close()

    def test_contact_dropout_is_recoverable_but_never_successful(self):
        h = handler()
        states = scene()
        states.objects["chair"].body_state[:, 0, :3] = torch.tensor([-.25, 0, .1])
        with patch.object(checkers, "common_chairman_checker", return_value=torch.tensor([False])), \
             patch.object(checkers, "get_batch_any_grasp_status", return_value=torch.tensor([False])):
            for _ in range(14):
                done, success = checkers.stege3_chacker(states, h, torch.tensor([True]))
                self.assertFalse(done.item())
                self.assertFalse(success.item())
            done, success = checkers.stege3_chacker(states, h, torch.tensor([True]))
            self.assertTrue(done.item())
            self.assertFalse(success.item())

    def test_final_stage_requires_open_fingers_and_allows_vertical_sway(self):
        h = handler()
        states = scene()
        robot = states.robots["g1_with_hands"]
        states.objects["chair"].body_state[:, 0, :3] = torch.tensor([-.25, 0, .1])
        robot.body_state[:, 0, 9] = .3  # vertical gait motion, zero XY velocity
        finger_ids = [i for i, n in enumerate(robot.joint_names) if "_hand_" in n]
        robot.joint_pos[:, finger_ids] = .2
        with patch.object(checkers, "common_chairman_checker", return_value=torch.tensor([False])):
            for _ in range(20):
                done, success = checkers.stege5_chacker(states, h, torch.tensor([True]))
                self.assertFalse(done.item())
                self.assertFalse(success.item())
            robot.joint_pos[:, finger_ids] = 0
            for _ in range(10):
                done, success = checkers.stege5_chacker(states, h, torch.tensor([True]))
            self.assertTrue(success.item())


if __name__ == "__main__":
    unittest.main()
