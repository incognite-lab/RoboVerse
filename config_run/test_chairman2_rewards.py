"""Stage-local shaping, proven walking behavior and contact gating."""
import copy
import math
import unittest
import torch
from config_run.test_chairman2_checkers import scene, contact
from metasim.cfg.tasks.humanoidbench import ChairMan_2 as r
from metasim.utils import chairman2_geometry as g


class Chairman2RewardsTest(unittest.TestCase):
    def test_stage0_rewards_walking_towards_goal_instead_of_waiting(self):
        states, h = scene()
        robot = states.robots[h.robot.name]
        robot.body_state[:, 0, 1] += 1.0
        reward = r.WalkToChairProgressReward()
        reward.actual_stage = torch.zeros(2, dtype=torch.long)
        robot.body_state[0, 0, 8] = -reward.target_speed
        robot.body_state[1, 0, 8] = 0.0
        result = reward(states, h.robot.name)
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 0)
        robot.body_state[0, 0, 8] = reward.target_speed
        self.assertLess(reward(states, h.robot.name)[0], result[1])
        reward.reset(torch.tensor([0]), states)
        self.assertTrue(torch.isnan(reward.prev_final_distance[0]))
        self.assertTrue(torch.isfinite(reward.prev_final_distance[1]))

    def test_walk_keeps_approaching_until_checker_tolerance_then_stops(self):
        states, h = scene()
        robot = states.robots[h.robot.name]
        reward = r.WalkToChairProgressReward()
        reward.actual_stage = torch.zeros(2, dtype=torch.long)
        robot.body_state[:, 0, 1] += 0.08
        robot.body_state[0, 0, 8] = -reward.min_walk_speed
        moving = reward(states, h.robot.name)
        self.assertGreater(moving[0], moving[1])
        robot.body_state[:, 0, 1] = g.APPROACH_DISTANCE
        reward.reset(torch.arange(2), states)
        arrival = reward(states, h.robot.name)
        self.assertGreater(arrival[1], arrival[0])
        reward.actual_stage[:] = 1
        self.assertFalse(reward(states, h.robot.name).any())

    def test_arm_reward_is_dense_from_initial_pose_and_checks_worst_joint(self):
        states, h = scene()
        robot = states.robots[h.robot.name]
        reward = r.Stage0ArmPos()
        reward.actual_stage = torch.zeros(2, dtype=torch.long)
        target = torch.tensor(list(g.STAGE0_JOINT_TARGETS.values()))
        initial = reward(states, h.robot.name)
        self.assertTrue((initial < 0).all())
        robot.joint_pos[:] = 0.5 * target
        self.assertTrue((reward(states, h.robot.name) > initial).all())
        robot.joint_pos[:] = target
        reached = reward(states, h.robot.name)
        self.assertTrue((reached > 0).all())
        # Holding the checker-valid pose remains worthwhile during its
        # consecutive-step success window.
        torch.testing.assert_close(reward(states, h.robot.name), torch.full((2,), reward.hold_bonus))
        robot.joint_pos[0, -1] += 0.4
        regressed = reward(states, h.robot.name)
        self.assertLess(regressed[0], 0)
        self.assertGreater(regressed[1], 0)

    def test_arm_reward_progress_regression_and_partial_reset(self):
        states, h = scene()
        robot = states.robots[h.robot.name]
        reward = r.Stage0ArmPos()
        reward.actual_stage = torch.zeros(2, dtype=torch.long)
        target = torch.tensor(list(g.STAGE0_JOINT_TARGETS.values()))

        reward(states, h.robot.name)
        robot.joint_pos[:] = 0.25 * target
        improved = reward(states, h.robot.name)
        self.assertTrue((improved > 0).all())

        unchanged = reward(states, h.robot.name)
        self.assertTrue((unchanged < 0).all())

        robot.joint_pos[:] = 0.10 * target
        regressed = reward(states, h.robot.name)
        self.assertTrue((regressed < unchanged).all())

        reward.reset(torch.tensor([0]), states)
        self.assertTrue(torch.isnan(reward.previous_score[0]))
        self.assertTrue(torch.isfinite(reward.previous_score[1]))
        first_after_reset = reward(states, h.robot.name)
        self.assertTrue(torch.isfinite(first_after_reset).all())

    def test_registration_and_direct_reward_classes(self):
        cfg = r.Chairman2Cfg()
        self.assertEqual(len(cfg.reward_functions), len(cfg.reward_weights))
        self.assertEqual(cfg.num_policy_stages, 5)
        for reward in cfg.reward_functions:
            self.assertIn('__call__', type(reward).__dict__)

    def test_stationary_unfinished_states_never_pay_positive_reward(self):
        states,h=scene()
        for cls in (r.ExtendArmsReward,r.PlaceHandsReward,r.PullChairReward,r.LiftHandsReward):
            reward=cls(); reward.actual_stage=torch.full((2,),reward.stage)
            for _ in range(3): self.assertTrue((reward(states,h.robot.name)<0).all(), cls.__name__)

    def test_progress_retreat_and_partial_reset(self):
        states,h=scene(); robot=states.robots[h.robot.name]
        reward=r.ExtendArmsReward(); reward.actual_stage=torch.ones(2,dtype=torch.long)
        target=torch.tensor(list(g.STAGE1_JOINT_TARGETS.values()))
        initial=reward(states,h.robot.name)
        robot.joint_pos[:]=target*.8
        improved=reward(states,h.robot.name)
        self.assertTrue((improved>initial).all())
        robot.joint_pos.zero_()
        self.assertTrue((reward(states,h.robot.name)<initial).all())
        reward.reset(torch.tensor([0]),states)
        self.assertTrue(torch.isnan(reward.previous_cost[0]))
        self.assertTrue(torch.isfinite(reward.previous_cost[1]))

    def test_heading_cost_has_no_flat_tolerance(self):
        angles=torch.tensor([0.,.01,1.,5.,10.,20.,45.,90.,180.])*math.pi/180
        cost=r.heading_cost({'heading':angles})
        self.assertTrue((cost[1:]>cost[:-1]).all())

    def test_pull_progress_cannot_pay_without_both_contacts(self):
        states,h=scene(); contact(states)
        base=g.measure(states,h.robot.name,h.task)
        for valid in (False,True):
            reward=r.PullChairReward(); reward.actual_stage=torch.full((2,),3,dtype=torch.long)
            m=copy.deepcopy(base); m['pull_error'][:]=1.; m['contact'][:]=valid
            reward.metrics=m; reward(states,h.robot.name)
            m['pull_error'][:]=.2
            value=reward(states,h.robot.name)
            if valid: self.assertTrue((value>0).all())
            else: self.assertTrue((value<0).all())

    def test_each_stage_is_masked_and_failure_overrides_completion(self):
        states,h=scene()
        for cls in (r.ExtendArmsReward,r.PlaceHandsReward,r.PullChairReward,r.LiftHandsReward):
            reward=cls(); reward.actual_stage=torch.full((2,),(reward.stage+1)%5)
            self.assertFalse(reward(states,h.robot.name).any())
        reward=r.StageOutcomeReward(); reward.actual_stage=torch.tensor([3,4]); reward.completed_stages=torch.ones(2,dtype=torch.long)
        self.assertEqual(reward(states,h.robot.name).tolist(),[10.,20.])
        reward.termination_events=torch.tensor([False,True])
        self.assertEqual(reward(states,h.robot.name).tolist(),[10.,-10.])


if __name__=='__main__': unittest.main()
