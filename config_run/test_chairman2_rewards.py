"""Reward regressions: no waiting income, signed progress and contact gating."""
import copy
import math
import unittest
import torch
from config_run.test_chairman2_checkers import scene, contact
from metasim.cfg.tasks.humanoidbench import ChairMan_2 as r
from metasim.utils import chairman2_geometry as g


class Chairman2RewardsTest(unittest.TestCase):
    def test_stationary_unfinished_states_never_pay_positive_reward(self):
        states,h=scene()
        for cls in (r.WalkToChairReward,r.ExtendArmsReward,r.PlaceHandsReward,r.PullChairReward,r.LiftHandsReward):
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
        for cls in (r.WalkToChairReward,r.ExtendArmsReward,r.PlaceHandsReward,r.PullChairReward,r.LiftHandsReward):
            reward=cls(); reward.actual_stage=torch.full((2,),(reward.stage+1)%5)
            self.assertFalse(reward(states,h.robot.name).any())
        reward=r.StageOutcomeReward(); reward.actual_stage=torch.tensor([3,4]); reward.completed_stages=torch.ones(2,dtype=torch.long)
        self.assertEqual(reward(states,h.robot.name).tolist(),[10.,20.])
        reward.termination_events=torch.tensor([False,True])
        self.assertEqual(reward(states,h.robot.name).tolist(),[10.,-10.])


if __name__=='__main__': unittest.main()
