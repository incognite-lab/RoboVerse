"""Reward/checker alignment tests without launching a simulator."""
import copy
from types import SimpleNamespace
import unittest

import torch

from config_run import test_SB3_chairman2_env as env_fixture
from metasim.cfg.checkers.checkers import _ChairMan2Checker
from metasim.cfg.checkers import stages_chairman2 as checks
from metasim.cfg.tasks.humanoidbench import ChairMan_2 as rewards
from metasim.cfg.tasks.humanoidbench import ChairMan_multi as proven
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


class Chairman2RewardsTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = env_fixture.Chairman2Test().make_env()
        self.handler = self.wrapper.env.env.handler
        self.states = self.handler.get_states()
        self.name = self.wrapper.robot_name
        self.robot = self.states.robots[self.name]
        # Keep both neck markers above the fall threshold.
        self.robot.body_state[:, :, 2] = 0.8
        self.robot.joint_pos_target = self.robot.joint_pos.clone()
        self.task = rewards.Chairman2Cfg()
        self.task.use_snapshot_curriculum = False
        self.handler.task = self.task
        self.handler.num_envs = 2
        self.handler.robot = self.wrapper.env.scenario.robots[0]
        self.handler.scenario = self.wrapper.env.scenario
        self.handler.scenario.task = self.task
        stages = torch.zeros(2, dtype=torch.long)
        completed = torch.zeros_like(stages)
        for reward in self.task.reward_functions:
            reward.actual_stage = stages
            reward.completed_stages = completed
            reset = getattr(reward, 'reset', None)
            if reset:
                reset(torch.arange(2), self.states)
        self.checker = _ChairMan2Checker()

    def activate(self, reward, stage):
        reward.actual_stage = torch.full((2,), stage)
        return reward

    def set_pose(self, target):
        names = list(self.robot.joint_names)
        for name, angle in target.items():
            self.robot.joint_pos[:, names.index(name)] = angle

    def test_joint_rewards_require_every_joint_and_use_checker_configuration(self):
        for stage, target in ((0, checks.STAGE0_JOINT_TARGETS), (2, checks.STAGE2_JOINT_TARGETS)):
            reward = self.activate(rewards.JointPoseReward(stage), stage)
            self.set_pose(target)
            torch.testing.assert_close(reward.score(self.states, self.name), torch.ones(2))
            index = list(self.robot.joint_names).index(next(iter(target)))
            self.robot.joint_pos[0, index] += 2 * checks.JOINT_POSITION_TOLERANCE
            score = reward.score(self.states, self.name)
            self.assertLess(score[0], score[1])
            value = reward(self.states, self.name)
            self.assertLess(value[0], 0)
            self.assertEqual(value[1], 0)
            reward.actual_stage[:] = 5
            self.assertFalse(reward(self.states, self.name).any())

    def test_progress_retreat_and_partial_reset(self):
        reward = self.activate(rewards.JointPoseReward(2), 2)
        self.set_pose(checks.STAGE2_JOINT_TARGETS)
        index = list(self.robot.joint_names).index('left_elbow_joint')
        self.robot.joint_pos[:, index] += 1
        first = reward(self.states, self.name)
        self.robot.joint_pos[:, index] -= 0.5
        better = reward(self.states, self.name)
        self.assertTrue((better > first).all())
        self.robot.joint_pos[:, index] += 0.5
        worse = reward(self.states, self.name)
        self.assertTrue((worse < first).all())
        reward.reset(torch.tensor([0]), self.states)
        self.assertTrue(torch.isnan(reward.previous_score[0]))
        self.assertTrue(torch.isfinite(reward.previous_score[1]))
        value = reward(self.states, self.name)
        self.assertTrue(torch.isfinite(value).all())

    def test_palms_follow_rotated_chair_and_lift_requires_both_hands(self):
        chair = self.states.objects['chair']
        chair.body_state[:, 0, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
        indices = [self.robot.body_names.index(f'{s}_hand_palm_link') for s in ('left', 'right')]
        targets = chair.body_state[:, 1:, :3].clone()
        targets[:, :, 0] -= checks.PALM_FRONT_OFFSET
        self.robot.body_state[:, indices, :3] = targets
        lower = self.activate(rewards.LowerPalmsReward(), 3)
        torch.testing.assert_close(lower.score(self.states, self.name), torch.ones(2))
        self.assertTrue(checks.stege3_chacker(self.states, self.handler, torch.ones(2, dtype=torch.bool))[1].all())
        self.robot.body_state[0, indices[1], 0] += 0.3
        self.assertLess(lower.score(self.states, self.name)[0], 1)
        targets = chair.body_state[:, 1:, :3].clone()
        targets[:, :, 2] += checks.PALM_LIFT_HEIGHT + 0.01
        self.robot.body_state[:, indices, :3] = targets
        lift = self.activate(rewards.LiftPalmsReward(), 5)
        torch.testing.assert_close(lift.score(self.states, self.name), torch.ones(2))
        self.robot.body_state[0, indices[1], 2] -= 0.3
        self.assertLess(lift.score(self.states, self.name)[0], 1)

    def test_pull_direction_lateral_error_and_both_speeds(self):
        reward = self.activate(rewards.PullAndStopReward(), 4)
        chair = self.states.objects['chair'].body_state[:, 0]
        reward.chair_anchor = chair[:, :3].clone()
        reward.pull_direction = torch.tensor([[0., 1.], [0., 1.]])
        initial = reward.score(self.states, self.name)
        chair[:, 1] += checks.CHAIR_PULL_DISTANCE_THRESHOLD
        torch.testing.assert_close(reward.score(self.states, self.name), torch.ones(2))
        self.assertTrue((initial < 1).all())
        chair[0, 7] = 0.3
        pelvis = self.robot.body_names.index('pelvis')
        self.robot.body_state[1, pelvis, 7] = 0.3
        self.assertTrue((reward.score(self.states, self.name) < 1).all())
        chair[0, 7] = 0
        self.robot.body_state[1, pelvis, 7] = 0
        chair[0, 0] += 0.5
        chair[1, 1] -= 2 * checks.CHAIR_PULL_DISTANCE_THRESHOLD
        self.assertTrue((reward.score(self.states, self.name) < 1).all())

    def test_walking_matches_original_terms_and_stage_gate(self):
        for cls in (proven.WalkToChairProgressReward, proven.FaceChairReward,
                    proven.KeepChairStillPenalty, proven.OpenGraspReward,
                    proven.LocomotionCommandPenalty, proven.Stage0ArmPos):
            original = cls()
            if isinstance(original, proven.Stage0ArmPos):
                original.required_pos.update(checks.STAGE0_JOINT_TARGETS)
            adapted = rewards.WalkingStageReward(copy.deepcopy(original))
            original.actual_stage = torch.zeros(2, dtype=torch.long)
            adapted.actual_stage = torch.tensor([1, 2])
            if hasattr(original, 'set_control_context'):
                commands = torch.tensor([[0.3, 0., 0.], [0.3, 0., 0.]])
                original.set_control_context(commands, torch.zeros_like(commands))
                adapted.set_control_context(commands, torch.zeros_like(commands))
            expected = original(self.states, self.name)
            actual = adapted(self.states, self.name)
            torch.testing.assert_close(actual[0], expected[0])
            self.assertEqual(actual[1], 0)

    def test_anchor_penalties_start_at_zero_and_follow_checker_distances(self):
        stages = self.task.reward_functions[0].actual_stage
        stages[:] = 2
        self.checker.check(self.handler)
        robot_penalty = next(r for r in self.task.reward_functions if isinstance(r, rewards.RobotAnchorPenalty))
        chair_penalty = next(r for r in self.task.reward_functions if isinstance(r, rewards.ChairAnchorPenalty))
        self.assertFalse(robot_penalty(self.states, self.name).any())
        self.assertFalse(chair_penalty(self.states, self.name).any())
        pelvis = self.robot.body_names.index('pelvis')
        self.robot.body_state[0, pelvis, 0] += 0.51
        self.states.objects['chair'].body_state[1, 0, 0] += 0.06
        self.assertGreater(robot_penalty(self.states, self.name)[0], 1)
        self.assertGreater(chair_penalty(self.states, self.name)[1], 1)
        self.assertTrue(self.checker.check(self.handler).all())
        self.assertFalse(self.task.reward_functions[0].completed_stages.any())

    def test_pull_commands_are_allowed_until_braking_near_goal(self):
        penalty = self.activate(rewards.ManipulationCommandPenalty(), 4)
        chair = self.states.objects['chair'].body_state[:, 0]
        penalty.chair_anchor = chair[:, :3].clone()
        penalty.pull_direction = torch.tensor([[0., 1.], [0., 1.]])
        command = torch.tensor([[0.3, 0., 0.], [0.3, 0., 0.]])
        penalty.set_control_context(command, command)
        self.assertFalse(penalty(self.states, self.name).any())
        chair[:, 1] += checks.CHAIR_PULL_DISTANCE_THRESHOLD
        self.assertTrue((penalty(self.states, self.name) > 0).all())
        penalty.reset(torch.tensor([0]), self.states)
        self.assertEqual(penalty(self.states, self.name)[0], 0)
        torch.testing.assert_close(command[0], torch.tensor([0.3, 0., 0.]))

    def test_timeout_has_failure_penalty_and_no_completion_bonus(self):
        self.checker.check(self.handler)
        self.task.stage_steps[:] = 100000
        self.assertTrue(self.checker.check(self.handler).all())
        self.assertTrue(self.task.reward_functions[0](self.states, self.name).bool().all())
        self.assertFalse(self.task.reward_functions[-1](self.states, self.name).any())

    def test_checker_publishes_failure_completion_and_original_stage(self):
        stages = self.task.reward_functions[0].actual_stage
        stages[:] = torch.tensor([0, 2])
        self.set_pose(checks.STAGE0_JOINT_TARGETS)
        neck = self.robot.body_names.index('left_shoulder_roll_link')
        neck2 = self.robot.body_names.index('right_shoulder_roll_link')
        self.robot.body_state[1, [neck, neck2], 2] = 0.1
        failed = self.checker.check(self.handler)
        torch.testing.assert_close(failed, torch.tensor([False, True]))
        torch.testing.assert_close(self.task.reward_stage, torch.tensor([0, 2]))
        torch.testing.assert_close(stages, torch.tensor([1, 2]))
        penalty = self.task.reward_functions[0](self.states, self.name)
        torch.testing.assert_close(penalty, torch.tensor([0., 1.]))
        bonus = self.task.reward_functions[-1]
        torch.testing.assert_close(bonus(self.states, self.name), torch.tensor([1., 0.]))
        self.assertFalse(bonus(self.states, self.name).any())

    def test_checker_executes_stage4_and_stage5(self):
        stages = self.task.reward_functions[0].actual_stage
        stages[:] = 4
        self.checker.check(self.handler)  # Record pull anchor.
        self.states.objects['chair'].body_state[:, 0, 1] += checks.CHAIR_PULL_DISTANCE_THRESHOLD
        for _ in range(checks.STAGE4_HOLD_STEPS):
            self.checker.check(self.handler)
        torch.testing.assert_close(stages, torch.tensor([5, 5]))
        self.assertFalse(self.task.just_finished.any())
        chair = self.states.objects['chair']
        for side, target_index in (('left', 1), ('right', 2)):
            index = self.robot.body_names.index(f'{side}_hand_palm_link')
            self.robot.body_state[:, index, :3] = chair.body_state[:, target_index, :3]
            self.robot.body_state[:, index, 2] += checks.PALM_LIFT_HEIGHT + 0.01
        self.checker.check(self.handler)
        self.assertTrue(self.task.just_finished.all())
        torch.testing.assert_close(self.task.completed_stage_events, torch.tensor([5, 5]))

    def test_all_rewards_reset_and_evaluate_for_every_stage(self):
        stages = self.task.reward_functions[0].actual_stage
        reward_env = SimpleNamespace(env=SimpleNamespace(handler=self.handler),
                                     scenario=self.handler.scenario, num_envs=2)
        for stage in range(6):
            stages[:] = stage
            self.checker.check(self.handler)
            for reward in self.task.reward_functions:
                setter = getattr(reward, 'set_control_context', None)
                if setter:
                    setter(torch.zeros(2, 3), torch.zeros(2, 3))
            total = MetaSimVecEnv._calculate_rewards(reward_env)
            self.assertEqual(total.shape, (2,))
            self.assertTrue(torch.isfinite(total).all())
            for reward in self.task.reward_functions:
                self.assertIs(reward.actual_stage, stages)
                reset = getattr(reward, 'reset', None)
                if reset:
                    reset(torch.tensor([1]), self.states)


if __name__ == '__main__':
    unittest.main()
