"""CPU checks for body reward adaptation and action-owning stage weighting."""
import copy
import unittest
from types import SimpleNamespace

import torch

from metasim.cfg.tasks.humanoidbench import ChairMan_multi as multi
from metasim.cfg.tasks.humanoidbench import ChairMan_separate as separate
from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS_BY_SIDE
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


ROBOT = "g1_with_hands"


def states_for_stages():
    names = list(multi._ARM_TARGETS["left"]) + list(multi._ARM_TARGETS["right"])
    names += list(multi._FINGER_TARGETS["left"]) + list(multi._FINGER_TARGETS["right"])
    names += ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
    bodies = torch.zeros(6, 4, 13)
    bodies[..., 3] = 1.0
    bodies[:, 3, 2] = 0.8
    joints = torch.linspace(-0.3, 0.4, len(names)).repeat(6, 1)
    robot = SimpleNamespace(
        body_names=["pelvis", "left_endeffector", "endeffector", "torso_link"],
        body_state=bodies, joint_names=names, joint_pos=joints,
        joint_vel=torch.full_like(joints, 0.15), joint_pos_target=joints.clone(),
    )
    chair_bodies = torch.zeros(6, 3, 13)
    chair_bodies[..., 3] = 1.0
    chair_bodies[:, 0, 0] = 0.75
    chair_bodies[:, 0, 2] = 0.1
    chair_bodies[:, 1, 0] = 0.05
    chair_bodies[:, 2, 0] = 0.1
    chair = SimpleNamespace(
        body_names=["base_link", "target_hand_left", "target_hand_right"],
        body_state=chair_bodies,
    )
    return SimpleNamespace(robots={ROBOT: robot}, objects={"chair": chair})


class ChairmanMultiAdaptedRewardsTest(unittest.TestCase):
    def test_body_rewards_match_separate_over_motion_and_partial_reset(self):
        # Compare the independent implementations across all six stages,
        # including histories, hand-specific motion and per-env resets.
        names = [type(r).__name__ for r in multi.ChairmanmultiCfg().reward_functions]
        names.remove("MultiPolicyStageCompletionReward")
        states = states_for_stages()
        stages = torch.arange(6)
        for name in names:
            with self.subTest(reward=name):
                a, b = getattr(multi, name)(), getattr(separate, name)()
                a.actual_stage, b.actual_stage = stages.clone(), stages.clone()
                if name == "TerminationCfg":
                    a.termination_events = b.termination_events = stages == 3
                for frame in range(3):
                    current = copy.deepcopy(states)
                    current.robots[ROBOT].joint_pos += frame * 0.02
                    current.robots[ROBOT].body_state[:, 1, 0] += frame * 0.03
                    current.objects["chair"].body_state[:, 0, 0] -= frame * 0.02
                    left, right = a(current, ROBOT), b(current, ROBOT)
                    self.assertTrue(torch.isfinite(left).all())
                    torch.testing.assert_close(left, right)
                    if frame == 1 and hasattr(a, "reset"):
                        a.reset(torch.tensor([2, 4]), current)
                        b.reset(torch.tensor([2, 4]), current)

    def test_stage2_rewards_require_both_hands_to_improve(self):
        states = states_for_stages()
        robot = states.robots[ROBOT]
        for side, targets in STAGE2_FINGER_JOINT_TARGETS_BY_SIDE.items():
            for name, value in targets.items():
                robot.joint_pos[:, robot.joint_names.index(name)] = value
        robot.joint_pos[0, [robot.joint_names.index(n) for n in multi._FINGER_TARGETS["left"]]] += 0.5
        rewards = []
        for cls in (multi.LeftStage2FingerJointPositionReward, multi.RightStage2FingerJointPositionReward):
            reward = cls()
            reward.actual_stage = torch.full((6,), 2)
            rewards.append(reward(states, ROBOT))
        self.assertLess(rewards[0][0].item(), 0.2)
        self.assertEqual(rewards[1][0].item(), 1.0)
        self.assertLess((rewards[0] + rewards[1])[0], (rewards[0] + rewards[1])[1])

    def test_registered_weights_and_terminal_scale(self):
        cfg = multi.ChairmanmultiCfg()
        names = [type(r).__name__ for r in cfg.reward_functions]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), len(cfg.reward_weights))
        self.assertTrue(all(w == 0 for w in cfg.reward_weights))
        used = set()
        for stage, weights in cfg.stage_reward_weights.items():
            self.assertFalse(set(weights) - set(names))
            used.update(weights)
            self.assertEqual(weights["MultiPolicyStageCompletionReward"], 500.0)
            self.assertEqual(weights["TerminationCfg"], -10.0 if stage < 3 else -250.0)
            # All active dense functions are bounded by one in magnitude.
            # At gamma=.995 this upper bound is below the per-step discount
            # cost of postponing the +500 success event (2.5).
            positive = sum(max(w, 0) for n, w in weights.items() if n != "MultiPolicyStageCompletionReward")
            self.assertLess(positive, 2.5)
        self.assertEqual(used, set(names))
        self.assertFalse(any(n.endswith("StageProgressReward") for n in names))

    def test_actual_reward_pipeline_uses_stage_before_success(self):
        states = states_for_stages()
        cfg = multi.ChairmanmultiCfg()
        cfg.reward_functions = copy.deepcopy(cfg.reward_functions)
        before = torch.arange(6)
        after = (before + 1).clamp_max(5)
        cfg.reward_stage = before
        cfg.completed_stage_events = before.clone()
        # Keep state histories inspectable after evaluation in this test.
        cfg.reset_rewards_on_stage_change = False
        for reward in cfg.reward_functions:
            reward.actual_stage = after
            reward.completed_stages = torch.ones(6, dtype=torch.long)
            if isinstance(reward, multi.TerminationCfg):
                reward.termination_events = torch.zeros(6, dtype=torch.bool)
        wrapper = object.__new__(MetaSimVecEnv)
        wrapper.num_envs = 6
        wrapper.scenario = SimpleNamespace(task=cfg, robots=[SimpleNamespace(name=ROBOT)])
        wrapper.env = SimpleNamespace(
            handler=SimpleNamespace(device="cpu", get_states=lambda: states),
            close=lambda: None,
        )
        total = wrapper._calculate_rewards()
        expected = torch.zeros(6)
        for name, raw in cfg.last_raw_reward_terms.items():
            weights = torch.tensor([cfg.stage_reward_weights[s].get(name, 0.0) for s in range(6)])
            contribution = weights * raw
            torch.testing.assert_close(contribution, cfg.last_reward_terms[name])
            expected += contribution
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(cfg.last_reward_terms["MultiPolicyStageCompletionReward"], torch.full((6,), 500.0))
        torch.testing.assert_close(cfg.completed_stage_events, before)
        self.assertTrue(all(r.actual_stage is after for r in cfg.reward_functions))
        # Completion flags are consumed exactly once.
        wrapper._calculate_rewards()
        self.assertTrue((cfg.last_reward_terms["MultiPolicyStageCompletionReward"] == 0).all())


if __name__ == "__main__":
    unittest.main()
