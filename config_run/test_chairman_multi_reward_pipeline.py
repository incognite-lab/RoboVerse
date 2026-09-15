"""CPU audit of completion credit, with simulated stage-success predicates.

Run: python -m unittest config_run.test_chairman_multi_reward_pipeline -v
No simulator, snapshots, or trained checkpoint is required.
"""

import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config_run.multi_ppo_trainer import MultiPPOTrainer
from config_run.SB3_chairman_multi_env import StableBaseline3VecEnv
from config_run.test_multi_ppo_trainer import FakeStageVecEnv
from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.checkers import stages_chairman
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import (
    MULTI_POLICY_STAGE_COMPLETION_WEIGHT,
    MultiPolicyStageCompletionReward,
)
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


class CompletionEnv(FakeStageVecEnv):
    """Real checker/reward aggregation, synthetic physics, optional reset."""

    def __init__(self, reset_on_success=False):
        super().__init__()
        self.reset_on_success = reset_on_success
        self.reward = MultiPolicyStageCompletionReward()
        self.reward.actual_stage = torch.ones(self.num_envs, dtype=torch.long)
        self.reward.completed_stages = torch.zeros(self.num_envs, dtype=torch.long)
        task = SimpleNamespace(
            reward_functions=[self.reward],
            reward_weights=[MULTI_POLICY_STAGE_COMPLETION_WEIGHT],
            snapshot_save_probability=0.0,
            use_snapshot_curriculum=False,
        )
        self.handler = SimpleNamespace(
            task=task, num_envs=self.num_envs, device=self.torch_device,
            get_states=lambda: None,
        )
        self.reward_env = SimpleNamespace(
            num_envs=self.num_envs,
            env=SimpleNamespace(handler=self.handler),
            scenario=SimpleNamespace(task=task, robots=[SimpleNamespace(name="g1_with_hands")]),
        )

    def reset(self):
        self.stages.fill(1)
        self.reward.actual_stage.fill_(1)
        return self._obs()

    def get_current_stages_torch(self):
        # Match the real wrapper: routing must survive in-place stage updates.
        return self.reward.actual_stage.clone()

    def torch_step(self, actions):
        def complete(states, handler, mask):
            return mask.clone(), mask.clone()

        with ExitStack() as stack:
            for stage in range(6):
                stack.enter_context(patch.object(stages_chairman, f"stege{stage}_chacker", complete))
            dones = _ChairManChecker().check(self.handler)
        rewards = MetaSimVecEnv._calculate_rewards(self.reward_env)
        events = self.handler.task.completed_stage_events.clone()
        after = self.reward.actual_stage.clone()
        self.stages[:] = after.numpy()
        if self.reset_on_success:
            self.reset()
            dones.fill_(True)
        return torch.from_numpy(self._obs()), rewards, dones, {
            "completed_stage": events, "stage_after_event": after,
        }


class CompletionPipelineTest(unittest.TestCase):
    def test_real_torch_wrapper_preserves_bonus_and_event_across_reset(self):
        wrapper = StableBaseline3VecEnv.__new__(StableBaseline3VecEnv)
        wrapper.torch_device = torch.device("cpu")
        wrapper.timesteps = torch.zeros(2)
        stage = torch.ones(2, dtype=torch.long)
        task = SimpleNamespace(
            train_stage=1, just_finished=torch.zeros(2, dtype=torch.bool),
            completed_stage_events=torch.full((2,), -1, dtype=torch.long),
        )

        def step(actions):
            stage.fill_(2)
            task.completed_stage_events.fill_(1)
            return torch.zeros(2, 4), torch.full((2,), MULTI_POLICY_STAGE_COMPLETION_WEIGHT), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

        def reset(env_ids):
            stage.fill_(1)
            task.completed_stage_events.fill_(-1)
            return torch.ones(2, 4), {}

        wrapper.env = SimpleNamespace(
            env=SimpleNamespace(handler=SimpleNamespace(task=task)), step=step, reset=reset,
        )
        wrapper.get_current_stages_torch = lambda: stage.clone()
        wrapper._compose_robot_targets_torch = lambda actions: actions
        wrapper._reset_motion_state_torch = lambda ids: None
        wrapper.add_extra_to_obs_torch = lambda obs: obs
        wrapper._update_reach_waypoint_visualization = lambda: None
        obs, rewards, dones, metadata = wrapper.torch_step(torch.zeros(2, 2))
        torch.testing.assert_close(rewards, torch.full((2,), MULTI_POLICY_STAGE_COMPLETION_WEIGHT))
        torch.testing.assert_close(metadata["completed_stage"], torch.ones(2, dtype=torch.long))
        torch.testing.assert_close(metadata["stage_after_event"], torch.full((2,), 2, dtype=torch.long))
        torch.testing.assert_close(obs, torch.ones(2, 4))
        self.assertTrue(dones.all())

    def test_checker_bonus_survives_stage_change_and_is_consumed_once(self):
        env = CompletionEnv()
        env.reset()
        _, rewards, dones, metadata = env.torch_step(None)
        torch.testing.assert_close(rewards, torch.full((4,), MULTI_POLICY_STAGE_COMPLETION_WEIGHT))
        torch.testing.assert_close(metadata["completed_stage"], torch.ones(4, dtype=torch.long))
        torch.testing.assert_close(env.reward.actual_stage, torch.full((4,), 2, dtype=torch.long))
        self.assertFalse(dones.any())
        torch.testing.assert_close(MetaSimVecEnv._calculate_rewards(env.reward_env), torch.zeros(4))
        torch.testing.assert_close(env.handler.task.completed_stage_events, torch.ones(4, dtype=torch.long))

    def test_completion_returns_reach_stage1_optimizer_with_and_without_reset(self):
        for reset in (False, True):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as tmp:
                env = CompletionEnv(reset_on_success=reset)
                trainer = MultiPPOTrainer(env, {
                    "task": "chairmanmulti", "train_only": reset, "train_stage": 1,
                    "total_timesteps": 8, "n_steps": 2, "batch_size": 4,
                    "n_epochs": 1, "stage_rollout_samples": 4,
                    "net_arch": [8], "gamma": 0.995, "gae_lambda": 0.97,
                    "model_save_path": tmp, "terminal_tables": False,
                })
                try:
                    trainer._collect_rollout(env.torch_reset())
                    batch = trainer.pending[1].pop_all()
                    # Completion terminates credit: no V(stage 2) or V(reset)
                    # appears in the terminal return, even on immediate reset.
                    torch.testing.assert_close(batch.returns, torch.full_like(batch.returns, MULTI_POLICY_STAGE_COMPLETION_WEIGHT))
                    self.assertEqual(len(batch), 8 if reset else 4)
                    self.assertEqual(trainer.pending[2].num_samples, 0 if reset else 4)
                    before = {k: v.clone() for k, v in trainer.models[1].policy.state_dict().items()}
                    trainer._ppo_update(1, batch)
                    self.assertEqual(trainer.updates[1], 1)
                    self.assertTrue(any(not torch.equal(before[k], v) for k, v in trainer.models[1].policy.state_dict().items()))
                finally:
                    trainer.writer.close()


if __name__ == "__main__":
    unittest.main()
