"""Actor handoff, categorical observation remapping, and resume regressions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from gymnasium import spaces
import numpy as np

from config_run.multi_ppo_trainer import MultiPPOTrainer, _new_stage_model, transfer_actor
from config_run.test_multi_ppo_trainer import FakeStageVecEnv


def config(root, **extra):
    return {
        "task": "chairmanmulti", "total_timesteps": 100,
        "n_steps": 4, "batch_size": 4, "n_epochs": 1,
        "net_arch": [8], "stage_rollout_samples": 4,
        "model_save_path": str(root), "terminal_tables": False,
        "freeze_learned_policies": False, **extra,
    }


def actor_state(model):
    return {k: v.clone() for k, v in model.policy.state_dict().items()
            if k.startswith(("mlp_extractor.policy_net.", "action_net.")) or k == "log_std"}


class ActorInheritanceTest(unittest.TestCase):
    def test_actor_output_matches_after_stage_change_but_critic_is_untouched(self):
        env = FakeStageVecEnv()
        env.observation_space = spaces.Box(-10, 10, (12,), dtype=np.float32)
        src = _new_stage_model(env, config("unused"), 0, "cpu")
        dst = _new_stage_model(env, config("unused"), 1, "cpu")
        before = {k: v.clone() for k, v in dst.policy.state_dict().items()}
        parameter = next(dst.policy.parameters())
        dst.policy.optimizer.state[parameter] = {"step": torch.tensor(1.)}
        transfer_actor(src, dst, 0, 1, tuple(range(4, 11)))
        x = torch.randn(5, 12)
        x[:, 4:11] = 0
        x[:, 4] = 1
        y = x.clone()
        y[:, 4] = 0
        y[:, 5] = 1
        with torch.no_grad():
            src_dist = src.policy.get_distribution(x).distribution
            dst_dist = dst.policy.get_distribution(y).distribution
            torch.testing.assert_close(src_dist.mean, dst_dist.mean)
            torch.testing.assert_close(src_dist.stddev, dst_dist.stddev)
        for key, value in dst.policy.state_dict().items():
            if key.startswith(("mlp_extractor.value_net.", "value_net.")):
                torch.testing.assert_close(value, before[key])
        self.assertFalse(dst.policy.optimizer.state)
        copied = actor_state(dst)
        with torch.no_grad():
            src.policy.action_net.bias.add_(100)
        for key, value in actor_state(dst).items():
            torch.testing.assert_close(value, copied[key])

    def test_transfer_occurs_before_first_stage1_action_and_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(FakeStageVecEnv(), config(tmp))
            try:
                source = actor_state(trainer.models[0])
                original_forward = trainer.models[1].policy.forward

                def check_first_action(*args, **kwargs):
                    self.assertTrue(trainer.actor_initialized[1])
                    for key, value in actor_state(trainer.models[1]).items():
                        torch.testing.assert_close(value, source[key])
                    return original_forward(*args, **kwargs)

                with patch.object(trainer.models[1].policy, "forward", side_effect=check_first_action) as called:
                    trainer._collect_rollout(trainer._env_reset_torch())
                self.assertGreater(called.call_count, 0)
                self.assertEqual(trainer.actor_initialization[1]["source_stage"], 0)
                with torch.no_grad():
                    trainer.models[0].policy.action_net.bias.add_(10)
                trainer._initialize_stage_actor(1)
                for key, value in actor_state(trainer.models[1]).items():
                    torch.testing.assert_close(value, source[key])
            finally:
                trainer.writer.close()

    def test_inherited_actor_without_samples_survives_save_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(FakeStageVecEnv(), config(Path(tmp) / "first"))
            try:
                trainer.samples[0] = 4
                trainer._initialize_stage_actor(1)
                trainer.save("test")
                before = actor_state(trainer.models[1])
                resumed = MultiPPOTrainer(FakeStageVecEnv(), config(Path(tmp) / "resume"), resume_path=trainer.run_dir)
                try:
                    self.assertEqual(resumed.samples[1], 0)
                    self.assertTrue(resumed.actor_initialized[1])
                    with torch.no_grad():
                        resumed.models[0].policy.action_net.bias.add_(10)
                    resumed._initialize_stage_actor(1)
                    for key, value in actor_state(resumed.models[1]).items():
                        torch.testing.assert_close(value, before[key])
                finally:
                    resumed.writer.close()
            finally:
                trainer.writer.close()

    def test_snapshot_start_without_source_and_disabled_transfer(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                trainer = MultiPPOTrainer(FakeStageVecEnv(), config(tmp, inherit_stage_actor=enabled))
                try:
                    before = actor_state(trainer.models[2])
                    trainer._initialize_stage_actor(2)
                    for key, value in actor_state(trainer.models[2]).items():
                        torch.testing.assert_close(value, before[key])
                    if enabled:
                        self.assertEqual(trainer.actor_initialization[2]["kind"], "independent")
                finally:
                    trainer.writer.close()

    def test_legacy_trained_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(FakeStageVecEnv(), config(Path(tmp) / "first"))
            try:
                trainer.samples[0] = trainer.samples[1] = 4
                trainer.save("test")
                before = actor_state(trainer.models[1])
                resumed = MultiPPOTrainer(FakeStageVecEnv(), config(Path(tmp) / "resume"), resume_path=trainer.run_dir)
                try:
                    resumed._initialize_stage_actor(1)
                    for key, value in actor_state(resumed.models[1]).items():
                        torch.testing.assert_close(value, before[key])
                finally:
                    resumed.writer.close()
            finally:
                trainer.writer.close()

    def test_frozen_checkpoint_is_preserved_when_selected_stage_is_unfrozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MultiPPOTrainer(FakeStageVecEnv(), config(Path(tmp) / "first"))
            try:
                trainer.samples[0] = 4
                trainer.frozen[1] = True
                trainer.save("test")
                before = actor_state(trainer.models[1])
                resumed = MultiPPOTrainer(
                    FakeStageVecEnv(), config(Path(tmp) / "resume", train_only=True, train_stage=1),
                    resume_path=trainer.run_dir,
                )
                try:
                    self.assertFalse(resumed.frozen[1])
                    resumed._initialize_stage_actor(1)
                    for key, value in actor_state(resumed.models[1]).items():
                        torch.testing.assert_close(value, before[key])
                finally:
                    resumed.writer.close()
            finally:
                trainer.writer.close()


if __name__ == "__main__":
    unittest.main()
