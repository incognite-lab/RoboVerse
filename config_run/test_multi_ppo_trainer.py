from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

try:
    from .multi_ppo_trainer import (
        MultiPPOTrainer,
        MultiPolicyRouter,
        RaggedStageRollout,
        StageStep,
        policy_stages_with_training_data,
    )
except ImportError:
    from multi_ppo_trainer import (
        MultiPPOTrainer,
        MultiPolicyRouter,
        RaggedStageRollout,
        StageStep,
        policy_stages_with_training_data,
    )


class FakeStageVecEnv(VecEnv):
    """Small deterministic six-stage env for trainer integration tests."""

    def __init__(self, num_envs: int = 4):
        self.torch_device = torch.device("cpu")
        self.stages = np.zeros(num_envs, dtype=np.int64)
        self.stage_steps = np.zeros(num_envs, dtype=np.int64)
        self.actions = None
        super().__init__(
            num_envs,
            spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        )

    def get_current_stages(self):
        return self.stages.copy()

    def get_current_stages_torch(self):
        return torch.as_tensor(self.stages, dtype=torch.long)

    def reset(self):
        self.stages.fill(0)
        self.stage_steps.fill(0)
        return self._obs()

    def torch_reset(self):
        return torch.from_numpy(self.reset())

    def _obs(self):
        obs = np.zeros((self.num_envs, 4), dtype=np.float32)
        obs[:, 0] = self.stages
        obs[:, 1] = self.stage_steps
        return obs

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        before = self.stages.copy()
        self.stage_steps += 1
        completed = self.stage_steps >= 2
        self.stages[completed] += 1
        self.stage_steps[completed] = 0
        dones = self.stages == 6
        infos = []
        for env_id in range(self.num_envs):
            event = int(before[env_id]) if completed[env_id] else -1
            infos.append(
                {
                    "stage_before": int(before[env_id]),
                    "stage_after": int(self.stages[env_id]),
                    "completed_stage": event,
                    "policy_terminal": bool(completed[env_id]),
                    "is_success": bool(dones[env_id]),
                }
            )
        self.stages[dones] = 0
        rewards = completed.astype(np.float32)
        return self._obs(), rewards, dones, infos

    def torch_step(self, actions):
        obs, rewards, dones, infos = self.step(actions.detach().cpu().numpy())
        return (
            torch.from_numpy(obs),
            torch.from_numpy(rewards),
            torch.from_numpy(dones),
            {
                "completed_stage": torch.tensor(
                    [info["completed_stage"] for info in infos], dtype=torch.long
                ),
                "stage_after_event": torch.tensor(
                    [info["stage_after"] for info in infos], dtype=torch.long
                ),
            },
        )

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class MultiPPOTrainerTest(unittest.TestCase):
    def test_inference_router_selects_policy_by_stage(self):
        class FakeModel:
            def __init__(self, stage):
                self.stage = stage

            def predict(self, observations, deterministic=True):
                return (
                    np.full((len(observations), 2), self.stage, dtype=np.float32),
                    None,
                )

        action_space = spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        router = MultiPolicyRouter(
            {stage: FakeModel(stage) for stage in range(6)}, action_space
        )
        actions = router.predict(
            np.zeros((6, 4), dtype=np.float32), np.arange(6)
        )
        np.testing.assert_array_equal(actions[:, 0], np.arange(6))
        with self.assertRaises(ValueError):
            router.predict(np.zeros((1, 4), dtype=np.float32), np.array([6]))

    def test_training_data_stages_support_partial_and_legacy_bundles(self):
        manifest = {
            "stages": {
                "0": {"samples": 100, "updates": 1},
                "1": {"samples": 0, "updates": 0},
                "2": {},
            }
        }
        self.assertEqual(policy_stages_with_training_data(manifest), {0, 2})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_rollout_storage_with_cuda_default_device(self):
        class FakePolicy:
            @staticmethod
            def predict_values(observations):
                return torch.ones(
                    (len(observations), 1), device=observations.device
                )

        class FakeModel:
            policy = FakePolicy()

        previous_device = torch.get_default_device()
        torch.set_default_device("cuda")
        try:
            trainer = MultiPPOTrainer.__new__(MultiPPOTrainer)
            trainer.device = "cuda"
            trainer.models = {0: FakeModel()}
            values = trainer._next_values(
                0,
                np.zeros((2, 4), dtype=np.float32),
                np.array([0, 1]),
                np.array([False, True]),
            )
            self.assertEqual(values.device.type, "cpu")
            torch.testing.assert_close(values, torch.tensor([1.0, 0.0], device="cpu"))

            rollout = RaggedStageRollout(num_envs=1, gamma=0.99, gae_lambda=0.95)
            rollout.add(
                StageStep(
                    global_step=0,
                    env_ids=torch.tensor([0], device="cpu"),
                    observations=torch.zeros((1, 2), device="cpu"),
                    actions=torch.zeros((1, 1), device="cpu"),
                    rewards=torch.ones(1, device="cpu"),
                    values=torch.zeros(1, device="cpu"),
                    old_log_prob=torch.zeros(1, device="cpu"),
                    next_values=torch.zeros(1, device="cpu"),
                    terminals=torch.tensor([True], device="cpu"),
                )
            )
            self.assertEqual(rollout.finish().advantages.device.type, "cpu")
        finally:
            torch.set_default_device(previous_device)

    def test_gae_stops_at_policy_boundary(self):
        rollout = RaggedStageRollout(num_envs=1, gamma=1.0, gae_lambda=1.0)
        for step, terminal in enumerate((False, True)):
            rollout.add(
                StageStep(
                    global_step=step,
                    env_ids=torch.tensor([0]),
                    observations=torch.zeros((1, 2)),
                    actions=torch.zeros((1, 1)),
                    rewards=torch.ones(1),
                    values=torch.zeros(1),
                    old_log_prob=torch.zeros(1),
                    next_values=torch.zeros(1),
                    terminals=torch.tensor([terminal]),
                )
            )
        batch = rollout.finish()
        self.assertIsNotNone(batch)
        torch.testing.assert_close(batch.advantages, torch.tensor([2.0, 1.0]))

    def test_all_six_policies_collect_and_update(self):
        env = FakeStageVecEnv()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "task": "chairmanmulti",
                "total_timesteps": 48,
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "stage_rollout_samples": 4,
                "model_save_path": tmpdir,
                "tensorboard_log": str(Path(tmpdir) / "tb"),
                "net_arch": [8],
                "progress_bar": False,
                "freeze_learned_policies": False,
                "model_save_freq": 0,
                "terminal_tables": False,
            }
            trainer = MultiPPOTrainer(env, config)
            run_dir = trainer.learn()
            manifest = json.loads(
                (run_dir / "multi_policy_manifest.json").read_text()
            )
            self.assertEqual(manifest["num_stage_policies"], 6)
            for stage in range(6):
                self.assertGreater(manifest["stages"][str(stage)]["samples"], 0)
                self.assertGreater(manifest["stages"][str(stage)]["updates"], 0)
                self.assertTrue(
                    (run_dir / f"stage_{stage}" / "model_final.zip").is_file()
                )

            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            event_dir = Path(config["tensorboard_log"]) / run_dir.name
            accumulator = EventAccumulator(str(event_dir))
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags()["scalars"])
            self.assertIn("time/fps", scalar_tags)
            self.assertIn("time/sim_steps_per_second", scalar_tags)
            self.assertIn("stage_0/episode/mean_return", scalar_tags)
            self.assertIn("stage_0/action/clip_fraction", scalar_tags)
            self.assertIn("stage_0/train/explained_variance", scalar_tags)


if __name__ == "__main__":
    unittest.main()
