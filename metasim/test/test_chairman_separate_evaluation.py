"""Regression tests for checkpoint SR curves without running a simulator."""
import csv
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config_run.separate_evaluation import (
    evaluate_rollouts, legacy_checkpoint_step, ppo_checkpoints, save_evaluation,
)
from config_run.dagger_recurrent.evaluation import evaluate_policy
from config_run.stage_diagnostics import save_stage_diagnostics


class StageEnvironment:
    num_envs = 2
    action_space = SimpleNamespace(low=-np.ones(1), high=np.ones(1))

    def reset(self):
        self.steps = 0
        return np.zeros((2, 1))

    def get_current_stages(self):
        return np.array([0, min(self.steps, 2)])

    def step(self, actions):
        self.steps += 1
        # Env 0 fails immediately and keeps auto-resetting. Env 1 completes
        # stages 0 and 1 before failing in stage 2 on its third step.
        infos = [dict(stage_before=0, completed_stage=-1, is_success=False),
                 dict(stage_before=min(self.steps - 1, 2),
                      completed_stage=self.steps - 1 if self.steps < 3 else -1,
                      is_success=False)]
        infos[0].update(failure_reasons=("fall", "chair_displacement"),
                        stage_endpoint_geometry={"pelvis_chair_xy_m": 1.3,
                                                 "pelvis_chair_lateral_m": .5,
                                                 "pelvis_chair_back_m": 1.2,
                                                 "walk_goal_error_m": .8})
        return np.zeros((2, 1)), np.ones(2), np.array([True, self.steps == 3]), infos

    def stage_endpoint_geometry(self, ids):
        return {i: {"pelvis_chair_xy_m": .75} for i in ids}


class DummyStudent:
    num_actions = 1

    def eval(self):
        return self

    def initial_hidden(self, num_envs):
        return torch.zeros(1, num_envs, 1)


class SeparateEvaluationTest(unittest.TestCase):
    def test_stage_sr_counts_completed_stages_even_if_episode_fails_later(self):
        env = StageEnvironment()
        row = evaluate_rollouts(env, lambda obs: np.zeros((2, 1)), 2, 5)
        self.assertEqual(env.steps, 3)  # Wait for the slower episode.
        self.assertEqual(row['episodes_finished'], 2)
        self.assertEqual(row['success_rate'], 0.)
        self.assertEqual(row['stage_0_episode_success_rate'], .5)
        self.assertEqual(row['stage_1_episode_success_rate'], .5)
        self.assertEqual(row['stage_2_episode_success_rate'], 0.)
        self.assertEqual(row['stage_5_episode_success_rate'], 0.)

    def test_caps_keep_completed_stages_and_count_all_assigned_episodes(self):
        for episodes, cap, expected in ((2, 2, .5), (6, 1, 1/6)):
            with self.subTest(episodes=episodes):
                row = evaluate_rollouts(StageEnvironment(), lambda obs: np.zeros((2, 1)), episodes, cap)
                self.assertEqual(row['episodes_finished'], episodes)
                self.assertEqual(row['stage_0_episode_success_rate'], expected)
                self.assertEqual(row['stage_0_episode_success_count'], 1)
                self.assertEqual(row['success_rate'], 0.)
        row = evaluate_rollouts(StageEnvironment(), lambda obs: np.zeros((2, 1)), 2, 2,
                                count_timeouts=False)
        self.assertEqual(row['episodes_finished'], 1)
        self.assertEqual(row['stage_0_episode_success_rate'], 0.)

    def test_recurrent_and_shared_evaluation_agree_on_stage_sr(self):
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.zeros((2, 1)))
        for cap in (1, 2, 5):
            recurrent = evaluate_policy(DummyStudent(), None, StageEnvironment(), 2, cap, teacher=teacher)
            shared = evaluate_rollouts(StageEnvironment(), lambda obs: np.zeros((2, 1)), 2, cap)
            for stage in range(6):
                key = f'stage_{stage}_episode_success_rate'
                self.assertEqual(shared[key], recurrent[key])
            # Existing conditional stage SR is preserved independently.
            self.assertIn('stage_0_success_rate', recurrent)

    def test_diagnostics_record_overlapping_causes_and_only_started_caps(self):
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.zeros((2, 1)))
        records, recurrent_records = [], []
        row = evaluate_rollouts(StageEnvironment(), lambda obs: np.zeros((2, 1)), 6, 1,
                                stage_records=records)
        evaluate_policy(DummyStudent(), None, StageEnvironment(), 6, 1, teacher=teacher,
                        stage_records=recurrent_records)
        self.assertEqual(records, recurrent_records)
        self.assertEqual([(event["stage"], event["outcome"], event["stage_steps"]) for event in records],
                         [(0, "failed", 1), (0, "completed", 1), (1, "capped_failure", 0)])
        self.assertEqual(records[0]["reasons"], "fall|chair_displacement")
        self.assertEqual(records[0]["pelvis_chair_xy_m"], 1.3)
        self.assertEqual(records[-1]["pelvis_chair_xy_m"], .75)
        self.assertEqual(row["capped_attempts"], 5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = [(dict(checkpoint="20", x_step=20, **row), records)]
            save_stage_diagnostics(report, root, "eval_ppo", num_stages=3)
            with (root / "eval_ppo_stage_failure_reasons.csv").open() as stream:
                counts = {row["reason"]: row for row in csv.DictReader(stream)}
            self.assertEqual(counts["fall"]["distance_median_m"], "1.3")
            self.assertEqual(counts["chair_displacement"]["count"], "1")
            self.assertEqual(counts["evaluation_cap"]["stage"], "1")
            with (root / "eval_ppo_stage_summary.csv").open() as stream:
                stages = list(csv.DictReader(stream))
            self.assertEqual(stages[0]["failed"], "1")
            self.assertEqual(stages[1]["capped_failure"], "1")
            self.assertEqual(stages[2]["entered"], "0")
            self.assertEqual(stages[2]["completion_rate"], "")
            self.assertTrue((root / "eval_ppo_stage_2_diagnostics.png").is_file())

    def test_wrapper_samples_geometry_and_masks_before_automatic_reset(self):
        from config_run.SB3_chairman_env import StableBaseline3VecEnv as Base
        from config_run.SB3_chairman_separate import StableBaseline3VecEnv
        from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS

        env = object.__new__(StableBaseline3VecEnv)
        env.num_envs = 1
        env.robot_name = "robot"
        env.raw_actions = []
        env.timesteps = torch.zeros(1)
        env.add_extra_to_obs = lambda obs: obs
        env._update_motion_diagnostics = lambda unsuccessful: None
        env._reset_motion_state = lambda ids: None
        robot = SimpleNamespace(
            body_names=["pelvis", "left_endeffector", "endeffector",
                        "left_shoulder_roll_link", "right_shoulder_roll_link"],
            joint_names=[*STAGE2_FINGER_JOINT_TARGETS, "left_elbow_joint"],
            body_state=torch.zeros(1, 5, 13),
            joint_pos=torch.zeros(1, len(STAGE2_FINGER_JOINT_TARGETS) + 1),
        )
        robot.body_state[0, 0, :3] = torch.tensor([3., 4., .2])
        chair = SimpleNamespace(body_names=["base_link", "target_hand_left", "target_hand_right"],
                                body_state=torch.zeros(1, 3, 13))
        chair.body_state[0, 0, 3] = 1
        states = SimpleNamespace(robots={"robot": robot}, objects={"chair": chair})
        task = SimpleNamespace(collect_stage_diagnostics=True, completed_stage_events=torch.tensor([-1]),
                               failure_masks={"fall": torch.tensor([True]),
                                              "stage_timeout": torch.tensor([True])},
                               just_finished=torch.tensor([False]))
        handler = SimpleNamespace(task=task, get_states=lambda: states)
        obs = torch.zeros((1, 1))

        def reset(env_ids):
            robot.body_state[0, 0, :3] = 0
            task.failure_masks.clear()
            return obs, {}

        env.env = SimpleNamespace(
            scenario=SimpleNamespace(sim="genesis", task=task), env=SimpleNamespace(handler=handler),
            step=lambda actions: (obs, torch.zeros(1), torch.tensor([True]), torch.tensor([False]), {}),
            reset=reset,
        )
        _, _, dones, infos = Base.step_wait(env)
        self.assertTrue(dones[0])
        self.assertEqual(infos[0]["failure_reasons"], ("fall", "stage_timeout"))
        geometry = infos[0]["stage_endpoint_geometry"]
        self.assertEqual(geometry["pelvis_chair_xy_m"], 5.)
        self.assertEqual(geometry["pelvis_chair_lateral_m"], 3.)
        self.assertEqual(geometry["pelvis_chair_back_m"], 4.)
        self.assertEqual(float(robot.body_state[0, 0, 0]), 0.)

        task.eval_goal_stage = 4
        task.just_finished.fill_(True)
        task.completed_stage_events.fill_(3)
        env.env.step = lambda actions: (obs, torch.zeros(1), torch.tensor([False]), torch.tensor([True]), {})
        with patch.object(env.env, "reset", wraps=reset) as reset_call:
            _, _, dones, infos = Base.step_wait(env)
        self.assertTrue(dones[0])
        self.assertTrue(infos[0]["is_success"])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        reset_call.assert_called_once_with(env_ids=[0])

    def test_final_stage_success_and_repeated_events_count_once_per_episode(self):
        from config_run.separate_evaluation import StageEpisodeMetrics
        metrics = StageEpisodeMetrics(1)
        metrics.step([dict(completed_stage=0)], [False], [True])
        metrics.step([dict(completed_stage=0)], [False], [True])
        metrics.step([dict(completed_stage=-1, stage_before=5, is_success=True)], [True], [True])
        self.assertEqual(metrics.row()['stage_0_episode_success_count'], 1)
        self.assertEqual(metrics.row()['stage_5_episode_success_rate'], 1.)
        metrics.step([dict(completed_stage=-1, is_success=False)], [True], [True])
        self.assertEqual(metrics.row()['stage_0_episode_success_rate'], .5)

    def test_ppo_uses_complete_bundles_in_numeric_order_and_final_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name in ('waist', 'right_arm', 'left_arm', 'right_fingers', 'left_fingers', 'direction'):
                policy = Path(directory) / name
                policy.mkdir()
                for label in ('100', '20', 'final'):
                    (policy / f'model_{label}.zip').touch()
                paths[name] = str(policy / 'model_final.zip')
            (Path(directory) / 'waist/model_50.zip').touch()
            manifest = dict(global_timesteps=125)
            self.assertEqual(ppo_checkpoints(paths, manifest), [(20, '20'), (100, '100'), (125, 'final')])
            self.assertEqual(ppo_checkpoints(paths, manifest, stride=2), [(20, '20'), (125, 'final')])
            self.assertEqual(ppo_checkpoints(paths, manifest, 100, stride=2), [(100, '100')])
            self.assertEqual(ppo_checkpoints(paths, manifest, 'final'), [(125, 'final')])

    def test_legacy_final_uses_saved_step_and_reports_all_stage_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / 'student_model_final.pth'
            torch.save(dict(next_step=300), f'{final}.training.pth')
            self.assertEqual(legacy_checkpoint_step(final, 101), 300)
            rows = []
            for step in (100, 20):
                result = evaluate_rollouts(StageEnvironment(), lambda obs: np.zeros((2, 1)), 2, 5)
                rows.append(dict(checkpoint=str(step), x_step=step, **result))
            save_evaluation(rows, directory, 'eval_ppo', 'PPO training transitions')
            with (Path(directory) / 'eval_ppo_results.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([row['x_step'] for row in saved], ['20', '100'])
            self.assertEqual(float(saved[0]['stage_1_episode_success_rate']), .5)
            for suffix in ['reward', 'success', 'length'] + [f'stage_{s}_success' for s in range(6)]:
                self.assertGreater((Path(directory) / f'eval_ppo_{suffix}_plot.png').stat().st_size, 1000)

    def test_ppo_entrypoint_loads_each_bundle_and_saves_stage_curves(self):
        # Match the script's sibling imports used by the existing entrypoint.
        script_dir = Path(__file__).resolve().parents[2] / 'config_run'
        with patch.object(sys, 'path', [str(script_dir), *sys.path]):
            from config_run import main_separate
        names = ('waist', 'right_arm', 'left_arm', 'right_fingers', 'left_fingers', 'direction')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                (root / name).mkdir()
                for label in ('20', '40', '60', '100', 'final'):
                    (root / name / f'model_{label}.zip').touch()
            manifest = dict(policy_names=list(names), global_timesteps=125,
                            policies={name: dict(model=f'{name}/model_final.zip') for name in names})
            (root / 'separate_policy_manifest.json').write_text(json.dumps(manifest))
            router = SimpleNamespace(predict=lambda obs, deterministic: np.zeros((2, 1)))
            with patch.object(main_separate, 'load_policy_router', return_value=(router, manifest)) as load:
                rows = main_separate.eval_ppo(dict(load_model_path=directory, eval_episodes=2,
                                                   eval_max_steps=5, log_stage_transitions=False,
                                                   eval_checkpoint_stride=3),
                                               StageEnvironment())
            self.assertEqual([call.kwargs['checkpoint'] for call in load.call_args_list], ['20', '100', 'final'])
            self.assertEqual([row['x_step'] for row in rows], [20, 100, 125])
            self.assertTrue((root / 'eval_ppo_stage_5_success_plot.png').is_file())
            self.assertTrue((root / 'eval_ppo_stage_5_diagnostics.png').is_file())


if __name__ == '__main__':
    unittest.main()
