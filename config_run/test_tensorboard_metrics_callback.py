from types import SimpleNamespace

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import metasim.cfg.checkers.stages_chairman as chairman_stages
from config_run.callbacks import TensorboardMetricsCallback
from metasim.cfg.checkers.checkers import _ChairManChecker


class _LoggerStub:
    def record(self, *args, **kwargs):
        pass

    def dump(self, *args, **kwargs):
        pass


def test_completed_stage_events_are_counted_even_after_reward_flag_is_cleared(tmp_path):
    reward = SimpleNamespace(
        actual_stage=torch.tensor([0, 1, 3, 3]),
        completed_stages=torch.zeros(4, dtype=torch.long),
    )
    task = SimpleNamespace(
        reward_functions=[reward],
        completed_stage_events=torch.tensor([-1, 0, 2, 2]),
    )
    handler = SimpleNamespace(task=task)
    training_env = SimpleNamespace(
        num_envs=4,
        env=SimpleNamespace(env=SimpleNamespace(handler=handler)),
    )

    callback = TensorboardMetricsCallback(
        log_dir=str(tmp_path), log_interval=100, max_stage=3, verbose=0
    )
    callback.model = SimpleNamespace(
        get_env=lambda: training_env,
        logger=_LoggerStub(),
    )
    callback._on_training_start()
    callback.locals = {
        "rewards": np.zeros(4),
        "dones": np.zeros(4, dtype=bool),
        "infos": [{} for _ in range(4)],
    }
    # Deliberately overshoot the interval: vectorized training commonly does
    # not land on an exact multiple of the configured TensorBoard interval.
    callback.num_timesteps = 101
    callback._on_step()

    np.testing.assert_array_equal(
        callback.stage_completed_window_counts, [0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        callback.stage_completed_total_counts, [1, 0, 2, 0]
    )
    callback.writer.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert events.Scalars("stage_completed/window_stage_0_count")[-1].value == 1
    assert events.Scalars("stage_completed/window_stage_2_count")[-1].value == 2
    assert events.Scalars("stage_completed/total_stage_2_count")[-1].value == 2


def test_chairman_checker_publishes_completed_stage_before_increment(monkeypatch):
    for stage_index in range(6):
        def fake_stage_checker(states, handler, mask, stage_index=stage_index):
            success = mask.clone() if stage_index == 1 else torch.zeros_like(mask)
            return torch.zeros_like(mask), success

        monkeypatch.setattr(
            chairman_stages, f"stege{stage_index}_chacker", fake_stage_checker
        )
    monkeypatch.setattr(chairman_stages, "save_snapshot_chairman", lambda *args: None)

    actual_stage = torch.tensor([0, 1, 2])
    reward = SimpleNamespace(
        actual_stage=actual_stage,
        completed_stages=torch.zeros(3, dtype=torch.long),
    )
    task = SimpleNamespace(reward_functions=[reward])
    handler = SimpleNamespace(
        task=task,
        num_envs=3,
        device=torch.device("cpu"),
        get_states=lambda: SimpleNamespace(),
    )

    _ChairManChecker().check(handler)

    np.testing.assert_array_equal(actual_stage, [0, 2, 2])
    np.testing.assert_array_equal(task.completed_stage_events, [-1, 1, -1])
