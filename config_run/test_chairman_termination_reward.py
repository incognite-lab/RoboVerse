from types import SimpleNamespace
from unittest.mock import patch

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.checkers import stages_chairman as stages_module
from metasim.cfg.tasks.humanoidbench.ChairMan_multi import TerminationCfg


def _checker_result(terminated, success):
    def result(states, handler, mask):
        device = mask.device
        return (
            torch.tensor(terminated, dtype=torch.bool, device=device),
            torch.tensor(success, dtype=torch.bool, device=device),
        )

    return result


def test_checker_termination_penalty_includes_non_fall_failures():
    reward = TerminationCfg()
    reward.actual_stage = torch.tensor([1, 1], dtype=torch.long)
    reward.completed_stages = torch.zeros(2, dtype=torch.long)

    robot = SimpleNamespace(joint_pos=torch.zeros((2, 1)))
    states = SimpleNamespace(robots={"g1_with_hands": robot})
    task = SimpleNamespace(
        reward_functions=[reward],
        snapshot_save_probability=0.0,
        use_snapshot_curriculum=False,
        verbose_stage_events=False,
    )
    handler = SimpleNamespace(
        task=task,
        num_envs=2,
        device=torch.device("cpu"),
        get_states=lambda: states,
    )

    no_event = _checker_result([False, False], [False, False])
    stage1_event = _checker_result([True, False], [False, True])
    with (
        patch.object(stages_module, "stege0_chacker", no_event),
        patch.object(stages_module, "stege1_chacker", stage1_event),
        patch.object(stages_module, "stege2_chacker", no_event),
        patch.object(stages_module, "stege3_chacker", no_event),
        patch.object(stages_module, "stege4_chacker", no_event),
        patch.object(stages_module, "stege5_chacker", no_event),
    ):
        terminated = _ChairManChecker().check(handler)

    # Env 0 represents a checker failure such as excessive chair movement.
    # Env 1 successfully completed its stage and must not be penalized.
    torch.testing.assert_close(terminated, torch.tensor([True, False]))
    torch.testing.assert_close(
        reward(states, "g1_with_hands"), torch.tensor([1.0, 0.0])
    )


def test_termination_reward_reset_clears_selected_events():
    reward = TerminationCfg()
    reward.termination_events = torch.tensor([True, True])
    states = SimpleNamespace(
        robots={"g1_with_hands": SimpleNamespace(joint_pos=torch.zeros((2, 1)))}
    )

    reward.reset(torch.tensor([0]), states)

    torch.testing.assert_close(
        reward(states, "g1_with_hands"), torch.tensor([0.0, 1.0])
    )
