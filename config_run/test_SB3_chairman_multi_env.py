from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

try:
    from . import SB3_chairman_env as chairman_module
    from .SB3_chairman_env import StableBaseline3VecEnv as BaseChairmanVecEnv
    from .SB3_chairman_multi_env import StableBaseline3VecEnv
    from .test_SB3_chairman_env import FakeMotionPolicy, make_fake_metasim_env
    from metasim.cfg.checkers import stages_chairman as stages_module
except ImportError:
    import SB3_chairman_env as chairman_module
    from SB3_chairman_env import StableBaseline3VecEnv as BaseChairmanVecEnv
    from SB3_chairman_multi_env import StableBaseline3VecEnv
    from test_SB3_chairman_env import FakeMotionPolicy, make_fake_metasim_env
    from metasim.cfg.checkers import stages_chairman as stages_module


class Container:
    pass


class ChairmanMultiVecEnvTest(unittest.TestCase):
    def test_stage_policy_commands_pretrained_walking_policy(self):
        with patch.object(chairman_module, "G1MotionPolicy", FakeMotionPolicy):
            fake_env = make_fake_metasim_env()
            # The real MetaSim object exposes the task through both the
            # scenario and the handler.  The shared lightweight fixture only
            # needs the handler path for the original wrapper tests.
            fake_env.scenario.task = fake_env.env.handler.task
            wrapper = StableBaseline3VecEnv(fake_env)
            upper = np.zeros(
                (wrapper.num_envs, wrapper.num_upper_body_actions),
                dtype=np.float32,
            )
            upper[:, 0] = 0.2
            commands = np.tile(
                np.array([[0.4, -0.1, 0.3]], dtype=np.float32),
                (wrapper.num_envs, 1),
            )
            actions = np.concatenate((upper, commands), axis=1)

            targets = wrapper._compose_robot_targets(actions)

        self.assertEqual(wrapper.action_space.shape, (34,))
        self.assertEqual(wrapper.num_upper_body_actions, 31)
        np.testing.assert_allclose(
            wrapper.last_stage_policy_walk_commands, commands
        )
        np.testing.assert_allclose(
            wrapper.motion_policy.calls[0]["command"], commands
        )
        np.testing.assert_allclose(
            targets[:, wrapper._upper_state_indices], upper
        )
        np.testing.assert_allclose(
            targets[:, wrapper._leg_state_indices],
            np.tile(
                (FakeMotionPolicy.DEFAULT_ANGLES + 0.01)[None, :],
                (wrapper.num_envs, 1),
            ),
        )

    def test_stage_change_is_policy_terminal_but_not_physical_done(self):
        wrapper = StableBaseline3VecEnv.__new__(StableBaseline3VecEnv)
        wrapper.num_envs = 2
        wrapper._stage_before_step = np.array([0, 1], dtype=np.int64)

        wrapper.env = Container()
        wrapper.env.env = Container()
        wrapper.env.env.handler = Container()
        wrapper.env.env.handler.task = Container()
        reward = Container()
        reward.actual_stage = torch.tensor([1, 1])
        wrapper.env.env.handler.task.reward_functions = [reward]
        wrapper.env.env.handler.task.completed_stage_events = torch.tensor([0, -1])

        parent_result = (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            [{}, {}],
        )
        with patch.object(BaseChairmanVecEnv, "step_wait", return_value=parent_result):
            _, _, dones, infos = wrapper.step_wait()

        self.assertFalse(dones.any())
        self.assertTrue(infos[0]["stage_changed"])
        self.assertTrue(infos[0]["policy_terminal"])
        self.assertFalse(infos[0]["physical_done"])
        self.assertEqual(infos[0]["completed_stage"], 0)
        self.assertFalse(infos[1]["policy_terminal"])

    def test_reset_accepts_stage_restored_from_snapshot_curriculum(self):
        wrapper = StableBaseline3VecEnv.__new__(StableBaseline3VecEnv)
        wrapper.num_envs = 2
        wrapper.env = Container()
        wrapper.env.env = Container()
        wrapper.env.env.handler = Container()
        wrapper.env.env.handler.task = Container()
        reward = Container()
        reward.actual_stage = torch.tensor([0, 2])
        wrapper.env.env.handler.task.reward_functions = [reward]

        expected = np.zeros((2, 3), dtype=np.float32)
        with patch.object(BaseChairmanVecEnv, "reset", return_value=expected):
            observation = wrapper.reset()

        self.assertIs(observation, expected)
        np.testing.assert_array_equal(wrapper._stage_before_step, [0, 2])

    def test_central_reset_restores_available_snapshot_stage(self):
        handler = Container()
        handler.num_envs = 2
        handler.device = torch.device("cpu")
        handler.robot = Container()
        handler.robot.name = "g1_with_hands"
        handler.task = Container()
        handler.task.reset_to_stage0 = False
        handler.task.use_snapshot_curriculum = True

        reward = Container()
        reward.actual_stage = torch.zeros(2, dtype=torch.long)
        reward.completed_stages = torch.zeros(2, dtype=torch.long)
        reward.reset = lambda **_: None
        handler.task.reward_functions = [reward]
        handler.get_states = lambda: Container()

        applied = {}
        handler.set_states = lambda *, states, env_ids: applied.update(
            states=states, env_ids=env_ids
        )
        stage0_state = object()
        stage1_snapshot = object()
        buffers = {1: [stage1_snapshot], 2: [], 3: [], 4: [], 5: []}

        with (
            patch.object(stages_module, "init_ram_buffer"),
            patch.object(stages_module, "stage0_init", return_value=stage0_state),
            patch.object(
                stages_module,
                "load_snapshot_chairman",
                return_value=stage1_snapshot,
            ),
            patch.object(stages_module.random, "randint", return_value=1),
            patch.object(stages_module, "RAM_SNAPSHOT_BUFFER", buffers),
        ):
            stages_module.reset_chairman(handler, env_ids=[0])

        self.assertEqual(int(reward.actual_stage[0]), 1)
        self.assertIs(applied["states"][0], stage1_snapshot)
        self.assertEqual(applied["env_ids"], [0])


if __name__ == "__main__":
    unittest.main()
