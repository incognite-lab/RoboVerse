"""Chairman2 action routing using the real URDF and a fake motion policy."""
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

from config_run import SB3_chairman_env as base
from config_run.SB3_chairman2_env import StableBaseline3VecEnv
from config_run.test_SB3_chairman_env import FakeMotionPolicy, make_fake_metasim_env
from metasim.cfg.robots.g1_cfg_without_hands import G1WithoutHandsCfg


class MotionPolicy(FakeMotionPolicy):
    def predict_joint_positions_torch(self, **kwargs):
        return torch.as_tensor(self.predict_joint_positions(**kwargs))


class Chairman2Test(unittest.TestCase):
    def make_env(self):
        env = make_fake_metasim_env()
        cfg = G1WithoutHandsCfg()
        cfg.fix_base_link = False
        states = env.env.handler.get_states()
        robot = states.robots.pop(env.scenario.robots[0].name)
        states.robots[cfg.name] = robot
        movable = {j.attrib['name'] for j in ET.parse(cfg.urdf_path).getroot().findall('joint')
                   if j.attrib['type'] != 'fixed'}
        # Different simulator order verifies that routing uses names.
        indices = [i for i, name in enumerate(robot.joint_names) if name in movable][::-1]
        robot.joint_names = robot.joint_names[indices]
        robot.joint_pos = robot.joint_pos[:, indices]
        robot.joint_vel = robot.joint_vel[:, indices]
        robot.body_names.append('right_hand_palm_link')
        extra = torch.zeros(2, 1, 13)
        extra[:, :, 3] = 1
        robot.body_state = torch.cat((robot.body_state, extra), dim=1)
        env.scenario.robots = [cfg]
        env.scenario.task = env.env.handler.task
        with patch.object(base, 'G1MotionPolicy', MotionPolicy):
            wrapper = StableBaseline3VecEnv(env)
        return wrapper

    def test_layout_matches_movable_urdf_joints(self):
        wrapper = self.make_env()
        self.assertEqual(wrapper.action_space.shape, (13,))
        self.assertEqual(len(wrapper.robot_joint_names), 25)
        self.assertEqual(wrapper.num_stages, 6)
        self.assertEqual(wrapper.action_names[-3:], ('walk_vx', 'walk_vy', 'walk_yaw_rate'))
        self.assertFalse(any('waist' in n or 'hand_' in n or 'wrist_pitch' in n or 'wrist_yaw' in n
                             for n in wrapper.upper_body_joint_names))
        self.assertEqual(len(wrapper._held_state_indices), 3)

    def test_numpy_and_torch_route_arms_legs_and_hold_waist(self):
        for use_torch in (False, True):
            wrapper = self.make_env()
            actions = np.full((2, 13), 0.1, dtype=np.float32)
            actions[:, -3:] = [0.4, -0.1, 0.3]
            robot = wrapper.env.env.handler.get_states().robots[wrapper.robot_name]
            robot.joint_pos[:, wrapper._held_state_indices] = 0.4
            compose = wrapper._compose_robot_targets_torch if use_torch else wrapper._compose_robot_targets
            targets = compose(torch.from_numpy(actions) if use_torch else actions)
            np.testing.assert_allclose(targets[:, wrapper._upper_state_indices], actions[:, :10])
            np.testing.assert_allclose(targets[:, wrapper._leg_state_indices],
                                       np.tile(MotionPolicy.DEFAULT_ANGLES + 0.01, (2, 1)))
            np.testing.assert_allclose(targets[:, wrapper._held_state_indices], 0.0)
            np.testing.assert_allclose(wrapper.motion_policy.calls[0]['command'], actions[:, -3:])
            compose(actions)
            self.assertEqual(len(wrapper.motion_policy.calls), 1)  # 100 Hz env, 50 Hz motion

    def test_observations_match_on_numpy_and_torch(self):
        wrapper = self.make_env()
        obs = np.zeros((2, 25), dtype=np.float32)
        numpy_obs = wrapper.add_extra_to_obs(obs)
        torch_obs = wrapper.add_extra_to_obs_torch(torch.from_numpy(obs))
        self.assertEqual(numpy_obs.shape, (2, wrapper.observation_space.shape[0]))
        np.testing.assert_allclose(numpy_obs, torch_obs, atol=1e-6)
        self.assertTrue(np.isfinite(numpy_obs).all())

    def test_invalid_actions_and_clipping(self):
        for use_torch in (False, True):
            wrapper = self.make_env()
            compose = wrapper._compose_robot_targets_torch if use_torch else wrapper._compose_robot_targets
            for bad in (np.zeros((2, 5)), np.full((2, 13), np.nan)):
                with self.assertRaises(ValueError):
                    compose(bad)
            targets = compose(np.full((2, 13), 100.0))
            np.testing.assert_allclose(targets[:, wrapper._upper_state_indices],
                                       np.tile(wrapper.action_space.high[:10], (2, 1)))
            np.testing.assert_allclose(wrapper.motion_policy.calls[0]['command'],
                                       np.tile(MotionPolicy.MAX_COMMAND, (2, 1)))

    def test_partial_reset_keeps_other_env_memory(self):
        wrapper = self.make_env()
        wrapper._cached_leg_targets[:] = 123
        wrapper._reset_motion_state([1])
        np.testing.assert_allclose(wrapper._cached_leg_targets[0], 123)
        np.testing.assert_allclose(wrapper._cached_leg_targets[1], MotionPolicy.DEFAULT_ANGLES)
        wrapper._cached_leg_targets_torch[:] = 123
        wrapper._reset_motion_state_torch(torch.tensor([1]))
        np.testing.assert_allclose(wrapper._cached_leg_targets_torch[0], 123)
        np.testing.assert_allclose(wrapper._cached_leg_targets_torch[1], MotionPolicy.DEFAULT_ANGLES)

    def test_hand_contacts_include_palms_and_fixed_fingers_in_either_order(self):
        wrapper = self.make_env()
        states = wrapper.env.env.handler.get_states()
        robot = states.robots[wrapper.robot_name]
        states.extras['global_link_map'] = {
            1: (wrapper.robot_name, 'left_hand_palm_link'),
            2: (wrapper.robot_name, 'left_hand_index_1_link'),
            3: (wrapper.robot_name, 'right_hand_palm_link'),
            4: ('chair', 'base_link'),
            5: ('ground', 'plane'),
        }
        robot.contact = {
            'link_a': torch.tensor([[1, 4, 3, 1, 1], [1, 4, 3, 1, 1]]),
            'link_b': torch.tensor([[4, 2, 4, 5, 4], [4, 2, 4, 5, 4]]),
            'valid_mask': torch.tensor([[1, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool),
            'force_b': torch.tensor([[[10., 0, 0], [0, 20., 0], [0, 0, 30.],
                                      [100., 100., 100.], [100., 100., 100.]]] * 2),
        }
        forces = wrapper._fingertip_chair_forces(states, robot)
        torch.testing.assert_close(forces, torch.tensor([[-.2, .4, 0, 0, 0, -.6], [0.] * 6]))
        obs = torch.zeros(2, 25)
        np.testing.assert_allclose(wrapper.add_extra_to_obs(obs.numpy()),
                                   wrapper.add_extra_to_obs_torch(obs), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
