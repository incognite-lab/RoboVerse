"""Contact batch/metadata regression without initializing Genesis or a GPU."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import torch


class ContactStateTest(unittest.TestCase):
    def test_contacts_use_same_environment_rows_as_robot_state(self):
        with patch('genesis.init'):
            from metasim.sim.genesis.genesis import GenesisHandler
        contact = {
            'link_a': torch.tensor([[10], [11], [12]]),
            'link_b': torch.tensor([[20], [21], [22]]),
            'valid_mask': torch.tensor([[True], [False], [True]]),
            'position': torch.arange(9).reshape(3, 1, 3).float(),
            'force_b': torch.arange(9).reshape(3, 1, 3).float(),
        }
        inst = Mock()
        inst.base_link.is_fixed = True
        inst.get_contacts.return_value = contact
        for getter, width in (('get_pos', 3), ('get_quat', 4), ('get_vel', 3),
                              ('get_ang', 3), ('get_dofs_position', 1), ('get_dofs_velocity', 1)):
            setattr(inst, getter, lambda envs_idx, w=width: torch.zeros(len(envs_idx), w))
        handler = SimpleNamespace(
            num_envs=3, objects=[], cameras=[], scenario=SimpleNamespace(sensors=[]),
            robot=SimpleNamespace(name='robot'), object_inst_dict={'robot': inst},
            global_link_map={10: ('robot', 'left_hand_palm_link'), 20: ('chair', 'base_link')},
            num_bodies_per_env=23, _previous_dof_pos_target={'robot': torch.zeros(2, 1)},
            get_joint_names=lambda name: ['joint'], get_joint_reindex=lambda name: np.array([0]),
            get_body_names=lambda name: ['pelvis'],
            get_body_states=lambda name, envs_idx: torch.zeros(len(envs_idx), 1, 13),
            _get_effort_targets=lambda: torch.zeros(2, 1),
        )
        state = GenesisHandler._get_states(handler, [2, 0])
        self.assertEqual(state.robots['robot'].joint_pos.shape[0], 2)
        for key, value in contact.items():
            torch.testing.assert_close(state.robots['robot'].contact[key], value[[2, 0]])
        self.assertEqual(state.extras['global_link_map'], handler.global_link_map)
        self.assertEqual(state.extras['num_bodies_per_env'], 23)


if __name__ == '__main__':
    unittest.main()
