"""Nyx compatibility must not reinitialize mesh buffers while rendering."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

# Load the adapter without importing the GenesisHandler package initializer,
# which initializes the simulator/GPU as a side effect.
_spec = importlib.util.spec_from_file_location(
    "nyx_compat_under_test",
    Path(__file__).resolve().parents[1] / "sim/genesis/gaussian_splat.py",
)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)


class NyxSolverCompatTest(unittest.TestCase):
    def test_public_geometry_api_never_reinitializes_legacy_fields(self):
        solver = SimpleNamespace(
            get_vgeoms_pos=Mock(), get_vgeoms_quat=Mock(),
            _init_vgeom_fields=Mock(), _init_vvert_fields=Mock(),
        )
        for _ in range(5):
            self.assertTrue(_adapter._patch_one_nyx_rigid_solver(solver))
        solver._init_vgeom_fields.assert_not_called()
        solver._init_vvert_fields.assert_not_called()

    def test_legacy_fields_still_update_geometry(self):
        solver = SimpleNamespace(
            vgeoms_state=object(), vverts_state=object(), update_vgeoms=Mock(),
        )
        self.assertTrue(_adapter._patch_one_nyx_rigid_solver(solver))
        solver.update_vgeoms.assert_called_once_with()

    def test_nested_legacy_fields_are_exposed(self):
        fields = SimpleNamespace(vgeoms_state=object(), vverts_state=object())
        solver = SimpleNamespace(data_manager=fields, update_vgeoms=Mock())
        self.assertTrue(_adapter._patch_one_nyx_rigid_solver(solver))
        self.assertIs(solver.vgeoms_state, fields.vgeoms_state)
        self.assertIs(solver.vverts_state, fields.vverts_state)
        solver.update_vgeoms.assert_called_once_with()

    def test_missing_solver_or_incomplete_api_is_not_marked_compatible(self):
        self.assertFalse(_adapter._patch_one_nyx_rigid_solver(None))
        self.assertFalse(_adapter._patch_one_nyx_rigid_solver(
            SimpleNamespace(get_vgeoms_pos=Mock())
        ))


if __name__ == "__main__":
    unittest.main()
