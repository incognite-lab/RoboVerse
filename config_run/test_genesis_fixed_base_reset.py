"""Reset dispatch regression tests without constructing a GPU simulator."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("with_velocity", [True, False])
def test_packed_reset_restores_fixed_root_and_preserves_floating_qpos(fixed, with_velocity):
    # The handler initializes Genesis at import time; suppress that side effect.
    with patch("genesis.init"):
        from metasim.sim.genesis.genesis import GenesisHandler

    inst = Mock()
    inst.base_link.is_fixed = fixed
    handler = SimpleNamespace(
        num_envs=3,
        device="cpu",
        objects=[],
        robot=SimpleNamespace(name="robot"),
        object_inst_dict={"robot": inst},
        _previous_dof_pos_target={},
    )
    pos = torch.tensor([[0.0, 0.0, 0.8]])
    quat = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])
    joints = torch.tensor([[0.0, -1.5, 0.3, -1.2, -1.2]])
    qpos = joints if fixed else torch.cat((pos, quat, joints), dim=1)
    entity = {"pos": pos, "quat": quat, "qpos": qpos, "joint_pos": joints}
    if with_velocity:
        entity["qvel"] = torch.ones((1, 5 if fixed else 11))

    GenesisHandler.set_packed_state_batch(handler, {"robot": entity}, env_ids=[1])

    if fixed:
        for setter, expected in ((inst.set_pos, pos), (inst.set_quat, quat)):
            setter.assert_called_once()
            torch.testing.assert_close(setter.call_args.args[0], expected)
            torch.testing.assert_close(setter.call_args.kwargs["envs_idx"], torch.tensor([1]))
            assert setter.call_args.kwargs["relative"] is False
        assert [call[0] for call in inst.method_calls[:3]] == ["set_pos", "set_quat", "set_qpos"]
    else:
        inst.set_pos.assert_not_called()
        inst.set_quat.assert_not_called()
    torch.testing.assert_close(inst.set_qpos.call_args.args[0], qpos)
    # The velocity setter does not refresh link positions. Reset anchors and
    # observations must already see the new pose before the first scene step.
    assert inst.set_qpos.call_args.kwargs["skip_forward"] is False
    if with_velocity:
        torch.testing.assert_close(inst.set_dofs_velocity.call_args.args[0], entity["qvel"])
    else:
        inst.set_dofs_velocity.assert_not_called()
    targets = handler._previous_dof_pos_target["robot"]
    torch.testing.assert_close(targets[1], joints[0])
    torch.testing.assert_close(targets[[0, 2]], torch.zeros((2, 5)))
