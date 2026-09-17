"""Regression tests for lazily unlocked snapshot tensor banks."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from metasim.cfg.checkers import stages_chairman as stages


def pack(rows):
    return {"robot": {"pos": torch.tensor(rows, dtype=torch.float32).reshape(-1, 1)}}


def test_unlock_stage_without_reservoir_version_change():
    buffer = {stage: [] for stage in range(1, 6)}
    buffer[1] = [1, 2, 3]
    handler = SimpleNamespace(pack_state_batch=pack)
    with patch.object(stages, "RAM_SNAPSHOT_BUFFER", buffer), patch.object(stages, "SNAPSHOT_BUFFER_VERSION", 7):
        assert stages._snapshot_tensor_banks(handler, 0) == {}
        bank = stages._snapshot_tensor_banks(handler, 1)[1]
        torch.testing.assert_close(bank["robot"]["pos"], pack(buffer[1])["robot"]["pos"])


def test_first_write_to_omitted_stage_then_append_and_replace():
    buffer = {stage: [] for stage in range(1, 6)}
    buffer[1] = list(range(22))
    handler = SimpleNamespace(pack_state_batch=pack, _chairman_snapshot_tensor_cache=(7, {}))
    with patch.object(stages, "RAM_SNAPSHOT_BUFFER", buffer), patch.object(stages, "SNAPSHOT_BUFFER_VERSION", 7), patch.object(stages, "ENABLE_DISK_SNAPSHOT_SAVE", False), patch.object(stages, "UNSAVED_COUNT", 0), patch.object(stages, "MAX_SNAPSHOTS", 24):
        for value in (22, 23, 99):
            old_version = stages.SNAPSHOT_BUFFER_VERSION
            index = stages._store_snapshot(1, value)
            stages._update_snapshot_tensor_cache(handler, 1, index, value, old_version)
            version, banks = handler._chairman_snapshot_tensor_cache
            assert version == stages.SNAPSHOT_BUFFER_VERSION
            torch.testing.assert_close(banks[1]["robot"]["pos"], pack(buffer[1])["robot"]["pos"])


def test_incomplete_bank_is_rebuilt_instead_of_writing_out_of_bounds():
    buffer = {stage: [] for stage in range(1, 6)}
    buffer[1] = list(range(24))
    handler = SimpleNamespace(pack_state_batch=pack, _chairman_snapshot_tensor_cache=(7, {1: pack([22])}))
    with patch.object(stages, "RAM_SNAPSHOT_BUFFER", buffer), patch.object(stages, "SNAPSHOT_BUFFER_VERSION", 8):
        stages._update_snapshot_tensor_cache(handler, 1, 23, 23, 7)
        torch.testing.assert_close(handler._chairman_snapshot_tensor_cache[1][1]["robot"]["pos"], pack(buffer[1])["robot"]["pos"])
