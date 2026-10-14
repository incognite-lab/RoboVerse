"""Explicit stage evaluation must work when disk curriculum is disabled."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metasim.cfg.checkers import stages_chairman as stages


class RequestedSnapshotTest(unittest.TestCase):
    def test_explicit_stage_loads_only_requested_stage_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = {"robots": {"g1_with_hands": {}}, "objects": {"chair": {}}}
            for stage in (1, 2):
                (root / f"stage_{stage}").mkdir()
                (root / f"stage_{stage}" / "snapshot.pkl").write_bytes(
                    pickle.dumps(snapshot)
                )
            buffer = {stage: [] for stage in range(1, 6)}
            with patch.object(stages, "SNAPSHOT_DIR", root), \
                 patch.object(stages, "RAM_SNAPSHOT_BUFFER", buffer), \
                 patch.object(stages, "SNAPSHOT_BUFFER_VERSION", 10), \
                 patch.object(stages, "ENABLE_DISK_SNAPSHOT_LOAD", False):
                self.assertEqual(stages.load_requested_stage_snapshots(2), 1)
                self.assertEqual(buffer[1], [])
                self.assertEqual(buffer[2], [snapshot])
                self.assertEqual(stages.SNAPSHOT_BUFFER_VERSION, 11)
                self.assertEqual(stages.load_requested_stage_snapshots(2), 1)
                self.assertEqual(stages.SNAPSHOT_BUFFER_VERSION, 11)

    def test_missing_snapshot_remains_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(stages, "SNAPSHOT_DIR", Path(directory)), \
                 patch.object(stages, "RAM_SNAPSHOT_BUFFER", {2: []}):
                self.assertEqual(stages.load_requested_stage_snapshots(2), 0)


if __name__ == "__main__":
    unittest.main()
