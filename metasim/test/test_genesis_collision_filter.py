import unittest
from types import SimpleNamespace

from metasim.utils.collision_filter import apply_genesis_link_self_collision_filter


def _link(name: str, n_geoms: int = 1) -> SimpleNamespace:
    geoms = [SimpleNamespace(_contype=7, _conaffinity=7) for _ in range(n_geoms)]
    return SimpleNamespace(name=name, geoms=geoms)


class GenesisCollisionFilterTest(unittest.TestCase):
    def test_keeps_only_matching_self_collision_geometries(self) -> None:
        torso = _link("torso_link", 2)
        elbow = _link("left_elbow_link")
        knee = _link("left_knee_link")
        entity = SimpleNamespace(links=[torso, elbow, knee], _is_local_collision_mask=False)

        result = apply_genesis_link_self_collision_filter(entity, ("torso_link", "*_elbow_link"))

        self.assertEqual(result, (2, 3))
        self.assertIs(entity._is_local_collision_mask, True)
        self.assertEqual(
            [(geom._contype, geom._conaffinity) for geom in torso.geoms + elbow.geoms],
            [(1, 1)] * 3,
        )
        self.assertEqual([(geom._contype, geom._conaffinity) for geom in knee.geoms], [(0, 0)])

    def test_rejects_a_filter_without_collision_geometries(self) -> None:
        entity = SimpleNamespace(links=[_link("left_knee_link")], _is_local_collision_mask=False)

        with self.assertRaisesRegex(ValueError, "matched no collision geometry"):
            apply_genesis_link_self_collision_filter(entity, ("torso_link",))


if __name__ == "__main__":
    unittest.main()
