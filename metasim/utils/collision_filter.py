from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Iterable


def apply_genesis_link_self_collision_filter(entity: Any, link_patterns: Iterable[str]) -> tuple[int, int]:
    """Restrict a Genesis entity's self collisions without affecting world contacts.

    Genesis creates its valid collision-pair table during ``Scene.build``.
    Assigning local contype/conaffinity masks before that point removes
    unwanted pairs from the table, so there is no Python-side filtering in the
    simulation loop.  A local mask is deliberately ignored for pairs belonging
    to different entities, preserving contacts with the ground and objects.

    Returns the number of enabled links and enabled collision geometries.
    """
    patterns = tuple(link_patterns)
    if not patterns:
        raise ValueError("link_patterns must not be empty")

    # Genesis has no public pre-build setters for these two properties.  They
    # are consumed by Collider._init_collision_fields during Scene.build.
    entity._is_local_collision_mask = True

    enabled_links = 0
    enabled_geoms = 0
    for link in entity.links:
        enabled = any(fnmatchcase(link.name, pattern) for pattern in patterns)
        if enabled and link.geoms:
            enabled_links += 1
        for geom in link.geoms:
            mask = 1 if enabled else 0
            geom._contype = mask
            geom._conaffinity = mask
            enabled_geoms += mask

    if enabled_geoms == 0:
        raise ValueError(f"self-collision patterns matched no collision geometry: {patterns!r}")

    return enabled_links, enabled_geoms
