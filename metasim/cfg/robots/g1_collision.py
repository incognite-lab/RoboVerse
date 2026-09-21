"""Shared, inexpensive self-collision selection for G1 upper-body robots."""

# Finger links are intentionally omitted.  The important arm-vs-torso,
# arm-vs-arm and hand-vs-head contacts remain active without adding the many
# quadratic finger-pair checks.  Genesis independently removes adjacent links.
G1_UPPER_BODY_SELF_COLLISION_LINK_PATTERNS: tuple[str, ...] = (
    "waist_*_link",
    "torso_link",
    "head_link",
    "*_shoulder_*_link",
    "*_elbow_link",
    "*_wrist_*_link",
    "*_hand_palm_link",
)
