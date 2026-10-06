"""Shared ChairMan grasp targets used by rewards and stage checkers."""

STAGE2_FINGER_JOINT_TARGETS_BY_SIDE = {
    "left": {
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.14,
        "left_hand_thumb_2_joint": 0.0,
        "left_hand_index_0_joint": -1.16,
        "left_hand_index_1_joint": 0.0,
        "left_hand_middle_0_joint": -1.16,
        "left_hand_middle_1_joint": 0.0,
    },
    "right": {
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": -0.14,
        "right_hand_thumb_2_joint": 0.0,
        "right_hand_index_0_joint": 1.16,
        "right_hand_index_1_joint": 0.0,
        "right_hand_middle_0_joint": 1.16,
        "right_hand_middle_1_joint": 0.0,
    },
}

STAGE2_FINGER_JOINT_TARGETS = {
    **STAGE2_FINGER_JOINT_TARGETS_BY_SIDE["left"],
    **STAGE2_FINGER_JOINT_TARGETS_BY_SIDE["right"],
}
