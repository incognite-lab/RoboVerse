
import torch
def stage0_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.0,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state
def stage1_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.1,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state
def stage2_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.2,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state
def stage3_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.3,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state
def stage4_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.4,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state
def stage5_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-0.5,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.5,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    # Left hand fingers
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.0,
                    "left_hand_middle_0_joint": 0.0,
                    "left_hand_middle_1_joint": 0.0,
                    "left_hand_index_0_joint": 0.0,
                    "left_hand_index_1_joint": 0.0,
                    # Right hand fingers
                    "right_hand_thumb_0_joint": 0.0,
                    "right_hand_thumb_1_joint": 0.0,
                    "right_hand_thumb_2_joint": 0.0,
                    "right_hand_middle_0_joint": 0.0,
                    "right_hand_middle_1_joint": 0.0,
                    "right_hand_index_0_joint": 0.0,
                    "right_hand_index_1_joint": 0.0,

                }
            }
        },
        "objects": {
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }
    return state
