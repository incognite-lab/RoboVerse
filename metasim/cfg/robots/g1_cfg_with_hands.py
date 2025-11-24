from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.sensors.gyro import GyroSensorCfg

@configclass
class G1WithHandsCfg(BaseRobotCfg):
    name: str = "g1_with_hands"
    num_joints: int = 41,
    usd_path: str = "urdf2usd_convert/g1/usd/g1_mygym.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_mygym.xml"
    #mjcf_path: str = "roboverse_data/robots/g1/mjcf/g1_mygym.xml"
    mjcf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_with_hand_rev_1_0.xml"

    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_mygym.urdf"
    #urdf_path: str = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"

    #urdf to chain pelvis --> right_hand_palm_link
    ik_urdf_path: str = "roboverse_data/robots/g1/IK_data/urdf_pelvis_to_RHPL.urdf"

    enabled_gravity: bool = True
    fix_base_link: bool = True
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = False

    # if not fix_base_link and not collapse_fixed_joints:
    #     urdf_path: str = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
    # else:
    #     urdf_path: str = "roboverse_data/robots/g1/urdf/g1_mygym.urdf"




    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "waist_yaw_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "waist_roll_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        # Left hand fingers
        "left_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "left_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        # Right hand fingers
        "right_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=0.5),
        "right_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=0.5),


    }
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
        "waist_yaw_joint": (-2.618, 2.618),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222054, 1.972222054),
        "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
        "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_roll_joint": (-1.972222054, 1.972222054),
        "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
        "right_wrist_yaw_joint": (-1.614429558, 1.614429558),
        # Left hand fingers
        "left_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "left_hand_thumb_1_joint": (-0.72431163, 1.04719755),
        "left_hand_thumb_2_joint": (0.0, 1.74532925),
        "left_hand_middle_0_joint": (-1.57079632, 0.0),
        "left_hand_middle_1_joint": (-1.74532925, 0.0),
        "left_hand_index_0_joint": (-1.57079632, 0.0),
        "left_hand_index_1_joint": (-1.74532925, 0.0),
        # Right hand fingers
        "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "right_hand_thumb_1_joint": (-1.04719755, 0.72431163),
        "right_hand_thumb_2_joint": (-1.74532925, 0.0),
        "right_hand_middle_0_joint": (0.0, 1.57079632),
        "right_hand_middle_1_joint": (0.0, 1.74532925),
        "right_hand_index_0_joint": (0.0, 1.57079632),
        "right_hand_index_1_joint": (0.0, 1.74532925),
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "left_hip_pitch_joint": 88,
        "left_hip_roll_joint": 139,
        "left_hip_yaw_joint": 88,
        "left_knee_joint": 139,
        "left_ankle_pitch_joint": 50,
        "left_ankle_roll_joint": 50,
        "right_hip_pitch_joint": 88,
        "right_hip_roll_joint": 139,
        "right_hip_yaw_joint": 88,
        "right_knee_joint": 139,
        "right_ankle_pitch_joint": 50,
        "right_ankle_roll_joint": 50,
        "waist_yaw_joint": 88,
        "waist_roll_joint": 88,
        "waist_pitch_joint": 88,
        "left_shoulder_pitch_joint": 25,
        "left_shoulder_roll_joint": 25,
        "left_shoulder_yaw_joint": 25,
        "left_elbow_joint": 25,
        "left_wrist_roll_joint": 20,
        "left_wrist_pitch_joint": 20,
        "left_wrist_yaw_joint": 20,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
        "right_wrist_roll_joint": 20,
        "right_wrist_pitch_joint": 20,
        "right_wrist_yaw_joint": 20,
        # Left hand fingers
        "left_hand_thumb_0_joint": 2.45,
        "left_hand_thumb_1_joint": 1.4,
        "left_hand_thumb_2_joint": 1.4,
        "left_hand_middle_0_joint": 1.4,
        "left_hand_middle_1_joint": 1.4,
        "left_hand_index_0_joint": 1.4,
        "left_hand_index_1_joint": 1.4,
        # Right hand fingers
        "right_hand_thumb_0_joint": 2.45,
        "right_hand_thumb_1_joint": 1.4,
        "right_hand_thumb_2_joint": 1.4,
        "right_hand_middle_0_joint": 1.4,
        "right_hand_middle_1_joint": 1.4,
        "right_hand_index_0_joint": 1.4,
        "right_hand_index_1_joint": 1.4,
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
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

    control_type: dict[str, Literal["position", "effort"]] = {
    "left_hip_pitch_joint": "position",
    "left_hip_roll_joint": "position",
    "left_hip_yaw_joint": "position",
    "left_knee_joint": "position",
    "left_ankle_pitch_joint": "position",
    "left_ankle_roll_joint": "position",
    "right_hip_pitch_joint": "position",
    "right_hip_roll_joint": "position",
    "right_hip_yaw_joint": "position",
    "right_knee_joint": "position",
    "right_ankle_pitch_joint": "position",
    "right_ankle_roll_joint": "position",
    "waist_yaw_joint": "position",
    "waist_roll_joint": "position",
    "waist_pitch_joint": "position",
    "left_shoulder_pitch_joint": "position",
    "left_shoulder_roll_joint": "position",
    "left_shoulder_yaw_joint": "position",
    "left_elbow_joint": "position",
    "left_wrist_roll_joint": "position",
    "left_wrist_pitch_joint": "position",
    "left_wrist_yaw_joint": "position",
    "right_shoulder_pitch_joint": "position",
    "right_shoulder_roll_joint": "position",
    "right_shoulder_yaw_joint": "position",
    "right_elbow_joint": "position",
    "right_wrist_roll_joint": "position",
    "right_wrist_pitch_joint": "position",
    "right_wrist_yaw_joint": "position",
    # Left hand fingers
    "left_hand_thumb_0_joint": "position",
    "left_hand_thumb_1_joint": "position",
    "left_hand_thumb_2_joint": "position",
    "left_hand_middle_0_joint": "position",
    "left_hand_middle_1_joint": "position",
    "left_hand_index_0_joint": "position",
    "left_hand_index_1_joint": "position",
    # Right hand fingers
    "right_hand_thumb_0_joint": "position",
    "right_hand_thumb_1_joint": "position",
    "right_hand_thumb_2_joint": "position",
    "right_hand_middle_0_joint": "position",
    "right_hand_middle_1_joint": "position",
    "right_hand_index_0_joint": "position",
    "right_hand_index_1_joint": "position",
}

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["rubber_hand"]
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]
    right_palm_links: list[str] = [""]
    joint_names_right_hand_and_torso = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        #"right_hand_palm_joint",
        #"endeffector_joint"
    ]
    joint_names_right_and_left_hand_and_torso = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        #"right_hand_palm_joint",
        #"endeffector_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        #"left_hand_palm_joint"
    ]

    # joint substrings, to find indices of joints.

    left_yaw_roll_joints = ["left_hip_yaw", "left_hip_roll"]
    right_yaw_roll_joints = ["right_hip_yaw", "right_hip_roll"]
    upper_body_joints = ["shoulder", "elbow", "torso"]

    """
        pelvis
        pelvis_contour_link
        left_hip_pitch_link
        left_hip_roll_link
        left_hip_yaw_link
        left_knee_link
        left_ankle_pitch_link
        left_ankle_roll_link
        right_hip_pitch_link
        right_hip_roll_link
        right_hip_yaw_link
        right_knee_link
        right_ankle_pitch_link
        right_ankle_roll_link
        waist_yaw_link
        waist_roll_link
        torso_link
        logo_link
        head_link
        imu_in_torso
        imu_in_pelvis
        d435_link
        mid360_link
        left_shoulder_pitch_link
        left_shoulder_roll_link
        left_shoulder_yaw_link
        left_elbow_link
        left_wrist_roll_link
        left_wrist_pitch_link
        left_wrist_yaw_link
        left_hand_palm_link
        left_hand_thumb_0_link
        left_hand_thumb_1_link
        left_hand_thumb_2_link
        left_hand_middle_0_link
        left_hand_middle_1_link
        left_hand_index_0_link
        left_hand_index_1_link
        right_shoulder_pitch_link
        right_shoulder_roll_link
        right_shoulder_yaw_link
        right_elbow_link
        right_wrist_roll_link
        right_wrist_pitch_link
        right_wrist_yaw_link
        right_hand_palm_link
        endeffector
        right_hand_thumb_0_link
        right_hand_thumb_1_link
        right_hand_thumb_2_link
        right_hand_middle_0_link
        right_hand_middle_1_link
        right_hand_index_0_link
        right_hand_index_1_link

    """
