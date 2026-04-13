from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.sensors.gyro import GyroSensorCfg

@configclass
class G1SliderSimpleCfg(BaseRobotCfg):
    name: str = "g1_slider_simple"
    num_joints: int = 5,
    usd_path: str = "urdf2usd_convert/g1/usd/g1_mygym.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_mygym.xml"
    #mjcf_path: str = "roboverse_data/robots/g1/mjcf/g1_mygym.xml"
    mjcf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_with_hand_rev_1_0.xml"

    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_mygym_rotslide_simple.urdf"
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
        # --- Base Joints (New Prismatic/Revolute) ---
        "baseslide_joint": BaseActuatorCfg(stiffness=200, damping=200),
        "baseslide_joint2": BaseActuatorCfg(stiffness=200, damping=200),
        "baserot_joint": BaseActuatorCfg(stiffness=200, damping=200),


        # -- Left Arm ---
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),


        # --- Right Arm (Left is fixed in URDF) ---
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
    }
    joint_limits: dict[str, tuple[float, float]] = {

        "baseslide_joint": (-1.5, 1.5),  # Example limits for the first prismatic joint
        "baseslide_joint2": (-2.0, 0.1),
        "baserot_joint": (-2.618, 2.618),


        "left_shoulder_pitch_joint": (-3.0892, 2.6704),

        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "baseslide_joint": 100,
        "baseslide_joint2": 100,
        "baserot_joint": 100,


        "left_shoulder_pitch_joint": 25,

        "right_shoulder_pitch_joint": 25,
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "baseslide_joint": 0.0, #y
        "baseslide_joint2": 0.6, #x
        "baserot_joint": 0.0,

        "left_shoulder_pitch_joint": 0.0,

        "right_shoulder_pitch_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "baseslide_joint": "position",
        "baseslide_joint2": "position",
        "baserot_joint": "position",

        "left_shoulder_roll_joint": "position",

        "right_shoulder_pitch_joint": "position",

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
