from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.sensors.gyro import GyroSensorCfg

@configclass
class G1NoHandsCfg(BaseRobotCfg):
    name: str = "g1_no_hands"
    num_joints: int = 29,
    usd_path: str = "roboverse_data/robots/g1/usd/g1_29dof_rev_1_0.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_29dof.xml"
    mjcf_path: str = "roboverse_data/robots/g1/mjcf/g1_29dof.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof.urdf"
    #urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_with_hand_rev_1_0.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True



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
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "left_hip_pitch_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "left_ankle_roll_joint": -1.5,
        "right_hip_pitch_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "right_ankle_roll_joint": -1.5,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.55,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.43,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.55,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 1.43,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "left_hip_pitch": "position",
        "left_hip_roll": "position",
        "left_hip_yaw": "position",
        "left_knee": "position",
        "left_ankle_pitch": "position",
        "left_ankle_roll": "position",
        "right_hip_pitch": "position",
        "right_hip_roll": "position",
        "right_hip_yaw": "position",
        "right_knee": "position",
        "right_ankle_pitch": "position",
        "right_ankle_roll": "position",
        "waist_yaw": "position",
        "left_shoulder_pitch": "position",
        "left_shoulder_roll": "position",
        "left_shoulder_yaw": "position",
        "left_elbow": "position",
        "right_shoulder_pitch": "position",
        "right_shoulder_roll": "position",
        "right_shoulder_yaw": "position",
        "right_elbow": "position",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["rubber_hand"]
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    # joint substrings, to find indices of joints.

    left_yaw_roll_joints = ["left_hip_yaw", "left_hip_roll"]
    right_yaw_roll_joints = ["right_hip_yaw", "right_hip_roll"]
    upper_body_joints = ["shoulder", "elbow", "torso"]
