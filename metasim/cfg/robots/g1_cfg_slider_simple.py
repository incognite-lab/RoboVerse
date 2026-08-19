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
        "baseslide_joint": BaseActuatorCfg(stiffness=40, damping=20),
        "baseslide_joint2": BaseActuatorCfg(stiffness=20, damping=10),
        "baserot_joint": BaseActuatorCfg(stiffness=60, damping=20),


        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=140, damping=30, torque_limit=80),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=140, damping=30, torque_limit=80),
    }
    joint_limits: dict[str, tuple[float, float]] = {

        "baseslide_joint": (-1.5, 1.5),
        "baseslide_joint2": (-1.6, 0.1),
        "baserot_joint": (-2.618, 2.618),


        "left_shoulder_pitch_joint": (-3.0892, 2.6704),

        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "baseslide_joint": 100,
        "baseslide_joint2": 100,
        "baserot_joint": 30,
        "left_shoulder_pitch_joint": 80,
        "right_shoulder_pitch_joint": 80,
        }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "baseslide_joint": 0.0, #y
        "baseslide_joint2": 0.0, #x
        "baserot_joint": 0.0,

        "left_shoulder_pitch_joint": 0.0,

        "right_shoulder_pitch_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "baseslide_joint": "position",
        "baseslide_joint2": "position",
        "baserot_joint": "position",

        "left_shoulder_pitch_joint": "position",

        "right_shoulder_pitch_joint": "position",

}
