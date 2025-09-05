"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from my_env.utils import ObsSaver


@configclass
class Args:
    """Arguments for the static scene."""

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

stiff = 100.0  # set your desired stiffness value here
damp = 10   # set your desired damping value here

robot = BaseRobotCfg(
    name="new_robot_g1",
    num_joints=29,  # maybe 32 (29dof)
    usd_path="my_env/example_assets/g1/usd/g1_29dof_rev_1_0.usd",
    mjcf_path="my_env/example_assets/g1/mjcf/g1_29dof.xml",
    #mjcf_path="my_env/example_assets/g1/urdf/g1_29dof.mjcf",
    urdf_path="my_env/example_assets/g1/urdf/g1_29dof.urdf",
    enabled_gravity=True,
    fix_base_link = False,
    enabled_self_collisions=True,
    isaacgym_flip_visual_attachments=False,
    collapse_fixed_joints=True,
    actuators={
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_knee_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_knee_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "waist_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "waist_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_elbow_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_elbow_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=stiff, damping=damp),
    },
    joint_limits={
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
    },
)
# initialize scenario
scenario = ScenarioCfg(
    robots=[robot],
    try_add_table=True,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)
from scipy.spatial.transform import Rotation as R

quat_xyzw = R.from_euler("xyz", [0, 60, 0], degrees=True).as_quat()
quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # convert to wxyz
translation = (0.1, 0.0, 0.9)


# add cameras
scenario.cameras = [
    PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0)),
    PinholeCameraCfg(
        name="camera_first_person",
        width=1024,
        height=1024,
        pos=(1.5, -1.5, 1.5),
        look_at=(0.0, 0.0, 0.0),
        mount_to=robot.name,
        mount_link="torso_link",
        mount_pos=translation,
        mount_quat=quat,
    ),
]

# add objects
scenario.objects = [
    # PrimitiveCubeCfg(
    #     name="cube",
    #     size=(0.1, 0.1, 0.1),
    #     color=[1.0, 0.0, 0.0],
    #     physics=PhysicStateType.RIGIDBODY,
    # ),
    # PrimitiveSphereCfg(
    #     name="sphere",
    #     radius=0.1,
    #     color=[0.0, 0.0, 1.0],
    #     physics=PhysicStateType.RIGIDBODY,
    # ),
    # RigidObjCfg(
    #     name="bbq_sauce",
    #     scale=(2, 2, 2),
    #     physics=PhysicStateType.RIGIDBODY,
    #     usd_path="my_env/example_assets/bbq_sauce/usd/bbq_sauce.usd",
    #     urdf_path="my_env/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
    #     mjcf_path="my_env/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    # ),
    # RigidObjCfg(
    #    name="my_box",
    #    scale=(2, 2, 2),
    #   physics=PhysicStateType.RIGIDBODY,
    #   usd_path="my_env/example_assets/my_box/usd/box.usdc",
    #  urdf_path="my_env/example_assets/my_box/urdf/box.urdf",
    #  mjcf_path="my_env/example_assets/my_box/mjcf/box.xml",
    # ),
    # ArticulationObjCfg(
    #     name="box_base",
    #     fix_base_link=True,
    #     usd_path="my_env/example_assets/box_base/usd/box_base.usd",
    #     urdf_path="my_env/example_assets/box_base/urdf/box_base_unique.urdf",
    #     mjcf_path="my_env/example_assets/box_base/mjcf/box_base_unique.mjcf",
    # ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            # "cube": {
            #     "pos": torch.tensor([0.3, -0.2, 0.05]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },
            # "sphere": {
            #     "pos": torch.tensor([0.4, -0.6, 0.05]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },
            # "bbq_sauce": {
            #     "pos": torch.tensor([0.7, -0.3, 0.14]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },
            # "my_box": {
            #    "pos": torch.tensor([1.5, -1.5, 0.5]),
            #    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },scenario
            # "box_base": {
            #     "pos": torch.tensor([0.6, 0.2, 0.1]),
            #     "bbq_sauce": {
            #         "pos": torch.tensor([0.7, -0.3, 0.14]),
            #         "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            #     },
            #     "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
            #     "dof_pos": {"box_joint": 0.0},
            # },
        },
        "robots": {
            "new_robot_g1": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([0.0, 0.0, 0.0, 0.5]),
                "dof_pos": {
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
                },
            },
        },
    }
]
assert list(init_states[0]["robots"].keys()) == [robot.name], (
    f"Robot name mismatch: scenario has '{robot.name}', init_states has {list(init_states[0]['robots'].keys())}"
)

obs, extras = env.reset(states=init_states)
os.makedirs("my_env/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"my_env/output/added_Camera2_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot = scenario.robots[0]
for _ in range(100):
    log.debug(f"Step {step}")
    actions = [
            {
            robot.name: {
                "dof_pos_target": {
                    "left_hip_pitch_joint": 0.0,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.04696,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.04696,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 1.55,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 1.43,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": -1.55,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 1.43,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                }
            }
        }
    ]
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs)
    step += 1

obs_saver.save()
