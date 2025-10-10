"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

# try:
#     import isaacgym  # noqa: F401
# except ImportError:
#     pass

import os

import matplotlib.pyplot as plt
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler
import numpy as np
from ikpy.chain import Chain

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from my_env.utils import ObsSaver
from metasim.wrapper.gym_vec_env import MetaSimVecEnv




@configclass
class Args:
    """Arguments for the static scene."""
    task: str = "reach"
    robot: str = "g1_with_hands"
    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "genesis"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

from ikpy.link import OriginLink

def ik_solver(robot_cfg: str, target_name: str, env: MetaSimVecEnv) -> dict:
    """
    Calculates inverse kinematics for the robot using URDF parsing and a simple geometric approach.
    This is a simplified IK solver and may not be accurate for complex poses.

    Args:
        urdf_path: Path to the URDF file.
        target_pos: np.array([x, y, z]) target position of the end-effector.
        ee_link: Name of the end-effector link (default: "right_hand_palm_link").

    Returns:
        dict: Dictionary of joint names and their corresponding angles.
    """
    states = env.env.handler.get_states()


    pose_cube = states.objects[target_name].body_state[0,0,:3]

    ori_cube = states.objects[target_name].body_state[0,0,3:7]

    # Example: set a target position for the right palm
    target_pos = pose_cube
    target_orient = ori_cube
    if target_orient.device != 'cpu':
        target_orient = target_orient.cpu()
    if target_pos.device != 'cpu':
        target_pos = target_pos.cpu()
    rot = R.from_quat([target_orient[0], target_orient[1], target_orient[2], target_orient[3]])  # [x, y, z, w]
    target_rot_matrix = rot.as_matrix()
    target_frame = np.eye(4)
    target_frame[:3, :3] = target_rot_matrix
    target_frame[:3, 3] = target_pos.numpy()
    chain = Chain.from_urdf_file(robot_cfg.ik_urdf_path, base_elements=["pelvis"])
    chain.active_links_mask[0] = True  # Make sure base is not actuated
    #print(chain.forward_kinematics([0]*len(chain.links), full_kinematics=True))  # should be identity matrix
    body_names = states.robots[robot_cfg.name].body_names

    joint_pos = states.robots[robot_cfg.name].joint_pos
    body_states = states.robots[robot_cfg.name].body_state


    joint_names = list(robot_cfg.joint_limits.keys())
    joint_reindex = env.env.handler.get_joint_reindex(robot_cfg.name)
    joint_names = [joint_names[i] for i in joint_reindex]
    # Get pelvis pose in world frame
    pelvis_idx = body_names.index("pelvis")
    pelvis_pos = body_states[0,pelvis_idx, :3]      # x, y, z
    pelvis_quat = body_states[0,pelvis_idx, 3:7]
    if pelvis_pos.device != 'cpu':
        pelvis_pos = pelvis_pos.cpu()
    if pelvis_quat.device != 'cpu':
        pelvis_quat = pelvis_quat.cpu()
    rot = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])  # [x, y, z, w]
    pelvis_rot_matrix = rot.as_matrix()
    base_frame = np.eye(4)
    base_frame[:3, :3] = pelvis_rot_matrix
    base_frame[:3, 3] = pelvis_pos
    # base_frame: 4x4 matrix, pelvis pose in world
    # target_frame: 4x4 matrix, target pose in world
    # Compute the target pose in the pelvis (chain) frame
    target_frame_in_chain = np.linalg.inv(base_frame)
    target_frame_in_chain = target_frame_in_chain @ target_frame
    #print("target_frame_in_chain", target_frame_in_chain)
    #print("base_frame", base_frame)
    #print(chain.links[-1])

    joint_angles = chain.inverse_kinematics_frame(target_frame_in_chain, initial_position=None)
    angles = dict(zip(robot_cfg.joint_names_right_hand_and_torso, joint_angles[1:-2]))




    # Create a dict with all joints set to zero
    full_joint_dict = {name: 0.0 for name in joint_names}

    # Update with your specific joint positions
    full_joint_dict.update(angles)
    # Apply joint positions to the robot in the environment
    action_dict = {
        robot_cfg.name: {
            "dof_pos_target": full_joint_dict
        }
    }
    return action_dict



def run_ik():
    """Run IK solver for reaching task."""
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        sim=args.sim,
        num_envs=args.num_envs,
        headless=args.headless
    )
    scenario.episode_length = 500
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(3, -3, 3), look_at=(0.0, 0.0, 0.0))]


    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)
    #metasim_env = get_sim_env_class(args.sim)
    env = metasim_env
    obs, _ = env.reset()


    # Get robot and target info
    robot_cfg = scenario.robots[0]



    os.makedirs("get_started/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/g1_ik_solver_{args.sim}.mp4")
    obs_orin = metasim_env.env.handler.get_states()
    obs_saver.add(obs_orin)
    for _ in range(50):


        for step in range(100):
            if step % 1 == 0:
                joint_positions = ik_solver(robot_cfg, "cube_1", env)
            obs, reward, done, info, extra = env.step([joint_positions])
            obs_orin = metasim_env.env.handler.get_states()
            if done:
                break
            obs_saver.add(obs_orin)
        env.reset()



    # Optionally render or visualize
    obs_saver.save()
    env.close()

if __name__ == "__main__":
    run_ik()
