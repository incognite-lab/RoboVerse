

from typing import Literal

import torch
from loguru import logger as log
import numpy as np


from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R




class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robots[0].joint_limits
        scale = 0.5
        joints_name_arms_torso = env.scenario.robots[0].joint_names_right_and_left_hand_and_torso
        #joint_limits_arms_torso = {name: joint_limits[name] for name in joints_name_arms_torso}
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            #low=-scale,
            #high=scale,
            shape=(len(joint_limits),), # joints(left arm and right arm and torso)
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits)+6+6,),  # joints(left and right arm and torso) + XYZ + orientation of cube + XYZ + orientation of endeffector
            dtype=np.float32,
        )
        self.env = env
        #self.sensors = [GyroSensor(cfg, env.env.handler) for cfg in env.scenario.sensors]
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))

        super().__init__(env.num_envs, self.observation_space, self.action_space)
    def _odd_cube_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        cube_pos = self.env.env.handler.get_states().objects["cube_1"].body_state[:,0,:3].cpu().numpy()
        cube_ori = self.env.env.handler.get_states().objects["cube_1"].body_state[:,0,3:7].cpu().numpy()
        ee_index = self.env.env.handler.get_states().robots[self.env.scenario.robots[0].name].body_names.index("endeffector")
        ee_pos = self.env.env.handler.get_states().robots[self.env.scenario.robots[0].name].body_state[:,ee_index,:3].cpu().numpy()
        ee_ori = self.env.env.handler.get_states().robots[self.env.scenario.robots[0].name].body_state[:,ee_index,3:7].cpu().numpy()
        ee_ori_euler = R.from_quat(ee_ori).as_euler('xyz', degrees=False)
        cube_ori_euler = R.from_quat(cube_ori).as_euler('xyz', degrees=False)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, cube_pos, cube_ori_euler,ee_pos,ee_ori_euler], axis=1).astype(np.float32)
        #return np.concatenate([obs, cube_pos, cube_ori], axis=1).astype(np.float32)
    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
        obs = self._odd_cube_obs(obs)
        self.timesteps.zero_()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        self.action_dicts = [
            {
                self.env.scenario.robots[0].name: {
                    "dof_pos_target": dict(zip(self.env.scenario.robots[0].joint_limits.keys(), action))
                    #"dof_pos_target": self.env.scenario.robots[0].default_joint_positions

                }
            }
            for action in actions
        ]

    def step_wait(self):
        """Wait for the step to complete."""
        obs, rewards, success, timeout, _ = self.env.step(self.action_dicts)
        #print("FULL Rewards:", rewards)
        obs = obs.cpu().numpy()
        obs = self._odd_cube_obs(obs)
        dones = timeout.to(success.device) | success

        self.timesteps += (~success).float()
        extra = [{} for _ in range(self.num_envs)]
        if dones.any():
            for i in range(self.num_envs):
                if dones[i]:
                    # naplníme info dict (callback pak ví, že epizoda skončila)
                    extra[i]["episode"] = {
                        "r": float(rewards[i].cpu().item()),    # reward této epizody
                        "l": int(self.timesteps[i].item()),     # délka epizody
                    }
                    extra[i]["is_success"] = bool(success[i].item())

        if dones.any():
            self.env.reset(env_ids=dones.nonzero().squeeze(-1).tolist())
            self.timesteps[dones.cpu()] = 0
        if success.any():
            self.timesteps[success.cpu()] = 0
            rewards[success] = 10.0
            self.env.reset(env_ids=success.nonzero(as_tuple=False).squeeze(-1).tolist())




        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), extra

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        """Get images from the environment."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set an attribute of the environment."""
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call a method of the environment."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if the environment is wrapped by a given wrapper class."""
        raise NotImplementedError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_chain_frames(frames, ax=None, label_prefix="", target_frame=None):
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')

    # vykresli řetězec (translace)
    points = [f[:3,3] for f in frames]
    pts = np.vstack(points)
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', label=label_prefix+'chain')

    # vykresli osy pro každý frame (krátké šipky)
    scale = 0.1
    for i, f in enumerate(frames):
        origin = f[:3,3]
        Rmat = f[:3,:3]
        x_axis = origin + Rmat[:,0] * scale
        y_axis = origin + Rmat[:,1] * scale
        z_axis = origin + Rmat[:,2] * scale
        ax.plot([origin[0], x_axis[0]],[origin[1], x_axis[1]],[origin[2], x_axis[2]], color='r', linewidth=1)
        ax.plot([origin[0], y_axis[0]],[origin[1], y_axis[1]],[origin[2], y_axis[2]], color='g', linewidth=1)
        ax.plot([origin[0], z_axis[0]],[origin[1], z_axis[1]],[origin[2], z_axis[2]], color='b', linewidth=1)

    # vykresli target pozici a jeho orientaci (pokud poskytnut)
    if target_frame is not None:
        # podpora numpy i torch
        if hasattr(target_frame, "cpu"):
            T = target_frame.cpu().numpy()
        else:
            T = np.array(target_frame)
        origin = T[:3,3]
        ax.plot([origin[0]],[origin[1]],[origin[2]], marker='x', markersize=8, color='k', label='target')
        # target axes (větší škála než osy linků)
        tscale = max(scale * 2.0, 0.15)
        Rmat = T[:3,:3]
        tx = origin + Rmat[:,0] * tscale
        ty = origin + Rmat[:,1] * tscale
        tz = origin + Rmat[:,2] * tscale
        ax.plot([origin[0], tx[0]],[origin[1], tx[1]],[origin[2], tx[2]], color='r', linewidth=2, label='target X')
        ax.plot([origin[0], ty[0]],[origin[1], ty[1]],[origin[2], ty[2]], color='g', linewidth=2, label='target Y')
        ax.plot([origin[0], tz[0]],[origin[1], tz[1]],[origin[2], tz[2]], color='b', linewidth=2, label='target Z')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); ax.set_box_aspect([1,1,1])
    plt.show()

from ikpy.chain import Chain

def ik_solver(robot_cfg: str, target_object_name: str, env: StableBaseline3VecEnv) -> dict:

    states = env.env.env.handler.get_states()
    # for obj_name, obj in states.objects.items():
    #     for i, link_name in enumerate(obj.body_names):
    #         pos = obj.body_state[0, i, :3]
    #         print(f"{obj_name}:{link_name} -> z={float(pos[2]):.4f}  pos={pos.numpy()}")
    eeefektor_idx = states.robots[robot_cfg.name].body_names.index("endeffector")
    eeefektor_pos = states.robots[robot_cfg.name].body_state[0,eeefektor_idx,:3]
    print(f"Endeffektor position: {eeefektor_pos}")
    if target_object_name == "cube_1":
        target_pos = states.objects[target_object_name].body_state[0,0,:3]
        target_orient = states.objects[target_object_name].body_state[0,0,3:7]
    elif target_object_name == "door":
        door_handle_idx = states.objects["door"].body_names.index("door_handle")
        target_pos = states.objects["door"].body_state[0,door_handle_idx,:3]
        target_orient = states.objects["door"].body_state[0,door_handle_idx,3:7]
        #target_orient = torch.tensor([1.0,0.0, 0.0,0.0],       dtype=torch.float32)
    if target_orient.device != 'cpu':
        target_orient = target_orient.cpu()
    if target_pos.device != 'cpu':
        target_pos = target_pos.cpu()


    rot = R.from_quat([target_orient[3], target_orient[1], target_orient[2], target_orient[0]])  # [x, y, z, w]
    #rot = R.from_quat([1,0,0,0])  # [x, y, z, w]
    target_rot_matrix = rot.as_matrix()

    target_frame = np.eye(4)
    target_frame[:3, :3] = target_rot_matrix
    target_frame[:3, 3] = target_pos.numpy()

    #target_frame[1, 3] *= -1.0          # flip Y translation


    chain = Chain.from_urdf_file(robot_cfg.ik_urdf_path, base_elements=["pelvis"])
    body_names = states.robots[robot_cfg.name].body_names
    body_states = states.robots[robot_cfg.name].body_state


    joint_names = list(robot_cfg.joint_limits.keys())
    # joint_reindex = env.env.env.handler.get_joint_reindex(robot_cfg.name)
    # joint_names = [joint_names[i] for i in joint_reindex]
    # Get pelvis pose in world frame
    pelvis_idx = body_names.index("pelvis")
    pelvis_pos = body_states[0,pelvis_idx, :3]      # x, y, z
    pelvis_quat = body_states[0,pelvis_idx, 3:7]


    #print(ee_quat)
    #print(right_hand_wrist_roll)


    if pelvis_pos.device != 'cpu':
        pelvis_pos = pelvis_pos.cpu()
    if pelvis_quat.device != 'cpu':
        pelvis_quat = pelvis_quat.cpu()
    rot = R.from_quat([pelvis_quat[3],pelvis_quat[1], pelvis_quat[2], pelvis_quat[0]])  # [x, y, z, w]
    pelvis_rot_matrix = rot.as_matrix()
    base_frame = np.eye(4)
    base_frame[:3, :3] = pelvis_rot_matrix
    base_frame[:3, 3] = pelvis_pos
    # base_frame: 4x4 matrix, pelvis pose in world
    # target_frame: 4x4 matrix, target pose in world
    # Compute the target pose in the pelvis (chain) frame
    target_frame_in_chain = np.linalg.inv(base_frame)

    target_frame_in_chain = target_frame_in_chain @ target_frame
    joint_angles = chain.inverse_kinematics_frame(
        target_frame_in_chain,
        initial_position=None,
        orientation_mode="all",        # enforce full orientation
        optimizer="least_squares"      # recommended solver
    )
    #plot_chain_frames(positions, label_prefix="ikpy ",target_frame=target_frame_in_chain)

    angles = dict(zip(robot_cfg.joint_names_right_hand_and_torso, joint_angles[1:-2]))
    # Create a dict with all joints set to zero
    full_joint_dict = {name: 0.0 for name in joint_names}
    forword_poses = chain.forward_kinematics(joint_angles)
    #print("IKPY Endeffektor pos:", forword_poses[:3,3])
    # Update with your specific joint positions
    full_joint_dict.update(angles)
    # Apply joint positions to the robot in the environment
    return full_joint_dict
