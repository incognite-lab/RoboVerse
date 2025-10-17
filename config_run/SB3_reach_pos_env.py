from __future__ import annotations

from typing import Literal

import torch
from loguru import logger as log
import numpy as np


from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces


class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robots[0].joint_limits
        scale = 0.5
        joints_name_arms_torso = env.scenario.robots[0].joint_names_right_and_left_hand_and_torso
        joint_limits_arms_torso = {name: joint_limits[name] for name in joints_name_arms_torso}
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
            shape=(len(joint_limits)+3,),  # joints(left and right arm and torso) + XYZ + orientation
            dtype=np.float32,
        )
        self.env = env
        #self.sensors = [GyroSensor(cfg, env.env.handler) for cfg in env.scenario.sensors]
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))

        super().__init__(env.num_envs, self.observation_space, self.action_space)
    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        gyrodata = self.sensors[0].get_data()  # shape (num_envs, 3)

        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata], axis=1).astype(np.float32)
    def _odd_cube_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        cube_pos = self.env.env.handler.get_states().objects["cube_1"].body_state[:,0,:3].cpu().numpy()
        #cube_ori = self.env.env.handler.get_states().objects["cube_1"].body_state[:,0,3:7].cpu().numpy()

        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, cube_pos], axis=1).astype(np.float32)
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
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
        obs = self._odd_cube_obs(obs)
        dones = timeout.to(success.device) | success

        self.timesteps += (~success).float()

        if dones.any():
            self.env.reset(env_ids=dones.nonzero().squeeze(-1).tolist())
            self.timesteps[success.cpu()] = 0
        if success.any():
            self.timesteps[success.cpu()] = 0
            rewards[success] = 10.0
            self.env.reset(env_ids=success.nonzero(as_tuple=False).squeeze(-1).tolist())
            extra = [{} for _ in range(self.num_envs)]
            return obs, rewards.cpu().numpy(), dones.cpu().numpy(), extra
        # reward vynásobíme časem

        #time_factors = self.timesteps.to(rewards.device)
        #rewards = rewards * time_factors


        extra = [{} for _ in range(self.num_envs)]
        # for env_id in range(self.num_envs):
        #     if dones[env_id]:
        #         extra[env_id]["terminal_observation"] = obs[env_id].cpu().numpy()
        #     extra[env_id]["TimeLimit.truncated"] = timeout[env_id].item() and not unsuccess[env_id].item()

        #obs = self.env.unwrapped._get_obs()

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
