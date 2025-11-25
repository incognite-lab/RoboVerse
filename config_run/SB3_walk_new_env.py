
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
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            shape=(len(joint_limits),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits)+3+17+3,),  # joints + XYZ gyro + extra + command to go
            dtype=np.float32,
        )
        self.env = env
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))
        self.current_commands = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.timesteps.device)

        self.command_duration = 200     # kolik kroků command držíme
        self.command_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=self.timesteps.device)

        # režim: "fixed" = pevný command, "random" = různé směry pro každé env
        self.command_mode = "fixed"
        # external forces
        self.external_force_enabled = env.scenario.force
        self.force_x_min = env.scenario.force_x_min
        self.force_x_max = env.scenario.force_x_max
        self.force_y_min = env.scenario.force_y_min
        self.force_y_max = env.scenario.force_y_max
        self.cached_forces = np.zeros((env.num_envs, 2), dtype=np.float32)  # cache pro síly
        # fixed command pokud je zvolen fixed mode
        self.fixed_command = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32, device=self.timesteps.device)
        super().__init__(env.num_envs, self.observation_space, self.action_space)

    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states, gyro data a command pro všechna envs."""
        states = self.env.env.handler.get_states()
        gyrodata = states.sensors["gyro0"]  # shape (num_envs, 3)
        gyrodata = gyrodata.reshape(self.num_envs, 3).cpu().numpy()
        command_data = states.sensors["command0"]  # shape (num_envs, 3)
        command_data = command_data.reshape(self.num_envs, 3).cpu().numpy()
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata,command_data], axis=1).astype(np.float32)
    def add_extra_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """extend obs with extra data."""
        states = self.env.env.handler.get_states()
        right_ankle_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("right_ankle_roll_link")
        left_ankle_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("left_ankle_roll_link")
        right_ankle_posori = states.robots[self.env.scenario.robots[0].name].body_state[:,right_ankle_idx,:7].cpu().numpy()
        left_ankle_posori = states.robots[self.env.scenario.robots[0].name].body_state[:,left_ankle_idx,:7].cpu().numpy()
        torso_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("torso_link")
        torso_pos = states.robots[self.env.scenario.robots[0].name].body_state[:,torso_idx,:3].cpu().numpy()
        other_pos = np.concatenate([right_ankle_posori,left_ankle_posori,torso_pos],axis=1)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, other_pos], axis=1).astype(np.float32)
    def reset(self,env_ids: list[int] | None = None):
        """Reset the environment."""
        self.command_timer.zero_()
        self.current_commands = self.generate_commands()
        self.set_command(self.current_commands)
        self.generate_external_forces()

        obs, _ = self.env.reset(env_ids=env_ids)
        obs = obs.cpu().numpy()
        obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)
        self.timesteps.zero_()
        return obs
    def set_command(self,command):
        self.env.env.handler.sensors[1].set_command(command)


    def generate_external_forces(self):
        """Generuje a uloží síly pro všechny envs – volá se pouze při resetu."""
        if not self.external_force_enabled:
            self.cached_forces[:] = 0.0
            return

        fx = np.random.uniform(self.force_x_min, self.force_x_max, size=self.num_envs)
        fy = np.random.uniform(self.force_y_min, self.force_y_max, size=self.num_envs)

        self.cached_forces = np.stack([fx, fy], axis=1)
        print("Generated external forces (fx, fy):", self.cached_forces)
    def apply_external_forces(self):
        """Apply cached external forces (same until next reset)."""
        if not self.external_force_enabled:
            return

        states = self.env.env.handler.get_states()
        torso_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("torso_link")
        robot_instance = self.env.env.handler.robot_inst.solver

        for env_id in range(self.env.num_envs):
            fx, fy = self.cached_forces[env_id]

            robot_instance.apply_links_external_force(
                force=np.array([[fx, fy, 0.0]]),
                links_idx=np.array([torso_idx]),
                envs_idx=np.array([env_id]),
            )


    def generate_commands(self):
        """Generuje commandy podle zvoleného režimu.
        - "fixed": všechny envs dostanou stejný command
        - "random": každé env má svůj náhodný command
        """

        if self.command_mode == "fixed":
            cmd = self.fixed_command.unsqueeze(0).repeat(self.num_envs, 1)
            return cmd

        elif self.command_mode == "random":
            # Náhodná rychlost v rozsahu [-0.5, 2.0] pro x
            cx = torch.rand(self.num_envs, device=self.timesteps.device) * 2.5 - 0.5

            # Náhodné boční pohyby (-0.5, 0.5)
            cy = torch.rand(self.num_envs, device=self.timesteps.device) - 0.5

            # Náhodný yaw
            cyaw = (torch.rand(self.num_envs, device=self.timesteps.device) - 0.5) * 1.0

            return torch.stack([cx, cy, cyaw], dim=1)
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
        # --- UPDATE COMMANDS KAŽDÝCH N KROKŮ ---
        self.command_timer += 1

        update_mask = self.command_timer >= self.command_duration
        if torch.any(update_mask):
            # envs které dosáhly délky trvání → nový command
            new_cmds = self.generate_commands()
            self.current_commands[update_mask] = new_cmds[update_mask]
            self.command_timer[update_mask] = 0

        # nastav komandy do simulace (pro všechna env najednou)
        self.set_command(self.current_commands)
        #----------------------------------------------------------------
        #-------------apply external forces to torso--------------
        #----------------------------------------------------------------
        self.apply_external_forces()

        obs, rewards, unsuccess, timeout, _ = self.env.step(self.action_dicts)
        obs = obs.cpu().numpy()
        obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)

        # --- Done flag ---
        dones = timeout.to(unsuccess.device) | unsuccess

        # --- Update time counters ---
        self.timesteps += (~unsuccess).float()

        # --- Připrav info dicty ---
        infos = [{} for _ in range(self.num_envs)]

        # --- Masky ---
        unsuccess_mask = unsuccess.cpu().numpy().astype(bool)
        timeout_mask = timeout.cpu().numpy().astype(bool)

        # --- Reset neúspěšných envů ---
        if unsuccess_mask.any():
            rewards[unsuccess_mask] = -1.0
            self.timesteps[unsuccess_mask] = 0.0
            unsuccess_ids = np.nonzero(unsuccess_mask)[0].tolist()
            self.reset(env_ids=unsuccess_ids)
            for i in unsuccess_ids:
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = False

        # --- Reset úspěšných envů (timeout = úspěch) ---
        if timeout_mask.any():
            self.timesteps[timeout_mask] = 0.0
            timeout_ids = np.nonzero(timeout_mask)[0].tolist()
            self.reset(env_ids=timeout_ids)
            for i in timeout_ids:
                infos[i]["is_success"] = True
                infos[i]["TimeLimit.truncated"] = True

        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
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
