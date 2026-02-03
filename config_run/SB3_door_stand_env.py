
from __future__ import annotations

from typing import Literal

import torch
from loguru import logger as log
import numpy as np


from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

FORCE_TO_KEEP_DOOR_CLOSE = 10000.0 # Síla pro udržení dveří zavřených
HANDLE_SPRING_FORCE = -0.5        # Síla vracející kliku nahoru
ANGLE_TO_UNLOCK_HANDLE = 0.4        # Úhel kliky pro odemčení
ANGLE_TO_CONSIDER_DOOR_CLOSED = 0.2 # Úhel pantu, kdy se dveře považují za zavřené


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
            shape=(len(joint_limits)+14,),  # joints + pos and ori of endeffector and door handle
            dtype=np.float32,
        )
        self.env = env
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))

        super().__init__(env.num_envs, self.observation_space, self.action_space)
    def add_extra_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """extend obs with extra data."""
        states = self.env.env.handler.get_states()
        endeffektor_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("endeffector")
        endeffektor_pos_ori = states.robots[self.env.scenario.robots[0].name].body_state[:,endeffektor_idx,:7].cpu().numpy()
        door_handle_idx = states.objects["door"].body_names.index("door_handle")
        door_handle_pos_ori = states.objects["door"].body_state[:,door_handle_idx,:7].cpu().numpy()
        other_pos = np.concatenate([endeffektor_pos_ori,door_handle_pos_ori],axis=1)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, other_pos], axis=1).astype(np.float32)

    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        states = self.env.env.handler.get_states()
        gyrodata = states.sensors["gyro0"].cpu().numpy()  # shape (num_envs, 3)
        gyrodata = gyrodata.reshape(self.num_envs, 3)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata], axis=1).astype(np.float32)
    def door_mechanism(self):
        # Načtení stavů
        states = self.env.env.handler.get_states()
        door_state = states.objects['door']

        # Získání pozic a rychlostí (potřebujeme rychlost pro tlumení)
        handle_angle = door_state.joint_pos[:, 1]
        hinge_angle = door_state.joint_pos[:, 0]
        hinge_vel = door_state.joint_vel[:, 0]

        if self.env.scenario.sim == 'genesis':
            # PARAMETRY

            device = "cuda" if self.env.scenario.sim == 'isaaclab' or self.env.scenario.sim == 'genesis' else "cpu"
            door_inst = self.env.env.handler.object_inst_dict['door']

            # 1. Cache indexů (bez nefunkčního nastavování KD/KP)
            if not hasattr(self, '_door_indices_cache'):
                handle_dof_idx = None
                hinge_dof_idx = None
                for joint in door_inst.joints:
                    if joint.name == 'door_handle_joint':
                        handle_dof_idx = joint.dofs_idx_local[0]
                    elif joint.name == 'door_hinge':
                        hinge_dof_idx = joint.dofs_idx_local[0]
                self._door_indices_cache = (handle_dof_idx, hinge_dof_idx)

            handle_idx, hinge_idx = self._door_indices_cache

            # 2. Logika sil

            # A) KLIKA (pružina)
            forces_handle = torch.full((self.num_envs, 1), HANDLE_SPRING_FORCE, device=device)

            # B) PANT (Zámek / Poziční držení)
            # Logika: Dveře držíme zavřené JEN TEHDY, pokud jsou už fyzicky u rámu
            # (tolerance 0.05 rad) A klika není stisknutá.

            # Jsou dveře fyzicky zavřené (blízko 0)?
            is_door_near_frame = torch.abs(hinge_angle) < ANGLE_TO_CONSIDER_DOOR_CLOSED
            # Je klika v horní poloze?
            is_handle_locked = handle_angle < ANGLE_TO_UNLOCK_HANDLE

            # Výsledná maska: Zamčeno
            is_locked = is_handle_locked & is_door_near_frame

            # Inicializace nulové síly (odemčeno = volný pohyb)
            forces_hinge = torch.zeros((self.num_envs, 1), device=device)

            # Pro zamčené prostředí aplikujeme "Poziční zámek"
            if is_locked.any():

                control_force = FORCE_TO_KEEP_DOOR_CLOSE
                forces_hinge[is_locked] = control_force

            # 3. Aplikace (Batching)
            # Musíme poslat obě síly najednou
            if handle_idx is not None and hinge_idx is not None:
                combined_forces = torch.cat([forces_hinge, forces_handle], dim=1)
                combined_indices = [hinge_idx, handle_idx]
                door_inst.control_dofs_force(
                    force=combined_forces,
                    dofs_idx_local=combined_indices
                )
        # --- SAPIEN 3 ---
        elif 'sapien' in self.env.scenario.sim:

            door_inst = self.env.env.handler.object_ids['door']
            handle_joint_idx = -1
            hinge_joint_idx = -1
            active_joints = door_inst.get_active_joints()
            for i, joint in enumerate(active_joints):
                if joint.name == 'door_handle_joint':
                    handle_joint_idx = i
                elif joint.name == 'door_hinge':
                    hinge_joint_idx = i

            if handle_joint_idx != -1 and hinge_joint_idx != -1:
                qf = door_inst.qf
                if isinstance(qf, torch.Tensor):
                    qf[handle_joint_idx] += HANDLE_SPRING_FORCE
                else:
                    # Fallback pro CPU NumPy backend
                    qf[handle_joint_idx] = HANDLE_SPRING_FORCE
                if handle_angle[0] < ANGLE_TO_UNLOCK_HANDLE and torch.abs(hinge_angle[0]) < ANGLE_TO_CONSIDER_DOOR_CLOSED:
                    # Přidáme sílu pro zavření dveří
                    qf[hinge_joint_idx] = FORCE_TO_KEEP_DOOR_CLOSE
                else:
                    qf[hinge_joint_idx] = 0.0
                # 4. Aplikujeme sílu zpět
                door_inst.set_qf(qf)



    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)
        self.timesteps.zero_()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        self.action_dicts = [
            {
                self.env.scenario.robots[0].name: {
                    #"dof_pos_target": dict(zip(self.env.scenario.robots[0].joint_limits.keys(), action))
                    "dof_pos_target": self.env.scenario.robots[0].default_joint_positions

                }
            }
            for action in actions
        ]
    def step_wait(self):
        """Wait for the step to complete."""
        obs, rewards, unsuccess, timeout, _ = self.env.step(self.action_dicts)
        obs = obs.cpu().numpy()
        obs = self.add_extra_to_obs(obs)
        #obs = self._combine_obs(obs)
        self.door_mechanism() # function to simulate door mechanism


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
            self.timesteps[unsuccess_mask] = 0.0
            unsuccess_ids = np.nonzero(unsuccess_mask)[0].tolist()
            self.env.reset(env_ids=unsuccess_ids)
            for i in unsuccess_ids:
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = False

        # --- Reset úspěšných envů (timeout = úspěch) ---
        if timeout_mask.any():
            self.timesteps[timeout_mask] = 0.0
            timeout_ids = np.nonzero(timeout_mask)[0].tolist()
            self.env.reset(env_ids=timeout_ids)
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
