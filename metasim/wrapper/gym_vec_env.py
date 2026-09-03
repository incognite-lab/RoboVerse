from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
import time

class MetaSimVecEnv(VectorEnv):
    """Vectorized environment for MetaSim that supports parallel RL training."""

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        sim: str = "isaaclab",
        task_name: str | None = None,
        num_envs: int | None = 4,
    ):
        if scenario is None:
            scenario = ScenarioCfg(task=task_name, robots=["franka"])
            scenario.num_envs = num_envs
            scenario = ScenarioCfg(**vars(scenario))
        self.num_envs = scenario.num_envs
        env_class = get_sim_env_class(SimType(sim))
        env = env_class(scenario)
        self.env: EnvWrapper[BaseSimHandler] = env
        self.render_mode = None  # XXX
        self.scenario = scenario

        # Get candidate states
        self.candidate_init_states, _, _ = get_traj(scenario.task, scenario.robots[0])

        # TODO: modify to give more meaningful space
        self.single_observation_space = spaces.Box(-np.inf, np.inf)
        self.single_action_space = spaces.Box(-np.inf, np.inf)
        self.observation_space = spaces.Box(-np.inf, np.inf)
        self.action_space = spaces.Box(-np.inf, np.inf)


    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self, env_ids: list[int] | torch.Tensor | None = None, seed: int | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.env.handler.device, dtype=torch.long
            )
        checker_owns_state = self.env.handler.checker.handles_state_reset()
        init_states = None if checker_owns_state else self.unwrapped._get_default_states(seed)
        #self.env.handler.checker.reset(self.env.handler)
        #tic = time.time()
        self.env.reset(states=init_states, env_ids=env_ids)
        #toc = time.time()
        #print(f"Reset took {toc - tic:.2f} seconds")
        return self.unwrapped._get_obs(), {}

    def step(self, actions: list[dict]):
        """Take a step in the environment."""
        _, _, success, timeout, _ = self.env.step(actions)
        obs = self.unwrapped._get_obs()
        #tic = time.time()
        if self.scenario.dagger == 1:
            rewards = torch.zeros(self.num_envs, device=obs.device)
        else:
            rewards = self.unwrapped._calculate_rewards()
        #toc = time.time()
        #print(f"Reward calculation took {toc - tic:.2f} seconds")
        return obs, rewards, success, timeout, {}

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Helper methods
    ############################################################
    def _get_obs(self):
        """Get current observations for all environments."""
        ## TODO: make this method generalizable
        states = self.env.handler.get_states()
        robot_name = self.scenario.robots[0].name
        robot_cfg = self.scenario.robots[0]
        joint_pos = states.robots[robot_name].joint_pos
        if robot_name == "franka":
            # Add end-effector position for franka
            panda_hand_index = states.robots[robot_name].body_names.index(robot_cfg.ee_body_name)
            ee_pos = states.robots[robot_name].body_state[:, panda_hand_index, :3]
            return torch.cat([joint_pos, ee_pos], dim=1)
        else:
            return joint_pos

    def _calculate_rewards(self):
        """Calculate rewards based on distance to origin."""
        states = self.env.handler.get_states()
        tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)

        from dataclasses import _MISSING_TYPE

        if isinstance(self.scenario.task.reward_functions, _MISSING_TYPE):
            return tot_reward

        reward_functions = self.scenario.task.reward_functions
        reward_stage = getattr(self.scenario.task, "reward_stage", None)
        original_stages = None
        if reward_stage is not None:
            # The staged checker advances actual_stage before rewards are
            # calculated.  Dense shaping and the completion bonus must still
            # describe the transition made by the policy that just acted.
            original_stages = [
                getattr(reward_fn, "actual_stage", None)
                for reward_fn in reward_functions
            ]
            for reward_fn in reward_functions:
                if hasattr(reward_fn, "actual_stage"):
                    reward_fn.actual_stage = reward_stage

        try:
            for reward_fn, weight in zip(reward_functions, self.scenario.task.reward_weights):
                reward_fn_ret = reward_fn(states, self.scenario.robots[0].name)
                tot_reward += weight * reward_fn_ret
        finally:
            if original_stages is not None:
                for reward_fn, actual_stage in zip(reward_functions, original_stages):
                    if hasattr(reward_fn, "actual_stage"):
                        reward_fn.actual_stage = actual_stage
        #print(tot_reward)
        return tot_reward

    def _get_default_states(self, seed: int | None = None):
        """Generate default reset states."""
        ## TODO: use non-reqeatable random choice when there is enough candidate states?
        return random.Random(seed).choices(self.candidate_init_states, k=self.num_envs)
