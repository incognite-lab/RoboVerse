from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random
import os
from dataclasses import _MISSING_TYPE

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
        self.profile_enabled = os.getenv("ROBO_WALK_PROFILE", "1") != "0"
        self.profile_interval = int(os.getenv("ROBO_WALK_PROFILE_INTERVAL", "1000"))
        self.profile_sync_cuda = os.getenv("ROBO_WALK_PROFILE_SYNC", "0") == "1"
        self._profile_totals = {}
        self._profile_window_steps = 0
        self._profile_window_total = 0.0

    def _profile_now(self) -> float:
        if self.profile_enabled and self.profile_sync_cuda:
            device = getattr(self.env.handler, "device", None)
            if device is not None and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        return time.perf_counter()

    def _profile_add(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self._profile_totals[name] = self._profile_totals.get(name, 0.0) + elapsed

    def _profile_report(self, total_elapsed: float) -> None:
        if not self.profile_enabled:
            return

        self._profile_window_steps += 1
        self._profile_window_total += total_elapsed
        if self._profile_window_steps < self.profile_interval:
            return

        steps = self._profile_window_steps
        avg_total_ms = 1000.0 * self._profile_window_total / steps
        vec_steps_per_sec = steps / max(self._profile_window_total, 1e-9)
        samples_per_sec = (steps * self.num_envs) / max(self._profile_window_total, 1e-9)
        parts = []
        for name, value in sorted(self._profile_totals.items(), key=lambda item: item[1], reverse=True):
            avg_ms = 1000.0 * value / steps
            pct = 100.0 * value / max(self._profile_window_total, 1e-9)
            parts.append(f"{name}={avg_ms:.3f}ms/{pct:.0f}%")

        print(
            f"[MetaSimProfile] envs={self.num_envs} steps={steps} "
            f"avg_step={avg_total_ms:.3f}ms "
            f"({vec_steps_per_sec:.1f} vec_steps/s, {samples_per_sec:.0f} samples/s) "
            f"sync_cuda={int(self.profile_sync_cuda)} | " + " ".join(parts),
            flush=True,
        )
        self._profile_totals.clear()
        self._profile_window_steps = 0
        self._profile_window_total = 0.0


    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self, env_ids: list[int] | None = None, seed: int | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        init_states = self.unwrapped._get_default_states(seed)
        #self.env.handler.checker.reset(self.env.handler)
        profile_total_t0 = self._profile_now()

        profile_t0 = self._profile_now()
        init_states = self.unwrapped._get_default_states(seed, num_states=len(env_ids))
        self._profile_add("reset_get_default_states", self._profile_now() - profile_t0)

        #tic = time.time()
        profile_t0 = self._profile_now()
        states, _ = self.env.reset(states=init_states, env_ids=env_ids)
        self._profile_add("reset_handler_reset", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.handler.device)
        reward_functions = getattr(self.scenario.task, "reward_functions", [])
        if isinstance(reward_functions, _MISSING_TYPE) or reward_functions is None:
            reward_functions = []
        for reward_fn in reward_functions:
            reset_fn = getattr(reward_fn, "reset", None)
            if callable(reset_fn):
                reset_fn(env_ids_tensor, states)
        self._profile_add("reset_reward_hooks", self._profile_now() - profile_t0)
        #toc = time.time()
        #print(f"Reset took {toc - tic:.2f} seconds")
        profile_t0 = self._profile_now()
        obs = self.unwrapped._get_obs(states=states)
        self._profile_add("reset_get_obs", self._profile_now() - profile_t0)
        self._profile_add("reset_total", self._profile_now() - profile_total_t0)
        return obs, {"states": states}

    def step(self, actions: list[dict]):
        """Take a step in the environment."""
        profile_total_t0 = self._profile_now()

        profile_t0 = self._profile_now()
        _, _, success, timeout, _ = self.env.step(actions)
        self._profile_add("handler_step", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        states = self.env.handler.get_states()
        self._profile_add("get_states", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs = self.unwrapped._get_obs(states=states)
        self._profile_add("get_obs_from_states", self._profile_now() - profile_t0)

        #tic = time.time()
        profile_t0 = self._profile_now()
        if self.scenario.dagger == 1:
            rewards = torch.zeros(self.num_envs, device=obs.device)
        else:
            rewards = self.unwrapped._calculate_rewards(states=states)
        self._profile_add("calculate_rewards", self._profile_now() - profile_t0)
        #toc = time.time()
        #print(f"Reward calculation took {toc - tic:.2f} seconds")
        self._profile_report(self._profile_now() - profile_total_t0)
        return obs, rewards, success, timeout, {"states": states}

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Helper methods
    ############################################################
    def _get_obs(self, states=None):
        """Get current observations for all environments."""
        ## TODO: make this method generalizable
        if states is None:
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

    def _calculate_rewards(self, states=None):
        """Calculate rewards based on distance to origin."""
        if states is None:
            states = self.env.handler.get_states()
        tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)

        if isinstance(self.scenario.task.reward_functions, _MISSING_TYPE):
            return tot_reward

        for reward_fn, weight in zip(self.scenario.task.reward_functions, self.scenario.task.reward_weights):

            profile_t0 = self._profile_now()
            reward_fn_ret = reward_fn(states, self.scenario.robots[0].name)
            self._profile_add(f"reward:{reward_fn.__class__.__name__}", self._profile_now() - profile_t0)
            tot_reward += weight * reward_fn_ret
        #print(tot_reward)
        return tot_reward

    def _get_default_states(self, seed: int | None = None, num_states: int | None = None):
        """Generate default reset states."""
        ## TODO: use non-reqeatable random choice when there is enough candidate states?
        if num_states is None:
            num_states = self.num_envs
        return random.Random(seed).choices(self.candidate_init_states, k=num_states)
