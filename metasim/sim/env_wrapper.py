"""Gym-like environment wrapper."""

from __future__ import annotations

import time
import os
from typing import Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch
from loguru import logger as log

from metasim.sim import BaseSimHandler
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, TimeOut

THandler = TypeVar("THandler", bound=BaseSimHandler)


class EnvWrapper(Generic[THandler]):
    """Gym-like environment wrapper."""

    handler: THandler

    def __init__(self, *args, **kwargs) -> None: ...
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]: ...
    def render(self) -> None: ...
    def close(self) -> None: ...

    @property
    def episode_length_buf(self) -> list[int]: ...


def IdentityEnvWrapper(cls: type[BaseSimHandler]) -> type[EnvWrapper[BaseSimHandler]]:
    """Gym-like environment wrapper for IsaacLab."""

    class IdentityEnv(EnvWrapper[BaseSimHandler]):
        def __init__(self, *args, **kwargs):
            self.handler = cls(*args, **kwargs)
            self.handler.launch()

        def reset(self, states: list[EnvState] | None = None, env_ids: list[int] | None = None) -> tuple[Obs, Extra]:
            if env_ids is None:
                env_ids = list(range(self.handler.num_envs))

            if states is not None:
                self.handler.set_states(states, env_ids=env_ids)
            return self.handler.reset(env_ids=env_ids)

        def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            return self.handler.step(action)

        def render(self) -> None:
            log.warning("render() is not implemented yet")

        def close(self) -> None:
            self.handler.close()

        @property
        def episode_length_buf(self) -> list[int]:
            return self.handler.episode_length_buf

        @property
        def observation_space(self) -> gym.Space:
            return self.handler.scenario.task.observation_space

        @property
        def action_space(self) -> gym.Space:
            action_low = torch.tensor(
                [limit[0] for limit in self.handler.scenario.robots[0].joint_limits.values()], dtype=torch.float32
            )
            action_high = torch.tensor(
                [limit[1] for limit in self.handler.scenario.robots[0].joint_limits.values()], dtype=torch.float32
            )
            return gym.spaces.Box(
                low=action_low.cpu().numpy(), high=action_high.cpu().numpy(), shape=(len(action_low),), dtype=np.float32
            )

    return IdentityEnv


def GymEnvWrapper(cls: type[THandler]) -> type[EnvWrapper[THandler]]:
    """Gym-like environment wrapper for IsaacGym, MuJoCo, Pybullet, SAPIEN, Genesis, etc."""

    class GymEnv:
        def __init__(self, *args, **kwargs):
            self.handler = cls(*args, **kwargs)
            self.handler.launch()
            self._episode_length_buf = torch.zeros(self.handler.num_envs, dtype=torch.int32, device=self.handler.device)
            self._profile_enabled = os.getenv("ROBO_WALK_PROFILE", "1") != "0"
            self._profile_interval = int(os.getenv("ROBO_WALK_PROFILE_INTERVAL", "1000"))
            self._profile_sync_cuda = os.getenv("ROBO_WALK_PROFILE_SYNC", "0") == "1"
            self._profile_totals = {}
            self._profile_steps = 0
            self._profile_total_time = 0.0

        def _profile_now(self) -> float:
            if self._profile_enabled and self._profile_sync_cuda:
                device = getattr(self.handler, "device", None)
                if device is not None and torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
            return time.perf_counter()

        def _profile_add(self, name: str, elapsed: float) -> None:
            if self._profile_enabled:
                self._profile_totals[name] = self._profile_totals.get(name, 0.0) + elapsed

        def _profile_report(self, elapsed: float) -> None:
            if not self._profile_enabled:
                return
            self._profile_steps += 1
            self._profile_total_time += elapsed
            if self._profile_steps < self._profile_interval:
                return

            steps = self._profile_steps
            parts = []
            for name, value in sorted(self._profile_totals.items(), key=lambda item: item[1], reverse=True):
                avg_ms = 1000.0 * value / steps
                pct = 100.0 * value / max(self._profile_total_time, 1e-9)
                parts.append(f"{name}={avg_ms:.3f}ms/{pct:.0f}%")
            avg_total_ms = 1000.0 * self._profile_total_time / steps
            print(
                f"[EnvWrapperProfile] envs={self.handler.num_envs} steps={steps} "
                f"avg_step={avg_total_ms:.3f}ms | " + " ".join(parts),
                flush=True,
            )
            self._profile_totals.clear()
            self._profile_steps = 0
            self._profile_total_time = 0.0

        def reset(self, states: list[EnvState] | None = None, env_ids: list[int] | None = None) -> tuple[Obs, Extra]:
            profile_total_t0 = self._profile_now()
            if env_ids is None:
                env_ids = list(range(self.handler.num_envs))

            profile_t0 = self._profile_now()
            self._episode_length_buf[env_ids] = 0
            self._profile_add("reset_episode_buf", self._profile_now() - profile_t0)
            if states is not None:
                profile_t0 = self._profile_now()
                self.handler.set_states(states, env_ids=env_ids)
                self._profile_add("reset_set_states", self._profile_now() - profile_t0)

            profile_t0 = self._profile_now()
            self.handler.checker.reset(self.handler, env_ids=env_ids) # zde reset přec checker
            self._profile_add("reset_checker", self._profile_now() - profile_t0)
            #print(self.handler.physics.contexts)#zázračný print bez kterého to nefunguje
            if hasattr(self.handler, 'physics'):
                contexts = self.handler.physics.contexts
            profile_t0 = self._profile_now()
            self.handler.refresh_render()
            self._profile_add("reset_refresh_render", self._profile_now() - profile_t0)

            profile_t0 = self._profile_now()
            states = self.handler.get_states()
            self._profile_add("reset_get_states_after", self._profile_now() - profile_t0)
            self._profile_report(self._profile_now() - profile_total_t0)
            return states, None

        def step(self, actions: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            profile_total_t0 = self._profile_now()
            self._episode_length_buf += 1

            profile_t0 = self._profile_now()
            for robot in self.handler.robots:
                self.handler.set_dof_targets(robot.name, actions)
            self._profile_add("set_dof_targets", self._profile_now() - profile_t0)

            profile_t0 = self._profile_now()
            self.handler.simulate()
            simulate_elapsed = self._profile_now() - profile_t0
            self._profile_add("simulate", simulate_elapsed)
            log.trace(f"Time taken to handler.simulate(): {simulate_elapsed:.4f}s")
            reward = None

            profile_t0 = self._profile_now()
            success = self.handler.checker.check(self.handler)
            checker_elapsed = self._profile_now() - profile_t0
            self._profile_add("checker_check", checker_elapsed)
            log.trace(f"Time taken to handler.checker.check(): {checker_elapsed:.4f}s")

            time_out = self._episode_length_buf >= self.handler.scenario.episode_length
            self._profile_report(self._profile_now() - profile_total_t0)
            return None, reward, success, time_out, None

        def step_actions(self, actions) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            self._episode_length_buf += 1
            self.handler.set_actions(self.handler.robot.name, actions)
            self.handler.simulate()
            reward = None
            success = self.handler.checker.check(self.handler)
            states = self.handler.get_states()
            time_out = self._episode_length_buf >= self.handler.scenario.episode_length
            return states, reward, success, time_out, None

        def render(self) -> None:
            log.warning("render() is not implemented yet")
            pass

        def close(self) -> None:
            self.handler.close()

        def _get_reward(self) -> Reward:
            if hasattr(self.handler.task, "reward_fn"):
                # XXX: compatible with old states format
                states = [{**state["robots"], **state["objects"]} for state in self.handler.get_states()]
                return self.handler.task.reward_fn(states)
            else:
                return None

        @property
        def episode_length_buf(self) -> list[int]:
            return self._episode_length_buf.tolist()

        @property
        def episode_length_buf_tensor(self) -> list[int]:
            return self._episode_length_buf

        @property
        def action_space(self) -> gym.Space:
            action_low = torch.tensor(
                [limit[0] for limit in self.handler.scenario.robots[0].joint_limits.values()], dtype=torch.float32
            )
            action_high = torch.tensor(
                [limit[1] for limit in self.handler.scenario.robots[0].joint_limits.values()], dtype=torch.float32
            )
            return gym.spaces.Box(
                low=action_low.cpu().numpy(), high=action_high.cpu().numpy(), shape=(len(action_low),), dtype=np.float32
            )

        @property
        def observation_space(self) -> gym.Space:
            # Handle simple format like {"shape": [48]} used by IsaacGym tasks
            if "shape" in self.handler.scenario.task.observation_space:
                shape = self.handler.scenario.task.observation_space["shape"]
                return gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(shape), dtype=np.float32)

            # Handle nested format for more complex observation spaces
            observation_space = {}
            for obj in self.handler.scenario.task.observation_space.keys():
                if obj == "robot":
                    for joint in self.handler.scenario.robots[0].joint_names:
                        observation_space[joint] = gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        )
                else:
                    for key, value in self.handler.scenario.task.observation_space[obj].items():
                        observation_space[obj][key] = gym.spaces.Box(
                            low=value["low"], high=value["high"], shape=value["shape"], dtype=value["dtype"]
                        )
            return gym.spaces.Dict(observation_space)

    return GymEnv
