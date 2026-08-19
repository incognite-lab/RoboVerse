from __future__ import annotations

import os
import time

import numpy as np
import torch
from loguru import logger as log
from tensordict import TensorDict


class RSLRLMetaSimEnv:
    """MetaSim wrapper with the VecEnv interface expected by RSL-RL."""

    def __init__(self, env):
        self.env = env
        device = "cuda" if env.scenario.sim in ["isaaclab", "genesis"] else "cpu"
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.use_vision = False
        self.extras = {}

        joint_limits = env.scenario.robots[0].joint_limits
        self.num_actions = len(joint_limits)
        self.num_obs = (2 * self.num_actions) + 3 + 3
        self.num_privileged_obs = self.num_obs + 27
        self.cfg = {
            "num_obs": self.num_obs,
            "num_privileged_obs": self.num_privileged_obs,
            "num_actions": self.num_actions,
        }

        self.action_low = torch.tensor(
            [lim[0] for lim in joint_limits.values()],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_high = torch.tensor(
            [lim[1] for lim in joint_limits.values()],
            dtype=torch.float32,
            device=self.device,
        )

        self.policy_obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float32, device=self.device)
        self.obs_buf = TensorDict(
            {
                "policy": self.policy_obs_buf,
                "critic": self.critic_obs_buf,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        self.privileged_obs_buf = self.critic_obs_buf
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_length = int(
            getattr(env.scenario.task, "episode_length", getattr(env.scenario, "episode_length", 1000))
        )

        self.current_commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_duration = 800
        self.command_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.command_step = 0
        self.command_mode = "random"
        self.fixed_command = torch.tensor([0.8, 0.0, 0.0], dtype=torch.float32, device=self.device)

        self.external_force_enabled = env.scenario.force
        self.force_x_min = env.scenario.force_x_min
        self.force_x_max = env.scenario.force_x_max
        self.force_y_min = env.scenario.force_y_min
        self.force_y_max = env.scenario.force_y_max
        self.force_interval_steps = max(1, int(getattr(env.scenario, "force_interval_steps", 50)))
        self.force_duration_steps = max(1, int(getattr(env.scenario, "force_duration_steps", 1)))
        self.force_duration_steps = min(self.force_duration_steps, self.force_interval_steps)
        self.force_resample_on_impulse = bool(getattr(env.scenario, "force_resample_on_impulse", True))
        self.external_force_step = 0
        self.external_force_was_active = False
        self.cached_forces = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.cached_forces_3d = np.zeros((self.num_envs, 1, 3), dtype=np.float32)
        self.zero_forces_3d = np.zeros((self.num_envs, 1, 3), dtype=np.float32)
        self.external_force_env_ids = np.arange(self.num_envs, dtype=np.int32)
        self.external_force_link_ids = None
        self.external_force_vectorized_failed = False
        self._body_cache_key = None
        self.right_ankle_idx = None
        self.left_ankle_idx = None
        self.torso_idx = None

        self.profile_enabled = os.getenv("ROBO_WALK_PROFILE", "1") != "0"
        self.profile_interval = int(os.getenv("ROBO_WALK_PROFILE_INTERVAL", "1000"))
        self.profile_sync_cuda = os.getenv("ROBO_WALK_PROFILE_SYNC", "0") == "1"
        self._profile_totals = {}
        self._profile_window_steps = 0
        self._profile_window_total = 0.0

    def _profile_now(self) -> float:
        if self.profile_enabled and self.profile_sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
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
        samples_per_sec = (steps * self.num_envs) / max(self._profile_window_total, 1e-9)
        parts = []
        for name, value in sorted(self._profile_totals.items(), key=lambda item: item[1], reverse=True):
            avg_ms = 1000.0 * value / steps
            pct = 100.0 * value / max(self._profile_window_total, 1e-9)
            parts.append(f"{name}={avg_ms:.3f}ms/{pct:.0f}%")

        print(
            f"[RSLWalkProfile] envs={self.num_envs} steps={steps} "
            f"avg_step={avg_total_ms:.3f}ms ({samples_per_sec:.0f} samples/s) | "
            + " ".join(parts),
            flush=True,
        )
        self._profile_totals.clear()
        self._profile_window_steps = 0
        self._profile_window_total = 0.0

    def _as_env_ids(self, env_ids=None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.detach().cpu().long().tolist()
        return list(env_ids)

    def _cache_body_indices(self, robot):
        body_names = list(robot.body_names)
        cache_key = tuple(body_names)
        if self._body_cache_key == cache_key:
            return

        self.right_ankle_idx = body_names.index("right_ankle_roll_link")
        self.left_ankle_idx = body_names.index("left_ankle_roll_link")
        self.torso_idx = body_names.index("torso_link")
        self._body_cache_key = cache_key

    def _build_observations(self, obs: torch.Tensor | np.ndarray, states=None) -> TensorDict:
        if states is None:
            states = self.env.env.handler.get_states()
        robot = states.robots[self.env.scenario.robots[0].name]
        self._cache_body_indices(robot)

        joint_pos = torch.as_tensor(obs, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        joint_vel = robot.joint_vel.reshape(self.num_envs, -1).to(device=self.device, dtype=torch.float32)
        gyro = states.sensors["gyro0"].reshape(self.num_envs, 3).to(device=self.device, dtype=torch.float32)
        command = states.sensors["command0"].reshape(self.num_envs, 3).to(device=self.device, dtype=torch.float32)

        actor_obs = torch.cat([joint_pos, joint_vel, gyro, command], dim=1)

        right_ankle_posori = robot.body_state[:, self.right_ankle_idx, :7].to(device=self.device, dtype=torch.float32)
        left_ankle_posori = robot.body_state[:, self.left_ankle_idx, :7].to(device=self.device, dtype=torch.float32)
        torso_posori = robot.body_state[:, self.torso_idx, :7].to(device=self.device, dtype=torch.float32)
        root_lin_vel = robot.root_state[:, 7:10].to(device=self.device, dtype=torch.float32)
        root_ang_vel = robot.root_state[:, 10:13].to(device=self.device, dtype=torch.float32)
        critic_extra = torch.cat(
            [right_ankle_posori, left_ankle_posori, torso_posori, root_lin_vel, root_ang_vel],
            dim=1,
        )
        critic_obs = torch.cat([actor_obs, critic_extra], dim=1)
        self.policy_obs_buf = actor_obs
        self.critic_obs_buf = critic_obs
        self.privileged_obs_buf = critic_obs
        self.obs_buf = TensorDict(
            {
                "policy": actor_obs,
                "critic": critic_obs,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        return self.obs_buf

    def reset(self, env_ids=None):
        env_ids_list = self._as_env_ids(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids_list, dtype=torch.long, device=self.device)

        self.command_timer[env_ids_tensor] = 0
        self.current_commands[env_ids_tensor] = self.generate_commands(num_commands=len(env_ids_list))
        self.set_command(self.current_commands)
        self.generate_external_forces(env_ids=env_ids_list)

        obs, _ = self.env.reset(env_ids=env_ids_list)
        self.set_command(self.current_commands)
        self.obs_buf = self._build_observations(obs)

        self.episode_length_buf[env_ids_tensor] = 0
        self.reset_buf[env_ids_tensor] = False
        return self.obs_buf

    def step(self, actions: torch.Tensor):
        profile_total_t0 = self._profile_now()

        profile_t0 = self._profile_now()
        actions = actions.to(device=self.device, dtype=torch.float32)
        actions = torch.clamp(actions, self.action_low.unsqueeze(0), self.action_high.unsqueeze(0))
        if self.env.scenario.sim in ["genesis", "isaacgym"]:
            sim_actions = actions
        else:
            sim_actions = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        self._profile_add("prepare_actions", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        self.command_step += 1
        if self.command_step >= self.command_duration:
            self.current_commands = self.generate_commands(num_commands=self.num_envs)
            self.command_timer.zero_()
            self.command_step = 0
        else:
            self.command_timer += 1
        self.set_command(self.current_commands)
        self._profile_add("command_update", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        self.apply_external_forces()
        self._profile_add("external_forces", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs, rewards, unsuccess, timeout, extra = self.env.step(sim_actions)
        self._profile_add("metasim_step_total", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        self.obs_buf = self._build_observations(obs, states=extra.get("states") if isinstance(extra, dict) else None)
        self._profile_add("build_obs", self._profile_now() - profile_t0)

        rewards = rewards.to(device=self.device, dtype=torch.float32)
        unsuccess = unsuccess.to(device=self.device).bool()
        timeout = timeout.to(device=self.device).bool()
        dones = unsuccess | timeout
        self.episode_length_buf += 1
        self.rew_buf = rewards
        self.reset_buf = dones

        infos = {"time_outs": timeout}

        rewards[unsuccess] = -1.0
        self.rew_buf = rewards

        reset_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            terminal_obs = self.obs_buf[reset_ids].detach().clone()
            terminal_privileged_obs = self.privileged_obs_buf[reset_ids].detach().clone()
            self.reset(env_ids=reset_ids)
            infos["terminal_observation"] = terminal_obs
            infos["terminal_privileged_observation"] = terminal_privileged_obs

        self._profile_report(self._profile_now() - profile_total_t0)
        return self.obs_buf, rewards, dones, infos

    def get_observations(self) -> TensorDict:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf

    def set_command(self, command: torch.Tensor):
        self.env.env.handler.sensors[1].set_command(command)

    def generate_commands(self, num_commands: int | None = None) -> torch.Tensor:
        if num_commands is None:
            num_commands = self.num_envs

        if self.command_mode == "fixed":
            return self.fixed_command.unsqueeze(0).repeat(num_commands, 1)

        if self.command_mode == "random":
            command_list = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    [0.6, 0.0, 0.5],
                    [0.6, 0.0, -0.5],
                    [-0.2, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            idx = torch.randint(0, command_list.shape[0], (num_commands,), device=self.device)
            return command_list[idx]

        raise ValueError(f"Unknown command_mode: {self.command_mode}")

    def generate_external_forces(self, env_ids=None):
        if not self.external_force_enabled:
            self.cached_forces[:] = 0.0
            self.cached_forces_3d[:] = 0.0
            return

        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        else:
            env_ids = np.asarray(list(env_ids), dtype=np.int64)

        self.cached_forces[env_ids, 0] = np.random.uniform(
            self.force_x_min,
            self.force_x_max,
            size=len(env_ids),
        )
        self.cached_forces[env_ids, 1] = np.random.uniform(
            self.force_y_min,
            self.force_y_max,
            size=len(env_ids),
        )
        self.cached_forces_3d[env_ids, 0, :2] = self.cached_forces[env_ids]
        self.cached_forces_3d[env_ids, 0, 2] = 0.0

    def _ensure_external_force_targets(self):
        if self.external_force_link_ids is None:
            states = self.env.env.handler.get_states()
            torso_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("torso_link")
            self.external_force_link_ids = np.array([torso_idx], dtype=np.int32)

    def _apply_external_force_array(self, force_array: np.ndarray):
        self._ensure_external_force_targets()
        robot_instance = self.env.env.handler.robot_inst.solver

        if not self.external_force_vectorized_failed:
            try:
                robot_instance.apply_links_external_force(
                    force=force_array,
                    links_idx=self.external_force_link_ids,
                    envs_idx=self.external_force_env_ids,
                )
                return
            except Exception as exc:
                self.external_force_vectorized_failed = True
                log.warning(
                    f"Vectorized external force application failed once, "
                    f"falling back to per-env calls: {exc}"
                )

        torso_idx = int(self.external_force_link_ids[0])
        for env_id in range(self.env.num_envs):
            fx, fy, fz = force_array[env_id, 0]
            robot_instance.apply_links_external_force(
                force=np.array([[fx, fy, fz]], dtype=np.float32),
                links_idx=np.array([torso_idx], dtype=np.int32),
                envs_idx=np.array([env_id], dtype=np.int32),
            )

    def apply_external_forces(self):
        if not self.external_force_enabled:
            return

        phase = self.external_force_step % self.force_interval_steps
        impulse_start = phase == 0
        impulse_active = phase < self.force_duration_steps

        if impulse_start and self.force_resample_on_impulse:
            self.generate_external_forces()

        if impulse_active:
            self._apply_external_force_array(self.cached_forces_3d)
            self.external_force_was_active = True
        elif self.external_force_was_active:
            self._apply_external_force_array(self.zero_forces_3d)
            self.external_force_was_active = False

        self.external_force_step += 1

    def close(self):
        self.env.close()
