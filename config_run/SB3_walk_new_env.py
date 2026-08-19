
from __future__ import annotations

import os
import time
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
        num_joints = len(joint_limits)
        obs_dim = (2 * num_joints) + 3 + 3 + 27
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            shape=(num_joints,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.env = env
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))
        self.current_commands = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.timesteps.device)

        self.command_duration = 200     # kolik policy kroků command držíme
        self.command_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=self.timesteps.device)

        # režim: "fixed" = pevný command, "random" = různé směry pro každé env
        self.command_mode = "random"
        self.stand_command_threshold = 0.05
        self.last_command_tracking_metrics: dict[str, float] = {}
        # external forces
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
        self.cached_forces = np.zeros((env.num_envs, 2), dtype=np.float32)  # cache pro síly
        self.cached_forces_3d = np.zeros((env.num_envs, 1, 3), dtype=np.float32)
        self.zero_forces_3d = np.zeros((env.num_envs, 1, 3), dtype=np.float32)
        self.external_force_env_ids = np.arange(env.num_envs, dtype=np.int32)
        self.external_force_link_ids = None
        self.external_force_vectorized_failed = False
        # fixed command pokud je zvolen fixed mode
        self.fixed_command = torch.tensor([0.8, 0.0, 0.0], dtype=torch.float32, device=self.timesteps.device)
        self.profile_enabled = os.getenv("ROBO_WALK_PROFILE", "1") != "0"
        self.profile_interval = int(os.getenv("ROBO_WALK_PROFILE_INTERVAL", "1000"))
        self.profile_sync_cuda = os.getenv("ROBO_WALK_PROFILE_SYNC", "0") == "1"
        self._profile_totals = {}
        self._profile_counts = {}
        self._profile_window_steps = 0
        self._profile_window_total = 0.0
        super().__init__(env.num_envs, self.observation_space, self.action_space)

    def _profile_now(self) -> float:
        if self.profile_enabled and self.profile_sync_cuda and self.timesteps.device.type == "cuda":
            torch.cuda.synchronize(self.timesteps.device)
        return time.perf_counter()

    def _profile_add(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self._profile_totals[name] = self._profile_totals.get(name, 0.0) + elapsed

    def _profile_count(self, name: str, value: int | float) -> None:
        if self.profile_enabled:
            self._profile_counts[name] = self._profile_counts.get(name, 0.0) + float(value)

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
        count_parts = []
        for name, value in sorted(self._profile_counts.items()):
            count_parts.append(f"{name}={value / steps:.1f}/step")

        print(
            f"[SB3WalkProfile] envs={self.num_envs} steps={steps} "
            f"avg_step={avg_total_ms:.3f}ms "
            f"({vec_steps_per_sec:.1f} vec_steps/s, {samples_per_sec:.0f} samples/s) "
            f"sync_cuda={int(self.profile_sync_cuda)} | " + " ".join(parts)
            + (" | counts " + " ".join(count_parts) if count_parts else ""),
            flush=True,
        )
        self._profile_totals.clear()
        self._profile_counts.clear()
        self._profile_window_steps = 0
        self._profile_window_total = 0.0

    def _combine_obs(self, obs: np.ndarray, states=None) -> np.ndarray:
        """Spojí joint states, gyro data a command pro všechna envs."""
        if states is None:
            states = self.env.env.handler.get_states()
        robot = states.robots[self.env.scenario.robots[0].name]
        gyrodata = states.sensors["gyro0"]  # shape (num_envs, 3)
        gyrodata = gyrodata.reshape(self.num_envs, 3).cpu().numpy()
        command_data = states.sensors["command0"]  # shape (num_envs, 3)
        command_data = command_data.reshape(self.num_envs, 3).cpu().numpy()
        joint_vel = robot.joint_vel.reshape(self.num_envs, -1).cpu().numpy()
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, joint_vel, gyrodata, command_data], axis=1).astype(np.float32)
    def add_extra_to_obs(self, obs: np.ndarray, states=None) -> np.ndarray:
        """extend obs with extra data."""
        if states is None:
            states = self.env.env.handler.get_states()
        robot = states.robots[self.env.scenario.robots[0].name]
        right_ankle_idx = robot.body_names.index("right_ankle_roll_link")
        left_ankle_idx = robot.body_names.index("left_ankle_roll_link")
        torso_idx = robot.body_names.index("torso_link")
        right_ankle_posori = robot.body_state[:, right_ankle_idx, :7].cpu().numpy()
        left_ankle_posori = robot.body_state[:, left_ankle_idx, :7].cpu().numpy()
        torso_posori = robot.body_state[:, torso_idx, :7].cpu().numpy()
        root_lin_vel = robot.root_state[:, 7:10].cpu().numpy()
        root_ang_vel = robot.root_state[:, 10:13].cpu().numpy()
        other_pos = np.concatenate(
            [right_ankle_posori, left_ankle_posori, torso_posori, root_lin_vel, root_ang_vel],
            axis=1,
        )
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, other_pos], axis=1).astype(np.float32)

    @staticmethod
    def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q.unbind(-1)
        return torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @classmethod
    def _local_xy_velocity(cls, root_state: torch.Tensor) -> torch.Tensor:
        vel_world = root_state[:, 7:10]
        yaw = cls._yaw_from_quat(root_state[:, 3:7])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx = vel_world[:, 0] * cos_yaw + vel_world[:, 1] * sin_yaw
        vy = -vel_world[:, 0] * sin_yaw + vel_world[:, 1] * cos_yaw
        return torch.stack((vx, vy), dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(torch.any(mask).item()):
            return 0.0
        return float(values[mask].mean().detach().cpu().item())

    def _update_command_tracking_metrics(self, states=None) -> None:
        """Store aggregate diagnostics for how closely the robot follows commands."""
        if states is None:
            states = self.env.env.handler.get_states()
        robot = states.robots[self.env.scenario.robots[0].name]

        root_state = robot.root_state
        cmd = states.sensors["command0"].reshape(self.num_envs, 3).to(
            device=root_state.device,
            dtype=root_state.dtype,
        )

        local_vel_xy = self._local_xy_velocity(root_state)
        yaw_rate = root_state[:, 12]
        target_xy = cmd[:, :2]
        target_yaw_rate = cmd[:, 2]

        xy_error_vec = target_xy - local_vel_xy
        xy_error_sq = torch.sum(torch.square(xy_error_vec), dim=1)
        xy_error = torch.sqrt(xy_error_sq)
        yaw_error = target_yaw_rate - yaw_rate
        yaw_error_abs = torch.abs(yaw_error)
        tracking_score = torch.exp(-2.0 * xy_error_sq) * torch.exp(-1.5 * torch.square(yaw_error))

        command_norm = torch.linalg.norm(cmd, dim=1)
        standing_mask = command_norm < self.stand_command_threshold
        moving_mask = ~standing_mask
        speed_xy = torch.linalg.norm(local_vel_xy, dim=1)

        self.last_command_tracking_metrics = {
            "tracking/xy_error_mean": float(xy_error.mean().detach().cpu().item()),
            "tracking/xy_error_rmse": float(torch.sqrt(xy_error_sq.mean()).detach().cpu().item()),
            "tracking/x_error_abs_mean": float(torch.abs(xy_error_vec[:, 0]).mean().detach().cpu().item()),
            "tracking/y_error_abs_mean": float(torch.abs(xy_error_vec[:, 1]).mean().detach().cpu().item()),
            "tracking/yaw_error_abs_mean": float(yaw_error_abs.mean().detach().cpu().item()),
            "tracking/yaw_error_rmse": float(torch.sqrt(torch.square(yaw_error).mean()).detach().cpu().item()),
            "tracking/score_mean": float(tracking_score.mean().detach().cpu().item()),
            "tracking/cmd_x_mean": float(target_xy[:, 0].mean().detach().cpu().item()),
            "tracking/cmd_y_mean": float(target_xy[:, 1].mean().detach().cpu().item()),
            "tracking/cmd_yaw_mean": float(target_yaw_rate.mean().detach().cpu().item()),
            "tracking/vel_x_mean": float(local_vel_xy[:, 0].mean().detach().cpu().item()),
            "tracking/vel_y_mean": float(local_vel_xy[:, 1].mean().detach().cpu().item()),
            "tracking/yaw_rate_mean": float(yaw_rate.mean().detach().cpu().item()),
            "tracking/moving_xy_error_mean": self._masked_mean(xy_error, moving_mask),
            "tracking/standing_speed_mean": self._masked_mean(speed_xy, standing_mask),
            "tracking/moving_env_ratio": float(moving_mask.float().mean().detach().cpu().item()),
        }

    def get_command_tracking_metrics(self) -> dict[str, float]:
        return dict(self.last_command_tracking_metrics)

    def reset(self,env_ids: list[int] | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.timesteps.device)
        self.command_timer[env_ids_tensor] = 0
        self.current_commands[env_ids_tensor] = self.generate_commands(num_commands=len(env_ids))
        self.set_command(self.current_commands)
        self.generate_external_forces(env_ids=env_ids)

        obs, extra = self.env.reset(env_ids=env_ids)
        states = extra.get("states") if isinstance(extra, dict) else None
        self.set_command(self.current_commands)
        obs = obs.cpu().numpy()
        obs = self._combine_obs(obs, states=states)
        obs = self.add_extra_to_obs(obs, states=states)
        self.timesteps[env_ids_tensor] = 0.0
        return obs
    def set_command(self,command):
        self.env.env.handler.sensors[1].set_command(command)


    def generate_external_forces(self, env_ids = None):
        """Generuje a uloží síly pro všechny envs – volá se pouze při resetu."""
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
                links_idx=np.array([torso_idx]),
                envs_idx=np.array([env_id]),
            )

    def apply_external_forces(self):
        """Apply short external-force impulses instead of a permanent per-step push."""
        if not self.external_force_enabled:
            return

        phase = self.external_force_step % self.force_interval_steps
        impulse_start = phase == 0
        impulse_active = phase < self.force_duration_steps

        if impulse_start and self.force_resample_on_impulse:
            self.generate_external_forces()

        if impulse_active:
            #print("šťouch")
            self._apply_external_force_array(self.cached_forces_3d)
            self.external_force_was_active = True
        elif self.external_force_was_active:
            self._apply_external_force_array(self.zero_forces_3d)
            self.external_force_was_active = False

        self.external_force_step += 1


    def generate_commands(self, num_commands: int | None = None):
        """Generuje commandy podle zvoleného režimu.
        - "fixed": všechny envs dostanou stejný command
        - "random": každé env má svůj náhodný command
        """
        if num_commands is None:
            num_commands = self.num_envs

        if self.command_mode == "fixed":
            cmd = self.fixed_command.unsqueeze(0).repeat(num_commands, 1)
            return cmd

        elif self.command_mode == "random":


            # ---- seznam povolených movement commandů ----
            command_list = torch.tensor([
                [0.0, 0.0, 0.0],    # stůj
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.8, 0.0, 0.0],    # rychleji vpřed
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.6, 0.0, 0.5],    # yaw
                [0.6, 0.0, -0.5],   # vpřed + yaw
                [0.6, 0.0, 1.0],    # yaw
                [0.6, 0.0, -1.0],   # vpřed + yaw
                [-0.2, 0.0, 0.0],    # vzad
                [-0.4, 0.0, 0.0],
                [-0.6, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, -0.4, 0.0],

            ], dtype=torch.float32, device=self.timesteps.device)

            # vyber náhodně pro každý env index z rozsahu [0, len(command_list)-1]
            idx = torch.randint(0, command_list.shape[0], (num_commands,), device=self.timesteps.device)
            print(f"generate commands: {command_list[idx].cpu().numpy()}")
            return command_list[idx]

        else:
            raise ValueError(f"Unknown command_mode: {self.command_mode}")

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        profile_t0 = self._profile_now()
        self.action_dicts = actions.astype(np.float32, copy=False)
        self._profile_add("step_async_actions", self._profile_now() - profile_t0)

    def step_wait(self):
        """Wait for the step to complete."""
        profile_total_t0 = self._profile_now()

        # --- UPDATE COMMANDS KAŽDÝCH N KROKŮ ---
        profile_t0 = self._profile_now()
        self.command_timer += 1

        update_mask = self.command_timer >= self.command_duration
        if bool(torch.any(update_mask).item()):
            # envs které dosáhly délky trvání → nový command
            new_cmds = self.generate_commands(num_commands=int(update_mask.sum().item()))
            self.current_commands[update_mask] = new_cmds
            self.command_timer[update_mask] = 0
        self._profile_add("command_update", self._profile_now() - profile_t0)

        # nastav komandy do simulace (pro všechna env najednou)
        profile_t0 = self._profile_now()
        self.set_command(self.current_commands)
        self._profile_add("set_command", self._profile_now() - profile_t0)
        #----------------------------------------------------------------
        #-------------apply external forces to torso--------------
        #----------------------------------------------------------------
        profile_t0 = self._profile_now()
        self.apply_external_forces()
        self._profile_add("external_forces", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs, rewards, unsuccess, timeout, extra = self.env.step(self.action_dicts)
        states = extra.get("states") if isinstance(extra, dict) else None
        self._profile_add("metasim_step_total", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs = obs.cpu().numpy()
        self._profile_add("base_obs_cpu_numpy", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs = self._combine_obs(obs, states=states)
        self._profile_add("combine_obs", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        obs = self.add_extra_to_obs(obs, states=states)
        self._profile_add("extra_obs", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        self._update_command_tracking_metrics(states=states)
        self._profile_add("tracking_metrics", self._profile_now() - profile_t0)

        # --- Done flag ---
        profile_t0 = self._profile_now()
        dones = timeout.to(unsuccess.device) | unsuccess

        # --- Update time counters ---
        self.timesteps += (~unsuccess).float()

        # --- Připrav info dicty ---
        infos = [{} for _ in range(self.num_envs)]

        # --- Masky ---
        unsuccess_mask = unsuccess.bool()
        timeout_mask = timeout.to(unsuccess.device).bool()
        unsuccess_mask_np = unsuccess_mask.cpu().numpy()
        timeout_mask_np = timeout_mask.cpu().numpy()
        self._profile_add("done_masks_infos", self._profile_now() - profile_t0)

        # --- Reset neúspěšných envů ---
        profile_t0 = self._profile_now()
        if bool(unsuccess_mask.any().item()):
            rewards[unsuccess_mask] = -1.0
            self.timesteps[unsuccess_mask] = 0.0
            unsuccess_ids = np.nonzero(unsuccess_mask_np)[0].tolist()
            self._profile_count("unsuccess_envs", len(unsuccess_ids))
            for i in unsuccess_ids:
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = False

        # --- Reset úspěšných envů (timeout = úspěch) ---
        if bool(timeout_mask.any().item()):
            self.timesteps[timeout_mask] = 0.0
            timeout_ids = np.nonzero(timeout_mask_np)[0].tolist()
            self._profile_count("timeout_envs", len(timeout_ids))
            for i in timeout_ids:
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["is_success"] = True
                infos[i]["TimeLimit.truncated"] = True

        reset_ids = np.nonzero((unsuccess_mask | timeout_mask).cpu().numpy())[0].tolist()
        if reset_ids:
            self._profile_count("reset_calls", 1)
            self._profile_count("reset_envs", len(reset_ids))
            reset_obs = self.reset(env_ids=reset_ids)
            obs[reset_ids] = reset_obs[reset_ids]
        self._profile_add("reset_done_envs", self._profile_now() - profile_t0)

        profile_t0 = self._profile_now()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()
        self._profile_add("reward_done_cpu_numpy", self._profile_now() - profile_t0)

        total_elapsed = self._profile_now() - profile_total_t0
        self._profile_report(total_elapsed)

        return obs, rewards_np, dones_np, infos
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
