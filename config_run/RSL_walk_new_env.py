import torch
import numpy as np


class RSLRLMetaSimEnv:
    """
    Environment wrapper for RSL-RL.
    Fully GPU-compatible, returns torch tensors, and follows the interface:
    - reset() -> obs (torch)
    - step(actions) -> obs, rewards, dones, infos
    """

    def __init__(self, env):
        self.env = env
        device = "cuda" if env.scenario.sim in ["isaaclab", "genesis"] else "cpu"
        self.device = torch.device(device)

        # --- Build action & observation size like in SB3 wrapper ---
        joint_limits = env.scenario.robots[0].joint_limits
        self.num_actions = len(joint_limits)
        self.num_obs = self.num_actions + 3 + 17 + 3   # podle tvého komba

        # Actions limits (needed by RSL-RL)
        self.action_space = torch.zeros(self.num_actions, 2, device=self.device)
        for i, lim in enumerate(joint_limits.values()):
            self.action_space[i, 0] = lim[0]
            self.action_space[i, 1] = lim[1]

        # internal counters
        self.num_envs = env.num_envs
        self.timesteps = torch.zeros(self.num_envs, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.command_duration = 200

        self.current_commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_mode = "fixed"
        self.fixed_command = torch.tensor([2.0, 0.0, 0.0], device=self.device)

        # external forces
        self.external_force_enabled = env.scenario.force
        self.force_x_min = env.scenario.force_x_min
        self.force_x_max = env.scenario.force_x_max
        self.force_y_min = env.scenario.force_y_min
        self.force_y_max = env.scenario.force_y_max
        self.cached_forces = torch.zeros((self.num_envs, 2), device=self.device)

    # -------------------------------------------------------------
    # --- Obs builder (torch version)
    # -------------------------------------------------------------
    def _combine_obs(self, obs):
        states = self.env.env.handler.get_states()
        gyro = states.sensors["gyro0"].reshape(self.num_envs, 3)
        cmd = states.sensors["command0"].reshape(self.num_envs, 3)
        obs = obs.reshape(self.num_envs, -1)
        return torch.cat([obs, gyro, cmd], dim=1)

    def add_extra_to_obs(self, obs):
        states = self.env.env.handler.get_states()
        robot = states.robots[self.env.scenario.robots[0].name]

        right_idx = robot.body_names.index("right_ankle_roll_link")
        left_idx = robot.body_names.index("left_ankle_roll_link")
        torso_idx = robot.body_names.index("torso_link")

        right = robot.body_state[:, right_idx, :7]
        left = robot.body_state[:, left_idx, :7]
        torso = robot.body_state[:, torso_idx, :3]

        extra = torch.cat([right, left, torso], dim=1)
        obs = obs.reshape(self.num_envs, -1)
        return torch.cat([obs, extra], dim=1)

    # -------------------------------------------------------------
    def reset(self, env_ids=None):
        self.command_timer.zero_()
        self.current_commands = self.generate_commands()
        self.set_command(self.current_commands)
        self.generate_external_forces(env_ids)

        obs, _ = self.env.reset(env_ids=env_ids)
        obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)

        self.timesteps.zero_()
        return obs.to(self.device)

    # -------------------------------------------------------------
    def step(self, actions):
        # Build action dict
        action_dicts = [
            {
                self.env.scenario.robots[0].name: {
                    "dof_pos_target": dict(zip(
                        self.env.scenario.robots[0].joint_limits.keys(),
                        actions[i].detach().cpu().numpy()
                    ))
                }
            }
            for i in range(self.num_envs)
        ]

        # --- Update commands ---
        self.command_timer += 1
        update_mask = self.command_timer >= self.command_duration
        if torch.any(update_mask):
            new_cmds = self.generate_commands()
            self.current_commands[update_mask] = new_cmds[update_mask]
            self.command_timer[update_mask] = 0

        self.set_command(self.current_commands)

        # external forces
        self.apply_external_forces()

        # Step simulation
        obs, rewards, unsuccess, timeout, _ = self.env.step(action_dicts)

        obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)

        dones = timeout | unsuccess
        self.timesteps += (~unsuccess).float()

        infos = {}

        # reset failed
        unsuccess_ids = torch.nonzero(unsuccess).squeeze(-1)
        if len(unsuccess_ids) > 0:
            rewards[unsuccess_ids] = -1.0
            self.timesteps[unsuccess_ids] = 0.0
            self.reset(env_ids=unsuccess_ids.tolist())

        # reset timeout
        timeout_ids = torch.nonzero(timeout).squeeze(-1)
        if len(timeout_ids) > 0:
            self.timesteps[timeout_ids] = 0.0
            self.reset(env_ids=timeout_ids.tolist())

        return (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
            infos,
        )

    # -------------------------------------------------------------
    def set_command(self, cmd):
        self.env.env.handler.sensors[1].set_command(cmd)

    # -------------------------------------------------------------
    def generate_commands(self):
        if self.command_mode == "fixed":
            return self.fixed_command.repeat(self.num_envs, 1)

        command_list = torch.tensor([
            [2.0,  0.0,  0.0],
            [-1.0, 0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0,  0.0, -1.0],
        ], device=self.device)

        idx = torch.randint(0, command_list.shape[0], (self.num_envs,), device=self.device)
        return command_list[idx]

    # -------------------------------------------------------------
    def generate_external_forces(self, env_ids=None):
        if not self.external_force_enabled:
            self.cached_forces.zero_()
            return

        if env_ids is None:
            env_ids = range(self.num_envs)

        for env_id in env_ids:
            fx = np.random.uniform(self.force_x_min, self.force_x_max)
            fy = np.random.uniform(self.force_y_min, self.force_y_max)
            self.cached_forces[env_id] = torch.tensor([fx, fy], device=self.device)

    def apply_external_forces(self):
        if not self.external_force_enabled:
            return

        states = self.env.env.handler.get_states()
        torso_idx = states.robots[self.env.scenario.robots[0].name].body_names.index("torso_link")
        robot_instance = self.env.env.handler.robot_inst.solver

        for env_id in range(self.env.num_envs):
            fx, fy = self.cached_forces[env_id]
            robot_instance.apply_links_external_force(
                force=np.array([[fx.item(), fy.item(), 0.0]]),
                links_idx=np.array([torso_idx]),
                envs_idx=np.array([env_id]),
            )
