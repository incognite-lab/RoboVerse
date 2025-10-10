"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

# try:
#     import isaacgym  # noqa: F401
# except ImportError:
#     pass

import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler
import numpy as np

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from my_env.utils import ObsSaver
from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class RewardPlotCallback(BaseCallback):
    """
    Callback pro logování akumulovaných rewardů do TensorBoard.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for r, d in zip(rewards, dones):
            if len(self.episode_rewards) == 0:
                self.episode_rewards.append(0.0)
            self.episode_rewards[-1] += r
            if d:
                ep_reward = self.episode_rewards[-1]
                self.logger.record("episode/accumulated_reward", ep_reward)
                self.episode_rewards.append(0.0)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()


@configclass
class Args:
    """Arguments for the static scene."""
    task: str = "stand"
    robot: str = "g1_no_hands"
    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robots[0].joint_limits
        scale = 0.5
        self.action_space = spaces.Box(
            #low=np.array([lim[0] for lim in joint_limits.values()]),
            #high=np.array([lim[1] for lim in joint_limits.values()]),
            low=-scale,
            high=scale,
            shape=(len(joint_limits),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits),),  # joints + XYZ gyro
            dtype=np.float32,
        )
        self.env = env
        #self.sensors = [GyroSensor(cfg, env.env.handler) for cfg in env.scenario.sensors]
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if args.sim == 'isaaclab' or args.sim == 'genesis' else "cpu"))

        super().__init__(env.num_envs, self.observation_space, self.action_space)
    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        gyrodata = self.sensors[0].get_data()  # shape (num_envs, 3)

        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata], axis=1).astype(np.float32)

    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
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
        obs, rewards, unsuccess, timeout, _ = self.env.step(self.action_dicts)
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)

        dones = timeout.to(unsuccess.device) | unsuccess

        self.timesteps += (~unsuccess).float()

        if dones.any():
            self.env.reset(env_ids=dones.nonzero().squeeze(-1).tolist())
            self.timesteps[unsuccess.cpu()] = 0
        if unsuccess.any():
            self.timesteps[unsuccess.cpu()] = 0
            rewards[unsuccess] = -10.0
            self.env.reset(env_ids=unsuccess.nonzero(as_tuple=False).squeeze(-1).tolist())
            extra = [{} for _ in range(self.num_envs)]
            return obs, rewards.cpu().numpy(), dones.cpu().numpy(), extra
        # reward vynásobíme časem

        time_factors = self.timesteps.to(rewards.device)
        rewards = rewards * time_factors


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

def train_ppo():
    """Train PPO for reaching task."""
    #Choice 1: use scenario config to initialize the environment
    scenario = ScenarioCfg(task=args.task, robots=[args.robot], sim=args.sim, num_envs=args.num_envs, headless=args.headless)
    scenario.episode_length = 500
    scenario.cameras = []  # XXX: remove cameras to avoid rendering to speed up
    """scenario.sensors = [GyroSensorCfg(
        name="gyro0",
        pos=(0.0, 0.0, 0.0),
        mount_to=args.robot,
        mount_link="torso_link"

        )
        ]"""
    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)

    env = StableBaseline3VecEnv(metasim_env)

    policy_kwargs = dict(
    net_arch=[256, 256, 128]  # dvě skryté vrstvy po 128 neuronech
    )
    # PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=50,
        batch_size=64,
        n_epochs=100,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="my_env/output/ppo_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
    )

    #načtení předtrénovaného modelu
    """task_name = scenario.task.__class__.__name__[:-3]
    model = PPO.load(f"my_env/output/g1_stand_{task_name}_{args.sim}")
    model.set_env(env)"""

    reward_callback = RewardPlotCallback("my_env/output/ppo_tensorboard/")
    #Start training
    model.learn(total_timesteps=100_000_000)

    #Save the model
    task_name = scenario.task.__class__.__name__[:-3]
    model.save(f"my_env/output/g1_stand_{task_name}_{args.sim}_3")
    log.info("Model saved.Ending the training and closing the environment.")
    env.close()

    quit()

if __name__ == "__main__":
    train_ppo()
