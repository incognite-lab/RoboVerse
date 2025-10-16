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
    robot: str = "g1_with_hands"
    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "genesis"
    model_path: str = "my_env/output/ppo_models/run_2025-10-15_20-22-31/model_80000000"
    video_path: str = "my_env/output/g1_stand_stand_genesis.mp4"
    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def video_ppo():
    args.num_envs = 1
    scenario = ScenarioCfg(task=args.task, robots=[args.robot], sim=args.sim, num_envs=args.num_envs)
    scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
    scenario.robots[0].fix_base_link = False
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    scenario.sensors = [GyroSensorCfg(
         name="gyro0",
         pos=(0.0, 0.0, 0.0),
         mount_to=args.robot,
         mount_link="torso_link"

         )
         ]
    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)

    env = StableBaseline3VecEnv(metasim_env)

    task_name = scenario.task.__class__.__name__[:-3]
    obs_saver = ObsSaver(video_path=args.video_path)

    # load the model
    log.info(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    #model.set_env(env)

    # inference
    obs = env.reset()
    obs_orin = metasim_env.env.handler.get_states()
    obs_saver.add(obs_orin)

    reward_acumulator = [0.0]
    # rollout
    for _ in range(1000):
        actions, _ = model.predict(obs, deterministic=False)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        reward_acumulator += rewards
        print(f"Step reward: {rewards}, Accumulated reward: {sum(reward_acumulator)}")
        plt.pause(0.01)

        obs_orin = metasim_env.env.handler.get_states()
        obs_saver.add(obs_orin)
    log.info(f"🎬 Video saved to {args.video_path}")
    obs_saver.save()


if __name__ == "__main__":
    from my_env.g1_stand_RL import StableBaseline3VecEnv
    video_ppo()
