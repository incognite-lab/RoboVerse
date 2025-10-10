"""Evaluation and video recording script for PPO policy."""

from __future__ import annotations
import os
import imageio
import torch
from loguru import logger as log
from stable_baselines3 import PPO
from metasim.cfg.scenario import ScenarioCfg
from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from g1_reach_RL import StableBaseline3VecEnv  # importuj třídu z tvého trénovacího skriptu
from metasim.utils import configclass
from typing import Literal


@configclass
class Args:
    task: str = "reach"
    robot: str = "g1_with_hands"
    sim: Literal["genesis", "isaaclab", "mujoco", "pybullet","sapien3"] = "genesis"
    num_envs: int = 1
    headless: bool = False
    model_path: str = "output/model_60002304.zip"
    video_path: str = "my_env/output/eval_video.mp4"
    steps: int = 1000


def evaluate_and_record(args: Args):
    # --- Inicializace prostředí ---
    scenario = ScenarioCfg(task=args.task, robots=[args.robot], sim=args.sim, num_envs=args.num_envs, headless=args.headless)
    scenario.episode_length = 500
    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)
    env = StableBaseline3VecEnv(metasim_env)

    # --- Načtení modelu ---
    log.info(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    # --- Nastavení videa ---
    os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
    writer = imageio.get_writer(args.video_path, fps=30)

    # --- Reset a rollout ---
    obs = env.reset()
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # Renderuj snímek z prostředí
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        if dones.any():
            log.info(f"Episode done at step {step}, resetting env...")
            obs = env.reset()

    writer.close()
    log.info(f"🎬 Video saved to {args.video_path}")
    env.close()


if __name__ == "__main__":
    args = Args()
    evaluate_and_record(args)
