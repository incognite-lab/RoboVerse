from __future__ import annotations
import torch
import math
import os
import sys
import time
from typing import Callable
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg

import numpy as np
import rootutils
import wandb
import yaml
from loguru import logger as log
from rich.logging import RichHandler
rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def load_config_from_yaml(config_name: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_name (str): Name of the YAML config file

    Returns:
        dict: The loaded config dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["batch_size"] = config["num_envs"] * config["n_steps"] // config["num_batch"]
    return config


def get_lr_schedule(config: dict) -> float | Callable:
    """
    Create a learning rate schedule based on configuration.

    Args:
        config (dict): Configuration dictionary containing learning rate settings

    Returns:
        Union[float, Callable]: Constant learning rate or schedule function
    """
    # Get base learning rate
    base_lr = config.get("learning_rate", 0.0003)

    # Check if learning rate schedule is enabled
    if not config.get("use_lr_schedule", False):
        return base_lr

    # Get schedule type
    schedule_type = config.get("lr_schedule_type", "linear")

    # Get final learning rate as a fraction of initial
    final_lr_fraction = config.get("final_lr_fraction", 0.1)
    final_lr = base_lr * final_lr_fraction

    # For linear schedule
    if schedule_type == "linear":
        from stable_baselines3.common.utils import get_linear_fn

        return get_linear_fn(base_lr, final_lr, 1.0)

    # For cosine schedule
    elif schedule_type == "cosine":

        def func(progress_remaining: float) -> float:
            progress = 1.0 - progress_remaining
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            return final_lr + cosine_factor * (base_lr - final_lr)

        return func

    # For constant schedule (explicitly handled)
    elif schedule_type == "constant":
        log.info("Using constant learning rate schedule")
        return base_lr

    # Default to constant if unknown schedule type
    log.warning(f"Unknown learning rate schedule type: {schedule_type}. Using constant learning rate.")
    return base_lr


def main():
    # if len(sys.argv) < 2:
    #     log.error("Please provide the config file path, e.g. python train_sb3.py configs/isaacgym.yaml")
    #     exit(1)
    # config_name = sys.argv[1]
    config_name = "genesis_reach"
    config = load_config_from_yaml(config_name)
    log.info(f"Load config: {config_name}")

    if config.get("sim") == "isaacgym":
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    if config.get("use_wandb") and config.get("train_or_eval") == "train":
        run = wandb.init(
            project=config.get("wandb_project", "humanoidbench_rl_training"),
            entity=config.get("wandb_entity"),
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
            name=f"SB3-{time.strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    else:
        from collections import namedtuple

        Run = namedtuple("Run", ["id"])
        run = Run(id=int(time.time()))

    # Create scenario config
    from metasim.cfg.scenario import ScenarioCfg

    scenario = ScenarioCfg(
        task=config.get("task"),
        robots=config.get("robots"),
        try_add_table=config.get("add_table", True),
        sim=config.get("sim"),
        num_envs=config.get("num_envs", 1),
        headless=config.get("headless", True),
        cameras=[],

    )
    scenario.sensors = [GyroSensorCfg(
        name="gyro0",
        pos=(0.0, 0.0, 0.0),
        mount_to='g1_no_hands',
        mount_link="torso_link"

        )]


    # For different simulators, the decimation factor is different, so we need to set it here
    scenario.task.decimation = config.get("decimation", 1)
    print("debug_point2")
    from roboverse_learn.humanoidbench_rl.wrapper_sb3 import Sb3EnvWrapper

    if config.get("sim") == "mujoco":
        if config.get("num_envs") > 1:
            log.error("Mujoco does not support multiple environments > 1")
            exit()
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "isaacgym":
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "isaaclab":
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "mjx":
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "genesis":
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "sapien3":
        env = Sb3EnvWrapper(scenario=scenario)
    else:
        raise ValueError(f"Invalid sim type: {config.get('sim')}")

    # Create learning rate schedule
    learning_rate = get_lr_schedule(config)
    if callable(learning_rate):
        log.info(f"Using {config.get('lr_schedule_type', 'linear')} learning rate schedule")
        log.info(f"Initial learning rate: {config.get('learning_rate', 0.0003)}")
        log.info(f"Final learning rate: {config.get('learning_rate', 0.0003) * config.get('final_lr_fraction', 0.1)}")
    else:
        log.info(f"Using constant learning rate: {learning_rate}")
    policy_kwargs = dict(
    net_arch=[512, 256, 128],
    activation_fn=torch.nn.ReLU,
    ortho_init=False,
    )
    print("debug_point3")
    # Initialize PPO algorithm
    from stable_baselines3 import PPO
    if config.get("load_model_path") == 'None':
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            verbose=1,
            ent_coef=config.get("ent_coef", 0.005),
            tensorboard_log=f"./ppo_logs/{run.id}",
            device="cuda",
            policy_kwargs=policy_kwargs,

        )
    else:
        model = PPO.load(
            config.get("loading_model_path"),
            env=env,
            learning_rate=learning_rate,
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            verbose=1,
            tensorboard_log=f"./ppo_logs/{run.id}",
            device="cpu",
        )

    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLogCallback(BaseCallback):
        """
        Callback for logging episode returns, lengths, and success to TensorBoard and W&B,
        compatible with multi-environment (VecEnv).
        """

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.returns_info = {
                "results/return": [],
                "results/episode_length": [],
                "results/success": [],
            }

        def _on_step(self) -> bool:
            # infos může být list of dicts (pro jedno env) nebo list of list of dicts (multi-env)
            infos = self.locals.get("infos", [])

            # Normalizujeme na list of dicts
            if len(infos) > 0 and isinstance(infos[0], list):
                # multi-env VecEnv
                flat_infos = [item for sublist in infos for item in sublist]
            else:
                flat_infos = infos

            for curr_info in flat_infos:
                if "episode" in curr_info:
                    ep_info = curr_info["episode"]
                    self.returns_info["results/return"].append(ep_info.get("r", 0.0))
                    self.returns_info["results/episode_length"].append(ep_info.get("l", 0))
                    self.returns_info["results/success"].append(curr_info.get("success", 0))

            return True

        def _on_rollout_end(self) -> None:
            global_step = self.model.num_timesteps
            log_dict = {}

            for key, values in self.returns_info.items():
                if len(values) > 0:
                    mean_value = np.mean(values)
                    # Zapiš do SB3 logger (TensorBoard)
                    self.logger.record(key, mean_value, global_step)
                    # Připrav pro W&B
                    log_dict[key] = mean_value
                    # Vyčisti seznam
                    self.returns_info[key] = []

            if wandb.run is not None and log_dict:
                log_dict["global_step"] = global_step
                wandb.log(log_dict, commit=True)  # commit=True zaručí zobrazení v W&B

    class SaveModelCallback(BaseCallback):
        """
        Callback for saving the model every 1M timesteps.

        Args:
            save_path (str): Path to the directory where models will be saved
            save_freq (int): Frequency in timesteps at which to save the model
            verbose (int): Verbosity level
        """

        def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.save_path = save_path
            self.save_freq = save_freq
            self.last_save_step = 0

        def _init_callback(self) -> None:
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            # Check if it's time to save the model
            if self.num_timesteps - self.last_save_step >= self.save_freq:
                path = os.path.join(self.save_path, f"model_{self.num_timesteps}")
                self.model.save(path)
                log.info(f"Model saved to {path}")
                self.last_save_step = self.num_timesteps
            return True

    if config.get("train_or_eval") == "train":
        log.info("Starting training...")

        # Set up save directory
        save_dir = f"{config.get('model_save_path')}/{run.id}"
        os.makedirs(save_dir, exist_ok=True)

        model.learn(
            total_timesteps=config.get("total_timesteps", 1_000_000),
            log_interval=1,
            callback=[
                EpisodeLogCallback(),
                SaveModelCallback(save_path=save_dir, save_freq=config.get("model_save_freq", 1_000_000)),
            ],
            progress_bar=True,
        )
    elif config.get("train_or_eval") == "eval":
        # Load the trained model
        model.load(config.get("eval_model_path"))

        # Evaluate the agent
        log.info("Starting evaluation...")
        obs = env.reset()
        total_rewards = []
        episode_rewards = 0

        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # pokud víc envs -> sum všech rewardů
            if isinstance(reward, (list, np.ndarray)):
                episode_rewards += np.sum(reward)
            else:
                episode_rewards += reward

            # reset pouze tam, kde je done
            if isinstance(done, (list, np.ndarray)):
                if np.any(done):
                    total_rewards.append(episode_rewards)
                    obs = env.reset()
                    episode_rewards = 0
            else:
                if done:
                    total_rewards.append(episode_rewards)
                    obs = env.reset()
                    episode_rewards = 0

        print("Mean reward:", np.mean(total_rewards))

    # Close environment and wandb
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
