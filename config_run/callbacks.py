
from stable_baselines3.common.callbacks import BaseCallback
import os
from loguru import logger as log
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


class TensorboardMetricsCallback(BaseCallback):
    """
    Callback pro logování užitečných metrik do TensorBoardu:
    - průměrný reward
    - success rate
    - průměrná délka epizody

    Funguje i s vektorizovanými prostředími (VecEnv).
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = np.zeros(0, dtype=np.float32)
        self.episode_lengths = np.zeros(0, dtype=np.int32)
        self.episode_success = np.zeros(0, dtype=np.int32)

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        self.episode_success = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape (num_envs,)
        dones = self.locals["dones"]      # shape (num_envs,)
        infos = self.locals.get("infos", [{}] * len(dones))

        self.episode_rewards += rewards
        self.episode_lengths += 1

        # Pokud prostředí vrací success flag v `infos`, můžeme ho použít
        for i, info in enumerate(infos):
            if "is_success" in info:
                self.episode_success[i] = int(info["is_success"])

        for i, done in enumerate(dones):
            if done:
                self.writer.add_scalar("episode/return", self.episode_rewards[i], self.num_timesteps)
                self.writer.add_scalar("episode/length", self.episode_lengths[i], self.num_timesteps)
                self.writer.add_scalar("episode/success", self.episode_success[i], self.num_timesteps)

                # Reset epizody
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.episode_success[i] = 0
        return True

    def _on_training_end(self) -> None:
        self.writer.close()

class SaveModelCallback(BaseCallback):
        """
        Callback for saving the model every 1M timesteps.

        Args:
            save_path (str): Path to the directory where models will be saved
            save_freq (int): Frequency in timesteps at which to save the model
            verbose (int): Verbosity level
        """

        def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0,task_name: str = None):
            super().__init__(verbose)
            self.save_path = save_path
            self.save_freq = save_freq
            self.last_save_step = 0
            self.task_name = task_name
            self.run_dir = None

        def _init_callback(self) -> None:
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)

            # Create unique subdirectory for this run
            timestamp = datetime.now().strftime("%Y-%m-%d_%H")
            self.run_dir = os.path.join(self.save_path, f"run_{timestamp}_{self.task_name}")
            os.makedirs(self.run_dir, exist_ok=True)

            log.info(f"Created new model save directory: {self.run_dir}")
        def _on_step(self) -> bool:
            # Save model every N steps
            if self.num_timesteps - self.last_save_step >= self.save_freq:
                model_path = os.path.join(self.run_dir, f"model_{self.num_timesteps}")
                self.model.save(model_path)
                log.info(f"✅ Model saved to {model_path}")
                self.last_save_step = self.num_timesteps
            return True

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
