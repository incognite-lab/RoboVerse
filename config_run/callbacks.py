
from stable_baselines3.common.callbacks import BaseCallback
import os
from loguru import logger as log
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


class TensorboardMetricsCallback(BaseCallback):
    """
    Callback pro logování průměrných metrik epizod do TensorBoardu a terminálu.
    Funguje s vektorizovanými env (VecEnv).

    Metriky:
    - průměrná rewarda přes všechny envy
    - průměrná délka epizody
    - průměrný success rate
    - max a min reward
    - počet dokončených epizod
    """
    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.verbose = verbose

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        self.episode_success = np.zeros(n_envs, dtype=np.int32)

        # historie dokončených epizod pro agregace
        self.completed_rewards = []
        self.completed_lengths = []
        self.completed_success = []

    def _on_step(self) -> bool:
        rewards = np.array(self.locals["rewards"])
        dones = np.array(self.locals["dones"])
        infos = self.locals.get("infos", [{}] * len(dones))

        # akumulace rewardů a délek pro každé env
        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i, info in enumerate(infos):
            if "is_success" in info:
                self.episode_success[i] = int(info["is_success"])

            if dones[i]:
                # uložíme dokončenou epizodu
                self.completed_rewards.append(self.episode_rewards[i])
                self.completed_lengths.append(self.episode_lengths[i])
                self.completed_success.append(self.episode_success[i])

                # reset counters pro dané env
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.episode_success[i] = 0

        # pokud máme dokončené epizody, vypočteme průměr přes všechny envy
        if self.completed_rewards:
            mean_r = np.mean(self.completed_rewards)
            mean_l = np.mean(self.completed_lengths)
            mean_s = np.mean(self.completed_success)
            max_r = np.max(self.completed_rewards)
            min_r = np.min(self.completed_rewards)
            num_ep = len(self.completed_rewards)

            # log do TensorBoardu
            self.writer.add_scalar("episode/mean_return", mean_r, self.num_timesteps)
            self.writer.add_scalar("episode/mean_length", mean_l, self.num_timesteps)
            self.writer.add_scalar("episode/mean_success", mean_s, self.num_timesteps)
            self.writer.add_scalar("episode/max_return", max_r, self.num_timesteps)
            self.writer.add_scalar("episode/min_return", min_r, self.num_timesteps)
            self.writer.add_scalar("episode/num_episodes", num_ep, self.num_timesteps)

            # histogram rewardů
            self.writer.add_histogram("episode/reward_hist", np.array(self.completed_rewards), self.num_timesteps)

            # log do terminálu
            if self.verbose > 0:
                self.logger.record("episode/mean_return", mean_r)
                self.logger.record("episode/mean_length", mean_l)
                self.logger.record("episode/mean_success", mean_s)
                self.logger.record("episode/max_return", max_r)
                self.logger.record("episode/min_return", min_r)
                self.logger.record("episode/num_episodes", num_ep)
                self.logger.dump(self.num_timesteps)

            # vyčistíme historii, aby nové dokončené epizody šly do nového kroku
            self.completed_rewards.clear()
            self.completed_lengths.clear()
            self.completed_success.clear()

        self.writer.flush()
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
