
from stable_baselines3.common.callbacks import BaseCallback
import os
from loguru import logger as log
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


class TensorboardMetricsCallbackOld(BaseCallback):
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

from stable_baselines3.common.callbacks import BaseCallback
import os
from loguru import logger as log
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


class TensorboardMetricsCallback(BaseCallback):
    """
    Callback pro logování metrik epizod + statistik stages do TensorBoardu a terminálu.

    Stage completion se NEODVOZUJE ze změny actual_stage,
    ale z explicitního signálu completed_stages z checkeru.
    """

    def __init__(self, log_dir: str, log_interval: int = 10000, max_stage: int = 6, verbose: int = 1):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_interval = log_interval
        self.max_stage = max_stage
        self.verbose = verbose

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs

        self.episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        self.episode_success = np.zeros(n_envs, dtype=np.int32)

        self.completed_rewards = []
        self.completed_lengths = []
        self.completed_success = []

        self.prev_stages = None
        self.prev_completed_flags = None

        self.stage_presence_counts = np.zeros(self.max_stage + 1, dtype=np.int64)
        self.stage_completed_window_counts = np.zeros(self.max_stage + 1, dtype=np.int64)
        self.stage_completed_total_counts = np.zeros(self.max_stage + 1, dtype=np.int64)

        self.max_stage_seen = 0

        log.info(
            f"TensorboardMetricsCallback started | "
            f"n_envs={n_envs}, log_interval={self.log_interval}, max_stage={self.max_stage}"
        )

    def _get_stage_data(self):
        """
        Vrátí:
        - current_stages: numpy array aktuálních stages
        - completed_flags: numpy array completed_stages
        """
        try:
            handler = self.training_env.env.env.handler
            current_stages = handler.task.reward_functions[0].actual_stage
            completed_flags = handler.task.reward_functions[0].completed_stages

            if current_stages is None or completed_flags is None:
                return None, None

            return (
                current_stages.detach().cpu().numpy().astype(int),
                completed_flags.detach().cpu().numpy().astype(int),
            )
        except Exception:
            return None, None

    def _on_step(self) -> bool:
        rewards = np.array(self.locals["rewards"])
        dones = np.array(self.locals["dones"])
        infos = self.locals.get("infos", [{}] * len(dones))

        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i, info in enumerate(infos):
            if "is_success" in info:
                self.episode_success[i] = int(info["is_success"])

            if dones[i]:
                self.completed_rewards.append(self.episode_rewards[i])
                self.completed_lengths.append(self.episode_lengths[i])
                self.completed_success.append(self.episode_success[i])

                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.episode_success[i] = 0

        # =========================
        # STAGE TRACKING
        # =========================
        current_stages, completed_flags = self._get_stage_data()
        if current_stages is not None and completed_flags is not None:
            current_stages = np.clip(current_stages, 0, self.max_stage)
            completed_flags = np.clip(completed_flags, 0, 1)

            if self.prev_stages is None:
                self.prev_stages = current_stages.copy()

            if self.prev_completed_flags is None:
                self.prev_completed_flags = completed_flags.copy()

            # aktuální obsazenost stages
            for s in range(self.max_stage + 1):
                self.stage_presence_counts[s] += int(np.sum(current_stages == s))

            # NOVÉ completion eventy: completed_stages přešlo 0 -> 1
            newly_completed_mask = (self.prev_completed_flags == 0) & (completed_flags == 1)
            newly_completed_indices = np.where(newly_completed_mask)[0]

            for idx in newly_completed_indices:
                completed_stage = int(current_stages[idx]) - 1
                if 0 <= completed_stage <= self.max_stage:
                    self.stage_completed_window_counts[completed_stage] += 1
                    self.stage_completed_total_counts[completed_stage] += 1

            self.max_stage_seen = max(self.max_stage_seen, int(np.max(current_stages)))

            self.prev_stages = current_stages.copy()
            self.prev_completed_flags = completed_flags.copy()

        # =========================
        # LOGGING
        # =========================
        if self.num_timesteps % self.log_interval == 0 and self.num_timesteps != 0:
            if len(self.completed_rewards) > 0:
                mean_r = float(np.mean(self.completed_rewards))
                mean_l = float(np.mean(self.completed_lengths))
                mean_s = float(np.mean(self.completed_success))
                max_r = float(np.max(self.completed_rewards))
                min_r = float(np.min(self.completed_rewards))
                success_count = int(np.sum(self.completed_success))
                fail_count = int(len(self.completed_success) - success_count)
                success_rate = 100.0 * mean_s
            else:
                mean_r = 0.0
                mean_l = 0.0
                mean_s = 0.0
                max_r = 0.0
                min_r = 0.0
                success_count = 0
                fail_count = 0
                success_rate = 0.0

            # episode logs
            self.writer.add_scalar("episode/mean_return", mean_r, self.num_timesteps)
            self.writer.add_scalar("episode/mean_length", mean_l, self.num_timesteps)
            self.writer.add_scalar("episode/success_rate_%", success_rate, self.num_timesteps)
            self.writer.add_scalar("episode/max_return", max_r, self.num_timesteps)
            self.writer.add_scalar("episode/min_return", min_r, self.num_timesteps)
            self.writer.add_scalar("episode/success_count", success_count, self.num_timesteps)
            self.writer.add_scalar("episode/fail_count", fail_count, self.num_timesteps)

            if len(self.completed_rewards) > 0:
                self.writer.add_histogram("episode/reward_hist", np.array(self.completed_rewards), self.num_timesteps)

            # stage logs
            if current_stages is not None:
                n_envs = len(current_stages)

                for s in range(self.max_stage + 1):
                    current_count = int(np.sum(current_stages == s))
                    current_ratio = current_count / max(n_envs, 1)

                    self.writer.add_scalar(f"stage/current_stage_{s}_envs", current_count, self.num_timesteps)
                    self.writer.add_scalar(f"stage/current_stage_{s}_ratio", current_ratio, self.num_timesteps)
                    self.writer.add_scalar(f"stage_completed/window_stage_{s}_count", int(self.stage_completed_window_counts[s]), self.num_timesteps)
                    self.writer.add_scalar(f"stage_completed/total_stage_{s}_count", int(self.stage_completed_total_counts[s]), self.num_timesteps)

                self.writer.add_scalar("stage/max_stage_seen_in_window", self.max_stage_seen, self.num_timesteps)

            # sb3 logger
            self.logger.record("episode/mean_return", mean_r)
            self.logger.record("episode/mean_length", mean_l)
            self.logger.record("episode/success_rate_%", success_rate)
            self.logger.record("episode/max_return", max_r)
            self.logger.record("episode/min_return", min_r)
            self.logger.record("episode/success_count", success_count)
            self.logger.record("episode/fail_count", fail_count)

            if current_stages is not None:
                n_envs = len(current_stages)
                for s in range(self.max_stage + 1):
                    current_count = int(np.sum(current_stages == s))
                    current_ratio = current_count / max(n_envs, 1)

                    self.logger.record(f"stage/current_stage_{s}_envs", current_count)
                    self.logger.record(f"stage/current_stage_{s}_ratio", current_ratio)
                    self.logger.record(f"stage_completed/window_stage_{s}_count", int(self.stage_completed_window_counts[s]))
                    self.logger.record(f"stage_completed/total_stage_{s}_count", int(self.stage_completed_total_counts[s]))

                self.logger.record("stage/max_stage_seen_in_window", int(self.max_stage_seen))

            self.logger.dump(self.num_timesteps)

            # terminal print
            if self.verbose > 0:
                msg = (
                    f"[TB] step={self.num_timesteps} | "
                    f"mean_return={mean_r:.3f} | mean_length={mean_l:.1f} | "
                    f"success_rate={success_rate:.2f}% | "
                    f"success={success_count} fail={fail_count}"
                )

                if current_stages is not None:
                    current_parts = []
                    completed_window_parts = []
                    completed_total_parts = []

                    for s in range(self.max_stage + 1):
                        current_count = int(np.sum(current_stages == s))
                        window_count = int(self.stage_completed_window_counts[s])
                        total_count = int(self.stage_completed_total_counts[s])

                        if current_count > 0:
                            current_parts.append(f"S{s}:{current_count}")
                        if window_count > 0:
                            completed_window_parts.append(f"S{s}:{window_count}")
                        if total_count > 0:
                            completed_total_parts.append(f"S{s}:{total_count}")

                    msg += " | current_stages=[" + ", ".join(current_parts) + "]"
                    msg += " | completed_window=[" + ", ".join(completed_window_parts) + "]"
                    msg += " | completed_total=[" + ", ".join(completed_total_parts) + "]"
                    msg += f" | max_stage_seen={self.max_stage_seen}"

                log.info(msg)

            # reset window buffers
            self.completed_rewards.clear()
            self.completed_lengths.clear()
            self.completed_success.clear()

            self.stage_presence_counts[:] = 0
            self.stage_completed_window_counts[:] = 0
            self.max_stage_seen = 0

            self.writer.flush()

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
        log.info("TensorboardMetricsCallback finished.")


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
class EvalCallback(BaseCallback):
    """
    Callback pro periodickou evaluaci modelu během tréninku.

    - Každých eval_freq kroků spustí n_eval_episodes epizod v eval_env.
    - Loguje metriky do TensorBoardu i SB3 loggeru.
    - Uloží nejlepší model (pokud save_best=True).
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 100000,
        n_eval_episodes: int = 5,
        log_dir: str = "./eval_logs",
        deterministic: bool = True,
        save_best: bool = True,
        best_model_dir: str = "./best_models",
        eval_max_steps: int = 1000,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.save_best = save_best
        self.best_model_dir = best_model_dir
        self.writer = SummaryWriter(log_dir)
        self.best_mean_reward = -np.inf
        self.eval_max_steps = eval_max_steps
        os.makedirs(self.best_model_dir, exist_ok=True)

    def _init_callback(self) -> None:
        log.info("EvalCallback initialized.")

    def _evaluate_policy(self):
        """Spustí evaluaci n_eval_episodes epizod a vrátí průměrné metriky."""
        episode_rewards = []
        episode_lengths = []
        success_flags = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            total_reward = 0.0
            ep_len = 0
            success = 0
            step = 0
            while not done.any() and step < self.eval_max_steps:

                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
                total_reward += np.mean(rewards)
                ep_len += 1
                done = dones
                step += 1

                if any("is_success" in info for info in infos):
                    success = int(any(info.get("is_success", False) for info in infos))

            episode_rewards.append(total_reward)
            episode_lengths.append(ep_len)
            success_flags.append(success)

        return (
            np.mean(episode_rewards),
            np.std(episode_rewards),
            np.mean(episode_lengths),
            np.mean(success_flags),
        )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            log.info(f"Running evaluation at {self.num_timesteps} timesteps...")
            mean_reward, std_reward, mean_length, success_rate = self._evaluate_policy()

            # Log do TensorBoardu
            self.writer.add_scalar("eval/mean_reward", mean_reward, self.num_timesteps)
            self.writer.add_scalar("eval/std_reward", std_reward, self.num_timesteps)
            self.writer.add_scalar("eval/mean_length", mean_length, self.num_timesteps)
            self.writer.add_scalar("eval/success_rate", success_rate, self.num_timesteps)
            self.writer.flush()

            # Log i do SB3
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_length", mean_length)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.dump(self.num_timesteps)

            # Ulož nejlepší model
            if self.save_best and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(
                    self.best_model_dir, f"best_model_{self.num_timesteps}.zip"
                )
                self.model.save(model_path)
                log.info(f"🌟 New best model saved: {model_path} (mean reward={mean_reward:.2f})")

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
        log.info("EvalCallback finished and TensorBoard writer closed.")
