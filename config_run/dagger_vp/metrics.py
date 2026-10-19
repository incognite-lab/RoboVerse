"""DAgger progress and wall-time telemetry without per-step GPU synchronization."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import csv
import io
import math
import subprocess
import threading
import time

import numpy as np
import torch


class HardwareMonitor:
    """Poll nvidia-smi off the training thread, selecting the actual CUDA UUID."""

    def __init__(self, device, interval=10.0):
        self.device = torch.device(device)
        self.interval = max(1.0, float(interval))
        self._sample = {}
        self._sample_time = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.process = None
        try:
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent()
        except ImportError:
            pass
        if self.device.type == 'cuda':
            properties = torch.cuda.get_device_properties(self.device)
            uuid = str(getattr(properties, 'uuid', ''))
            if uuid:
                self.uuid = uuid if uuid.startswith(('GPU-', 'MIG-')) else f'GPU-{uuid}'
                self._thread = threading.Thread(target=self._poll, daemon=True, name='dagger-gpu-monitor')
                self._thread.start()

    @staticmethod
    def parse_smi(output):
        """Ignore unsupported driver fields instead of reporting invented zeroes."""
        fields = ('utilization_percent', 'memory_controller_utilization_percent',
                  'device_memory_used_mib', 'device_memory_total_mib', 'power_watts', 'temperature_c')
        rows = list(csv.reader(io.StringIO(output.strip())))
        if len(rows) != 1 or len(rows[0]) != len(fields):
            return {}
        result = {}
        for key, value in zip(fields, rows[0]):
            try:
                number = float(value.strip())
                if math.isfinite(number):
                    result[f'GPU/{key}'] = number
            except ValueError:
                pass
        return result

    def _poll(self):
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ['nvidia-smi', f'--id={self.uuid}',
                     '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2, check=True,
                ).stdout
                sample = self.parse_smi(output)
            except (OSError, subprocess.SubprocessError):
                sample = {}
            with self._lock:
                self._sample = sample
                self._sample_time = time.perf_counter()
            self._stop.wait(self.interval)

    def snapshot(self):
        with self._lock:
            values = dict(self._sample)
            sampled_at = self._sample_time
        values['GPU/monitor_available'] = int(bool(values))
        if sampled_at is not None:
            values['GPU/sample_age_seconds'] = time.perf_counter() - sampled_at
        if self.device.type == 'cuda':
            for name, getter in (
                ('torch_allocated_mib', torch.cuda.memory_allocated),
                ('torch_reserved_mib', torch.cuda.memory_reserved),
                ('torch_peak_allocated_mib', torch.cuda.max_memory_allocated),
            ):
                values[f'GPU/{name}'] = getter(self.device) / 2**20
        if self.process is not None:
            try:
                values['Host/process_rss_mib'] = self.process.memory_info().rss / 2**20
                values['Host/process_cpu_percent_single_core'] = self.process.cpu_percent()
            except (OSError, RuntimeError):
                pass
        return values

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


class DaggerMetrics:
    """Count observed stage attempts; curriculum entries never imply completion."""

    def __init__(self, initial_stages, num_stages, control_dt, clock=time.perf_counter):
        self.clock = clock
        self.started = self.last_log = clock()
        self.stages = np.asarray(initial_stages, dtype=np.int64).copy()
        self.num_envs = len(self.stages)
        self.num_stages = num_stages
        self.control_dt = float(control_dt)
        self.stage_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_steps = self.stage_steps.copy()
        self.entered_at = np.full(self.num_envs, self.started)
        self.episode_at = self.entered_at.copy()
        self.counts = np.zeros((num_stages, 3), dtype=np.int64)  # entered, completed, failed
        self.recent_counts = self.counts.copy()
        self.unique_completed = np.zeros((self.num_envs, num_stages), dtype=bool)
        self.duration_steps = [deque(maxlen=256) for _ in range(num_stages)]
        self.duration_wall = [deque(maxlen=256) for _ in range(num_stages)]
        self.completed_step_sums = np.zeros(num_stages)
        self.completed_wall_sums = np.zeros(num_stages)
        self.failure_reasons = Counter()
        self.episode_lengths = deque(maxlen=256)
        self.episode_walls = deque(maxlen=256)
        self.episodes = self.successful_episodes = 0
        self.recent_episodes = self.recent_successful_episodes = 0
        self.steps = self.last_steps = 0
        self.stored = self.last_stored = 0
        self.updates = self.last_updates = 0
        self.trained_samples = self.last_trained_samples = 0
        self.expert_steps = self.last_expert_steps = 0
        self.losses = []
        self.predictions = []
        self.times = defaultdict(float)
        for stage in self.stages:
            self._enter(stage)

    def _enter(self, stage):
        if 0 <= stage < self.num_stages:
            self.counts[stage, 0] += 1
            self.recent_counts[stage, 0] += 1

    @contextmanager
    def measure(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.times[name] += self.clock() - start

    def observe(self, dones, infos, stored, executed_expert, failure_masks=None):
        now = self.clock()
        dones = np.asarray(dones, dtype=bool)
        self.stage_steps += 1
        self.episode_steps += 1
        self.steps += 1
        self.stored += stored
        self.expert_steps += int(executed_expert)
        for env, info in enumerate(infos):
            before = int(info['stage_before'])
            after = int(info['stage_after'])
            valid = 0 <= before < self.num_stages
            completed = valid and int(info.get('completed_stage', -1)) == before
            if valid and completed:
                self.counts[before, 1] += 1
                self.recent_counts[before, 1] += 1
                self.unique_completed[env, before] = True
                duration = int(self.stage_steps[env])
                wall = now - self.entered_at[env]
                self.duration_steps[before].append(duration)
                self.duration_wall[before].append(wall)
                self.completed_step_sums[before] += duration
                self.completed_wall_sums[before] += wall
            elif valid and dones[env]:
                self.counts[before, 2] += 1
                self.recent_counts[before, 2] += 1
                if info.get('TimeLimit.truncated', False):
                    self.failure_reasons['global_timeout'] += 1
                for reason, mask in (failure_masks or {}).items():
                    if mask[env]:
                        self.failure_reasons[reason] += 1
            if dones[env]:
                self.episodes += 1
                success = int(info.get('is_success', False))
                self.successful_episodes += success
                self.recent_episodes += 1
                self.recent_successful_episodes += success
                self.episode_lengths.append(int(self.episode_steps[env]))
                self.episode_walls.append(now - self.episode_at[env])
                self.episode_steps[env] = 0
                self.episode_at[env] = now
            if completed or dones[env] or after != before:
                self.stage_steps[env] = 0
                self.entered_at[env] = now
                self._enter(after)
            self.stages[env] = after

    def record_updates(self, losses, samples_per_update):
        self.losses.extend(losses)
        self.updates += len(losses)
        self.trained_samples += len(losses) * samples_per_update

    def record_predictions(self, student_actions, expert_actions, low, high):
        error = student_actions - expert_actions
        self.predictions.append((np.mean(error ** 2), np.mean(np.abs(error)),
                                 np.mean((student_actions < low) | (student_actions > high))))

    def write(self, writer, step, buffer, beta, learning_rate, hardware=None, end_step=None):
        now = self.clock()
        elapsed = max(now - self.started, 1e-9)
        interval = max(now - self.last_log, 1e-9)
        steps = self.steps - self.last_steps
        values = {
            'Progress/vector_steps_this_session': self.steps,
            'Progress/transitions_this_session': self.steps * self.num_envs,
            'Progress/stored_samples_this_session': self.stored,
            'Progress/gradient_updates_this_session': self.updates,
            'Progress/trained_samples_this_session': self.trained_samples,
            'Speed/vector_steps_per_second': steps / interval,
            'Speed/transitions_per_second': steps * self.num_envs / interval,
            'Speed/stored_samples_per_second': (self.stored - self.last_stored) / interval,
            'Speed/gradient_updates_per_second': (self.updates - self.last_updates) / interval,
            'Speed/trained_samples_per_second': (self.trained_samples - self.last_trained_samples) / interval,
            'Speed/vector_steps_per_second_session': self.steps / elapsed,
            'Time/elapsed_seconds': elapsed,
            'Time/control_dt_sim_seconds': self.control_dt,
            'Time/simulation_seconds_per_env': self.steps * self.control_dt,
            'Time/real_time_factor_per_env': steps * self.control_dt / interval,
            'Student/beta': beta,
            'Student/learning_rate': learning_rate,
            'Student/expert_action_fraction': (self.expert_steps - self.last_expert_steps) / max(1, steps),
            'Buffer/size': buffer.size,
            'Buffer/capacity': buffer.max_samples,
            'Buffer/fill_fraction': buffer.size / buffer.max_samples,
            'Buffer/allocated_mib': sum(t.numel() * t.element_size() for t in
                                        (buffer.imgs, buffer.joints, buffer.expert_actions)) / 2**20,
            'Episodes/completed_total': self.episodes,
            'Episodes/successful_from_reset_total': self.successful_episodes,
        }
        if end_step is not None and steps:
            values['Time/estimated_remaining_seconds'] = max(0, end_step - step) * interval / steps
        if self.recent_episodes:
            values['Episodes/success_rate_recent'] = self.recent_successful_episodes / self.recent_episodes
        if self.episode_lengths:
            values['Episodes/length_steps_recent_mean'] = np.mean(self.episode_lengths)
            values['Episodes/length_sim_seconds_recent_mean'] = np.mean(self.episode_lengths) * self.control_dt
            values['Episodes/length_wall_seconds_recent_mean'] = np.mean(self.episode_walls)
        if self.losses:
            values['Student/window_mse_loss'] = np.mean(self.losses)
        if self.predictions:
            mse, mae, outside = np.mean(self.predictions, axis=0)
            values['Student/rollout_action_mse'] = mse
            values['Student/rollout_action_mae'] = mae
            values['Student/action_out_of_bounds_fraction'] = outside
        for phase, seconds in self.times.items():
            values[f'Timing/{phase}_ms_per_vector_step'] = seconds * 1000 / max(1, steps)
            values[f'Timing/{phase}_wall_fraction'] = seconds / interval
        for stage in range(self.num_stages):
            prefix = f'Stages/stage_{stage}'
            entered, completed, failed = self.counts[stage]
            _, recent_completed, recent_failed = self.recent_counts[stage]
            active = self.stages == stage
            values.update({
                f'{prefix}/entered_total': entered,
                f'{prefix}/completed_total': completed,
                f'{prefix}/failed_total': failed,
                f'{prefix}/completed_recent': recent_completed,
                f'{prefix}/failed_recent': recent_failed,
                f'{prefix}/active_envs': np.count_nonzero(active),
                f'{prefix}/completed_env_slots_total': np.count_nonzero(self.unique_completed[:, stage]),
            })
            if completed + failed:
                values[f'{prefix}/success_rate_total'] = completed / (completed + failed)
            if recent_completed + recent_failed:
                values[f'{prefix}/success_rate_recent'] = recent_completed / (recent_completed + recent_failed)
            if active.any():
                values[f'{prefix}/active_age_steps_mean'] = np.mean(self.stage_steps[active])
            if completed:
                values[f'{prefix}/completion_steps_mean'] = self.completed_step_sums[stage] / completed
                values[f'{prefix}/completion_sim_seconds_mean'] = self.completed_step_sums[stage] * self.control_dt / completed
                values[f'{prefix}/completion_wall_seconds_mean'] = self.completed_wall_sums[stage] / completed
                values[f'{prefix}/completion_steps_recent_p50'] = np.percentile(self.duration_steps[stage], 50)
                values[f'{prefix}/completion_steps_recent_p95'] = np.percentile(self.duration_steps[stage], 95)
        for reason, count in self.failure_reasons.items():
            values[f'Failures/{reason}_total'] = count
        if hardware is not None:
            values.update(hardware.snapshot())
        for tag, value in values.items():
            writer.add_scalar(tag, float(value), step)
        self.last_log = now
        self.last_steps, self.last_stored = self.steps, self.stored
        self.last_updates, self.last_trained_samples = self.updates, self.trained_samples
        self.last_expert_steps = self.expert_steps
        self.recent_counts.fill(0)
        self.recent_episodes = self.recent_successful_episodes = 0
        self.losses.clear()
        self.predictions.clear()
        self.times.clear()
