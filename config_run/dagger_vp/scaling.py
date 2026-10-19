"""Scale all-env DAgger collection while preserving replay and sample budgets."""
from copy import deepcopy
import math
from statistics import median


def steady_sample_rate(events, log_every):
    """Discard the initial interval and partial tail of a throughput trace."""
    values = [event.value for previous, event in zip(events, events[1:])
              if event.step - previous.step == log_every]
    return float(median(values)) if values else None


def scale_dagger_config(config, num_envs):
    """Preserve total transitions and sampled training examples per new example.

    Batch size changes, so optimizer dynamics and the final policy still need
    evaluation. This is throughput tuning, not learning-quality optimization.
    """
    result = deepcopy(config)
    base_envs = int(config['num_envs'])
    if base_envs <= 0 or num_envs <= 0 or config['dagger_store_per_step'] != base_envs:
        raise ValueError('Scaling requires positive env counts and storing every base environment')
    ratio = num_envs / base_envs
    capacity = int(config['dagger_buffer_steps']) * base_envs
    if num_envs > capacity:
        raise ValueError('Environment batch exceeds replay capacity')
    result.update(
        num_envs=num_envs,
        dagger_store_per_step=num_envs,
        dagger_batch_size=max(1, round(config['dagger_batch_size'] * ratio)),
        dagger_buffer_steps=max(1, capacity // num_envs),
        total_timesteps=math.ceil(config['total_timesteps'] / ratio),
    )
    if config.get('beta_schedule', 'linear') == 'exponential':
        result['beta_decay'] = float(config.get('beta_decay', .9995)) ** ratio
    if config.get('model_save_freq', 0) > 0:
        result['model_save_freq'] = max(1, round(config['model_save_freq'] / ratio))
    return result
