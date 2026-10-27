"""Scale all-env DAgger collection while preserving replay and sample budgets."""
from copy import deepcopy
import math
from statistics import median
from fractions import Fraction


def recurrent_config(config):
    return '_new' in config.get('train_or_eval', '')


def replay_ratio(config):
    batch = int(config.get('dagger_new_batch_size', 8)) if recurrent_config(config) else int(config['dagger_batch_size'])
    length = int(config.get('dagger_new_sequence_length', 16)) if recurrent_config(config) else 1
    return Fraction(int(config['dagger_updates_per_train']) * batch * length,
                    int(config['dagger_train_every']) * int(config['dagger_store_per_step']))


def replay_sample_bytes(config):
    """Exact tensor storage per transition for current student replay layouts."""
    width, height = int(config.get('dagger_image_width', 128)), int(config.get('dagger_image_height', 128))
    joints = int(config.get('dagger_num_joints', 43))
    if recurrent_config(config):
        hidden = int(config.get('dagger_new_hidden_size', 256)) * int(config.get('dagger_new_num_layers', 1))
        return 3 * width * height + 4 * (2 * joints + 6 + 34 + 34 + hidden) + 1
    return 3 * width * height + 4 * (joints + 34)


def recurrent_candidate(config, num_envs, *, replay_gib=8., batch_size=None,
                        visual_batch_size=None, checkpoint_visual=True, threads=4,
                        replay_device='cuda'):
    """All-env collection with fixed data budget and EXACT trained-frame ratio.

    Adjust updates and the collection interval together when an optimizer batch
    does not divide the requested amount of training. No candidate wins by
    silently doing less learning. Replay allocation is identical in profiling
    and production (within whole environment rows of the byte budget).
    """
    if not recurrent_config(config) or num_envs <= 0 or replay_gib <= 0 or threads <= 0:
        raise ValueError('Positive recurrent candidate dimensions/budget required')
    result = deepcopy(config)
    ratio = replay_ratio(config)
    length = int(config.get('dagger_new_sequence_length', 16))
    burn = int(config.get('dagger_new_burn_in', 8))
    batch = (max(1, round(int(config.get('dagger_new_batch_size', 8))
                        * num_envs / int(config['dagger_store_per_step'])))
             if batch_size is None else batch_size)
    if batch <= 0:
        raise ValueError('Sequence batch must be positive')
    # Smallest update/interval multiplier preserving the original train interval.
    updates = ratio * num_envs * int(config['dagger_train_every']) / (batch * length)
    every = int(config['dagger_train_every']) * updates.denominator
    capacity = int(replay_gib * 2**30) // (num_envs * replay_sample_bytes(config))
    if capacity < length + burn:
        raise ValueError('Replay byte budget cannot fit burn-in + sequence for these environments')
    base_envs = int(config['num_envs'])
    result.update(num_envs=num_envs, train_or_eval='train_dagger_new',
                  dagger_store_per_step=num_envs, dagger_new_batch_size=batch,
                  dagger_updates_per_train=updates.numerator, dagger_train_every=every,
                  dagger_buffer_steps=capacity, dagger_new_buffer_device=replay_device,
                  dagger_buffer_device=replay_device, dagger_buffer_pin_memory=False,
                  dagger_new_checkpoint_visual=checkpoint_visual,
                  dagger_torch_threads=threads, dagger_defer_camera_rendering=True,
                  total_timesteps=math.ceil(int(config['total_timesteps']) * base_envs / num_envs))
    if visual_batch_size is not None:
        result['dagger_new_visual_batch_size'] = visual_batch_size
    if config.get('model_save_freq', 0) > 0:
        result['model_save_freq'] = max(1, round(config['model_save_freq'] * base_envs / num_envs))
    if config.get('beta_schedule', 'linear') == 'exponential':
        result['beta_decay'] = float(config.get('beta_decay', .9995)) ** (num_envs / base_envs)
    return result


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
    if recurrent_config(config):
        budget = int(config['dagger_buffer_steps']) * int(config['dagger_store_per_step']) * replay_sample_bytes(config) / 2**30
        return recurrent_candidate(config, num_envs, replay_gib=budget,
            replay_device=config.get('dagger_new_buffer_device', 'cpu'),
            threads=int(config.get('dagger_torch_threads', 4)),
            visual_batch_size=int(config.get('dagger_new_visual_batch_size', 4)))
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
