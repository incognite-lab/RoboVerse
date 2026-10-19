"""DAgger mixing ratio as a deterministic function of completed vector steps."""
from __future__ import annotations

import math


def dagger_target_steps(config, start_step=None):
    """total_timesteps is the absolute final vector step, also after resume."""
    target = int(config.get('total_timesteps', 100_000))
    if target <= 0:
        raise ValueError('DAgger total_timesteps must be positive')
    if start_step is not None and start_step >= target:
        raise ValueError(f'DAgger total_timesteps ({target}) must exceed resumed step ({start_step})')
    return target


def beta_at_step(config, completed_steps):
    """Default: beta_start at zero, beta_end exactly at total_timesteps."""
    target = dagger_target_steps(config)
    if completed_steps < 0:
        raise ValueError('DAgger completed_steps must be nonnegative')
    start = float(config.get('beta_start', 1.0))
    end = float(config.get('beta_end', 0.0))
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (start, end)) or end > start:
        raise ValueError('DAgger requires 0 <= beta_end <= beta_start <= 1')
    schedule = config.get('beta_schedule', 'linear')
    if schedule == 'linear':
        fraction = min(float(completed_steps) / target, 1.0)
        return start + (end - start) * fraction
    if schedule == 'exponential':
        decay = float(config.get('beta_decay', .9995))
        if not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError('DAgger beta_decay must be in [0, 1]')
        return max(end, start * decay ** completed_steps)
    raise ValueError(f'Unknown DAgger beta_schedule: {schedule!r}')
