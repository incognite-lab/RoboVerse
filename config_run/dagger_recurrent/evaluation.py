"""Fixed-cohort evaluations; no teacher actions enter student success metrics."""
from contextlib import contextmanager

import numpy as np
import torch

from .inputs import RolloutState


@contextmanager
def evaluation_resets(env, start_stage=0, stop_stage=None):
    handler = env.env.env.handler if hasattr(env, 'env') else None
    task = getattr(handler, 'task', None)
    changes = dict(eval_start_stage=start_stage, reset_to_stage0=start_stage == 0,
                   use_snapshot_curriculum=start_stage != 0, snapshot_save_probability=0.)
    if stop_stage is not None:
        changes.update(dagger_target_stage=stop_stage, curriculum_max_stage=stop_stage)
    saved = {}
    if task is not None:
        saved = {key: (hasattr(task, key), getattr(task, key, None)) for key in changes}
        for key, value in changes.items():
            setattr(task, key, value)
    try:
        yield
    finally:
        for key, (existed, value) in saved.items():
            if existed:
                setattr(task, key, value)
            else:
                delattr(task, key)


def evaluate_policy(student, adapter, env, episodes, max_steps, *, teacher=None, frame_callback=None,
                    count_timeouts=True):
    """Evaluate quotas assigned BEFORE rollout, avoiding fastest-failure bias.

    Caller chooses reset configuration. Environment is reset here and left at
    the evaluation's final state; a training caller MUST reset its rollout too.
    Unfinished assigned attempts at the cap are explicit failures by default.
    """
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("Evaluation episodes and max_steps must be positive")
    observation = env.reset()
    state = RolloutState(student, env.num_envs)
    student.eval()
    quotas = np.full(env.num_envs, episodes // env.num_envs, dtype=np.int64)
    quotas[:episodes % env.num_envs] += 1
    done_counts = np.zeros(env.num_envs, dtype=np.int64)
    returns = np.zeros(env.num_envs)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    successes, finished_returns, finished_lengths = [], [], []
    stage_counts = np.zeros(6, dtype=np.int64)
    stage_completions = np.zeros(6, dtype=np.int64)
    stage_failures = np.zeros(6, dtype=np.int64)
    stage_censored = np.zeros(6, dtype=np.int64)
    for _ in range(max_steps):
        if teacher is None:
            values = adapter.read(state.previous_actions)
            if frame_callback is not None:
                frame_callback(values[0])
            with torch.no_grad():
                prediction, hidden = student(values[0].float() / 255, *values[1:],
                                             hidden=state.hidden, reset_mask=state.reset_mask)
            actions = prediction.cpu().numpy()
        else:
            actions = teacher.predict(observation, deterministic=True)
            hidden = state.hidden
        actions = np.clip(actions, env.action_space.low, env.action_space.high)
        if not np.isfinite(actions).all():
            raise FloatingPointError('Non-finite evaluation action; simulator was not stepped')
        observation, rewards, dones, infos = env.step(actions)
        state.after_step(hidden, actions, dones)
        active = done_counts < quotas
        returns[active] += np.asarray(rewards)[active]
        lengths[active] += 1
        for index in np.flatnonzero(active):
            stage = int(infos[index].get('completed_stage', -1))
            if 0 <= stage < 6:
                stage_completions[stage] += 1
            elif dones[index]:
                before = int(infos[index].get('stage_before', -1))
                if 0 <= before < 6:
                    if infos[index].get('is_success', False):
                        stage_completions[before] += 1
                    else:
                        stage_failures[before] += 1
        for index in np.flatnonzero(np.asarray(dones) & active):
            successes.append(int(infos[index].get('is_success', False)))
            finished_returns.append(float(returns[index]))
            finished_lengths.append(int(lengths[index]))
            stage = int(np.clip(infos[index].get('stage_before', 0), 0, 5))
            stage_counts[stage] += 1
            done_counts[index] += 1
            returns[index] = lengths[index] = 0
        if np.all(done_counts >= quotas):
            break
    unfinished = int((quotas - done_counts).sum())
    stages = np.clip(env.get_current_stages(), 0, 5)
    for index in np.flatnonzero((done_counts < quotas) & (lengths > 0)):
        stage_censored[int(stages[index])] += 1
    if count_timeouts:
        stages = np.clip(env.get_current_stages(), 0, 5)
        for index in np.flatnonzero(done_counts < quotas):
            # First outstanding attempt has accumulated a return; later
            # assigned attempts never started and are counted as cap failures.
            remaining = int(quotas[index] - done_counts[index])
            successes.extend([0] * remaining)
            finished_returns.extend([float(returns[index])] + [0.] * (remaining - 1))
            finished_lengths.extend([int(lengths[index])] + [0] * (remaining - 1))
            stage_counts[int(stages[index])] += remaining
    row = dict(episodes_finished=len(successes), requested_episodes=episodes, capped_attempts=unfinished,
               mean_reward=float(np.mean(finished_returns)) if successes else 0.,
               std_reward=float(np.std(finished_returns)) if successes else 0.,
               success_rate=float(np.mean(successes)) if successes else 0., success_count=sum(successes),
               mean_length=float(np.mean(finished_lengths)) if successes else 0.,
               std_length=float(np.std(finished_lengths)) if successes else 0.)
    row.update({f'end_stage_{stage}_count': int(count) for stage, count in enumerate(stage_counts)})
    row.update({f'stage_{stage}_completions': int(count) for stage, count in enumerate(stage_completions)})
    for stage in range(6):
        attempts = int(stage_completions[stage] + stage_failures[stage])
        row[f'stage_{stage}_attempts'] = attempts
        row[f'stage_{stage}_failures'] = int(stage_failures[stage])
        row[f'stage_{stage}_censored'] = int(stage_censored[stage])
        row[f'stage_{stage}_success_rate'] = float(stage_completions[stage]/attempts) if attempts else None
    return row


class EvaluationBeta:
    """Bootstrap student states after warmup; gate further reductions by evaluation."""
    def __init__(self, config, saved=None, start_step=0):
        self.beta = float((saved or {}).get('beta', config.get('beta_start', 1.)))
        self.end = float(config.get('beta_end', 0.))
        self.decrement = float(config.get('dagger_beta_decrement', .1))
        self.threshold = float(config.get('dagger_eval_success_threshold', .8))
        self.required_passes = int(config.get('dagger_eval_required_passes', 2))
        self.failure_patience = int(config.get('dagger_eval_failure_patience', 2))
        self.maximum = float(config.get('beta_start', 1.))
        cap = float(config.get('dagger_beta_after_warmup', max(.9, self.end)))
        if not 0 <= cap <= 1:
            raise ValueError('dagger_beta_after_warmup must be in [0,1]')
        self.warmup_cap = min(self.maximum, cap)
        # Rebuild replay before any further reduction after a resume.
        warmup = int(config.get('dagger_teacher_warmup_steps', 5000))
        self.ready_step = start_step + warmup
        self.passes = self.failures = 0
        if (not 0 <= self.end <= self.beta <= 1 or not 0 < self.decrement <= 1
                or self.warmup_cap < self.end):
            raise ValueError('Invalid evaluation-gated beta bounds/decrement')
        if (not 0 < self.threshold <= 1 or min(self.required_passes, self.failure_patience) <= 0
                or warmup < 0 or not self.beta <= self.maximum <= 1):
            raise ValueError('Invalid evaluation success threshold/pass count')

    def beta_at_step(self, step):
        # A student trained only on teacher states cannot learn recovery from
        # its own errors. Retained demonstrations and teacher anchors protect
        # late-stage data while a small fraction of student actions explores.
        maximum = self.maximum if step < self.ready_step else self.warmup_cap
        self.beta = min(self.beta, maximum)
        return self.beta

    def observe(self, result, step):
        self.beta_at_step(step)
        valid = (step >= self.ready_step and result['episodes_finished'] == result['requested_episodes']
                 and result['success_rate'] >= self.threshold)
        self.passes = self.passes + 1 if valid else 0
        self.failures = self.failures + 1 if not valid and step >= self.ready_step else 0
        if self.passes >= self.required_passes:
            self.beta = max(self.end, self.beta - self.decrement)
            self.passes = 0
        elif self.failures >= self.failure_patience:
            self.beta = min(self.warmup_cap, self.beta + self.decrement)
            self.failures = 0
        return self.beta

    def state_dict(self):
        return dict(beta=self.beta)
