"""Linear beta decay with stage-relative corrections and recovery pauses."""
from __future__ import annotations

from copy import deepcopy
import math


def stage_measurement(row, stage):
    """Only resolved attempts count; an externally capped probe is inconclusive."""
    successes = int(row.get(f'stage_{stage}_completions', 0))
    failures = int(row.get(f'stage_{stage}_failures', 0))
    attempts = successes + failures
    return dict(successes=successes, failures=failures, attempts=attempts,
                censored=int(row.get(f'stage_{stage}_censored', 0)),
                capped_attempts=int(row.get('capped_attempts', 0)),
                success_rate=successes/attempts if attempts else None)


class StageSawtoothBeta:
    mode = 'stage_sawtooth'

    def __init__(self, config, saved=None, start_step=0):
        target_stage = config.get('dagger_target_stage', 4)
        if isinstance(target_stage, bool) or not isinstance(target_stage, int):
            raise ValueError('dagger_target_stage must be an integer from 0 to 5')
        self.target_stage = target_stage
        self.start = float(config.get('beta_start', 1.))
        self.end = float(config.get('beta_end', 0.))
        self.decay_steps = int(config.get('dagger_beta_decay_steps', 1000))
        self.tolerance = float(config.get('dagger_beta_stage_tolerance', .1))
        self.increment = float(config.get('dagger_beta_recovery_increment', .1))
        self.pause_steps = int(config.get('dagger_beta_recovery_pause_steps', 500))
        self.min_attempts = int(config.get('dagger_beta_min_stage_attempts', 30))
        warmup = int(config.get('dagger_teacher_warmup_steps', 5000))
        if not 0 <= self.target_stage <= 5 or isinstance(config.get('dagger_target_stage'), bool):
            raise ValueError('dagger_target_stage must be 0..5')
        if not all(math.isfinite(v) for v in (self.start,self.end,self.tolerance,self.increment)):
            raise ValueError('Stage beta settings must be finite')
        if not 0 <= self.end <= self.start <= 1 or not 0 <= self.tolerance < 1 or not 0 < self.increment <= 1:
            raise ValueError('Invalid stage beta bounds/tolerance/increment')
        if min(self.decay_steps,self.min_attempts) <= 0 or min(warmup,self.pause_steps) < 0:
            raise ValueError('Invalid stage beta decay/warmup/pause/sample counts')
        self.warmup_end = start_step + warmup
        self.last_step = start_step
        self.beta = self.start
        self.pause_until = 0
        self.last_evaluation_step = -1
        self.reference = None
        self.last_comparison = {}
        if saved is not None:
            if saved.get('mode') != self.mode:
                raise ValueError('Cannot change beta controller on resume; start a new run')
            if int(saved['target_stage']) != self.target_stage:
                raise ValueError('Cannot change dagger_target_stage on stage-controller resume')
            # Resume the actual curve, including a partially consumed pause.
            for key in ('start','end','decay_steps','tolerance','increment','pause_steps','min_attempts',
                        'warmup_end','last_step','beta','pause_until','last_evaluation_step','reference'):
                setattr(self,key,deepcopy(saved[key]))
            if self.last_step != start_step:
                raise ValueError('Stage beta state and checkpoint steps differ')

    @property
    def stages(self):
        return range(self.target_stage+1)

    def reliable(self, measurement):
        return (measurement is not None and measurement.get('success_rate') is not None
                and measurement.get('attempts',0) >= self.min_attempts
                and measurement.get('censored',0) == 0 and measurement.get('capped_attempts',0) == 0)

    def beta_at_step(self, step):
        if step < self.last_step:
            raise ValueError('Stage beta time cannot move backwards')
        if self.reference is not None:
            active_since = max(self.last_step,self.warmup_end,self.pause_until)
            self.beta = max(self.end,self.beta - max(0,step-active_since)*(self.start-self.end)/self.decay_steps)
        self.last_step = step
        return self.beta

    def set_reference(self, measurements, step):
        if self.reference is not None:
            raise ValueError('Teacher reference is already frozen')
        if step < self.warmup_end:
            raise ValueError('Teacher reference is finalized after teacher warmup')
        self.beta_at_step(step)
        self.reference = {int(stage):deepcopy(measurements.get(stage)) for stage in self.stages}

    def observe(self, measurements, step):
        self.beta_at_step(step)
        if self.reference is None or step < self.warmup_end or step <= self.last_evaluation_step:
            return []
        self.last_evaluation_step = step
        bad = []
        self.last_comparison = {}
        for stage in self.stages:
            teacher, student = self.reference.get(stage),measurements.get(stage)
            eligible = self.reliable(teacher) and self.reliable(student)
            drop = teacher['success_rate']-student['success_rate'] if eligible else None
            self.last_comparison[stage] = dict(eligible=eligible,drop=drop)
            if eligible and drop > self.tolerance + 1e-12:
                bad.append(stage)
        if bad:
            raised = min(self.start,self.beta+self.increment)
            # At the upper bound no actual correction happened: repeatedly
            # extending a pause there would reintroduce teacher-only deadlock.
            if raised > self.beta + 1e-12:
                self.beta = raised
                self.pause_until = step+self.pause_steps
        return bad

    def state_dict(self):
        keys = ('target_stage','start','end','decay_steps','tolerance','increment','pause_steps',
                'min_attempts','warmup_end','last_step','beta','pause_until','last_evaluation_step','reference')
        return dict(mode=self.mode,**{key:deepcopy(getattr(self,key)) for key in keys})

    def metrics(self, step):
        values = dict(beta_pause_steps_remaining=max(0,self.pause_until-step),
                      teacher_reference_ready=int(self.reference is not None))
        for stage in self.stages:
            reference = (self.reference or {}).get(stage)
            values[f'stage_{stage}/reference_valid'] = int(self.reliable(reference))
            if reference is not None:
                values[f'stage_{stage}/teacher_attempts'] = reference['attempts']
                if reference['success_rate'] is not None:
                    values[f'stage_{stage}/teacher_success_rate'] = reference['success_rate']
            comparison = self.last_comparison.get(stage,{})
            if comparison.get('drop') is not None:
                values[f'stage_{stage}/success_drop'] = comparison['drop']
        return values
