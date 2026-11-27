"""Inference-only policy playback for the interactive Nyx calibration window."""
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger as log


@dataclass(frozen=True)
class PreviewPolicySettings:
    enabled: bool = False
    type: str = 'student'
    path: str = ''
    checkpoint: str | int | None = None
    start_paused: bool = False

    @classmethod
    def from_config(cls, config):
        raw = config.get('nyx_preview_policy', {})
        if not isinstance(raw, dict):
            raise ValueError('nyx_preview_policy must be a mapping')
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown nyx_preview_policy options: {sorted(unknown)}')
        settings = cls(**raw)
        if type(settings.enabled) is not bool or type(settings.start_paused) is not bool:
            raise ValueError('Preview policy enabled/start_paused must be true or false')
        if settings.type not in ('student', 'teacher'):
            raise ValueError('Preview policy type must be student or teacher')
        if settings.enabled:
            if not isinstance(settings.path, str) or not settings.path.strip():
                raise ValueError('Enabled preview policy requires a checkpoint path')
            if not config.get('nyx_preview_controls', False):
                raise ValueError('Policy playback requires nyx_preview_controls: true (Space pauses physics)')
            if settings.type == 'student' and settings.checkpoint is not None:
                raise ValueError('For student, set path to the .pth file and leave checkpoint null')
        return settings


class PreviewPolicy:
    """One call advances exactly one control step. No call means no physics/RNN update.

    The student temporarily uses the same Nyx sensor at the robot optical pose.
    The inspection pose is restored before displaying the scene. Preview-only
    G/V augmentation never changes the policy's inputs.
    """
    def __init__(self, settings, base, env):
        self.settings, self.base, self.env = settings, base, env
        self.handler = base.env.handler
        self.student = self.adapter = self.state = self.teacher = None
        self.steps = 0
        self.episodes = 0
        device = str(env.torch_device)
        if settings.type == 'student':
            try:
                from .dagger_recurrent.checkpoint import load_student
                from .dagger_recurrent.inputs import RolloutState, StudentInputs
            except ImportError:
                from dagger_recurrent.checkpoint import load_student
                from dagger_recurrent.inputs import RolloutState, StudentInputs
            self.student, meta, step = load_student(settings.path, env, base, device)
            task = self.handler.task
            if meta.get('target_stage') is not None:
                task.dagger_target_stage = task.curriculum_max_stage = meta['target_stage']
            self.observation = env.reset()
            self.adapter = StudentInputs(base, env, device, meta['camera_name'], meta['joint_names'],
                                         image_size=tuple(meta['image_shape'][-2:]))
            self.state = RolloutState(self.student, env.num_envs)
            log.info('Nyx preview loaded student {} at step {}', settings.path, step)
        else:
            try:
                from .separate_ppo_trainer import load_policy_router
            except ImportError:
                from separate_ppo_trainer import load_policy_router
            self.teacher, manifest = load_policy_router(env, settings.path, device=device,
                                                        checkpoint=settings.checkpoint)
            self.observation = env.reset()
            log.info('Nyx preview loaded PPO teacher {} ({} transitions)',
                     settings.path, manifest.get('global_timesteps', '?'))

    @torch.no_grad()
    def step(self, controller):
        hidden = None
        if self.student is not None:
            # Never let free-flight navigation leak into the student's camera.
            controller.apply(robot_view=True)
            self.handler._camera_states_cache = None
            try:
                values = self.adapter.read(self.state.previous_actions)
                images = values[0].float().div_(255)
                action, hidden = self.student(images, *values[1:], hidden=self.state.hidden,
                                               reset_mask=self.state.reset_mask)
                action = action.cpu().numpy()
            finally:
                controller.apply()
                self.handler._camera_states_cache = None
        else:
            action = self.teacher.predict(self.observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.env.num_envs, *self.env.action_space.shape) or not np.isfinite(action).all():
            raise ValueError('Preview policy returned invalid/nonfinite actions')
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.observation, _, dones, _ = self.env.step(action)
        if self.state is not None:
            self.state.after_step(hidden, action, dones)
        self.steps += 1
        self.episodes += int(np.asarray(dones).sum())

    def caption(self, paused):
        status = 'PAUSED' if paused else 'RUNNING'
        return f'{self.settings.type} {status} (Space) | step={self.steps} episodes={self.episodes}'
