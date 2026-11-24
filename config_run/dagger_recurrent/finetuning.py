"""Two learning rates and an update-count warmup for the visual backbone."""
from __future__ import annotations

import math

import torch


class VisualFineTuning:
    """Optionally warm up policy layers, then unfreeze the pretrained ResNet.

    All parameters remain in the optimizer from the start, including frozen
    ones. AdamW skips parameters without gradients, preserving encoder weights
    exactly during warmup. BatchNorm running statistics stay frozen forever.
    """

    def __init__(self, student, config, *, pretrained=False, saved=None):
        self.student = student
        # Explicit configuration may freeze an already fine-tuned checkpoint.
        # Omitted settings preserve its mode; old checkpoints keep prior behavior.
        self.enabled = config.get('dagger_new_finetune_visual', (saved or {}).get('enabled', True))
        if not isinstance(self.enabled, bool):
            raise ValueError('dagger_new_finetune_visual must be true or false')
        if saved is None:
            self.pretrained = bool(pretrained)
            self.policy_lr = float(config.get('learning_rate', 1e-4))
            self.visual_lr = float(config.get('dagger_new_visual_learning_rate', self.policy_lr * .1))
            self.warmup_updates = int(config.get('dagger_new_visual_warmup_updates', 1000))
            self.completed_updates = 0
        else:
            self.pretrained = bool(saved['pretrained'])
            self.policy_lr = float(saved['policy_lr'])
            self.visual_lr = float(saved['visual_lr'])
            self.warmup_updates = int(saved['warmup_updates'])
            self.completed_updates = int(saved['completed_updates'])
            if self.pretrained != bool(pretrained):
                raise ValueError('Fine-tuning state disagrees with checkpoint visual initialization')
        if (not all(math.isfinite(x) and x > 0 for x in (self.policy_lr, self.visual_lr))
                or self.visual_lr > self.policy_lr):
            raise ValueError('Require 0 < dagger_new_visual_learning_rate <= learning_rate, both finite')
        if min(self.warmup_updates, self.completed_updates) < 0:
            raise ValueError('Visual warmup/update counts must be nonnegative')
        self._active = None
        self.apply()

    @property
    def active(self):
        return self.enabled and (not self.pretrained or self.completed_updates >= self.warmup_updates)

    def make_optimizer(self, weight_decay=1e-4):
        visual = list(self.student.visual_encoder.parameters())
        visual_ids = {id(parameter) for parameter in visual}
        policy = [parameter for parameter in self.student.parameters() if id(parameter) not in visual_ids]
        optimizer = torch.optim.AdamW([
            dict(name='policy', params=policy, lr=self.policy_lr),
            dict(name='visual', params=visual, lr=self.visual_lr if self.active else 0.),
        ], weight_decay=weight_decay)
        return optimizer

    def apply(self, optimizer=None):
        """Apply saved phase/LRs without reinitializing either weights or Adam."""
        if self._active != self.active:
            for parameter in self.student.visual_encoder.parameters():
                parameter.requires_grad_(self.active)
                parameter.grad = None
            self._active = self.active
        if optimizer is not None:
            for group in optimizer.param_groups:
                if group.get('name') == 'policy':
                    group['lr'] = self.policy_lr
                elif group.get('name') == 'visual':
                    group['lr'] = self.visual_lr if self.active else 0.
                else:
                    raise ValueError('Unexpected optimizer group in visual fine-tuning')

    def after_update(self, optimizer):
        """Called only AFTER a successful gradient update; unlock update N+1."""
        before = self.active
        self.completed_updates += 1
        self.apply(optimizer)
        return not before and self.active

    def state_dict(self):
        return dict(enabled=self.enabled, pretrained=self.pretrained, policy_lr=self.policy_lr, visual_lr=self.visual_lr,
                    warmup_updates=self.warmup_updates, completed_updates=self.completed_updates)

    def metrics(self):
        return dict(visual_finetuning_enabled=int(self.enabled), visual_finetuning_active=int(self.active),
                    visual_learning_rate=self.visual_lr if self.active else 0.,
                    completed_gradient_updates=self.completed_updates,
                    visual_warmup_updates_remaining=max(0, self.warmup_updates-self.completed_updates)
                    if self.pretrained and self.enabled else 0)
