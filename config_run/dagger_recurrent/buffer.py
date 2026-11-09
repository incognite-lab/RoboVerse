"""Time-major replay with persistent environment slots and masked episode resets."""
from __future__ import annotations

import torch


class SequenceBuffer:
    def __init__(self, capacity_steps, num_envs, image_shape, num_joints, hidden_shape,
                 device="cpu", storage_device="cpu", pin_memory=False,
                 demonstrations_per_stage=0, demonstration_length=24, demonstration_fraction=.5,
                 demonstration_burn_in=0):
        if capacity_steps <= 0 or num_envs <= 0:
            raise ValueError("Replay capacity and stored environment count must be positive")
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.max_samples = capacity_steps * num_envs
        self.position = self.filled_steps = 0
        kwargs = dict(device=self.storage_device,
                      pin_memory=bool(pin_memory and self.storage_device.type == "cpu"))
        shape = (capacity_steps, num_envs)
        self.imgs = torch.empty((*shape, *image_shape), dtype=torch.uint8, **kwargs)
        # Combined proprioception also makes the shared metrics count its bytes.
        self.joints = torch.empty((*shape, 2 * num_joints + 6 + 34), **kwargs)
        self.expert_actions = torch.empty((*shape, 34), **kwargs)
        self.hidden = torch.empty((*shape, *hidden_shape), **kwargs)  # [time,env,layer,hidden]
        self.resets = torch.empty(shape, dtype=torch.bool, **kwargs)
        if (demonstrations_per_stage < 0 or demonstration_length <= 0 or not 0 <= demonstration_fraction <= 1
                or not 0 <= demonstration_burn_in < demonstration_length):
            raise ValueError("Invalid demonstration archive settings")
        if demonstrations_per_stage and demonstration_length > capacity_steps:
            raise ValueError("Demonstration windows must fit inside the online replay")
        self.demonstrations_per_stage = demonstrations_per_stage
        self.demonstration_length = demonstration_length
        self.demonstration_fraction = demonstration_fraction
        self.demonstration_burn_in = demonstration_burn_in
        self.demonstration_stride = demonstration_length - demonstration_burn_in
        self.demonstrations = {}  # CPU, stage-balanced reservoir of complete teacher windows
        self.demonstrations_seen = {}
        self.teacher_masks = torch.zeros(shape, dtype=torch.bool, **kwargs)
        self.stages = torch.full(shape, -1, dtype=torch.int64, **kwargs)
        self.steps_added = 0

    @property
    def size(self):
        return self.filled_steps * self.num_envs

    @property
    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in
                   (self.imgs, self.joints, self.expert_actions, self.hidden, self.resets,
                    self.teacher_masks, self.stages))

    @property
    def demonstration_bytes(self):
        return sum(t.numel() * t.element_size() for windows in self.demonstrations.values()
                   for window in windows for t in window)

    def clear(self):
        """Invalidate replay, e.g. a fresh simulator run; no stale windows survive."""
        self.position = self.filled_steps = 0
        self.steps_added = 0
        self.teacher_masks.zero_()
        self.stages.fill_(-1)

    def add(self, inputs, expert_actions, hidden, reset_mask):
        images, *proprio = inputs
        expected = (self.num_envs,)
        if images.shape[:1] != expected or expert_actions.shape != (*expected, 34):
            raise ValueError("Every replay step must contain the same persistent environment slots")
        for target, value in (
            (self.imgs, images), (self.joints, torch.cat(proprio, dim=-1)),
            (self.expert_actions, expert_actions), (self.hidden, hidden.transpose(0, 1)),
            (self.resets, reset_mask),
        ):
            # copy_ completes the transfer before the caller may mutate inputs.
            target[self.position].copy_(value.detach())
        self.position = (self.position + 1) % self.capacity_steps
        self.filled_steps = min(self.filled_steps + 1, self.capacity_steps)
        self.steps_added += 1

    def record_demonstrations(self, stages, executed_expert):
        """Archive contiguous teacher-only windows before the ring overwrites them.

        One candidate per observed stage and window period bounds copy traffic.
        A separate reservoir per stage prevents stage 0 from evicting late stages.
        Episode reset masks are retained, as are the pre-observation hidden states.
        """
        last = (self.position - 1) % self.capacity_steps
        self.teacher_masks[last].copy_(torch.as_tensor(executed_expert, device=self.storage_device))
        self.stages[last].copy_(torch.as_tensor(stages, device=self.storage_device, dtype=torch.long))
        length = self.demonstration_length
        if (not self.demonstrations_per_stage or self.filled_steps < length
                or (self.steps_added - length) % self.demonstration_stride):
            return
        times = (self.position - length + torch.arange(length, device=self.storage_device)) % self.capacity_steps
        eligible = self.teacher_masks[times].all(0)
        # Label a window by EVERY stage represented in its supervised targets.
        # Overlap the burn-in context so supervised parts have no temporal gaps.
        # Otherwise brief phases could always land in an untrained burn-in gap.
        window_stages = self.stages[times[self.demonstration_burn_in:]]
        for stage in window_stages[:, eligible].unique().tolist():
            if not 0 <= stage < 6:
                continue
            candidates = (eligible & (window_stages == stage).any(0)).nonzero().flatten()
            env = candidates[torch.randint(len(candidates), ())].item()
            seen = self.demonstrations_seen.get(stage, 0) + 1
            self.demonstrations_seen[stage] = seen
            windows = self.demonstrations.setdefault(stage, [])
            slot = len(windows) if len(windows) < self.demonstrations_per_stage else int(torch.randint(seen, ()))
            if slot >= self.demonstrations_per_stage:
                continue
            window = tuple(t[times, env].detach().cpu().clone() for t in
                           (self.imgs, self.joints, self.expert_actions, self.hidden, self.resets))
            if slot == len(windows):
                windows.append(window)
            else:
                windows[slot] = window

    def archive_state(self):
        return dict(length=self.demonstration_length, burn_in=self.demonstration_burn_in,
                    image_shape=list(self.imgs.shape[2:]),
                    num_joints=self.num_joints, hidden_shape=list(self.hidden.shape[2:]),
                    windows=self.demonstrations, seen=self.demonstrations_seen)

    def load_archive(self, state):
        expected = self.archive_state()
        for key in ("length", "burn_in", "image_shape", "num_joints", "hidden_shape"):
            if state[key] != expected[key]:
                raise ValueError(f"Demonstration archive {key} differs from this run")
        self.demonstrations = {int(k): v[:self.demonstrations_per_stage] for k, v in state['windows'].items()}
        self.demonstrations_seen = dict(state['seen'])

    def sample(self, batch_size, sequence_length, burn_in=0):
        total = sequence_length + burn_in
        if batch_size <= 0 or sequence_length <= 0 or burn_in < 0:
            raise ValueError("Invalid sequence sampling dimensions")
        if self.filled_steps < total:
            raise ValueError("Not enough contiguous replay steps yet")
        # Logical positions run oldest -> newest, so windows never jump across
        # the ring's newest/oldest boundary or across different environments.
        oldest = (self.position - self.filled_steps) % self.capacity_steps
        starts = torch.randint(self.filled_steps - total + 1, (batch_size,), device=self.storage_device)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.storage_device)
        times = (oldest + starts[:, None] + torch.arange(total, device=self.storage_device)) % self.capacity_steps
        demo_rows = []
        available = [stage for stage, windows in self.demonstrations.items() if windows]
        if available:
            if total != self.demonstration_length:
                raise ValueError("Sample length differs from the archived demonstration length")
            for row in range(batch_size):
                if float(torch.rand(())) < self.demonstration_fraction:
                    stage = available[int(torch.randint(len(available), ()))]
                    windows = self.demonstrations[stage]
                    demo_rows.append((row, windows[int(torch.randint(len(windows), ()))]))
        def gather(value, field):
            result = value[times, envs[:, None]]
            for row, window in demo_rows:
                result[row].copy_(window[field].to(self.storage_device))
            return result.to(self.device)
        proprio = gather(self.joints, 1)
        q, dq, imu, previous = torch.split(proprio, [self.num_joints, self.num_joints, 6, 34], dim=-1)
        hidden = gather(self.hidden, 3)[:, 0].transpose(0, 1).contiguous()
        return ((gather(self.imgs, 0).float() / 255, q, dq, imu, previous),
                gather(self.expert_actions, 2), hidden, gather(self.resets, 4))


def train_sequence_step(student, optimizer, buffer, batch_size, sequence_length, burn_in=8, grad_clip=1.0,
                        diagnostics=None):
    values, target, hidden, resets = buffer.sample(batch_size, sequence_length, burn_in)
    student.train()
    if burn_in:
        # Warm up a replayed rollout state using current network parameters.
        with torch.no_grad():
            _, hidden = student(*(value[:, :burn_in] for value in values), hidden=hidden,
                                reset_mask=resets[:, :burn_in])
    prediction, _ = student(*(value[:, burn_in:] for value in values), hidden=hidden.detach(),
                            reset_mask=resets[:, burn_in:])
    loss = torch.nn.functional.mse_loss(prediction, target[:, burn_in:])
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite recurrent imitation loss; optimizer was not updated")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip, error_if_nonfinite=True)
    if diagnostics is not None:
        diagnostics['gradient_norm_before_clip'] = float(norm)
        for name, module in (("visual", student.visual_encoder), ("proprio", student.proprio_encoder),
                             ("gru", student.gru), ("heads", student.action_heads)):
            norms = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
            diagnostics[f'{name}_gradient_norm'] = float(torch.stack(norms).norm()) if norms else 0.
        diagnostics.update({k: float(v) for k,v in student.last_diagnostics.items()})
    optimizer.step()
    return float(loss.detach())
