"""Time-major replay with persistent environment slots and masked episode resets."""
from __future__ import annotations

import torch


class SequenceBuffer:
    def __init__(self, capacity_steps, num_envs, image_shape, num_joints, hidden_shape,
                 device="cpu", storage_device="cpu", pin_memory=False):
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

    @property
    def size(self):
        return self.filled_steps * self.num_envs

    @property
    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in
                   (self.imgs, self.joints, self.expert_actions, self.hidden, self.resets))

    def clear(self):
        """Invalidate replay, e.g. a fresh simulator run; no stale windows survive."""
        self.position = self.filled_steps = 0

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
        def gather(value):
            return value[times, envs[:, None]].to(self.device)
        proprio = gather(self.joints)
        q, dq, imu, previous = torch.split(proprio, [self.num_joints, self.num_joints, 6, 34], dim=-1)
        hidden = self.hidden[times[:, 0], envs].transpose(0, 1).contiguous().to(self.device)
        return (gather(self.imgs).float() / 255, q, dq, imu, previous), gather(self.expert_actions), hidden, gather(self.resets)


def train_sequence_step(student, optimizer, buffer, batch_size, sequence_length, burn_in=8, grad_clip=1.0):
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
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())
