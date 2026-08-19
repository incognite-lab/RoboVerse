# dagger/dagger_trainer.py
import torch
import torch.nn as nn

class DAggerBuffer:
    def __init__(
        self,
        max_samples,
        img_shape,
        num_joints,
        num_actions,
        device,
        storage_device="cpu",
        pin_memory=False,
    ):
        self.max_samples = max_samples
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.pin_memory = bool(pin_memory and self.storage_device.type == "cpu")

        c, h, w = img_shape

        try:
            alloc_kwargs = {"device": self.storage_device}
            if self.storage_device.type == "cpu":
                alloc_kwargs["pin_memory"] = self.pin_memory

            self.imgs = torch.empty((max_samples, c, h, w), dtype=torch.uint8, **alloc_kwargs)
            self.joints = torch.empty((max_samples, num_joints), dtype=torch.float32, **alloc_kwargs)
            self.expert_actions = torch.empty((max_samples, num_actions), dtype=torch.float32, **alloc_kwargs)
        except RuntimeError:
            if not self.pin_memory:
                raise

            self.pin_memory = False
            self.imgs = torch.empty((max_samples, c, h, w), dtype=torch.uint8, device=self.storage_device)
            self.joints = torch.empty((max_samples, num_joints), dtype=torch.float32, device=self.storage_device)
            self.expert_actions = torch.empty((max_samples, num_actions), dtype=torch.float32, device=self.storage_device)

        self.ptr = 0
        self.size = 0

    def add_batch(self, batch_imgs_uint8, batch_joints, batch_actions, store_count=None):
        """
        batch_imgs_uint8: [N, C, H, W]
        batch_joints:     [N, num_joints]
        batch_actions:    [N, num_actions]
        """
        num_envs = batch_imgs_uint8.shape[0]

        if store_count is None:
            store_count = num_envs

        store_count = min(store_count, num_envs)

        indices = torch.randperm(num_envs, device=batch_imgs_uint8.device)[:store_count]

        imgs = batch_imgs_uint8[indices].detach().to(
            device=self.storage_device,
            dtype=torch.uint8,
            non_blocking=self.pin_memory,
        )
        joints = batch_joints[indices].detach().to(
            device=self.storage_device,
            dtype=torch.float32,
            non_blocking=self.pin_memory,
        )
        actions = batch_actions[indices].detach().to(
            device=self.storage_device,
            dtype=torch.float32,
            non_blocking=self.pin_memory,
        )

        end = self.ptr + store_count
        if end <= self.max_samples:
            self.imgs[self.ptr:end] = imgs
            self.joints[self.ptr:end] = joints
            self.expert_actions[self.ptr:end] = actions
        else:
            first = self.max_samples - self.ptr
            second = store_count - first

            self.imgs[self.ptr:] = imgs[:first]
            self.joints[self.ptr:] = joints[:first]
            self.expert_actions[self.ptr:] = actions[:first]

            self.imgs[:second] = imgs[first:]
            self.joints[:second] = joints[first:]
            self.expert_actions[:second] = actions[first:]

        self.ptr = (self.ptr + store_count) % self.max_samples
        self.size = min(self.size + store_count, self.max_samples)

    def sample(self, batch_size):
        if self.size == 0:
            return None, None, None

        batch_size = min(batch_size, self.size)
        idx = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

        imgs = self.imgs[idx].to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.pin_memory,
        ) / 255.0
        joints = self.joints[idx].to(device=self.device, non_blocking=self.pin_memory)
        acts = self.expert_actions[idx].to(device=self.device, non_blocking=self.pin_memory)
        return imgs, joints, acts


def train_dagger_step(student_net, optimizer, buffer, batch_size=512):
    if buffer.size == 0:
        return 0.0

    student_net.train()
    imgs, joints, expert_acts = buffer.sample(batch_size)

    pred = student_net(imgs, joints)
    loss = nn.MSELoss()(pred, expert_acts)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()
