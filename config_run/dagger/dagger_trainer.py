# dagger/dagger_trainer.py
import torch
import torch.nn as nn

class DAggerBuffer:
    def __init__(self, max_samples, img_shape, num_joints, num_actions, device):
        self.max_samples = max_samples
        self.device = device

        c, h, w = img_shape

        self.imgs = torch.empty((max_samples, c, h, w), dtype=torch.uint8, device=device)
        self.joints = torch.empty((max_samples, num_joints), dtype=torch.float32, device=device)
        self.expert_actions = torch.empty((max_samples, num_actions), dtype=torch.float32, device=device)

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

        indices = torch.randperm(num_envs, device=self.device)[:store_count]

        imgs = batch_imgs_uint8[indices]
        joints = batch_joints[indices]
        actions = batch_actions[indices]

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
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        imgs = self.imgs[idx].float() / 255.0
        joints = self.joints[idx]
        acts = self.expert_actions[idx]
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
