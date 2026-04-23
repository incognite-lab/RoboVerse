# dagger/student_net.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class VisionStudent(nn.Module):
    def __init__(self, num_actions, num_joints, img_channels=3, init_log_std=-2.0):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.img_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        self.joint_head = nn.Sequential(
            nn.Linear(num_joints, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        # nový parametr pro stochastic policy
        self.log_std = nn.Parameter(torch.ones(num_actions) * init_log_std)

    def encode(self, img, joint_pos):
        img_feat = self.cnn(img)
        img_feat = self.img_head(img_feat)

        joint_feat = self.joint_head(joint_pos)

        fused = torch.cat([img_feat, joint_feat], dim=1)
        return fused

    def forward(self, img, joint_pos):
        # kvůli kompatibilitě s DAggerem vrací mean action
        fused = self.encode(img, joint_pos)
        mean = self.actor(fused)
        return mean

    def get_dist(self, img, joint_pos):
        mean = self.forward(img, joint_pos)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        return dist, mean

    def act(self, img, joint_pos, deterministic=False):
        dist, mean = self.get_dist(img, joint_pos)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, mean

    def evaluate_actions(self, img, joint_pos, actions):
        dist, mean = self.get_dist(img, joint_pos)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean
