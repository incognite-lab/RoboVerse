# dagger/student_net.py
import torch
import torch.nn as nn

class VisionStudent(nn.Module):
    def __init__(self, num_actions, num_joints, img_channels=3):
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

        # obraz -> 2048 feature
        self.img_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        # klouby -> menší embedding
        self.joint_head = nn.Sequential(
            nn.Linear(num_joints, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # fúze obou větví
        self.actor = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, img, joint_pos):
        img_feat = self.cnn(img)
        img_feat = self.img_head(img_feat)

        joint_feat = self.joint_head(joint_pos)

        fused = torch.cat([img_feat, joint_feat], dim=1)
        actions = self.actor(fused)
        return actions
