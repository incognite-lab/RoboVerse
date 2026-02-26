# config_run/dagger/student_net.py
import torch
import torch.nn as nn

class VisionStudent(nn.Module):
    def __init__(self, num_actions, img_channels=3):
        super().__init__()
        # Vstup očekává: [Batch, 3, 360, 640]
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Přidány další vrstvy pro lepší kompresi obrazu
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            # Bez ohledu na to, jak velký obrázek tam vleze, tato vrstva
            # z něj udělá mřížku 4x4 s 128 kanály.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # 128 kanálů * 4 výška * 4 šířka = 2048 vstupních rysů
        self.actor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, img):
        x = self.cnn(img)
        actions = self.actor(x)
        return actions
