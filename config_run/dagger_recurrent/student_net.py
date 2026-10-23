"""ResNet-18 + proprioception MLP + GRU for the 34-action ChairMan interface."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


# Matches SB3_chairman_env.upper_body_joint_names, followed by walk commands.
# This is action order, not the simulator's joint-observation order.
ACTION_GROUP_SIZES = (
    ("waist", 3),
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_fingers", 7),
    ("right_fingers", 7),
    ("direction", 3),
)


class RecurrentVisionStudent(nn.Module):
    """Consume one step or a batch-first sequence and return actions + memory.

    RGB is float RGB in [0, 1]. Proprioception consists of all joint angles
    [rad], joint velocities [rad/s], body-frame angular velocity [rad/s] plus
    projected unit gravity (six IMU features), and the previous *executed*
    34-action vector in the environment's physical units.

    Single step: images [B,3,H,W], other inputs [B,D], reset_mask [B].
    Sequence: images [B,T,3,H,W], other inputs [B,T,D], reset_mask [B,T].
    True resets memory BEFORE the corresponding observation, including resets
    inside sequences. Hidden state is explicit [num_layers,B,hidden_size]; no
    state is retained in this module. Detach it at truncated-BPTT boundaries.
    """

    num_actions = 34
    imu_dim = 6

    def __init__(
        self,
        num_joints: int = 43,
        hidden_size: int = 256,
        num_layers: int = 1,
        proprio_embedding_size: int = 128,
        pretrained_visual: bool = False,
    ):
        super().__init__()
        if min(num_joints, hidden_size, num_layers, proprio_embedding_size) <= 0:
            raise ValueError("Joint count and network dimensions must be positive")
        self.num_joints = num_joints
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proprio_dim = 2 * num_joints + self.imu_dim + self.num_actions

        # No download by default. Opting into ImageNet weights may download them.
        weights = ResNet18_Weights.DEFAULT if pretrained_visual else None
        self.visual_encoder = resnet18(weights=weights)
        self.visual_encoder.fc = nn.Identity()
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, proprio_embedding_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            512 + proprio_embedding_size, hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.action_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, size))
            for name, size in ACTION_GROUP_SIZES
        })
        self.action_slices = OrderedDict()
        start = 0
        for name, size in ACTION_GROUP_SIZES:
            self.action_slices[name] = slice(start, start + size)
            start += size

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("proprio_mean", torch.zeros(self.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(self.proprio_dim))
        self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self):
        # Avoid dependence on future frames / other environments through batch
        # statistics, and allow one-frame inference at small resolutions.
        # Affine parameters and all convolution weights remain trainable.
        for module in self.visual_encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_batch_norm_statistics()
        return self

    @torch.no_grad()
    def set_proprio_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Install training-set statistics; saved with the model state_dict.

        Feature order: positions, velocities, IMU, previous executed actions.
        Defaults are identity; validation/evaluation must reuse training stats.
        """
        mean = torch.as_tensor(mean, device=self.proprio_mean.device, dtype=self.proprio_mean.dtype)
        std = torch.as_tensor(std, device=self.proprio_std.device, dtype=self.proprio_std.dtype)
        if mean.shape != self.proprio_mean.shape or std.shape != self.proprio_std.shape:
            raise ValueError(f"Normalization statistics must have shape ({self.proprio_dim},)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization statistics must be finite with strictly positive std")
        self.proprio_mean.copy_(mean)
        self.proprio_std.copy_(std)

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        """Allocate zero memory on the model's current device and dtype."""
        return self.gru.weight_ih_l0.new_zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(
        self,
        images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        imu: torch.Tensor,
        previous_actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_step = images.ndim == 4
        if images.ndim not in (4, 5):
            raise ValueError("Images must be [B,3,H,W] or [B,T,3,H,W]")
        if not images.is_floating_point():
            raise ValueError("Images must be floating-point RGB in [0,1], not uint8")
        leading = images.shape[:-3]
        if images.shape[-3] != 3 or min(images.shape[-2:]) < 32 or min(leading) <= 0:
            raise ValueError("Expected a nonempty RGB batch with image height/width >= 32")
        for name, value, size in (
            ("joint_positions", joint_positions, self.num_joints),
            ("joint_velocities", joint_velocities, self.num_joints),
            ("imu", imu, self.imu_dim),
            ("previous_actions", previous_actions, self.num_actions),
        ):
            if value.shape != (*leading, size):
                raise ValueError(f"{name} must have shape {(*leading, size)}, got {tuple(value.shape)}")
        if reset_mask is not None:
            if reset_mask.shape != leading or reset_mask.dtype != torch.bool:
                raise ValueError(f"reset_mask must be boolean with shape {tuple(leading)}")
            reset_mask = reset_mask.to(device=images.device)
        batch_size = images.shape[0]
        steps = 1 if single_step else images.shape[1]
        if hidden is None:
            hidden = self.initial_hidden(batch_size)
        elif hidden.shape != (self.num_layers, batch_size, self.hidden_size):
            raise ValueError("Hidden state shape must be [num_layers,B,hidden_size]")

        rgb = images.reshape(-1, *images.shape[-3:])
        visual = self.visual_encoder((rgb - self.image_mean) / self.image_std)
        proprio = torch.cat((joint_positions, joint_velocities, imu, previous_actions), dim=-1)
        proprio = self.proprio_encoder((proprio - self.proprio_mean) / self.proprio_std)
        fused = torch.cat(
            (visual.reshape(batch_size, steps, 512), proprio.reshape(batch_size, steps, -1)), dim=-1
        )
        if reset_mask is None:
            features, hidden = self.gru(fused, hidden)
        else:
            resets = reset_mask.reshape(batch_size, steps)
            outputs = []
            for step in range(steps):
                hidden = hidden.masked_fill(resets[:, step].view(1, batch_size, 1), 0.0)
                output, hidden = self.gru(fused[:, step:step + 1], hidden)
                outputs.append(output)
            features = torch.cat(outputs, dim=1)
        actions = torch.cat([head(features) for head in self.action_heads.values()], dim=-1)
        # Raw physical action targets: clipping/rate limiting belongs to the
        # existing control wrapper. Output ranges mix angles and walk commands.
        return (actions[:, 0] if single_step else actions), hidden
