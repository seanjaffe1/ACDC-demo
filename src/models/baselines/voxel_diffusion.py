"""
Baseline 2: Voxel-space diffusion with a 3D U-Net score network.

Justifies mesh representation over voxel grids.
Operates on downsampled binary segmentation volumes (e.g., 32^3).
Uses DDPM-style diffusion (or can be swapped for flow matching on voxels).
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    """Simple 3D U-Net for voxel diffusion. Input: (B, 1, D, H, W) noisy volume."""

    def __init__(self, base_ch: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        ch = base_ch
        self.enc1 = ConvBlock3D(1, ch, time_dim)
        self.enc2 = ConvBlock3D(ch, ch * 2, time_dim)
        self.enc3 = ConvBlock3D(ch * 2, ch * 4, time_dim)

        self.bottleneck = ConvBlock3D(ch * 4, ch * 4, time_dim)

        self.dec3 = ConvBlock3D(ch * 8, ch * 2, time_dim)
        self.dec2 = ConvBlock3D(ch * 4, ch, time_dim)
        self.dec1 = ConvBlock3D(ch * 2, ch, time_dim)

        self.out = nn.Conv3d(ch, 1, 1)

        self.down = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embed(t.unsqueeze(-1))

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)

        b = self.bottleneck(self.down(e3), t_emb)

        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)

        return self.out(d1)


class VoxelFlowMatching(nn.Module):
    """Flow matching wrapper around UNet3D."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.unet = UNet3D(base_ch=base_ch)

    def forward(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """Compute CFM loss on voxel volumes."""
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None, None, None]) * x_0 + t[:, None, None, None, None] * x_1
        v_pred = self.unet(x_t, t)
        target = x_1 - x_0
        return ((v_pred - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device, steps: int = 100) -> Tensor:
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            v = self.unet(x, t)
            x = x + dt * v
        return x
