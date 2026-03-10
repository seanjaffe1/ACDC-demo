"""
Baseline 3: PointNet++ Autoencoder (no diffusion).

Justifies distributional modelling — shows that a deterministic encoder/decoder
without a learned distribution underperforms.
Uses a VAE loss (reconstruction + KL) without any flow/diffusion generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool


class SAModule(nn.Module):
    """Set Abstraction layer from PointNet++."""

    def __init__(self, ratio: float, r: float, nn_module: nn.Module):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn_module, add_self_loops=False)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2, nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64),
        ))
        self.sa2 = SAModule(0.25, 0.4, nn.Sequential(
            nn.Linear(64 + 3, 128), nn.ReLU(), nn.Linear(128, 128),
        ))
        self.sa3 = SAModule(0.25, 0.8, nn.Sequential(
            nn.Linear(128 + 3, 256), nn.ReLU(), nn.Linear(256, 256),
        ))
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        pos = data.pos
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        x = None

        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)

        pooled = global_max_pool(x, batch)
        return self.fc_mu(pooled), self.fc_logvar(pooled)


class FoldingDecoder(nn.Module):
    """FoldingNet-style decoder: latent → 2D grid → 3D surface."""

    def __init__(self, latent_dim: int = 64, num_points: int = 2500, grid_size: int = 50):
        super().__init__()
        self.num_points = num_points
        self.grid_size = grid_size

        # 2D grid
        xs = torch.linspace(-0.3, 0.3, grid_size)
        ys = torch.linspace(-0.3, 0.3, grid_size)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (G^2, 2)
        self.register_buffer("grid", grid)

        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3),
        )
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, latent_dim) → (B, G^2, 3)"""
        B = z.shape[0]
        G = self.grid.shape[0]
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, G^2, 2)
        z_rep = z.unsqueeze(1).expand(-1, G, -1)

        x = self.fold1(torch.cat([z_rep, grid], dim=-1))  # (B, G^2, 3)
        x = self.fold2(torch.cat([z_rep, x], dim=-1))
        return x


class PointNetAE(nn.Module):
    def __init__(self, latent_dim: int = 64, num_points: int = 2500):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = FoldingDecoder(latent_dim, num_points)
        self.latent_dim = latent_dim

    def forward(self, data: Data | Batch) -> dict:
        mu, logvar = self.encoder(data)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)  # (B, G^2, 3)

        # Chamfer reconstruction loss — split batched pos into (B, N_max, 3)
        B = mu.shape[0]
        batch_idx = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(data.pos.shape[0], dtype=torch.long, device=data.pos.device)
        per_graph = [data.pos[batch_idx == i] for i in range(B)]
        n_max = max(p.shape[0] for p in per_graph)
        target = torch.stack([F.pad(p, (0, 0, 0, n_max - p.shape[0])) for p in per_graph])
        chamfer = chamfer_loss(recon, target)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=-1).mean()
        loss = chamfer + 0.001 * kl

        return {"loss": loss, "chamfer": chamfer, "kl": kl, "mu": mu, "recon": recon}


def chamfer_loss(pred: Tensor, target: Tensor) -> Tensor:
    """pred, target: (B, N, 3). Returns mean Chamfer distance."""
    dist = torch.cdist(pred, target)  # (B, N, M)
    d1 = dist.min(dim=2).values.mean()
    d2 = dist.min(dim=1).values.mean()
    return d1 + d2
