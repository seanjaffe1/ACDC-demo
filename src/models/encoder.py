"""
Graph encoder: EGNN backbone → global latent code via pooling.
Used for conditioning the diffusion model and for latent space analysis.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

from src.models.egnn import EGNN


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_node_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
        pooling: str = "mean+max",
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.pooling = pooling
        pool_dim = hidden_dim * 2 if "+" in pooling else hidden_dim

        self.fc_mu = nn.Linear(pool_dim, latent_dim)
        self.fc_logvar = nn.Linear(pool_dim, latent_dim)

    def forward(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        h, _ = self.egnn(data)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        if self.pooling == "mean+max":
            pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        elif self.pooling == "mean":
            pooled = global_mean_pool(h, batch)
        else:
            pooled = global_max_pool(h, batch)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def encode(self, data: Data | Batch) -> Tensor:
        """Return mean latent code (no sampling)."""
        mu, _ = self.forward(data)
        return mu

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
