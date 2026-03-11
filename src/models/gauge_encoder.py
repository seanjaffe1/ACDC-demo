"""
Gauge-equivariant graph encoder.

Identical interface to GraphEncoder (encoder.py) but uses GaugeEGNN as
backbone instead of plain EGNN.  Designed for use with curvature-augmented
graphs (data/graphs_curvature/) where data.x contains four curvature
scalars: [pv1, pv2, H, K].

The only architectural difference from GraphEncoder:
  - backbone is GaugeEGNN (anisotropic message-passing)
  - in_node_dim defaults to 4 (curvature features)
  - exposes curvature_dim for the gauge frame construction
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax

from src.models.gauge_egnn import GaugeEGNN


class GaugeGraphEncoder(nn.Module):
    """
    VAE encoder: GaugeEGNN backbone → mean/logvar latent code.

    Args:
        in_node_dim   : dimension of raw node features (default 4 for
                        curvature graphs: pv1, pv2, H, K).
        hidden_dim    : hidden dimension of GaugeEGNN layers.
        latent_dim    : dimension of the VAE latent space (mu / logvar).
        n_layers      : number of GaugeEGNNLayers.
        curvature_dim : number of curvature channels in data.x (default 4).
        pooling       : global pooling strategy ("mean+max", "mean", "max").
    """

    def __init__(
        self,
        in_node_dim: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
        curvature_dim: int = 4,
        pooling: str = "mean+max",
    ):
        super().__init__()
        self.gauge_egnn = GaugeEGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            curvature_dim=curvature_dim,
        )
        self.pooling = pooling
        # h_aug = [h_out | dist_to_centroid] has dim hidden_dim+1
        # three pools: mean, max, attention → 3*(hidden_dim+1)
        aug_dim = hidden_dim + 1
        pool_dim = aug_dim * 3 if "+" in pooling else aug_dim

        self.attn_mlp = nn.Linear(aug_dim, 1)
        self.fc_mu = nn.Linear(pool_dim, latent_dim)
        self.fc_logvar = nn.Linear(pool_dim, latent_dim)

    def forward(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        h_out, x_out = self.gauge_egnn(data)
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h_out.shape[0], dtype=torch.long, device=h_out.device)
        )

        # Geometric pooling: per-vertex distance to graph centroid (invariant scalar)
        centroid = global_mean_pool(x_out, batch)[batch]          # (V, 3)
        dist_to_centroid = (x_out - centroid).norm(dim=-1, keepdim=True)  # (V, 1)
        h_aug = torch.cat([h_out, dist_to_centroid], dim=-1)      # (V, hidden_dim+1)

        if self.pooling == "mean+max":
            global_mean = global_mean_pool(h_aug, batch)          # (B, hidden_dim+1)
            global_max = global_max_pool(h_aug, batch)            # (B, hidden_dim+1)
            attn_logits = self.attn_mlp(h_aug)                    # (V, 1)
            attn_weights = scatter_softmax(attn_logits, batch, dim=0)  # (V, 1)
            attn_pool = global_add_pool(attn_weights * h_aug, batch)   # (B, hidden_dim+1)
            pooled = torch.cat([global_mean, global_max, attn_pool], dim=-1)
        elif self.pooling == "mean":
            pooled = global_mean_pool(h_aug, batch)
        else:
            pooled = global_max_pool(h_aug, batch)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled).clamp(-10.0, 10.0)
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
