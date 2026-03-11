"""
X-Eigenvector graph encoder.

Wraps XEigvecEGNN as a VAE encoder with the same augmented pooling scheme
as GaugeGraphEncoder: mean + max + attention pooling over per-vertex features
augmented by the distance-to-centroid (an SE(3)-invariant scalar).

Expects graphs built with the updated add_curvature.py that stores
data.principal_dir1 (V, 3) and data.principal_dir2 (V, 3).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax

from src.models.x_egnn import XEigvecEGNN


class XEigvecEncoder(nn.Module):
    """
    VAE encoder: XEigvecEGNN backbone → mean/logvar latent code.

    Uses three-way pooling (mean + max + attention) over node features
    augmented by the per-vertex distance to the graph centroid.

    Args:
        in_node_dim   : raw node-feature dimension (default 5: k1,k2,H,K,chi).
        hidden_dim    : hidden dimension of XEigvecEGNN layers.
        latent_dim    : dimension of the VAE latent space (mu / logvar).
        n_layers      : number of XEigvecEGNNLayers.
        curvature_dim : curvature scalar channels used in anisotropy weight.
        pooling       : global pooling strategy ("mean+max", "mean", "max").
    """

    def __init__(
        self,
        in_node_dim: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
        curvature_dim: int = 4,
        pooling: str = "mean+max",
    ):
        super().__init__()
        self.egnn = XEigvecEGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            curvature_dim=curvature_dim,
        )
        self.pooling = pooling

        # h_aug = [h_out | dist_to_centroid] has dim hidden_dim + 1
        # three pools: mean, max, attention → 3 × (hidden_dim + 1)
        aug_dim = hidden_dim + 1
        pool_dim = aug_dim * 3 if "+" in pooling else aug_dim

        self.attn_mlp = nn.Linear(aug_dim, 1)
        self.fc_mu = nn.Linear(pool_dim, latent_dim)
        self.fc_logvar = nn.Linear(pool_dim, latent_dim)

    def forward(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        h_out, x_out = self.egnn(data)
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h_out.shape[0], dtype=torch.long, device=h_out.device)
        )

        # Geometric pooling augmentation: SE(3)-invariant distance to centroid
        centroid = global_mean_pool(x_out, batch)[batch]               # (V, 3)
        dist_to_centroid = (x_out - centroid).norm(dim=-1, keepdim=True)  # (V, 1)
        h_aug = torch.cat([h_out, dist_to_centroid], dim=-1)           # (V, H+1)

        if self.pooling == "mean+max":
            global_mean = global_mean_pool(h_aug, batch)               # (B, H+1)
            global_max = global_max_pool(h_aug, batch)                 # (B, H+1)
            attn_logits = self.attn_mlp(h_aug)                        # (V, 1)
            attn_weights = scatter_softmax(attn_logits, batch, dim=0)  # (V, 1)
            attn_pool = global_add_pool(attn_weights * h_aug, batch)   # (B, H+1)
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
