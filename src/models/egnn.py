"""
SE(3)-Equivariant Graph Neural Network (EGNN).

Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
https://arxiv.org/abs/2102.09844

Each layer updates:
  h_i  ← φ_h(h_i, Σ_j m_ij)
  x_i  ← x_i + Σ_j (x_i - x_j) φ_x(m_ij)

where m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feat_dim: int = 0, act: str = "silu"):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]

        # Edge MLP: (h_i, h_j, ||x_i-x_j||^2, [edge_feats]) → m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )

        # Coordinate update MLP: m_ij → scalar weight
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Node update MLP: (h_i, Σ m_ij) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> tuple[Tensor, Tensor]:
        row, col = edge_index  # row=source, col=target

        # Pairwise squared distances
        diff = x[row] - x[col]  # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge features
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden)

        # Coordinate update
        coord_weight = self.coord_mlp(m_ij)  # (E, 1)
        coord_agg = torch.zeros_like(x).scatter_add(
            0, col.unsqueeze(-1).expand_as(diff), coord_weight * diff
        )
        x = x + coord_agg

        # Node update
        msg_agg = torch.zeros(h.shape[0], m_ij.shape[-1], device=h.device).scatter_add(
            0, col.unsqueeze(-1).expand_as(m_ij), m_ij
        )
        h_new = self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        h = self.norm(h + h_new)  # residual

        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_layers: int = 4,
        edge_feat_dim: int = 0,
        act: str = "silu",
    ):
        super().__init__()
        self.embed = nn.Linear(in_node_dim, hidden_dim) if in_node_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_feat_dim, act) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            data: PyG Data with pos, edge_index, (optional) x node features.
            h: optional pre-embedded node features (V, hidden_dim).
        Returns:
            h_out: (V, out_dim) node features
            x_out: (V, 3) updated positions (equivariant)
        """
        x = data.pos  # (V, 3)
        edge_index = data.edge_index

        if h is None:
            if hasattr(data, "x") and data.x is not None:
                h = self.embed(data.x)
            else:
                # Use ones as initial features
                h = self.embed(torch.ones(x.shape[0], self.embed.in_features if hasattr(self.embed, "in_features") else x.shape[0], device=x.device))

        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        return self.out_proj(h), x
