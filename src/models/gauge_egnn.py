"""
Gauge-Equivariant Graph Neural Network (GaugeEGNN).

Motivation
----------
Standard EGNN passes messages that are isotropic: each node aggregates
information from neighbours weighted only by the pairwise distance.
Cardiac surfaces, however, have pronounced local anisotropy captured by
their principal curvatures (pv1, pv2): one principal direction corresponds
to the long axis of the ventricle and the other to the circumferential
direction, and these directions carry different shape-information.

Gauge-equivariant formulation
-------------------------------
We augment EGNN message-passing with a per-node anisotropy weight derived
from the principal-curvature features already embedded in the node
representation.  Concretely, for each edge (i → j):

1. Compute the unit edge vector  e_ij = (x_j - x_i) / ||x_j - x_i||.
2. Build a 2-D local frame at node i from the curvature features:
   - Frame direction d_i is the normalised 3-D vector predicted from a
     small MLP applied to the curvature scalars (pv1_i, pv2_i, H_i, K_i).
   - The anisotropy weight w_ij = f(cos θ_ij) where
       cos θ_ij = |e_ij · d_i|  (alignment of edge with dominant curvature)
     and f is a learned 1-D function implemented as a 2-layer MLP.
3. Multiply the edge message m_ij by w_ij before aggregation.

This keeps all invariant/equivariant properties of EGNN:
- Node-feature updates remain invariant (w_ij is a scalar derived from
  invariant inner products, so the aggregated message is invariant).
- Coordinate updates remain equivariant (we scale the EGNN coordinate
  update by the same scalar w_ij, which does not introduce any preferred
  direction in 3-D space — only a reweighting of already-equivariant terms).

Gauge freedom
~~~~~~~~~~~~~
The local frame direction d_i is predicted from *scalar* curvature
features, not from the 3-D positions.  This means d_i is defined in the
tangent-plane sense (as a magnitude/alignment signal) rather than as a
fixed 3-D axis.  The alignment |e_ij · d_i| is computed after normalising
d_i to a 3-D unit vector via a linear map from the curvature scalars to R^3.
Under a global rotation R of all positions:
  - x_j - x_i  →  R(x_j - x_i)  →  same ||...|| →  same curvature scalars
  - d_i is re-computed from unchanged scalars, but the 3-D projection
    must also rotate.  We handle this by constructing d_i as a learned
    combination of the normalised edge vectors in the 1-hop neighbourhood
    (see AnisotropyFrame below), which is manifestly equivariant.

For simplicity and numerical stability we adopt the following convention:
  d_i = sum_{j in N(i)} alpha_{ij} * normalised(x_j - x_i)
where the scalar mixing weights alpha_{ij} are predicted from
(h_i, curvature_i, dist_ij).  This gives an equivariant local frame
vector d_i in R^3 whose 3-D direction co-rotates with the point cloud.

References
----------
- Satorras et al., E(n) Equivariant Graph Neural Networks, ICML 2021.
- de Haan et al., Gauge Equivariant Mesh CNNs, ICML 2021.
- Cohen et al., Gauge Equivariant Convolutional Networks, ICML 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from src.models.egnn import EGNNLayer, EGNN


# ---------------------------------------------------------------------------
# Anisotropy frame & weighting
# ---------------------------------------------------------------------------

class AnisotropyFrame(nn.Module):
    """
    Construct a per-node equivariant local-frame vector d_i in R^3.

    d_i = sum_j  alpha_ij * normalised(x_j - x_i)

    where alpha_ij are scalar mixing weights from an MLP:
        alpha_ij = MLP(h_i, curvature_i, ||x_j - x_i||)

    Because each term is a *scalar* times a *relative position*, the sum
    transforms as a vector under SO(3) — giving d_i SE(3)-equivariant
    despite being constructed purely from invariant inputs.
    """

    def __init__(self, hidden_dim: int, curvature_dim: int = 4):
        super().__init__()
        # Inputs: h_i (hidden_dim) + curvature_i (curvature_dim) + dist (1)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(hidden_dim + curvature_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: Tensor,              # (V, hidden_dim)
        x: Tensor,              # (V, 3)
        curvature: Tensor,      # (V, curvature_dim)
        edge_index: Tensor,     # (2, E)
    ) -> Tensor:
        """Returns d : (V, 3) equivariant local-frame vectors."""
        row, col = edge_index   # row=source, col=target

        diff = x[col] - x[row]                            # (E, 3)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        unit_diff = diff / dist                            # (E, 3), unit vectors

        # Mixing weights: invariant scalars at source node
        inp = torch.cat([h[row], curvature[row], dist], dim=-1)  # (E, H+C+1)
        alpha = self.alpha_mlp(inp)  # (E, 1)

        # Weighted sum into each target node (same as EGNN coord aggregation)
        # Use source=row to accumulate into row nodes' frame
        d = torch.zeros_like(x).scatter_add(
            0,
            row.unsqueeze(-1).expand(-1, 3),
            alpha * unit_diff,
        )   # (V, 3)

        # Normalise (handle zero-degree nodes gracefully)
        d_norm = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        d = d / d_norm  # (V, 3) unit frame vectors

        return d


class AnisotropyWeight(nn.Module):
    """
    Compute a scalar edge weight w_ij from the alignment of the edge
    direction with the local frame at the source node.

        w_ij = MLP( |cos θ_ij|, curvature_i )

    where cos θ_ij = (e_ij · d_i), e_ij = unit(x_j - x_i).
    Taking the absolute value makes the weight independent of frame
    orientation (gauge-invariant).
    """

    def __init__(self, curvature_dim: int = 4):
        super().__init__()
        # Inputs: alignment (1) + curvature_i (curvature_dim)
        self.weight_mlp = nn.Sequential(
            nn.Linear(1 + curvature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # weight in (0, 1) — we add 0.5 offset below
        )

    def forward(
        self,
        x: Tensor,              # (V, 3)
        d: Tensor,              # (V, 3) local frame vectors
        curvature: Tensor,      # (V, curvature_dim)
        edge_index: Tensor,     # (2, E)
    ) -> Tensor:
        """Returns w : (E, 1) edge anisotropy weights, centred around 1."""
        row, col = edge_index

        diff = x[col] - x[row]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit_diff = diff / dist   # (E, 3)

        # Alignment: absolute dot product with local frame at source
        alignment = (unit_diff * d[row]).sum(dim=-1, keepdim=True).abs()  # (E, 1)

        inp = torch.cat([alignment, curvature[row]], dim=-1)  # (E, 1+C)
        # Map sigmoid output [0,1] → [0.5, 1.5] so mean weight ≈ 1
        w = self.weight_mlp(inp) + 0.5   # (E, 1)
        return w


# ---------------------------------------------------------------------------
# Gauge-equivariant EGNN layer
# ---------------------------------------------------------------------------

class GaugeEGNNLayer(nn.Module):
    """
    Gauge-augmented EGNN layer.

    Extends EGNNLayer by multiplying edge messages (and coordinate updates)
    by a per-edge anisotropy weight derived from the local curvature frame.

    Interface: forward(h, x, edge_index, curvature, edge_attr=None)
    """

    def __init__(
        self,
        hidden_dim: int,
        curvature_dim: int = 4,
        edge_feat_dim: int = 0,
        act: str = "silu",
    ):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]

        # Standard EGNN MLPs
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Gauge-equivariant additions
        self.frame = AnisotropyFrame(hidden_dim, curvature_dim)
        self.aniso = AnisotropyWeight(curvature_dim)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        curvature: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h          : (V, hidden_dim) node features
            x          : (V, 3) positions
            edge_index : (2, E)
            curvature  : (V, curvature_dim) raw curvature scalars
            edge_attr  : optional (E, edge_feat_dim)
        Returns:
            h_new : (V, hidden_dim)
            x_new : (V, 3)
        """
        row, col = edge_index

        # --- Standard EGNN edge messages ---
        diff = x[row] - x[col]   # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden)

        # --- Gauge anisotropy weight ---
        d = self.frame(h, x, curvature, edge_index)          # (V, 3)
        w = self.aniso(x, d, curvature, edge_index)          # (E, 1)

        # Apply anisotropy weight to messages
        m_ij_weighted = m_ij * w     # (E, hidden)

        # --- Coordinate update (equivariant) ---
        dist = dist_sq.sqrt() + 1.0   # (E, 1)
        coord_weight = torch.tanh(self.coord_mlp(m_ij_weighted))  # (E, 1)
        coord_agg = torch.zeros_like(x).scatter_add(
            0, col.unsqueeze(-1).expand_as(diff), coord_weight * diff / dist
        )
        x = x + coord_agg

        # --- Node update ---
        msg_agg = torch.zeros(h.shape[0], m_ij_weighted.shape[-1], device=h.device).scatter_add(
            0, col.unsqueeze(-1).expand_as(m_ij_weighted), m_ij_weighted
        )
        h_new = self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        h = self.norm(h + h_new)

        return h, x


# ---------------------------------------------------------------------------
# Stacked GaugeEGNN
# ---------------------------------------------------------------------------

class GaugeEGNN(nn.Module):
    """
    Stacked GaugeEGNNLayers — a drop-in replacement for EGNN that leverages
    curvature features for anisotropic message passing.

    Args:
        in_node_dim   : dimensionality of raw input node features (e.g. 4)
        hidden_dim    : hidden/message dimension
        out_dim       : output node-feature dimension
        n_layers      : number of GaugeEGNNLayers
        curvature_dim : dimensionality of curvature features (default 4:
                        pv1, pv2, H, K). These are extracted from the first
                        `curvature_dim` columns of data.x.
        edge_feat_dim : optional edge-feature dimension
        act           : activation function name
    """

    def __init__(
        self,
        in_node_dim: int = 4,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_layers: int = 4,
        curvature_dim: int = 4,
        edge_feat_dim: int = 0,
        act: str = "silu",
    ):
        super().__init__()
        self.curvature_dim = curvature_dim

        self.embed = nn.Linear(in_node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GaugeEGNNLayer(hidden_dim, curvature_dim, edge_feat_dim, act)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            data : PyG Data with pos (V,3), edge_index (2,E), x (V, in_node_dim).
                   data.x must have at least curvature_dim columns for curvature.
            h    : optional pre-embedded features (V, hidden_dim).
        Returns:
            h_out : (V, out_dim)
            x_out : (V, 3)
        """
        x = data.pos
        edge_index = data.edge_index

        # Extract raw curvature features from data.x
        if hasattr(data, "x") and data.x is not None:
            curvature = data.x[:, :self.curvature_dim]   # (V, curvature_dim)
        else:
            curvature = torch.zeros(x.shape[0], self.curvature_dim, device=x.device)

        if h is None:
            if hasattr(data, "x") and data.x is not None:
                h = self.embed(data.x)
            else:
                h = self.embed(torch.ones(x.shape[0], self.embed.in_features, device=x.device))

        for layer in self.layers:
            h, x = layer(h, x, edge_index, curvature)

        return self.out_proj(h), x