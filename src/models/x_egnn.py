"""
Explicit Eigenvector EGNN (XEigvecEGNN).

Motivation
----------
GaugeEGNN constructs a local frame d_i by learning a weighted combination of
neighbouring relative positions — equivariant but an approximate proxy for the
true curvature frame.

XEigvecEGNN instead uses the *exact* principal-curvature eigenvectors
(pd1_i, pd2_i) stored in data.principal_dir1 / data.principal_dir2.  These
are the eigenvectors of the shape operator, computed by libigl and stored
during preprocessing (add_curvature.py).

Key ideas
---------
1. 2-D local frame at each node i:
       pd1_i  — direction of maximum principal curvature  (equivariant)
       pd2_i  — direction of minimum principal curvature  (equivariant)

2. For each edge (i → j), project the unit edge vector onto the frame:
       c1_ij = e_ij · pd1_i   (signed; invariant scalar)
       c2_ij = e_ij · pd2_i   (signed; invariant scalar)

3. Anisotropy weight (per edge, invariant scalar):
       w_ij = MLP(|c1|, |c2|, c1·k1_i, c2·k2_i, curvature_i, dist_ij) + 0.5
   Sigmoid-offset so w_ij ∈ (0.5, 1.5), mean ≈ 1.

4. Equivariant coordinate update — three independent contributions:
       Δx_i = Σ_j [α_base_ij · diff / dist_norm   (standard EGNN term)
                  + α1_ij    · c1_ij · pd1_i       (along principal dir 1)
                  + α2_ij    · c2_ij · pd2_i ]     (along principal dir 2)
   Because c1, c2 are invariant scalars and pd1, pd2 are equivariant vectors,
   every term is SE(3)-equivariant.

Equivariance proof sketch
--------------------------
Under a global rotation R and translation t:
   diff = x_j - x_i  →  R · diff       (equivariant, translation-invariant)
   pd1_i             →  R · pd1_i      (equivariant — stored and co-rotates)
   c1_ij = e_ij · pd1_i  is *invariant*  (dot product of two equivariant)
   α_base, α1, α2, w  are all invariant  (derived from invariant inputs)
   Δx_i              →  R · Δx_i       (equivariant) ✓
   x_i + t + Δx_i   →  R(x_i + Δx_i) + t ✓

Comparison with GaugeEGNN
--------------------------
| Property             | GaugeEGNN              | XEigvecEGNN              |
|----------------------|------------------------|--------------------------|
| Local frame          | Learned proxy (d_i)    | Exact curvature eigenvecs|
| Frame dimensions     | 1-D                    | 2-D (pd1 + pd2)          |
| Coord update terms   | 1 (diff only)          | 3 (diff + pd1 + pd2)     |
| Extra data required  | None                   | principal_dir1/dir2      |
| Parameters           | +AnisotropyFrame MLP   | −frame MLP, +2 coord MLPs|

Requires graphs built with the updated add_curvature.py that stores
data.principal_dir1 (V, 3) and data.principal_dir2 (V, 3) in addition
to data.x = [k1, k2, H, K, chi] (V, 5).

References
----------
- Satorras et al., E(n) Equivariant Graph Neural Networks, ICML 2021.
- de Haan et al., Gauge Equivariant Mesh CNNs, ICML 2021.
- Weiler et al., 3D Steerable CNNs, NeurIPS 2018.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Eigenvector anisotropy module
# ---------------------------------------------------------------------------

class EigvecAnisotropy(nn.Module):
    """
    Compute a per-edge anisotropy weight from the 2-D principal-curvature frame.

    For edge (i → j):
        c1_ij = e_ij · pd1_i   — alignment with max-curvature direction
        c2_ij = e_ij · pd2_i   — alignment with min-curvature direction

    Input to MLP: [|c1|, |c2|, c1·k1_i, c2·k2_i, curvature_i, dist_ij]
    Output: w ∈ (0.5, 1.5)  (Sigmoid + 0.5 offset)

    Returns w, c1, c2 so the caller can reuse c1/c2 for coordinate updates.
    """

    def __init__(self, curvature_dim: int = 4):
        super().__init__()
        # 4 frame-derived scalars + curvature_dim scalars + 1 distance
        in_dim = 4 + curvature_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        e_ij: Tensor,       # (E, 3) unit edge vectors
        pd1: Tensor,        # (V, 3) max-curvature directions
        pd2: Tensor,        # (V, 3) min-curvature directions
        curvature: Tensor,  # (V, C) curvature scalars
        dist: Tensor,       # (E, 1) edge lengths
        row: Tensor,        # (E,) source node indices
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            w   : (E, 1) anisotropy weight in (0.5, 1.5)
            c1  : (E, 1) signed projection onto pd1
            c2  : (E, 1) signed projection onto pd2
        """
        c1 = (e_ij * pd1[row]).sum(dim=-1, keepdim=True)  # (E, 1)
        c2 = (e_ij * pd2[row]).sum(dim=-1, keepdim=True)  # (E, 1)
        k1 = curvature[row, 0:1]  # (E, 1) max principal curvature
        k2 = curvature[row, 1:2]  # (E, 1) min principal curvature

        inp = torch.cat([
            c1.abs(), c2.abs(),      # magnitude of alignment
            c1 * k1,  c2 * k2,      # curvature-weighted projections
            curvature[row],          # all curvature scalars
            dist,                    # edge length
        ], dim=-1)
        w = self.mlp(inp) + 0.5    # (E, 1) in (0.5, 1.5)
        return w, c1, c2


# ---------------------------------------------------------------------------
# XEigvecEGNN layer
# ---------------------------------------------------------------------------

class XEigvecEGNNLayer(nn.Module):
    """
    EGNN layer that uses exact principal-curvature eigenvectors for
    anisotropic, equivariant message passing.

    Compared with GaugeEGNNLayer:
    - No AnisotropyFrame (uses pd1, pd2 directly from data).
    - 2-D frame: separate coord-update paths along pd1 and pd2.
    - Three equivariant coordinate-update contributions:
        diff/dist (standard), c1·pd1 (max-curv), c2·pd2 (min-curv).

    Interface: forward(h, x, edge_index, curvature, pd1, pd2, edge_attr=None)
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

        # Standard EGNN edge MLP: (h_i, h_j, dist^2, [edge_feats]) → m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )

        # Three independent coordinate-update heads (outputs tanh-bounded scalars)
        self.coord_mlp_base = nn.Sequential(   # standard diff/dist term
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.coord_mlp_pd1 = nn.Sequential(    # along max-curvature direction
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        self.coord_mlp_pd2 = nn.Sequential(    # along min-curvature direction
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Anisotropy weight from exact eigenvectors
        self.aniso = EigvecAnisotropy(curvature_dim)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        curvature: Tensor,
        pd1: Tensor,
        pd2: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h          : (V, hidden_dim) node features
            x          : (V, 3) positions
            edge_index : (2, E)
            curvature  : (V, curvature_dim) curvature scalars [k1, k2, H, K, ...]
            pd1        : (V, 3) max-curvature direction (equivariant, unit)
            pd2        : (V, 3) min-curvature direction (equivariant, unit)
            edge_attr  : optional (E, edge_feat_dim)
        Returns:
            h_new : (V, hidden_dim)
            x_new : (V, 3)
        """
        row, col = edge_index

        # --- Edge geometry ---
        diff = x[col] - x[row]                                  # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)         # (E, 1)
        dist = dist_sq.sqrt().clamp(min=1e-6)                    # (E, 1)
        e_ij = diff / dist                                       # (E, 3) unit edge vec

        # --- Standard EGNN edge message ---
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)                        # (E, hidden)

        # --- Anisotropy weight from eigenvector frame ---
        w, c1, c2 = self.aniso(e_ij, pd1, pd2, curvature, dist, row)
        m_ij_w = m_ij * w                                       # (E, hidden)

        # --- Equivariant coordinate update (3 contributions) ---
        dist_norm = dist + 1.0                                   # (E, 1) avoids /0

        # 1. Standard EGNN: along the edge direction
        alpha_base = torch.tanh(self.coord_mlp_base(m_ij_w))   # (E, 1)
        delta_base = alpha_base * diff / dist_norm              # (E, 3)

        # 2. Along max-curvature direction (pd1): magnitude set by edge's c1
        alpha_pd1 = torch.tanh(self.coord_mlp_pd1(m_ij_w))     # (E, 1)
        delta_pd1 = alpha_pd1 * c1 * pd1[row]                  # (E, 3) equivariant

        # 3. Along min-curvature direction (pd2): magnitude set by edge's c2
        alpha_pd2 = torch.tanh(self.coord_mlp_pd2(m_ij_w))     # (E, 2)
        delta_pd2 = alpha_pd2 * c2 * pd2[row]                  # (E, 3) equivariant

        coord_delta = delta_base + delta_pd1 + delta_pd2        # (E, 3)
        coord_agg = torch.zeros_like(x).scatter_add(
            0, col.unsqueeze(-1).expand_as(coord_delta), coord_delta
        )
        x = x + coord_agg

        # --- Node update ---
        msg_agg = torch.zeros(
            h.shape[0], m_ij_w.shape[-1], device=h.device
        ).scatter_add(
            0, col.unsqueeze(-1).expand_as(m_ij_w), m_ij_w
        )
        h_new = self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        h = self.norm(h + h_new)

        return h, x


# ---------------------------------------------------------------------------
# Stacked XEigvecEGNN
# ---------------------------------------------------------------------------

class XEigvecEGNN(nn.Module):
    """
    Stacked XEigvecEGNNLayers — uses exact curvature eigenvectors for
    fully 2-D anisotropic, SE(3)-equivariant message passing.

    Drop-in replacement for GaugeEGNN that requires graphs built with the
    updated add_curvature.py (data.principal_dir1, data.principal_dir2).
    Falls back gracefully to isotropic updates if the attributes are absent
    (pd1 = pd2 = 0 → c1 = c2 = 0 → pd1/pd2 coord terms vanish).

    Args:
        in_node_dim   : raw input node-feature dimension
                        (default 5: [k1, k2, H, K, chi])
        hidden_dim    : message / hidden dimension
        out_dim       : output node-feature dimension
        n_layers      : number of XEigvecEGNNLayers
        curvature_dim : number of curvature channels used for the anisotropy
                        weight (taken from data.x[:, :curvature_dim])
        edge_feat_dim : optional edge-feature dimension
        act           : activation name
    """

    def __init__(
        self,
        in_node_dim: int = 5,
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
            XEigvecEGNNLayer(hidden_dim, curvature_dim, edge_feat_dim, act)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            data : PyG Data with:
                     pos            (V, 3)
                     edge_index     (2, E)
                     x              (V, in_node_dim)  curvature scalars
                     principal_dir1 (V, 3)  max-curvature eigenvector
                     principal_dir2 (V, 3)  min-curvature eigenvector
            h    : optional pre-embedded features (V, hidden_dim)
        Returns:
            h_out : (V, out_dim)
            x_out : (V, 3)
        """
        x = data.pos
        edge_index = data.edge_index

        # Curvature scalar features
        if hasattr(data, "x") and data.x is not None:
            curvature = data.x[:, :self.curvature_dim]     # (V, curvature_dim)
        else:
            curvature = torch.zeros(x.shape[0], self.curvature_dim, device=x.device)

        # Principal direction vectors (equivariant, co-rotate with point cloud)
        if hasattr(data, "principal_dir1") and data.principal_dir1 is not None:
            pd1 = data.principal_dir1   # (V, 3)
            pd2 = data.principal_dir2   # (V, 3)
        else:
            # Graceful fallback: zero → pd1/pd2 coord terms vanish, w → 0.5+const
            pd1 = torch.zeros_like(x)
            pd2 = torch.zeros_like(x)

        # Re-normalise (igl returns unit vectors, but batching may not preserve norms)
        pd1_norm = pd1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pd2_norm = pd2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pd1 = pd1 / pd1_norm
        pd2 = pd2 / pd2_norm

        # Initial node embedding
        if h is None:
            if hasattr(data, "x") and data.x is not None:
                h = self.embed(data.x)
            else:
                h = self.embed(torch.ones(x.shape[0], self.embed.in_features, device=x.device))

        for layer in self.layers:
            h, x = layer(h, x, edge_index, curvature, pd1, pd2)

        return self.out_proj(h), x
