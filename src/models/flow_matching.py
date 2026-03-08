"""
Conditional Flow Matching over cardiac mesh vertex positions.

We treat vertex coordinates (V, 3) as the data space.
The velocity field v_θ(x_t, t, z) is parameterised by an EGNN,
where z is a conditioning latent code from GraphEncoder.

Training objective (optimal transport CFM):
    L = E_{t, x_0, x_1} || v_θ(x_t, t, z) - (x_1 - x_0) ||^2
    x_t = (1-t)*x_0 + t*x_1,  t ~ U[0,1]

Sampling uses Euler / RK4 integration from x_0 ~ N(0,I).

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from src.models.egnn import EGNN


class VelocityField(nn.Module):
    """EGNN-based velocity field v_θ(pos, t, z) → velocity on each vertex."""

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
        time_dim: int = 16,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Node features: time embedding + latent code projected to per-node
        self.cond_proj = nn.Linear(latent_dim + time_dim, hidden_dim)

        self.egnn = EGNN(
            in_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=3,  # velocity in R^3
            n_layers=n_layers,
        )

    def forward(self, data: Data, t: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            data : PyG Data with pos (V, 3) and edge_index. pos = x_t.
            t    : (B,) or scalar time in [0, 1].
            z    : (B, latent_dim) conditioning latent codes.
        Returns:
            velocity : (V, 3)
        """
        V = data.pos.shape[0]
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(V, dtype=torch.long, device=data.pos.device)

        # Broadcast time and latent to per-node
        t_node = t[batch] if t.ndim > 0 else t.expand(V)
        t_emb = self.time_embed(t_node.unsqueeze(-1))  # (V, time_dim)
        z_node = z[batch]  # (V, latent_dim)

        h = self.cond_proj(torch.cat([z_node, t_emb], dim=-1))  # (V, hidden)

        _, velocity = self.egnn(data, h=h)
        # egnn returns updated positions; we want the delta (velocity)
        velocity = velocity - data.pos  # treat position update as velocity
        return velocity


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.velocity_field = VelocityField(hidden_dim, latent_dim, n_layers)

    def forward(self, data_0: Data, data_1: Data, z: Tensor) -> Tensor:
        """
        Compute CFM training loss.

        data_0 : noise sample (Gaussian-initialised positions on same graph)
        data_1 : real cardiac mesh
        z      : conditioning latent (B, latent_dim)
        """
        t = torch.rand(1, device=data_1.pos.device)

        x_0 = data_0.pos
        x_1 = data_1.pos
        x_t = (1 - t) * x_0 + t * x_1

        target_velocity = x_1 - x_0  # straight-line OT path

        # Build interpolated data object
        data_t = data_1.clone()
        data_t.pos = x_t

        v_pred = self.velocity_field(data_t, t.expand(data_t.pos.shape[0]), z)
        loss = ((v_pred - target_velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def sample(self, data_template: Data, z: Tensor, steps: int = 100, method: str = "euler") -> Tensor:
        """
        Sample new mesh positions by integrating the velocity field from t=0 to t=1.

        data_template : graph structure (edge_index, faces) — positions will be overwritten
        z             : (B, latent_dim) conditioning code
        Returns:
            x_1 : (V, 3) generated vertex positions
        """
        x = torch.randn_like(data_template.pos)
        dt = 1.0 / steps
        data = data_template.clone()

        for i in range(steps):
            t_val = torch.tensor(i * dt, device=x.device)
            data.pos = x
            v = self.velocity_field(data, t_val.expand(x.shape[0]), z)

            if method == "euler":
                x = x + dt * v
            elif method == "rk4":
                data.pos = x
                k1 = self.velocity_field(data, t_val.expand(x.shape[0]), z)
                data.pos = x + 0.5 * dt * k1
                k2 = self.velocity_field(data, (t_val + 0.5 * dt).expand(x.shape[0]), z)
                data.pos = x + 0.5 * dt * k2
                k3 = self.velocity_field(data, (t_val + 0.5 * dt).expand(x.shape[0]), z)
                data.pos = x + dt * k3
                k4 = self.velocity_field(data, (t_val + dt).expand(x.shape[0]), z)
                x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
