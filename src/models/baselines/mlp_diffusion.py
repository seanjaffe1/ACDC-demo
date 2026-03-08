"""
Baseline 4: Flow matching with MLP message passing (no SE(3) equivariance).

Justifies geometric deep learning — same architecture as the main model
but EGNN equivariant layers are replaced with standard MLP node updates.
Training uses random rotation augmentation to compensate.

This is the CRITICAL baseline: if the equivariant model doesn't clearly
outperform this one (especially in low-data regimes), the geometric
prior is not justified.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing


class MLPMessagePassing(MessagePassing):
    """Standard message passing with MLP — no equivariance."""

    def __init__(self, hidden_dim: int, act: str = "silu"):
        super().__init__(aggr="add")
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU}[act]

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),  # concat h_i, h_j, x_i-x_j
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        agg = self.propagate(edge_index, h=h, pos=pos)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)

    def message(self, h_i: Tensor, h_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        return self.edge_mlp(torch.cat([h_i, h_j, pos_i - pos_j], dim=-1))


class MLPVelocityField(nn.Module):
    """MLP-based velocity field: no geometric equivariance."""

    def __init__(self, hidden_dim: int = 128, latent_dim: int = 64, n_layers: int = 4, time_dim: int = 16):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Linear(latent_dim + time_dim, hidden_dim)
        self.layers = nn.ModuleList([MLPMessagePassing(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, 3)

    def forward(self, data: Data, t: Tensor, z: Tensor) -> Tensor:
        V = data.pos.shape[0]
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(V, dtype=torch.long, device=data.pos.device)

        t_node = t[batch] if t.ndim > 0 else t.expand(V)
        t_emb = self.time_embed(t_node.unsqueeze(-1))
        z_node = z[batch]
        h = self.cond_proj(torch.cat([z_node, t_emb], dim=-1))

        for layer in self.layers:
            h = layer(h, data.pos, data.edge_index)

        return self.out_proj(h)


class MLPFlowMatchingModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.velocity_field = MLPVelocityField(hidden_dim, latent_dim, n_layers)

    def forward(self, data_0: Data, data_1: Data, z: Tensor) -> Tensor:
        t = torch.rand(1, device=data_1.pos.device)

        x_0 = data_0.pos
        x_1 = data_1.pos

        # Random rotation augmentation — compensation for lack of equivariance
        if self.training:
            R = random_rotation(device=x_1.device)
            x_0 = x_0 @ R.T
            x_1 = x_1 @ R.T

        x_t = (1 - t) * x_0 + t * x_1
        target = x_1 - x_0

        data_t = data_1.clone()
        data_t.pos = x_t

        v_pred = self.velocity_field(data_t, t.expand(V := data_t.pos.shape[0]), z)
        return ((v_pred - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, data_template: Data, z: Tensor, steps: int = 100) -> Tensor:
        x = torch.randn_like(data_template.pos)
        dt = 1.0 / steps
        data = data_template.clone()
        for i in range(steps):
            t_val = torch.tensor(i * dt, device=x.device)
            data.pos = x
            v = self.velocity_field(data, t_val.expand(x.shape[0]), z)
            x = x + dt * v
        return x


def random_rotation(device: torch.device) -> Tensor:
    """Sample a random SO(3) rotation matrix."""
    q = torch.randn(4, device=device)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)]),
        torch.stack([    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)]),
        torch.stack([    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]),
    ])
    return R
