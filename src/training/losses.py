"""Shared loss functions used across models."""

import torch
import torch.nn.functional as F
from torch import Tensor


def chamfer_distance(pred: Tensor, target: Tensor) -> Tensor:
    """
    Symmetric Chamfer distance.
    pred, target: (N, 3) or (B, N, 3)
    Returns scalar mean distance (in same units as input coordinates).
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    dist = torch.cdist(pred, target)  # (B, N, M)
    d1 = dist.min(dim=2).values.mean()   # pred → target
    d2 = dist.min(dim=1).values.mean()   # target → pred
    return (d1 + d2) / 2


def flow_matching_loss(v_pred: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
    """OT-CFM loss: MSE between predicted velocity and straight-line target."""
    target = x_1 - x_0
    return F.mse_loss(v_pred, target)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(q || N(0,I)) for VAE regularisation."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()


def surface_normal_consistency(pred_verts: Tensor, faces: Tensor) -> Tensor:
    """
    Penalise inconsistent face normals (smoothness prior).
    pred_verts: (V, 3), faces: (F, 3)
    """
    v0 = pred_verts[faces[:, 0]]
    v1 = pred_verts[faces[:, 1]]
    v2 = pred_verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = F.normalize(normals, dim=-1)
    # Penalise deviation from mean normal (global smoothness)
    mean_normal = normals.mean(0, keepdim=True)
    return (1 - (normals * mean_normal).sum(dim=-1)).mean()
