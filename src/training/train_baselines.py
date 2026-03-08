"""
Training script for all 4 baselines.

Usage:
    python -m src.training.train_baselines --config configs/baseline_mlp.yaml --baseline mlp
    python -m src.training.train_baselines --config configs/baseline_pca.yaml --baseline pca
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.training.train_main import CardiacGraphDataset, make_noise_data
from src.models.encoder import GraphEncoder


def train_pca(cfg: dict):
    from src.models.baselines.pca_model import PCAShapeModel

    graph_dir = Path(cfg["graph_dir"])
    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = cfg.get("struct", "lv")
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    shapes = []
    for e in entries:
        data = torch.load(graph_dir / e["file"], weights_only=False)
        shapes.append(data.pos.numpy())
    shapes = np.stack(shapes)

    model = PCAShapeModel(n_components=cfg.get("n_components", 50))
    model.fit(shapes)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"PCA model saved. Explained variance (top-10): {model.explained_variance_ratio()[:10].sum():.3f}")


def train_neural(cfg: dict, baseline: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_dir = Path(cfg["graph_dir"])
    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = cfg.get("struct", "lv")
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    split = int(0.8 * len(entries))
    train_ds = CardiacGraphDataset(graph_dir, entries[:split], struct)
    val_ds = CardiacGraphDataset(graph_dir, entries[split:], struct)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    if baseline == "mlp":
        from src.models.baselines.mlp_diffusion import MLPFlowMatchingModel
        from src.models.encoder import GraphEncoder

        encoder = GraphEncoder(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            n_layers=cfg["n_encoder_layers"],
        ).to(device)
        flow_model = MLPFlowMatchingModel(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            n_layers=cfg["n_flow_layers"],
        ).to(device)
        params = list(encoder.parameters()) + list(flow_model.parameters())

    elif baseline == "pointnet":
        from src.models.baselines.pointnet_ae import PointNetAE
        model = PointNetAE(latent_dim=cfg["latent_dim"], num_points=cfg.get("num_points", 2500)).to(device)
        params = list(model.parameters())

    elif baseline == "voxel":
        from src.models.baselines.voxel_diffusion import VoxelFlowMatching
        model = VoxelFlowMatching(base_ch=cfg.get("base_ch", 32)).to(device)
        params = list(model.parameters())

    optimizer = AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            if baseline == "mlp":
                encoder.train(); flow_model.train()
                mu, logvar = encoder(batch)
                z = GraphEncoder.reparameterise(mu, logvar)
                noise = make_noise_data(batch)
                loss = flow_model(noise, batch, z)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
                loss = loss + cfg.get("kl_weight", 0.001) * kl

            elif baseline == "pointnet":
                model.train()
                out = model(batch)
                loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if baseline == "mlp":
                    encoder.eval(); flow_model.eval()
                    mu, _ = encoder(batch)
                    noise = make_noise_data(batch)
                    loss = flow_model(noise, batch, mu)
                elif baseline == "pointnet":
                    model.eval()
                    loss = model(batch)["loss"]
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"[{baseline}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_dict = {"epoch": epoch, "cfg": cfg}
            if baseline == "mlp":
                save_dict.update({"encoder": encoder.state_dict(), "flow_model": flow_model.state_dict()})
            else:
                save_dict["model"] = model.state_dict()
            torch.save(save_dict, out_dir / f"best_{baseline}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True, choices=["pca", "voxel", "pointnet", "mlp"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.baseline == "pca":
        train_pca(cfg)
    else:
        train_neural(cfg, args.baseline)


if __name__ == "__main__":
    main()
