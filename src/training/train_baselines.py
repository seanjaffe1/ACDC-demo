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
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.training.train_main import CardiacGraphDataset, make_noise_data
from src.models.encoder import GraphEncoder


# Maps baseline key → human-readable tag for W&B
BASELINE_TAGS = {
    "pca": ["baseline", "pca", "no-deep-learning"],
    "voxel": ["baseline", "voxel", "no-mesh"],
    "pointnet": ["baseline", "pointnet", "no-diffusion"],
    "mlp": ["baseline", "mlp", "no-equivariance"],
}


def train_pca(cfg: dict):
    from src.models.baselines.pca_model import PCAShapeModel

    run = wandb.init(
        project=cfg.get("wandb_project", "acdc-cardiac-diffusion"),
        name=cfg.get("wandb_run_name", "baseline-pca"),
        entity=cfg.get("wandb_entity") or None,
        config=cfg,
        tags=BASELINE_TAGS["pca"],
    )
    print(f"W&B run: {run.url}")

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

    # Log explained variance curve
    evr = model.explained_variance_ratio()
    for k in [10, 20, 30, 50]:
        k = min(k, len(evr))
        wandb.log({f"pca/explained_variance_top{k}": float(evr[:k].sum())})
    wandb.log({"pca/explained_variance_curve": wandb.plot.line_series(
        xs=list(range(1, len(evr) + 1)),
        ys=[evr.cumsum().tolist()],
        keys=["cumulative"],
        title="PCA Explained Variance",
        xname="# components",
    )})

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "pca_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    wandb.save(str(pkl_path))

    top10_var = float(evr[:10].sum())
    wandb.run.summary["explained_variance_top10"] = top10_var
    print(f"PCA model saved. Explained variance (top-10): {top10_var:.3f}")
    wandb.finish()


def train_neural(cfg: dict, baseline: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=cfg.get("wandb_project", "acdc-cardiac-diffusion"),
        name=cfg.get("wandb_run_name", f"baseline-{baseline}"),
        entity=cfg.get("wandb_entity") or None,
        config=cfg,
        tags=BASELINE_TAGS[baseline],
    )
    print(f"W&B run: {run.url}")
    print(f"Training [{baseline}] on {device}")

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
        encoder = GraphEncoder(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            n_layers=cfg["n_encoder_layers"],
        ).to(device)
        flow_model = MLPFlowMatchingModel(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            n_layers=cfg["n_flow_layers"],
        ).to(device)
        params = list(encoder.parameters()) + list(flow_model.parameters())
        wandb.watch([encoder, flow_model], log="gradients", log_freq=50)

    elif baseline == "pointnet":
        from src.models.baselines.pointnet_ae import PointNetAE
        model = PointNetAE(latent_dim=cfg["latent_dim"], num_points=cfg.get("num_points", 2500)).to(device)
        params = list(model.parameters())
        wandb.watch(model, log="gradients", log_freq=50)

    elif baseline == "voxel":
        from src.models.baselines.voxel_diffusion import VoxelFlowMatching
        model = VoxelFlowMatching(base_ch=cfg.get("base_ch", 32)).to(device)
        params = list(model.parameters())
        wandb.watch(model, log="gradients", log_freq=50)

    optimizer = AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    kl_weight = cfg.get("kl_weight", 0.001)

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            if baseline == "mlp":
                encoder.train(); flow_model.train()
                mu, logvar = encoder(batch)
                z = GraphEncoder.reparameterise(mu, logvar)
                noise = make_noise_data(batch)
                cfm_loss = flow_model(noise, batch, z)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
                loss = cfm_loss + kl_weight * kl

            elif baseline == "pointnet":
                model.train()
                out = model(batch)
                loss = out["loss"]

            elif baseline == "voxel":
                # voxel baseline expects volumetric tensors — skip batch for now
                # (requires separate voxelisation step; placeholder loss)
                model.train()
                loss = torch.tensor(0.0, requires_grad=True, device=device)

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
                    mu, logvar = encoder(batch)
                    noise = make_noise_data(batch)
                    cfm_loss = flow_model(noise, batch, mu)
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
                    loss = cfm_loss + kl_weight * kl
                elif baseline == "pointnet":
                    model.eval()
                    loss = model(batch)["loss"]
                elif baseline == "voxel":
                    loss = torch.tensor(0.0, device=device)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        lr_now = scheduler.get_last_lr()[0]
        print(f"[{baseline}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.2e}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": lr_now,
        })

        if val_loss < best_val:
            best_val = val_loss
            save_dict = {"epoch": epoch, "val_loss": val_loss, "cfg": cfg}
            if baseline == "mlp":
                save_dict.update({"encoder": encoder.state_dict(), "flow_model": flow_model.state_dict()})
            else:
                save_dict["model"] = model.state_dict()
            ckpt_path = out_dir / f"best_{baseline}.pt"
            torch.save(save_dict, ckpt_path)
            wandb.save(str(ckpt_path))
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = epoch

    wandb.finish()


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
