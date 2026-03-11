"""
Training script for the X-Eigenvector EGNN flow-matching model.

Identical interface to train_gauge.py except:
  - Uses XEigvecEncoder (XEigvecEGNN backbone) instead of GaugeGraphEncoder
  - Loads graphs from data/graphs_curvature/ rebuilt with the updated
    add_curvature.py that stores data.principal_dir1 / data.principal_dir2
  - Checkpoint cfg includes model_type='x_eigvec' for eval scripts
  - in_node_dim defaults to 5 (k1, k2, H, K, chi)

Usage:
    python -m src.training.train_x_egnn --config configs/x_egnn_model.yaml

    # or override individual keys:
    python -m src.training.train_x_egnn --config configs/x_egnn_model.yaml \\
        --set kl_weight=0.005 out_dir=checkpoints/x_egnn_kl5e-3 \\
               wandb_run_name=x-eigvec-kl5e-3
"""

import argparse
import json
from pathlib import Path

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.x_encoder import XEigvecEncoder
from src.models.flow_matching import FlowMatchingModel


class CardiacGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir: str, manifest_entries: list[dict], struct: str = "lv"):
        self.graph_dir = Path(graph_dir)
        self.entries = [e for e in manifest_entries if e["struct"] == struct]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Data:
        entry = self.entries[idx]
        data = torch.load(self.graph_dir / entry["file"], weights_only=False)
        # Normalise positions to zero-mean unit-std so CFM loss is O(1)
        data.pos = (data.pos - data.pos.mean(0)) / (data.pos.std() + 1e-6)
        # principal_dir1/dir2 are direction vectors: normalisation doesn't affect them
        return data


def make_noise_data(real: Data) -> Data:
    """Create a noise graph with same topology as real but Gaussian positions."""
    noise = real.clone()
    noise.pos = torch.randn_like(real.pos)
    return noise


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = cfg.get("wandb_run_name", "x-eigvec-egnn-flow")
    if cfg.get("_name_suffix"):
        run_name = f"{run_name}-{cfg['_name_suffix']}"

    run = wandb.init(
        project=cfg.get("wandb_project", "acdc-cardiac-diffusion"),
        name=run_name,
        entity=cfg.get("wandb_entity") or None,
        config=cfg,
        tags=["x-eigvec", "egnn", "flow-matching", "curvature", "eigenvectors"],
    )
    print(f"W&B run: {run.url}")
    print(f"Training on {device}")

    graph_dir = Path(cfg["graph_dir"])
    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = cfg.get("struct", "lv")
    train_entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    split = int(0.8 * len(train_entries))
    train_ds = CardiacGraphDataset(graph_dir, train_entries[:split], struct)
    val_ds = CardiacGraphDataset(graph_dir, train_entries[split:], struct)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    in_node_dim = cfg.get("in_node_dim", 5)
    curvature_dim = cfg.get("curvature_dim", 4)

    encoder = XEigvecEncoder(
        in_node_dim=in_node_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_layers=cfg["n_encoder_layers"],
        curvature_dim=curvature_dim,
    ).to(device)

    flow_model = FlowMatchingModel(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_layers=cfg["n_flow_layers"],
    ).to(device)

    params = list(encoder.parameters()) + list(flow_model.parameters())
    optimizer = AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    wandb.watch([encoder, flow_model], log="gradients", log_freq=50)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    kl_weight_max = cfg.get("kl_weight", 0.001)
    kl_anneal_epochs = cfg.get("kl_anneal_epochs", 0)

    for epoch in range(1, cfg["epochs"] + 1):
        if kl_anneal_epochs > 0:
            kl_weight = kl_weight_max * min(1.0, epoch / kl_anneal_epochs)
        else:
            kl_weight = kl_weight_max

        encoder.train()
        flow_model.train()
        train_loss = train_cfm = train_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            noise_batch = make_noise_data(batch)

            mu, logvar = encoder(batch)
            z = XEigvecEncoder.reparameterise(mu, logvar)

            cfm_loss = flow_model(noise_batch, batch, z)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            loss = cfm_loss + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_cfm += cfm_loss.item()
            train_kl += kl.item()

        scheduler.step()
        n = len(train_loader)
        train_loss /= n
        train_cfm /= n
        train_kl /= n

        # Validation
        encoder.eval()
        flow_model.eval()
        val_loss = val_cfm = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                noise_batch = make_noise_data(batch)
                mu, logvar = encoder(batch)
                cfm_loss = flow_model(noise_batch, batch, mu)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
                val_loss += (cfm_loss + kl_weight * kl).item()
                val_cfm += cfm_loss.item()
        val_n = max(len(val_loader), 1)
        val_loss /= val_n
        val_cfm /= val_n

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} "
            f"(cfm {train_cfm:.4f} kl {train_kl:.4f}) | "
            f"val {val_loss:.4f} | lr {lr_now:.2e} | kl_w {kl_weight:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/cfm_loss": train_cfm,
            "train/kl": train_kl,
            "val/loss": val_loss,
            "val/cfm_loss": val_cfm,
            "lr": lr_now,
            "kl_weight": kl_weight,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pt"
            save_cfg = dict(cfg)
            save_cfg["model_type"] = "x_eigvec"
            torch.save({
                "encoder": encoder.state_dict(),
                "flow_model": flow_model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "cfg": save_cfg,
            }, ckpt_path)
            wandb.save(str(ckpt_path))
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/x_egnn_model.yaml")
    parser.add_argument(
        "--set", nargs="*", metavar="KEY=VALUE",
        help="Override config values, e.g. --set kl_weight=0.01 out_dir=checkpoints/exp1"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for item in (args.set or []):
        key, _, raw = item.partition("=")
        for cast in (int, float):
            try:
                raw = cast(raw)
                break
            except ValueError:
                pass
        cfg[key] = raw

    train(cfg)


if __name__ == "__main__":
    main()
