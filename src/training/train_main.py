"""
Main training script: SE(3)-equivariant flow matching model.

Usage:
    python -m src.training.train_main --config configs/main_model.yaml
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

from src.models.encoder import GraphEncoder
from src.models.flow_matching import FlowMatchingModel


class CardiacGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir: str, manifest_entries: list[dict], struct: str = "lv"):
        self.graph_dir = Path(graph_dir)
        self.entries = [e for e in manifest_entries if e["struct"] == struct]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Data:
        entry = self.entries[idx]
        return torch.load(self.graph_dir / entry["file"], weights_only=False)


def make_noise_data(real: Data) -> Data:
    """Create a noise graph with same topology as real but Gaussian positions."""
    noise = real.clone()
    noise.pos = torch.randn_like(real.pos)
    return noise


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=cfg.get("wandb_project", "acdc-cardiac-diffusion"),
        name=cfg.get("wandb_run_name", "main-egnn-flow"),
        entity=cfg.get("wandb_entity") or None,
        config=cfg,
        tags=["main", "egnn", "flow-matching"],
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

    encoder = GraphEncoder(
        in_node_dim=cfg.get("in_node_dim", 1),
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_layers=cfg["n_encoder_layers"],
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
    kl_weight = cfg.get("kl_weight", 0.001)

    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train()
        flow_model.train()
        train_loss = train_cfm = train_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            noise_batch = make_noise_data(batch)

            mu, logvar = encoder(batch)
            z = GraphEncoder.reparameterise(mu, logvar)

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
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} (cfm {train_cfm:.4f} kl {train_kl:.4f}) | val {val_loss:.4f} | lr {lr_now:.2e}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/cfm_loss": train_cfm,
            "train/kl": train_kl,
            "val/loss": val_loss,
            "val/cfm_loss": val_cfm,
            "lr": lr_now,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "encoder": encoder.state_dict(),
                "flow_model": flow_model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "cfg": cfg,
            }, ckpt_path)
            wandb.save(str(ckpt_path))
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_model.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
