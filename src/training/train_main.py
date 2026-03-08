"""
Main training script: SE(3)-equivariant flow matching model.

Usage:
    python -m src.training.train_main --config configs/main_model.yaml
"""

import argparse
import json
from pathlib import Path

import torch
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
    print(f"Training on {device}")

    # Load manifest and split
    graph_dir = Path(cfg["graph_dir"])
    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = cfg.get("struct", "lv")
    train_entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    # Simple 80/20 split by index
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

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train()
        flow_model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            noise_batch = make_noise_data(batch)

            mu, logvar = encoder(batch)
            z = GraphEncoder.reparameterise(mu, logvar)

            loss = flow_model(noise_batch, batch, z)
            # Optional KL regularisation
            kl_weight = cfg.get("kl_weight", 0.001)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            loss = loss + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        encoder.eval()
        flow_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                noise_batch = make_noise_data(batch)
                mu, _ = encoder(batch)
                loss = flow_model(noise_batch, batch, mu)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "flow_model": flow_model.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }, out_dir / "best_model.pt")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_model.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
