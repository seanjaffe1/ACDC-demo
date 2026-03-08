"""
Evaluate distribution quality:
  - 1-NNA: are generated shapes indistinguishable from real ones?
  - Pathology separation: linear probe on latent codes across 5 ACDC classes

Usage:
    python -m src.evaluation.eval_generation \
        --graph_dir data/graphs \
        --checkpoint checkpoints/main/best_model.pt \
        --out results/generation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.evaluation.metrics import one_nearest_neighbour_accuracy, linear_probe_accuracy
from src.training.train_main import CardiacGraphDataset


def collect_latents_and_labels(encoder, loader, device) -> tuple[np.ndarray, np.ndarray]:
    latents, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = encoder.encode(batch)
            latents.append(z.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    return np.concatenate(latents), np.concatenate(labels)


def generate_shapes(encoder, flow_model, loader, device, steps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    real_shapes, gen_shapes = [], []
    encoder.eval(); flow_model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = encoder.encode(batch)
            template = batch.clone()
            template.pos = torch.randn_like(batch.pos)
            gen_pos = flow_model.sample(template, z, steps=steps)
            # Flatten per-graph shapes
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                real_shapes.append(batch.pos[mask].cpu().numpy().flatten())
                gen_shapes.append(gen_pos[mask].cpu().numpy().flatten())
    return np.stack(real_shapes), np.stack(gen_shapes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--checkpoint", default="checkpoints/main/best_model.pt")
    parser.add_argument("--struct", default="lv")
    parser.add_argument("--out", default="results/generation.json")
    parser.add_argument("--n_generate", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = Path(args.graph_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.models.encoder import GraphEncoder
    from src.models.flow_matching import FlowMatchingModel

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    encoder = GraphEncoder(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_encoder_layers"]).to(device)
    flow_model = FlowMatchingModel(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_flow_layers"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == args.struct]
    ds = CardiacGraphDataset(graph_dir, entries, args.struct)
    loader = DataLoader(ds, batch_size=4)

    # 1-NNA
    real_shapes, gen_shapes = generate_shapes(encoder, flow_model, loader, device)
    nna = one_nearest_neighbour_accuracy(real_shapes, gen_shapes)

    # Pathology separation
    latents, labels = collect_latents_and_labels(encoder, loader, device)
    probe_acc = linear_probe_accuracy(latents, labels, n_classes=5)

    results = {
        "1NNA": float(nna),
        "1NNA_note": "0.5=perfect, 1.0=mode collapse",
        "pathology_probe_accuracy": float(probe_acc),
    }
    print(json.dumps(results, indent=2))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
