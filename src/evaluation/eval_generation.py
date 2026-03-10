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
import wandb
from torch_geometric.loader import DataLoader

from src.evaluation.metrics import one_nearest_neighbour_accuracy, linear_probe_accuracy
from src.training.train_main import CardiacGraphDataset
from src.utils.acdc_loader import PATHOLOGY_CLASSES


LABEL_NAMES = list(PATHOLOGY_CLASSES.keys())  # ["NOR", "MINF", "DCM", "HCM", "RV"]


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
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                real_shapes.append(batch.pos[mask].cpu().numpy().flatten())
                gen_shapes.append(gen_pos[mask].cpu().numpy().flatten())
    max_len = max(s.shape[0] for s in real_shapes)
    real_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in real_shapes])
    gen_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in gen_shapes])
    return real_arr, gen_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--checkpoint", default="checkpoints/main/best_model.pt")
    parser.add_argument("--struct", default="lv")
    parser.add_argument("--out", default="results/generation.json")
    parser.add_argument("--n_generate", type=int, default=100)
    parser.add_argument("--wandb_project", default="acdc-cardiac-diffusion")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name="eval-generation",
        job_type="evaluation",
        tags=["eval", "generation", "1nna", "latent-space"],
    )

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

    # Pathology separation (linear probe)
    latents, labels = collect_latents_and_labels(encoder, loader, device)
    probe_acc = linear_probe_accuracy(latents, labels, n_classes=5)

    results = {
        "1NNA": float(nna),
        "1NNA_note": "0.5=perfect generation, 1.0=mode collapse",
        "pathology_probe_accuracy": float(probe_acc),
    }
    print(json.dumps(results, indent=2))

    # W&B: scalar summaries
    wandb.run.summary.update({
        "1NNA": float(nna),
        "pathology_probe_accuracy": float(probe_acc),
    })
    wandb.log({
        "eval/1NNA": float(nna),
        "eval/pathology_probe_accuracy": float(probe_acc),
    })

    # W&B: latent space scatter (2D UMAP if available, else t-SNE)
    if len(latents) >= 5:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latents)
        except ImportError:
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1)).fit_transform(latents)

        scatter_table = wandb.Table(columns=["x", "y", "pathology"])
        for (x, y), lbl in zip(embedding, labels):
            name = LABEL_NAMES[int(lbl)] if int(lbl) < len(LABEL_NAMES) else str(lbl)
            scatter_table.add_data(float(x), float(y), name)
        wandb.log({
            "latent_space": wandb.plot.scatter(scatter_table, "x", "y", title="Latent Space by Pathology")
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
