"""
Evaluate reconstruction quality for main model and all baselines.

Outputs a JSON results file with Chamfer distance and Hausdorff distance per model.
All results are logged to W&B as a summary table.

Usage:
    python -m src.evaluation.eval_reconstruction \
        --graph_dir data/graphs \
        --checkpoint_dir checkpoints \
        --out results/reconstruction.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader

from src.training.train_main import CardiacGraphDataset
from src.evaluation.metrics import chamfer_distance_numpy, hausdorff_distance


def eval_main_model(checkpoint_path: Path, val_loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    from src.models.encoder import GraphEncoder
    from src.models.flow_matching import FlowMatchingModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = GraphEncoder(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_encoder_layers"]).to(device)
    flow_model = FlowMatchingModel(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_flow_layers"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])
    encoder.eval(); flow_model.eval()

    chamfers, hausdorffs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            z = encoder.encode(batch)
            template = batch.clone()
            template.pos = torch.randn_like(batch.pos)
            gen_pos = flow_model.sample(template, z, steps=100)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                pred = gen_pos[mask].cpu().numpy()
                tgt = batch.pos[mask].cpu().numpy()
                chamfers.append(chamfer_distance_numpy(pred, tgt))
                hausdorffs.append(hausdorff_distance(pred, tgt))

    return {
        "chamfer_mean": float(np.mean(chamfers)),
        "chamfer_std": float(np.std(chamfers)),
        "hausdorff_mean": float(np.mean(hausdorffs)),
        "hausdorff_std": float(np.std(hausdorffs)),
    }


def eval_pca_model(pca_path: Path, val_loader: DataLoader) -> dict:
    with open(pca_path, "rb") as f:
        model = pickle.load(f)

    v_max = model.mean_shape.shape[0] // 3
    chamfers, hausdorffs = [], []
    for batch in val_loader:
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            tgt = batch.pos[mask].numpy()
            v_orig = tgt.shape[0]
            tgt_padded = np.pad(tgt, ((0, v_max - v_orig), (0, 0))) if v_orig < v_max else tgt
            recon = model.reconstruct(tgt_padded)[:v_orig]
            chamfers.append(chamfer_distance_numpy(recon, tgt))
            hausdorffs.append(hausdorff_distance(recon, tgt))

    return {
        "chamfer_mean": float(np.mean(chamfers)),
        "chamfer_std": float(np.std(chamfers)),
        "hausdorff_mean": float(np.mean(hausdorffs)),
        "hausdorff_std": float(np.std(hausdorffs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out", default="results/reconstruction.json")
    parser.add_argument("--wandb_project", default="acdc-cardiac-diffusion")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name="eval-reconstruction",
        job_type="evaluation",
        tags=["eval", "reconstruction"],
    )

    graph_dir = Path(args.graph_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = "lv"
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    split = int(0.8 * len(entries))
    val_ds = CardiacGraphDataset(graph_dir, entries[split:], struct)
    val_loader = DataLoader(val_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    pca_path = ckpt_dir / "pca" / "pca_model.pkl"
    if pca_path.exists():
        results["pca"] = eval_pca_model(pca_path, val_loader)
        print(f"PCA: {results['pca']}")

    main_path = ckpt_dir / "main" / "best_model.pt"
    if main_path.exists():
        ckpt = torch.load(main_path, map_location="cpu", weights_only=False)
        results["main"] = eval_main_model(main_path, val_loader, device, ckpt["cfg"])
        print(f"Main: {results['main']}")

    # Log a W&B comparison table
    if results:
        table = wandb.Table(columns=["model", "chamfer_mean_mm", "chamfer_std_mm", "hausdorff_mean_mm", "hausdorff_std_mm"])
        for model_name, m in results.items():
            table.add_data(model_name, m["chamfer_mean"], m["chamfer_std"], m["hausdorff_mean"], m["hausdorff_std"])
        wandb.log({"reconstruction_comparison": table})

        # Also log flat metrics for easy filtering
        for model_name, m in results.items():
            for k, v in m.items():
                wandb.run.summary[f"{model_name}/{k}"] = v

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
