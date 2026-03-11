"""
Unified evaluation script for the gauge-equivariant model.

Collects all three metric groups into separate JSON files:
  results/gauge_reconstruction.json
  results/gauge_generation.json
  results/gauge_clinical.json

Usage:
    python -m src.evaluation.eval_gauge \
        --checkpoint checkpoints/gauge/best_model.pt \
        --graph_dir data/graphs_curvature

The script loads the GaugeGraphEncoder automatically based on the
model_type='gauge' key stored in the checkpoint cfg.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader

from src.evaluation.metrics import (
    chamfer_distance_numpy,
    hausdorff_distance,
    one_nearest_neighbour_accuracy,
    linear_probe_accuracy,
    ejection_fraction,
    mesh_volume,
)
from src.utils.acdc_loader import PATHOLOGY_CLASSES

LABEL_NAMES = list(PATHOLOGY_CLASSES.keys())


# ---------------------------------------------------------------------------
# Dataset helper (mirrors train_gauge.py CardiacGraphDataset)
# ---------------------------------------------------------------------------

class CardiacGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dir, manifest_entries, struct="lv"):
        self.graph_dir = Path(graph_dir)
        self.entries = [e for e in manifest_entries if e["struct"] == struct]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        import torch as _torch
        entry = self.entries[idx]
        data = _torch.load(self.graph_dir / entry["file"], weights_only=False)
        data.pos = (data.pos - data.pos.mean(0)) / (data.pos.std() + 1e-6)
        return data


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_models(ckpt_path: Path, device: torch.device):
    from src.models.gauge_encoder import GaugeGraphEncoder
    from src.models.flow_matching import FlowMatchingModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    encoder = GaugeGraphEncoder(
        in_node_dim=cfg.get("in_node_dim", 4),
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_layers=cfg["n_encoder_layers"],
        curvature_dim=cfg.get("curvature_dim", 4),
    ).to(device)

    flow_model = FlowMatchingModel(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_layers=cfg["n_flow_layers"],
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])
    encoder.eval()
    flow_model.eval()

    return encoder, flow_model, cfg


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def eval_reconstruction(encoder, flow_model, val_loader, device) -> dict:
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


# ---------------------------------------------------------------------------
# Generation / distribution quality
# ---------------------------------------------------------------------------

def eval_generation(encoder, flow_model, loader, device) -> dict:
    real_shapes, gen_shapes = [], []
    latents, labels = [], []
    encoder.eval()
    flow_model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = encoder.encode(batch)
            template = batch.clone()
            template.pos = torch.randn_like(batch.pos)
            gen_pos = flow_model.sample(template, z, steps=50)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                real_shapes.append(batch.pos[mask].cpu().numpy().flatten())
                gen_shapes.append(gen_pos[mask].cpu().numpy().flatten())
            latents.append(z.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    max_len = max(s.shape[0] for s in real_shapes)
    real_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in real_shapes])
    gen_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in gen_shapes])
    latents_arr = np.concatenate(latents)
    labels_arr = np.concatenate(labels)

    nna = one_nearest_neighbour_accuracy(real_arr, gen_arr)
    probe = linear_probe_accuracy(latents_arr, labels_arr, n_classes=5)

    return {
        "1NNA": float(nna),
        "1NNA_note": "0.5=perfect generation, 1.0=mode collapse",
        "pathology_probe_accuracy": float(probe),
    }


# ---------------------------------------------------------------------------
# Clinical
# ---------------------------------------------------------------------------

def eval_clinical(ckpt_path: Path, graph_dir: Path, struct: str, device: torch.device) -> dict:
    encoder, flow_model, cfg = load_models(ckpt_path, device)

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    results_by_frame = {}
    for frame in ["ed", "es"]:
        entries = [e for e in manifest if e["frame"] == frame and e["struct"] == struct]
        ds = CardiacGraphDataset(graph_dir, entries, struct)
        loader = DataLoader(ds, batch_size=1)

        frame_results = []
        for data, entry in zip(loader, entries):
            data = data.to(device)
            with torch.no_grad():
                z = encoder.encode(data)
                template = data.clone()
                template.pos = torch.randn_like(data.pos)
                gen_pos = flow_model.sample(template, z, steps=50)
            frame_results.append({
                "subject_id": entry["subject_id"],
                "frame": frame,
                "pred_verts": gen_pos.cpu().numpy(),
                "true_verts": data.pos.cpu().numpy(),
                "faces": data.face.T.cpu().numpy() if hasattr(data, "face") and data.face is not None else None,
            })
        results_by_frame[frame] = {r["subject_id"]: r for r in frame_results}

    ed_map = results_by_frame.get("ed", {})
    es_map = results_by_frame.get("es", {})
    common = set(ed_map) & set(es_map)

    ef_errors = []
    for subj_id in sorted(common):
        ed = ed_map[subj_id]
        es = es_map[subj_id]
        if ed["faces"] is None:
            continue
        pred_ef = ejection_fraction(ed["pred_verts"], ed["faces"], es["pred_verts"], es["faces"])
        true_ef = ejection_fraction(ed["true_verts"], ed["faces"], es["true_verts"], es["faces"])
        ef_errors.append(abs(pred_ef - true_ef))

    return {
        "ejection_fraction": {
            "ef_error_mean": float(np.mean(ef_errors)) if ef_errors else float("nan"),
            "ef_error_std": float(np.std(ef_errors)) if ef_errors else float("nan"),
            "ef_error_le5_pct": float(np.mean(np.array(ef_errors) <= 5.0) * 100) if ef_errors else float("nan"),
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gauge/best_model.pt")
    parser.add_argument("--graph_dir", default="data/graphs_curvature")
    parser.add_argument("--struct", default="lv")
    parser.add_argument("--out_recon", default="results/gauge_reconstruction.json")
    parser.add_argument("--out_gen", default="results/gauge_generation.json")
    parser.add_argument("--out_clin", default="results/gauge_clinical.json")
    parser.add_argument("--wandb_project", default="acdc-cardiac-diffusion")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name="eval-gauge",
        job_type="evaluation",
        tags=["eval", "gauge", "curvature"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    graph_dir = Path(args.graph_dir)

    Path(args.out_recon).parent.mkdir(parents=True, exist_ok=True)

    # Load once for reconstruction + generation
    encoder, flow_model, cfg = load_models(ckpt_path, device)

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    struct = args.struct
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == struct]
    split = int(0.8 * len(entries))

    val_ds = CardiacGraphDataset(graph_dir, entries[split:], struct)
    val_loader = DataLoader(val_ds, batch_size=4)
    all_ds = CardiacGraphDataset(graph_dir, entries, struct)
    all_loader = DataLoader(all_ds, batch_size=4)

    # --- Reconstruction ---
    print("Evaluating reconstruction...")
    recon = eval_reconstruction(encoder, flow_model, val_loader, device)
    print(json.dumps({"gauge": recon}, indent=2))
    with open(args.out_recon, "w") as f:
        json.dump({"gauge": recon}, f, indent=2)

    # --- Generation ---
    print("Evaluating generation / distribution quality...")
    gen = eval_generation(encoder, flow_model, all_loader, device)
    print(json.dumps(gen, indent=2))
    with open(args.out_gen, "w") as f:
        json.dump(gen, f, indent=2)

    # --- Clinical ---
    print("Evaluating clinical (EF)...")
    clin = eval_clinical(ckpt_path, graph_dir, "lv", device)
    print(json.dumps(clin, indent=2))
    with open(args.out_clin, "w") as f:
        json.dump(clin, f, indent=2)

    # Log to W&B
    wandb.run.summary.update({
        "gauge/chamfer_mean": recon["chamfer_mean"],
        "gauge/hausdorff_mean": recon["hausdorff_mean"],
        "gauge/1NNA": gen["1NNA"],
        "gauge/pathology_probe": gen["pathology_probe_accuracy"],
        "gauge/ef_error_mean": clin["ejection_fraction"]["ef_error_mean"],
    })
    wandb.finish()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
