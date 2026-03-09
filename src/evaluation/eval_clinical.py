"""
Evaluate clinical validity:
  - Ejection fraction error (target < 5% absolute)
  - Myocardial volume conservation across cardiac cycle

Usage:
    python -m src.evaluation.eval_clinical \
        --graph_dir data/graphs \
        --checkpoint checkpoints/main/best_model.pt \
        --out results/clinical.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader

from src.evaluation.metrics import ejection_fraction, mesh_volume
from src.training.train_main import CardiacGraphDataset


def reconstruct_meshes(ckpt_path: Path, graph_dir: Path, struct: str, device: torch.device) -> list[dict]:
    from src.models.encoder import GraphEncoder
    from src.models.flow_matching import FlowMatchingModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    encoder = GraphEncoder(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_encoder_layers"]).to(device)
    flow_model = FlowMatchingModel(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_flow_layers"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])
    encoder.eval(); flow_model.eval()

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    results = []
    for frame in ["ed", "es"]:
        entries = [e for e in manifest if e["frame"] == frame and e["struct"] == struct]
        ds = CardiacGraphDataset(graph_dir, entries, struct)
        loader = DataLoader(ds, batch_size=1)

        for data, entry in zip(loader, entries):
            data = data.to(device)
            with torch.no_grad():
                z = encoder.encode(data)
                template = data.clone()
                template.pos = torch.randn_like(data.pos)
                gen_pos = flow_model.sample(template, z, steps=50)

            results.append({
                "subject_id": entry["subject_id"],
                "frame": frame,
                "struct": struct,
                "pred_verts": gen_pos.cpu().numpy(),
                "true_verts": data.pos.cpu().numpy(),
                "faces": data.face.T.cpu().numpy() if hasattr(data, "face") and data.face is not None else None,
            })
    return results


def compute_ef_errors(lv_results: list[dict]) -> tuple[dict, list[dict]]:
    ed_by_subj = {r["subject_id"]: r for r in lv_results if r["frame"] == "ed"}
    es_by_subj = {r["subject_id"]: r for r in lv_results if r["frame"] == "es"}
    common = set(ed_by_subj) & set(es_by_subj)

    ef_errors, per_subject = [], []
    for subj_id in sorted(common):
        ed = ed_by_subj[subj_id]
        es = es_by_subj[subj_id]
        if ed["faces"] is None:
            continue
        pred_ef = ejection_fraction(ed["pred_verts"], ed["faces"], es["pred_verts"], es["faces"])
        true_ef = ejection_fraction(ed["true_verts"], ed["faces"], es["true_verts"], es["faces"])
        err = abs(pred_ef - true_ef)
        ef_errors.append(err)
        per_subject.append({"subject_id": subj_id, "true_ef": true_ef, "pred_ef": pred_ef, "abs_error": err})

    summary = {
        "ef_error_mean": float(np.mean(ef_errors)),
        "ef_error_std": float(np.std(ef_errors)),
        "ef_error_le5_pct": float(np.mean(np.array(ef_errors) <= 5.0) * 100),
    }
    return summary, per_subject


def compute_myo_volume_conservation(myo_results: list[dict]) -> tuple[dict, list[dict]]:
    ed_by_subj = {r["subject_id"]: r for r in myo_results if r["frame"] == "ed"}
    es_by_subj = {r["subject_id"]: r for r in myo_results if r["frame"] == "es"}
    common = set(ed_by_subj) & set(es_by_subj)

    vol_diffs, per_subject = [], []
    for subj_id in sorted(common):
        ed = ed_by_subj[subj_id]
        es = es_by_subj[subj_id]
        if ed["faces"] is None:
            continue
        v_ed = mesh_volume(ed["pred_verts"], ed["faces"])
        v_es = mesh_volume(es["pred_verts"], es["faces"])
        pct = abs(v_ed - v_es) / max(v_ed, 1e-6) * 100
        vol_diffs.append(pct)
        per_subject.append({"subject_id": subj_id, "vol_ed": v_ed, "vol_es": v_es, "pct_change": pct})

    summary = {
        "myo_vol_change_mean_pct": float(np.mean(vol_diffs)),
        "myo_vol_change_std_pct": float(np.std(vol_diffs)),
    }
    return summary, per_subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--checkpoint", default="checkpoints/main/best_model.pt")
    parser.add_argument("--out", default="results/clinical.json")
    parser.add_argument("--wandb_project", default="acdc-cardiac-diffusion")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name="eval-clinical",
        job_type="evaluation",
        tags=["eval", "clinical"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = Path(args.graph_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lv_results = reconstruct_meshes(Path(args.checkpoint), graph_dir, "lv", device)
    myo_results = reconstruct_meshes(Path(args.checkpoint), graph_dir, "myo", device)

    ef_summary, ef_per_subj = compute_ef_errors(lv_results)
    myo_summary, myo_per_subj = compute_myo_volume_conservation(myo_results)

    results = {"ejection_fraction": ef_summary, "myo_volume": myo_summary}
    print(json.dumps(results, indent=2))

    # W&B: scalar summaries
    wandb.run.summary.update({
        "ef_error_mean": ef_summary["ef_error_mean"],
        "ef_error_std": ef_summary["ef_error_std"],
        "ef_error_le5_pct": ef_summary["ef_error_le5_pct"],
        "myo_vol_change_mean_pct": myo_summary["myo_vol_change_mean_pct"],
    })

    # W&B: per-subject EF table
    ef_table = wandb.Table(columns=["subject_id", "true_ef", "pred_ef", "abs_error"])
    for row in ef_per_subj:
        ef_table.add_data(row["subject_id"], row["true_ef"], row["pred_ef"], row["abs_error"])
    wandb.log({"ejection_fraction_per_subject": ef_table})

    # W&B: EF error histogram
    if ef_per_subj:
        errors = [r["abs_error"] for r in ef_per_subj]
        wandb.log({"ef_error_histogram": wandb.Histogram(errors)})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
