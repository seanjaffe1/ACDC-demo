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


def compute_ef_errors(lv_results: list[dict]) -> dict:
    ed_by_subj = {r["subject_id"]: r for r in lv_results if r["frame"] == "ed"}
    es_by_subj = {r["subject_id"]: r for r in lv_results if r["frame"] == "es"}
    common = set(ed_by_subj) & set(es_by_subj)

    ef_errors = []
    for subj_id in common:
        ed = ed_by_subj[subj_id]
        es = es_by_subj[subj_id]
        if ed["faces"] is None:
            continue
        pred_ef = ejection_fraction(ed["pred_verts"], ed["faces"], es["pred_verts"], es["faces"])
        true_ef = ejection_fraction(ed["true_verts"], ed["faces"], es["true_verts"], es["faces"])
        ef_errors.append(abs(pred_ef - true_ef))

    return {
        "ef_error_mean": float(np.mean(ef_errors)),
        "ef_error_std": float(np.std(ef_errors)),
        "ef_error_le5_pct": float(np.mean(np.array(ef_errors) <= 5.0) * 100),
    }


def compute_myo_volume_conservation(myo_results: list[dict]) -> dict:
    """Check that predicted myocardial volume is conserved across ED/ES."""
    ed_by_subj = {r["subject_id"]: r for r in myo_results if r["frame"] == "ed"}
    es_by_subj = {r["subject_id"]: r for r in myo_results if r["frame"] == "es"}
    common = set(ed_by_subj) & set(es_by_subj)

    vol_diffs = []
    for subj_id in common:
        ed = ed_by_subj[subj_id]
        es = es_by_subj[subj_id]
        if ed["faces"] is None:
            continue
        v_ed = mesh_volume(ed["pred_verts"], ed["faces"])
        v_es = mesh_volume(es["pred_verts"], es["faces"])
        vol_diffs.append(abs(v_ed - v_es) / max(v_ed, 1e-6) * 100)  # % change

    return {
        "myo_vol_change_mean_pct": float(np.mean(vol_diffs)),
        "myo_vol_change_std_pct": float(np.std(vol_diffs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--checkpoint", default="checkpoints/main/best_model.pt")
    parser.add_argument("--out", default="results/clinical.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = Path(args.graph_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lv_results = reconstruct_meshes(Path(args.checkpoint), graph_dir, "lv", device)
    myo_results = reconstruct_meshes(Path(args.checkpoint), graph_dir, "myo", device)

    results = {}
    results["ejection_fraction"] = compute_ef_errors(lv_results)
    results["myo_volume"] = compute_myo_volume_conservation(myo_results)

    print(json.dumps(results, indent=2))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
