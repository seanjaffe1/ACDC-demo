"""
Standalone script to augment PyG graphs with curvature features.
Avoids module import issues that cause segfault with igl.

Usage:
    python scripts/build_curvature_graphs.py
"""

import json
import sys
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh

reg_dir = Path("data/registered")
graph_dir = Path("data/graphs")
out_dir = Path("data/graphs_curvature")
out_dir.mkdir(exist_ok=True)

with open(graph_dir / "manifest.json") as f:
    manifest = json.load(f)

print(f"Processing {len(manifest)} graphs...")
n_ok = n_fail = 0
new_manifest = []

for i, entry in enumerate(manifest):
    fname = entry["file"]
    subj_id = entry["subject_id"]
    frame = entry["frame"]
    struct = entry["struct"]
    mesh_path = reg_dir / subj_id / f"{frame}_{struct}.ply"

    if not mesh_path.exists():
        print(f"  SKIP (no mesh): {mesh_path}", flush=True)
        continue

    try:
        mesh = trimesh.load(str(mesh_path), process=False)
        V = mesh.vertices.astype(np.float64)
        F = mesh.faces.astype(np.int64)
        _, _, pv1, pv2, bad = igl.principal_curvature(V, F)
        H = (pv1 + pv2) / 2.0
        K = pv1 * pv2
        feats = np.stack([pv1, pv2, H, K], axis=1).astype(np.float32)
        if len(bad) > 0:
            feats[bad] = 0.0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.tensor(feats, dtype=torch.float32)
        data = torch.load(graph_dir / fname, weights_only=False)
        data.x = x
        torch.save(data, out_dir / fname)

        new_entry = dict(entry)
        new_entry["in_node_dim"] = 4
        new_manifest.append(new_entry)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  [{n_ok}/{len(manifest)}] processed ...", flush=True)
    except Exception as e:
        print(f"  FAIL: {fname} — {e}", flush=True)
        n_fail += 1

with open(out_dir / "manifest.json", "w") as f:
    json.dump(new_manifest, f, indent=2)

print(f"\nDone: {n_ok} augmented, {n_fail} failed")
print(f"Graphs saved to {out_dir}")
