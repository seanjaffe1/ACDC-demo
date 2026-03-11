"""
Process a single graph entry for curvature augmentation.
Called by build_curvature_batch.py as a subprocess to avoid igl memory issues.

Usage: python scripts/build_curvature_single.py <entry_json> <graph_dir> <reg_dir> <out_dir>
"""
import json, sys
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh

entry = json.loads(sys.argv[1])
graph_dir = Path(sys.argv[2])
reg_dir = Path(sys.argv[3])
out_dir = Path(sys.argv[4])

fname = entry["file"]
subj_id = entry["subject_id"]
frame = entry["frame"]
struct = entry["struct"]
mesh_path = reg_dir / subj_id / f"{frame}_{struct}.ply"

mesh = trimesh.load(str(mesh_path), process=False)
V = mesh.vertices.astype(np.float64)
F = mesh.faces.astype(np.int64)

pd1, pd2, pv1, pv2, bad = igl.principal_curvature(V, F)
H = (pv1 + pv2) / 2.0
K = pv1 * pv2

normals = np.array(mesh.vertex_normals, dtype=np.float64)
chi = np.sign(np.sum(np.cross(pd1, pd2) * normals, axis=1, keepdims=True))
chi[chi == 0] = 1.0

feats = np.concatenate(
    [np.stack([pv1, pv2, H, K], axis=1), chi], axis=1
).astype(np.float32)
if len(bad) > 0:
    feats[bad] = 0.0
feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

pd1 = np.nan_to_num(pd1.astype(np.float32), nan=0.0)
pd2 = np.nan_to_num(pd2.astype(np.float32), nan=0.0)
if len(bad) > 0:
    pd1[bad] = 0.0
    pd2[bad] = 0.0

data = torch.load(graph_dir / fname, weights_only=False)
data.x = torch.tensor(feats, dtype=torch.float32)
data.principal_dir1 = torch.tensor(pd1, dtype=torch.float32)
data.principal_dir2 = torch.tensor(pd2, dtype=torch.float32)
torch.save(data, out_dir / fname)
print(f"OK: {fname}", flush=True)
