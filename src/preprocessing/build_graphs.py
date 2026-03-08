"""
Build PyTorch Geometric graph objects from registered meshes.

Each graph stores:
  - pos  : (V, 3) float32 vertex positions
  - edge_index : (2, E) k-NN edges
  - face : (3, F) mesh faces (for surface area / Dice evaluation)
  - y    : int pathology class label
  - subject_id : str

Usage:
    python -m src.preprocessing.build_graphs \
        --reg_dir data/registered \
        --data_dir data/raw \
        --out_dir data/graphs \
        --k 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from src.utils.acdc_loader import ACDCDataset, PATHOLOGY_CLASSES


def mesh_to_graph(mesh: trimesh.Trimesh, k: int, y: int, subject_id: str, frame: str, struct: str) -> Data:
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.T, dtype=torch.long)  # (3, F)
    edge_index = knn_graph(pos, k=k, loop=False)

    return Data(
        pos=pos,
        edge_index=edge_index,
        face=faces,
        y=torch.tensor(y, dtype=torch.long),
        subject_id=subject_id,
        frame=frame,
        struct=struct,
        num_nodes=pos.shape[0],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_dir", default="data/registered")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/graphs")
    parser.add_argument("--k", type=int, default=8, help="k for k-NN graph")
    args = parser.parse_args()

    reg_dir = Path(args.reg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ACDCDataset(args.data_dir)
    pathology_map = {s.subject_id: s.pathology_label for s in dataset.subjects}

    manifest = []
    for subj_dir in sorted(reg_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name
        y = pathology_map.get(subj_id, 0)

        for frame in ["ed", "es"]:
            for struct in ["lv", "rv", "myo"]:
                mesh_path = subj_dir / f"{frame}_{struct}.ply"
                if not mesh_path.exists():
                    continue
                mesh = trimesh.load(str(mesh_path), process=False)
                graph = mesh_to_graph(mesh, k=args.k, y=y, subject_id=subj_id, frame=frame, struct=struct)

                out_name = f"{subj_id}_{frame}_{struct}.pt"
                torch.save(graph, out_dir / out_name)
                manifest.append({"file": out_name, "subject_id": subj_id, "frame": frame, "struct": struct, "y": y})
                print(f"  {out_name}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {len(manifest)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
