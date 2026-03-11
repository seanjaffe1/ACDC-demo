"""
Evaluate reconstruction quality for MLP, PointNet, and Voxel baselines.
Appends results to results/reconstruction.json and results/generation.json.

Usage:
    python scripts/eval_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_main import CardiacGraphDataset
from src.evaluation.metrics import (
    chamfer_distance_numpy,
    hausdorff_distance,
    one_nearest_neighbour_accuracy,
    linear_probe_accuracy,
)

GRAPH_DIR = Path("data/graphs")
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
VOXEL_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_val_loader():
    with open(GRAPH_DIR / "manifest.json") as f:
        manifest = json.load(f)
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == "lv"]
    split = int(0.8 * len(entries))
    val_ds = CardiacGraphDataset(GRAPH_DIR, entries[split:], "lv")
    return DataLoader(val_ds, batch_size=4)


# --------------------------------------------------------------------------- #
# MLP reconstruction
# --------------------------------------------------------------------------- #
def eval_mlp_reconstruction(val_loader):
    from src.models.encoder import GraphEncoder
    from src.models.baselines.mlp_diffusion import MLPFlowMatchingModel

    ckpt = torch.load(CKPT_DIR / "mlp" / "best_mlp.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["cfg"]
    encoder = GraphEncoder(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_encoder_layers"]
    ).to(DEVICE)
    flow_model = MLPFlowMatchingModel(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_flow_layers"]
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])
    encoder.eval(); flow_model.eval()

    chamfers, hausdorffs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
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


# --------------------------------------------------------------------------- #
# MLP generation (1-NNA + pathology probe)
# --------------------------------------------------------------------------- #
def eval_mlp_generation():
    from src.models.encoder import GraphEncoder
    from src.models.baselines.mlp_diffusion import MLPFlowMatchingModel

    with open(GRAPH_DIR / "manifest.json") as f:
        manifest = json.load(f)
    entries = [e for e in manifest if e["frame"] == "ed" and e["struct"] == "lv"]
    ds = CardiacGraphDataset(GRAPH_DIR, entries, "lv")
    loader = DataLoader(ds, batch_size=4)

    ckpt = torch.load(CKPT_DIR / "mlp" / "best_mlp.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["cfg"]
    encoder = GraphEncoder(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_encoder_layers"]
    ).to(DEVICE)
    flow_model = MLPFlowMatchingModel(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], n_layers=cfg["n_flow_layers"]
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    flow_model.load_state_dict(ckpt["flow_model"])
    encoder.eval(); flow_model.eval()

    real_shapes, gen_shapes, latents, labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            z = encoder.encode(batch)
            template = batch.clone()
            template.pos = torch.randn_like(batch.pos)
            gen_pos = flow_model.sample(template, z, steps=50)
            latents.append(z.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                real_shapes.append(batch.pos[mask].cpu().numpy().flatten())
                gen_shapes.append(gen_pos[mask].cpu().numpy().flatten())

    max_len = max(s.shape[0] for s in real_shapes)
    real_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in real_shapes])
    gen_arr = np.stack([np.pad(s, (0, max_len - len(s))) for s in gen_shapes])
    nna = one_nearest_neighbour_accuracy(real_arr, gen_arr)

    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    probe_acc = linear_probe_accuracy(latents, labels, n_classes=5)

    return {"1NNA": float(nna), "pathology_probe_accuracy": float(probe_acc)}


# --------------------------------------------------------------------------- #
# PointNet reconstruction
# --------------------------------------------------------------------------- #
def eval_pointnet_reconstruction(val_loader):
    from src.models.baselines.pointnet_ae import PointNetAE

    ckpt = torch.load(CKPT_DIR / "pointnet" / "best_pointnet.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["cfg"]
    model = PointNetAE(latent_dim=cfg["latent_dim"], num_points=cfg.get("num_points", 2500)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    chamfers, hausdorffs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            recon = out["recon"]  # (B, G^2, 3)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                tgt = batch.pos[mask].cpu().numpy()
                pred = recon[i].cpu().numpy()  # (G^2, 3)
                chamfers.append(chamfer_distance_numpy(pred, tgt))
                hausdorffs.append(hausdorff_distance(pred, tgt))

    return {
        "chamfer_mean": float(np.mean(chamfers)),
        "chamfer_std": float(np.std(chamfers)),
        "hausdorff_mean": float(np.mean(hausdorffs)),
        "hausdorff_std": float(np.std(hausdorffs)),
    }


# --------------------------------------------------------------------------- #
# Voxel reconstruction (nearest-neighbour coverage in dataset-normalised space)
# --------------------------------------------------------------------------- #
def eval_voxel_reconstruction(val_loader):
    """
    The voxel model is unconditional, so true per-shape reconstruction is not
    possible.  Instead we measure coverage-based reconstruction quality:
    generate N_GEN shapes from the model, then for each test shape find its
    nearest generated neighbour after rescaling the generated voxel point cloud
    to the test shape's bounding box.  All Chamfer/Hausdorff values are
    reported in the same dataset-normalised coordinate space as every other
    baseline.
    """
    from src.models.baselines.voxel_diffusion import VoxelFlowMatching

    ckpt = torch.load(CKPT_DIR / "voxel" / "best_voxel.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["cfg"]
    model = VoxelFlowMatching(base_ch=cfg.get("base_ch", 32)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Collect test shapes with their bounding boxes (in dataset-normalised space)
    test_info = []  # list of (pos_np, pos_min, pos_max)
    with torch.no_grad():
        for batch in val_loader:
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                pos = batch.pos[mask].numpy()
                test_info.append((pos, pos.min(0), pos.max(0)))

    n_test = len(test_info)

    # Generate unconditional voxel samples (2× test set, min 50)
    n_gen = max(n_test * 2, 50)
    batch_sz = 4
    gen_unit_clouds = []  # each in [0,1]^3
    with torch.no_grad():
        for _ in range(0, n_gen, batch_sz):
            vols = model.sample(
                (batch_sz, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), DEVICE, steps=50
            )
            for b in range(batch_sz):
                vol_np = vols[b, 0].cpu().numpy()
                indices = np.argwhere(vol_np > 0.5)
                if len(indices) == 0:
                    indices = np.zeros((1, 3))
                gen_unit_clouds.append((indices / (VOXEL_SIZE - 1)).astype(np.float32))

    # For each test shape, rescale every generated cloud to its bbox and take min Chamfer
    chamfers, hausdorffs = [], []
    for pos, pos_min, pos_max in test_info:
        best_cd = float("inf")
        best_hd = float("inf")
        for unit_cloud in gen_unit_clouds:
            # Rescale generated [0,1]^3 cloud into this test shape's coordinate space
            scaled = unit_cloud * (pos_max - pos_min) + pos_min
            cd = chamfer_distance_numpy(scaled, pos)
            if cd < best_cd:
                best_cd = cd
                best_hd = hausdorff_distance(scaled, pos)
        chamfers.append(best_cd)
        hausdorffs.append(best_hd)

    return {
        "chamfer_mean": float(np.mean(chamfers)),
        "chamfer_std": float(np.std(chamfers)),
        "hausdorff_mean": float(np.mean(hausdorffs)),
        "hausdorff_std": float(np.std(hausdorffs)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    val_loader = load_val_loader()

    # Load existing reconstruction results
    recon_path = RESULTS_DIR / "reconstruction.json"
    recon = json.loads(recon_path.read_text()) if recon_path.exists() else {}

    print("Evaluating MLP reconstruction...")
    recon["mlp"] = eval_mlp_reconstruction(val_loader)
    print(f"  MLP recon: {recon['mlp']}")

    print("Evaluating PointNet reconstruction...")
    recon["pointnet"] = eval_pointnet_reconstruction(val_loader)
    print(f"  PointNet recon: {recon['pointnet']}")

    print("Evaluating Voxel reconstruction...")
    recon["voxel"] = eval_voxel_reconstruction(val_loader)
    print(f"  Voxel recon: {recon['voxel']}")

    recon_path.write_text(json.dumps(recon, indent=2))
    print(f"Reconstruction results saved to {recon_path}")

    print("Evaluating MLP generation metrics...")
    mlp_gen = eval_mlp_generation()
    print(f"  MLP gen: {mlp_gen}")

    gen_path = RESULTS_DIR / "generation.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    gen["mlp"] = mlp_gen
    gen_path.write_text(json.dumps(gen, indent=2))
    print(f"Generation results saved to {gen_path}")


if __name__ == "__main__":
    main()
