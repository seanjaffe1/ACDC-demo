# ACDC Cardiac Shape Generation — Full Project Plan

## The Pipeline

```
ACDC MRI volumes
    ↓ marching cubes + smoothing          [src/preprocessing/extract_meshes.py]
cardiac meshes (LV, RV, myocardium)
    ↓ register to template, normalise     [src/preprocessing/register_template.py]
canonical pose meshes (~2500 vertices)
    ↓ build k-NN graph (k=8)              [src/preprocessing/build_graphs.py]
PyG graph objects (pos, edge_index, face, y)
    ↓ SE(3)-equivariant message passing   [src/models/egnn.py + encoder.py]
equivariant latent codes (dim=64)
    ↓ conditional flow matching (OT-CFM)  [src/models/flow_matching.py]
distribution over cardiac shapes
```

---

## What Is Being Measured

### Reconstruction Quality
- Chamfer distance (mm)
- 95th-percentile Hausdorff distance (mm)

### Clinical Validity
- Ejection fraction error — target < 5% absolute
- Myocardial volume conservation across cardiac cycle (% change ED→ES)

### Distribution Quality
- 1-NNA — 0.5 = perfect generation, 1.0 = mode collapse
- Pathology separation — 5-fold CV logistic regression accuracy on latent codes (5 ACDC classes)

---

## The Four Baselines

| # | Baseline | What it justifies |
|---|---|---|
| 1 | PCA shape model | Using deep learning at all |
| 2 | Voxel diffusion (3D U-Net) | Mesh representation over grids |
| 3 | PointNet++ VAE (no diffusion) | Distributional modelling |
| 4 | Same model, MLP instead of EGNN + rot-aug | **SE(3) equivariance** ← most important |

Baseline 4 is the critical ablation. If the equivariant model doesn't clearly beat it
(especially in low-data regimes), the geometric prior is not justified.

---

## Execution Steps

### Phase 1 — Environment & Data (Days 1–2)
1. `pip install -r requirements.txt`
2. Download ACDC from acdc.creatis.insa-lyon.fr (requires registration)
3. Place data in `data/raw/` — each subject as `patientNNN/` folder
4. Run `notebooks/01_data_exploration.ipynb` — verify label conventions (LV=3, RV=1, Myo=2)

### Phase 2 — Preprocessing (Days 3–5)
5. `python -m src.preprocessing.extract_meshes --data_dir data/raw --out_dir data/meshes`
6. `notebooks/02_mesh_inspection.ipynb` — verify vertex counts ~2000–3000, watertight meshes
7. `python -m src.preprocessing.register_template --mesh_dir data/meshes --out_dir data/registered`
8. `python -m src.preprocessing.build_graphs --reg_dir data/registered --data_dir data/raw --out_dir data/graphs --k 8`

### Phase 3 — Main Model (Days 6–10)
9. `python -m src.training.train_main --config configs/main_model.yaml`
   - Logs to W&B project `acdc-cardiac-diffusion`, run `main-egnn-flow`
   - Saves best checkpoint to `checkpoints/main/best_model.pt`

### Phase 4 — Baselines (Days 11–15)
10. Baseline 4 first (most important):
    `python -m src.training.train_baselines --config configs/baseline_mlp.yaml --baseline mlp`
11. `python -m src.training.train_baselines --config configs/baseline_pca.yaml --baseline pca`
12. `python -m src.training.train_baselines --config configs/baseline_pointnet.yaml --baseline pointnet`
13. `python -m src.training.train_baselines --config configs/baseline_voxel.yaml --baseline voxel`

### Phase 5 — Evaluation (Days 16–18)
14. `python -m src.evaluation.eval_reconstruction --checkpoint_dir checkpoints --out results/reconstruction.json`
15. `python -m src.evaluation.eval_clinical --checkpoint checkpoints/main/best_model.pt --out results/clinical.json`
16. `python -m src.evaluation.eval_generation --checkpoint checkpoints/main/best_model.pt --out results/generation.json`

### Phase 6 — Report (Days 19–21)
17. `notebooks/03_results_analysis.ipynb` — generate comparison table + bar charts
18. `notebooks/04_latent_space.ipynb` — UMAP latent space coloured by pathology
19. Write `report/main.tex` covering: intro → pipeline diagram → experiments → results table → Baseline 4 ablation → conclusion

---

## Key Design Decisions Made

- **Equivariance library**: EGNN (Satorras et al. 2021) — chosen over e3nn for simplicity (no spherical harmonics)
- **Diffusion method**: Optimal Transport Flow Matching (Lipman et al. 2023) — straight-line paths, faster than DDPM
- **Graph construction**: k-NN (k=8) on registered vertex positions
- **Latent space**: VAE-style encoder (mu + logvar), dim=64, KL weight=0.001
- **Template registration**: Procrustes (Kabsch) → ICP refinement → centroid normalisation
- **Structures**: LV, RV, myocardium extracted separately; LV is primary evaluation target

---

## W&B Setup
- Project: `acdc-cardiac-diffusion`
- Entity: set `wandb_entity` in configs or leave null (uses logged-in account)
- Run `wandb login` before training
