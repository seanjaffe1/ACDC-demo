# ACDC-demo Project Instructions

## Project Overview
SE(3)-equivariant cardiac shape generation using flow matching on the ACDC dataset.
Primary target structure: left ventricle (LV). The model is a VAE encoder (EGNN) + conditional flow matching decoder.

## Key Metrics (Current Baseline)
- Reconstruction: Chamfer 0.99 mm, Hausdorff 1.61 mm (good)
- Generation: 1-NNA 0.605 (target ~0.5), pathology probe 0.32 (low)
- Clinical EF error: 229% (severe — known issue)

## Architecture
- `src/models/egnn.py` — SE(3)-equivariant GNN layers
- `src/models/encoder.py` — VAE encoder (EGNN → mu/logvar)
- `src/models/flow_matching.py` — OT-CFM velocity field
- `src/training/train_main.py` — main training loop with KL annealing support
- `configs/main_model.yaml` — canonical hyperparameters

## Current Hyperparameters (main_model.yaml)
```
hidden_dim: 128, latent_dim: 64
n_encoder_layers: 4, n_flow_layers: 4
kl_weight: 0.001, lr: 0.0003, batch_size: 8, epochs: 200
```

## Running Experiments
Use `--set KEY=VALUE` to override config without editing YAML files:
```bash
python -m src.training.train_main --config configs/main_model.yaml \
  --set kl_weight=0.005 out_dir=checkpoints/exp1 wandb_run_name=exp1
```

Evaluation scripts:
```bash
python -m src.evaluation.eval_reconstruction
python -m src.evaluation.eval_generation
python -m src.evaluation.eval_clinical
```
Results land in `results/`.

## Workflow Preferences
- Ablations via `--set` overrides, not by editing YAML files directly
- Each experiment gets its own `out_dir` and `wandb_run_name`
- Evaluate after training completes; compare against baseline in `results/`
- Do not auto-commit
- Have a working ablations section in the paper with updated results. do not record every ablation, just the ones that are strucuturally succinct or informative and help paint an overall picture.

## Known Issues
- Clinical EF error (229%) is likely a model/data issue, not a bug in the eval script
- No KL annealing in baseline (kl_anneal_epochs defaults to 0)
- Pathology probe accuracy (0.32) suggests latent space is not well-structured

## Specific instuctions
