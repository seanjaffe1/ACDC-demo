#!/usr/bin/env bash
# =============================================================================
# ACDC Cardiac Shape Generation — Full Pipeline Runner
# Usage: bash run.sh [--skip-install] [--skip-preprocess] [--skip-train]
#                    [--skip-baselines] [--skip-eval] [--data-dir PATH]
#
# Assumes:
#   - Python 3.11+ available (conda or system)
#   - ACDC dataset already downloaded and placed in data/raw/
#     Each patient directory: data/raw/patientNNN/patientNNN_frame01.nii.gz, etc.
#   - Run from the project root: bash run.sh
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $*${NC}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $*${NC}"; exit 1; }

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_BASELINES=false
SKIP_EVAL=false
DATA_DIR="data/raw"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)     SKIP_INSTALL=true ;;
    --skip-preprocess)  SKIP_PREPROCESS=true ;;
    --skip-train)       SKIP_TRAIN=true ;;
    --skip-baselines)   SKIP_BASELINES=true ;;
    --skip-eval)        SKIP_EVAL=true ;;
    --data-dir)         DATA_DIR="$2"; shift ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"   # always run from project root

if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
  die "No data found at '$DATA_DIR'. Download the ACDC dataset first and place patient directories there."
fi

PATIENT_COUNT=$(ls -d "$DATA_DIR"/patient* 2>/dev/null | wc -l | tr -d ' ')
log "Found ${PATIENT_COUNT} patient directories in ${DATA_DIR}"
[[ "$PATIENT_COUNT" -lt 5 ]] && warn "Very few patients detected — double-check your data directory."

# ── Environment setup ─────────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == false ]]; then
  log "Step 0/5 — Installing Python dependencies"

  # Detect if inside a conda env; otherwise warn
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    log "Active conda env: ${CONDA_DEFAULT_ENV}"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log "Active venv: ${VIRTUAL_ENV}"
  else
    warn "No active conda/venv detected. Installing into system Python."
  fi

  # torch-geometric extension packages need the right CUDA/torch combination.
  # Install them via pip find-links if not already satisfied.
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
  if [[ -z "$TORCH_VERSION" ]]; then
    log "Installing PyTorch (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    log "PyTorch ${TORCH_VERSION} already installed — skipping torch install."
  fi

  log "Installing remaining requirements..."
  pip install -r requirements.txt

  # torch-geometric scatter/sparse/cluster often need special wheels
  log "Installing torch-geometric extras..."
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")+cu121.html \
    || warn "torch-geometric extras failed — if you are on CPU-only, install manually."

  pip install torch-geometric

  # Install this project as an editable package so `src.*` imports work
  pip install -e .

  ok "Dependencies installed."
else
  log "Skipping install (--skip-install)."
fi

# ── Weights & Biases login ────────────────────────────────────────────────────
if python -c "import wandb" 2>/dev/null; then
  if wandb status 2>&1 | grep -q "not logged in"; then
    warn "W&B not logged in. Run 'wandb login' to enable experiment tracking."
    warn "Continuing with W&B offline mode."
    export WANDB_MODE=offline
  else
    log "W&B authenticated."
  fi
else
  warn "wandb not importable — experiment tracking will be disabled."
  export WANDB_MODE=disabled
fi

# ── Preprocessing ─────────────────────────────────────────────────────────────
if [[ "$SKIP_PREPROCESS" == false ]]; then
  log "Step 1/5 — Preprocessing: extract meshes"
  python -m src.preprocessing.extract_meshes \
    --data_dir "$DATA_DIR" \
    --out_dir  data/meshes \
    --target_vertices 2500
  ok "Meshes extracted → data/meshes/"

  log "Step 2/5 — Preprocessing: register to template"
  python -m src.preprocessing.register_template \
    --mesh_dir data/meshes \
    --out_dir  data/registered
  ok "Meshes registered → data/registered/"

  log "Step 3/5 — Preprocessing: build graphs"
  python -m src.preprocessing.build_graphs \
    --reg_dir  data/registered \
    --data_dir "$DATA_DIR" \
    --out_dir  data/graphs \
    --k 8
  ok "Graphs built → data/graphs/"
else
  log "Skipping preprocessing (--skip-preprocess)."
  [[ -f data/graphs/manifest.json ]] || die "data/graphs/manifest.json not found — run preprocessing first."
fi

# ── Main model training ───────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" == false ]]; then
  log "Step 4/5 — Training: main EGNN flow-matching model"
  python -m src.training.train_main --config configs/main_model.yaml
  ok "Main model trained → checkpoints/main/best_model.pt"
else
  log "Skipping main model training (--skip-train)."
fi

# ── Baseline training ─────────────────────────────────────────────────────────
if [[ "$SKIP_BASELINES" == false ]]; then
  log "Step 4b — Training: PCA baseline"
  python -m src.training.train_baselines --config configs/baseline_pca.yaml --baseline pca
  ok "PCA baseline done → checkpoints/pca/"

  log "Step 4b — Training: MLP flow-matching baseline (ablation: no equivariance)"
  python -m src.training.train_baselines --config configs/baseline_mlp.yaml --baseline mlp
  ok "MLP baseline done → checkpoints/mlp/"

  log "Step 4b — Training: PointNet++ VAE baseline (ablation: no diffusion)"
  python -m src.training.train_baselines --config configs/baseline_pointnet.yaml --baseline pointnet
  ok "PointNet baseline done → checkpoints/pointnet/"

  log "Step 4b — Training: Voxel diffusion baseline (ablation: grid vs mesh)"
  python -m src.training.train_baselines --config configs/baseline_voxel.yaml --baseline voxel
  ok "Voxel baseline done → checkpoints/voxel/"
else
  log "Skipping baselines (--skip-baselines)."
fi

# ── Evaluation ────────────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" == false ]]; then
  mkdir -p results

  MAIN_CKPT="checkpoints/main/best_model.pt"
  [[ -f "$MAIN_CKPT" ]] || die "Main checkpoint not found at ${MAIN_CKPT}. Train first."

  log "Step 5/5 — Evaluation: reconstruction (Chamfer + Hausdorff)"
  python -m src.evaluation.eval_reconstruction \
    --checkpoint_dir checkpoints \
    --out results/reconstruction.json
  ok "Reconstruction metrics → results/reconstruction.json"

  log "Step 5/5 — Evaluation: clinical metrics (EF, myocardial volume)"
  python -m src.evaluation.eval_clinical \
    --checkpoint "$MAIN_CKPT" \
    --out results/clinical.json
  ok "Clinical metrics → results/clinical.json"

  log "Step 5/5 — Evaluation: generation quality (1-NNA, latent probe)"
  python -m src.evaluation.eval_generation \
    --checkpoint "$MAIN_CKPT" \
    --out results/generation.json
  ok "Generation metrics → results/generation.json"

  log "─────────────────────────────────────────────────"
  log "Results summary:"
  python - <<'PYEOF'
import json, os

for name, path in [
    ("Reconstruction", "results/reconstruction.json"),
    ("Clinical",       "results/clinical.json"),
    ("Generation",     "results/generation.json"),
]:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"\n{name}:")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        print(f"  {k}/{kk}: {vv:.4f}")
    else:
        print(f"\n{name}: results file not found")
PYEOF

else
  log "Skipping evaluation (--skip-eval)."
fi

ok "Pipeline complete."
echo
echo "  Checkpoints : checkpoints/"
echo "  Results     : results/"
echo "  W&B runs    : https://wandb.ai (project: acdc-cardiac-diffusion)"