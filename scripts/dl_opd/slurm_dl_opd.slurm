#!/bin/bash
#SBATCH --job-name=dlopd_full
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dl_opd/slurm_dl_opd_%j.out
#SBATCH --error=logs/dl_opd/slurm_dl_opd_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
mkdir -p logs/dl_opd

if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -d sam2_repo ]]; then pip install -e sam2_repo -q; elif [[ -d sam2 ]]; then pip install -e sam2 -q; fi

# Outer Data-Policy + optional inner (set RUN_INNER=1)
POLICY_ARGS=(--config configs/dl_opd/dl_opd.yaml --n_outer "${N_OUTER:-4}")
if [[ "${RUN_INNER:-0}" == "1" ]]; then
  POLICY_ARGS+=(--run_inner --inner_extra_args "${INNER_EXTRA_ARGS:---steps 500}")
else
  POLICY_ARGS+=(--policy_only)
fi

python -u scripts/dl_opd/train_dl_opd.py "${POLICY_ARGS[@]}"
