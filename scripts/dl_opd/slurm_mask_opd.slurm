#!/bin/bash
#SBATCH --job-name=dlopd_mask
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_opd/slurm_mask_opd_%j.out
#SBATCH --error=logs/dl_opd/slurm_mask_opd_%j.err

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

python -u scripts/dl_opd/train_mask_opd.py \
  --config configs/dl_opd/dl_opd.yaml \
  ${EXTRA_ARGS:-}
