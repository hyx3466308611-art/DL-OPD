#!/bin/bash
# fix_v1。7B：sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm
#
#SBATCH --job-name=corr_fix_v1
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/teacher_correction/fix_v1/slurm_train_%j.out
#SBATCH --error=logs/teacher_correction/fix_v1/slurm_train_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT

# shellcheck disable=SC1091
source scripts/teacher_correction/fix_v1/_env_fix_v1.sh
mkdir -p logs/teacher_correction/fix_v1 "$LOG_DIR" "$CORR_OUT"

activate_converseg
install_sam2_editable
check_assets
check_fix_init

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$DATA_ROOT}"
JSONL="${DATASET_JSONL:-$PHASE2_JSONL_REL}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${HUMAN_REPEAT}" != "0" ]]; then
  echo "REFUSE: HUMAN_REPEAT=${HUMAN_REPEAT}; ConverSeg eval set must not enter training"
  exit 1
fi

echo "=== Correction Teacher fix_v1 ==="
echo "TEACHER_BACKBONE=$TEACHER_BACKBONE BS=$CORR_BS ACC=$CORR_ACC"
echo "INIT=$INIT_CORR_DIR OUT=$CORR_OUT NAME=$CORR_NAME STEPS=$FIX_STEPS"
echo "DATA=${DATA_DIR}/${JSONL}"
echo "human=${CONVERSEG_EVAL_ROOT}/${HUMAN_JSONL_REL} x${HUMAN_REPEAT}"
echo "negative=${DATA_DIR}/${NEGATIVE_JSONL_REL} prob=0.175 dual_mode=0.5 HeadV2 final-only"

python -u scripts/teacher_correction/fix_v1/train_fix_v1.py \
  "${TRAIN_FIX_COMMON[@]}" \
  --data-dir "$DATA_DIR" \
  --dataset-jsonl "$JSONL" \
  --converseg-root "$CONVERSEG_EVAL_ROOT" \
  2>&1 | tee -a "$TRAIN_LOG"

echo "Done. Final weights only: ${CORR_OUT}/${CORR_NAME}_final.torch"
