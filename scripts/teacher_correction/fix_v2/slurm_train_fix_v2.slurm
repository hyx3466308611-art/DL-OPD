#!/bin/bash
# fix_v2 training. Example:
#   sbatch --export=ALL,TEACHER_BACKBONE=author3b scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm
#
#SBATCH --job-name=corr_fix_v2
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/teacher_correction/fix_v2/slurm_train_%j.out
#SBATCH --error=logs/teacher_correction/fix_v2/slurm_train_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT

# shellcheck disable=SC1091
source scripts/teacher_correction/fix_v2/_env_fix_v2.sh
mkdir -p "$LOG_DIR" "$CORR_OUT"

activate_converseg
install_sam2_editable
check_assets
check_fix_v2_init

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-$DATA_ROOT}"
JSONL="${DATASET_JSONL:-$PHASE2_JSONL_REL}"

if [[ "${HUMAN_REPEAT}" != "0" ]]; then
  echo "REFUSE: HUMAN_REPEAT=${HUMAN_REPEAT}; ConverSeg eval set must not enter training"
  exit 1
fi

echo "=== Correction Teacher fix_v2 ==="
echo "BACKBONE=$TEACHER_BACKBONE INIT=$INIT_CORR_DIR"
echo "OUT=$CORR_OUT NAME=$CORR_NAME STEPS=$FIX_V2_STEPS BS=$CORR_BS ACC=$CORR_ACC"
echo "DATA=${DATA_DIR}/${JSONL}"
echo "human=${CONVERSEG_EVAL_ROOT}/${HUMAN_JSONL_REL} x${HUMAN_REPEAT}"
echo "full_gt=1 fake_negative=0.10 dropout_B/C=0.15/0.30 empty_bce=3 identity=0.5"

python -u scripts/teacher_correction/fix_v2/train_fix_v2.py \
  --data-dir "$DATA_DIR" \
  --dataset-jsonl "$JSONL" \
  --converseg-root "$CONVERSEG_EVAL_ROOT" \
  --human-jsonl "$HUMAN_JSONL_REL" \
  --human-repeat "$HUMAN_REPEAT" \
  --model-cfg "$SAM_CFG" \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --device cuda \
  --steps "$FIX_V2_STEPS" \
  --lr 2e-5 \
  --wd 0.05 \
  --batch-size "$CORR_BS" \
  --acc "$CORR_ACC" \
  --max-dim 1024 \
  --workers 0 \
  --log-every 10 \
  --lora-r 16 \
  --lora-alpha 32 \
  --plm-gradient-checkpointing \
  --qwen-image-size 256 \
  --fake-negative-prob 0.10 \
  --empty-bce-weight 3.0 \
  --identity-weight 0.5 \
  --preserve-iou-thresh 0.85 \
  --out "$CORR_OUT" \
  --name "$CORR_NAME" \
  --resume-sam-path "$SAM_FIN" \
  --resume-plm-path "$PLM_FIN" \
  --resume-head-path "$HEAD_FIN" \
  --resume-lora-path "$LORA_FIN" \
  2>&1 | tee -a "$TRAIN_LOG"

echo "Done: ${CORR_OUT}/${CORR_NAME}_head_final.torch"
