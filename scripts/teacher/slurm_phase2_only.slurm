#!/bin/bash
# 教师仅 Phase2（Phase1 已在其它作业完成）
#   sbatch --dependency=afterok:<JOB1> scripts/teacher/slurm_phase2_only.slurm

#SBATCH --job-name=converseg-t-p2
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/teacher/slurm-%x-%j.out
#SBATCH --error=logs/teacher/slurm-%x-%j.err

set -euo pipefail

ROOT=/home/hanyuxi/ConverSeg-main
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher/_train_env.sh"

mkdir -p "$LOG_DIR" "$STAGE2_OUT"
cd "$ROOT"
activate_converseg

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

install_sam2_editable
check_teacher_assets

FINAL_SAM="${STAGE1_OUT}/${NAME1}_final.torch"
FINAL_PLM="${STAGE1_OUT}/${NAME1}_plm_final.torch"
FINAL_LORA="${STAGE1_OUT}/lora_plm_adapter_final"
for f in "$FINAL_SAM" "$FINAL_PLM"; do
  [[ -f "$f" ]] || { echo "Missing Phase1 ckpt: $f"; exit 1; }
done

python -u train.py \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase2/phase2_mixed.jsonl \
  "${TRAIN_COMMON[@]}" \
  --steps 90000 \
  --save-every 50000 \
  --resume-sam-path "$FINAL_SAM" \
  --resume-plm-path "$FINAL_PLM" \
  --resume-lora-path "$FINAL_LORA" \
  --out "$STAGE2_OUT" \
  --name "$NAME2" \
  2>&1 | tee -a "$PHASE2_LOG"
