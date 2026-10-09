#!/bin/bash
# 教师 Phase2 final 在 ConVerSeg benchmark 上评测（与学生 eval 相同数据）

#SBATCH --job-name=converseg-t-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/teacher/slurm-%x-%j.out
#SBATCH --error=logs/teacher/slurm-%x-%j.err

set -euo pipefail

ROOT=/home/hanyuxi/ConverSeg-main
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher/_train_env.sh"

CKPT="$STAGE2_OUT"
EVAL_ROOT="${ROOT}/data/ConverSeg"
EVAL_LOG="${LOG_DIR}/eval_teacher_stage2_bf16.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"
activate_converseg

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
install_sam2_editable

for f in \
  "${CKPT}/${NAME2}_final.torch" \
  "${CKPT}/${NAME2}_plm_final.torch" \
  "${CKPT}/lora_plm_adapter_final" \
  "${EVAL_ROOT}/sam_seeded.jsonl" \
  "${EVAL_ROOT}/human_annotated.jsonl" \
  "${PLM}/config.json" \
  "$SAM_CKPT"
do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done

python -u eval.py \
  --final_ckpt "${CKPT}/${NAME2}_final.torch" \
  --plm_ckpt "${CKPT}/${NAME2}_plm_final.torch" \
  --lora_ckpt "${CKPT}/lora_plm_adapter_final" \
  --plm-model-name "$PLM" \
  --base_ckpt "$SAM_CKPT" \
  --model_cfg "$SAM_CFG" \
  --device cuda \
  --precision bf16 \
  --eval_splits sam_seeded,human_annotated \
  2>&1 | tee -a "$EVAL_LOG"

echo "Done. grep JSON: $EVAL_LOG"
