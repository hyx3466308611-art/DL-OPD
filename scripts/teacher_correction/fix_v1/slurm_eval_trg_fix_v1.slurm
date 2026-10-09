#!/bin/bash
# fix_v1 TRG。主协议 Mixed-TRG（默认 EVAL_STAGE=MIX）：
#   sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_ERODE_GT=1 scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm
# 附录 B/C：
#   sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_STAGE=B,EVAL_ERODE_GT=1 scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm

#SBATCH --job-name=fix_v1_trg
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/teacher_correction/fix_v1/slurm_trg_%j.out
#SBATCH --error=logs/teacher_correction/fix_v1/slurm_trg_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT

# shellcheck disable=SC1091
source scripts/teacher_correction/fix_v1/_env_fix_v1.sh

EVAL_STAGE="${EVAL_STAGE:-MIX}"
# 与 baseline / Mixed 主表对比：erode GT（默认）；纯看 fix_v1 训练对齐：EVAL_ERODE_GT=0
EVAL_ERODE_GT="${EVAL_ERODE_GT:-1}"

OUT_DIR="${CORR_OUT}/eval_recovery_stage${EVAL_STAGE}"
FLAG_PATH="${CORR_OUT}/teacher_frozen_ready_stage${EVAL_STAGE}.flag"
STAGE_LOG="${LOG_DIR}/eval_trg_fix_v1_stage${EVAL_STAGE}.log"

mkdir -p logs/teacher_correction/fix_v1 "$OUT_DIR" "$LOG_DIR"

activate_converseg
install_sam2_editable
check_assets

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SAM_FIN="${CORR_OUT}/${CORR_NAME}_final.torch"
PLM_FIN="${CORR_OUT}/${CORR_NAME}_plm_final.torch"
HEAD_FIN="${CORR_OUT}/${CORR_NAME}_head_final.torch"
LORA_FIN="${CORR_OUT}/lora_plm_adapter_final"

for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
  [[ -f "$f" ]] || { echo "Missing: $f (train fix_v1 first)"; exit 1; }
done

FULL_GT_FLAG=(--full-gt)
if [[ "$EVAL_ERODE_GT" == "1" ]]; then
  FULL_GT_FLAG=()
fi

echo "=== fix_v1 TRG ==="
echo "TEACHER_BACKBONE=$TEACHER_BACKBONE CKPT=$CORR_OUT NAME=$CORR_NAME"
echo "EVAL_STAGE=$EVAL_STAGE EVAL_ERODE_GT=$EVAL_ERODE_GT OUT=$OUT_DIR"

python -u scripts/teacher_correction/fix_v1/eval_trg_fix_v1.py \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$SAM_FIN" \
  --resume-plm-path "$PLM_FIN" \
  --resume-head-path "$HEAD_FIN" \
  --resume-lora-path "$LORA_FIN" \
  --eval-stage "$EVAL_STAGE" \
  "${FULL_GT_FLAG[@]}" \
  --splits sam_seeded,human_annotated \
  --out-dir "$OUT_DIR" \
  --flag-path "$FLAG_PATH" \
  2>&1 | tee -a "$STAGE_LOG"

echo "Report: $OUT_DIR/trg_report.json"
echo "Flag: $FLAG_PATH"
echo "Log: $STAGE_LOG"
