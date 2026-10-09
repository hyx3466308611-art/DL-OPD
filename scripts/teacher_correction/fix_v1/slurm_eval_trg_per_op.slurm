#!/bin/bash
# 诊断协议：一次加载权重，按算子扫 PER_OP_DIAGNOSIS_OPS（severity 用 Stage C）
#   sbatch --export=ALL,TEACHER_BACKBONE=author3b,EVAL_ERODE_GT=1 \
#     scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm
# 单算子：
#   sbatch --export=ALL,TEACHER_BACKBONE=author3b,FORCE_OP=empty_gt,EVAL_ERODE_GT=1 \
#     scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm
#
#SBATCH --job-name=fix_v1_per_op
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/teacher_correction/fix_v1/slurm_per_op_%j.out
#SBATCH --error=logs/teacher_correction/fix_v1/slurm_per_op_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT

# shellcheck disable=SC1091
source scripts/teacher_correction/fix_v1/_env_fix_v1.sh

EVAL_STAGE="${EVAL_STAGE:-C}"
EVAL_ERODE_GT="${EVAL_ERODE_GT:-1}"
FORCE_OP="${FORCE_OP:-ALL}"

OUT_DIR="${CORR_OUT}/eval_recovery_per_op"
FLAG_PATH="${CORR_OUT}/teacher_frozen_ready_per_op.flag"
STAGE_LOG="${LOG_DIR}/eval_trg_per_op_${TEACHER_BACKBONE}.log"

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

echo "=== fix_v1 Per-op TRG diagnosis ==="
echo "TEACHER_BACKBONE=$TEACHER_BACKBONE CKPT=$CORR_OUT"
echo "EVAL_STAGE=$EVAL_STAGE FORCE_OP=$FORCE_OP EVAL_ERODE_GT=$EVAL_ERODE_GT OUT=$OUT_DIR"

python -u scripts/teacher_correction/fix_v1/eval_trg_fix_v1.py \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$SAM_FIN" \
  --resume-plm-path "$PLM_FIN" \
  --resume-head-path "$HEAD_FIN" \
  --resume-lora-path "$LORA_FIN" \
  --eval-stage "$EVAL_STAGE" \
  --force-op "$FORCE_OP" \
  "${FULL_GT_FLAG[@]}" \
  --splits sam_seeded,human_annotated \
  --out-dir "$OUT_DIR" \
  --flag-path "$FLAG_PATH" \
  2>&1 | tee -a "$STAGE_LOG"

if [[ "$FORCE_OP" == "ALL" ]]; then
  python -u scripts/teacher_correction/merge_per_op_reports.py --root "$OUT_DIR" \
    2>&1 | tee -a "$STAGE_LOG"
fi

echo "Suite root: $OUT_DIR"
echo "Summary: $OUT_DIR/per_op_summary.json"
echo "Log: $STAGE_LOG"
