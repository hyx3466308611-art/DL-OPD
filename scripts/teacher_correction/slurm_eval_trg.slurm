#!/bin/bash
# 主协议 Mixed-TRG（默认 EVAL_STAGE=MIX，A:B:C=1:2:2）：
#   sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/slurm_eval_trg.slurm
# 附录难度扫描：
#   sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_STAGE=B scripts/teacher_correction/slurm_eval_trg.slurm
#
#SBATCH --job-name=corr_trg
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/teacher_correction/slurm_trg_%j.out
#SBATCH --error=logs/teacher_correction/slurm_trg_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT

# shellcheck disable=SC1091
source scripts/teacher_correction/_env.sh
mkdir -p logs/teacher_correction "$EVAL_RECOVERY_DIR"

activate_converseg
install_sam2_editable
check_assets

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SAM_FIN="${CORR_OUT}/${CORR_NAME}_final.torch"
PLM_FIN="${CORR_OUT}/${CORR_NAME}_plm_final.torch"
HEAD_FIN="${CORR_OUT}/${CORR_NAME}_head_final.torch"
LORA_FIN="${CORR_OUT}/lora_plm_adapter_final"

for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
  [[ -f "$f" ]] || { echo "Missing checkpoint: $f (run slurm_train_correction first)"; exit 1; }
done

EVAL_STAGE="${EVAL_STAGE:-MIX}"
OUT_DIR="${CORR_OUT}/eval_recovery_stage${EVAL_STAGE}"
FLAG_PATH="${CORR_OUT}/teacher_frozen_ready_stage${EVAL_STAGE}.flag"
STAGE_LOG="${LOG_DIR}/eval_trg_${TEACHER_BACKBONE}_stage${EVAL_STAGE}.log"
mkdir -p "$OUT_DIR"

python eval_teacher_recovery.py \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$SAM_FIN" \
  --resume-plm-path "$PLM_FIN" \
  --resume-head-path "$HEAD_FIN" \
  --resume-lora-path "$LORA_FIN" \
  --eval-stage "$EVAL_STAGE" \
  --splits sam_seeded,human_annotated \
  --out-dir "$OUT_DIR" \
  --flag-path "$FLAG_PATH" \
  2>&1 | tee -a "$STAGE_LOG"

echo "Report: $OUT_DIR/trg_report.json"
echo "Flag: $FLAG_PATH (CORR_OUT=$CORR_OUT TEACHER_BACKBONE=$TEACHER_BACKBONE)"
