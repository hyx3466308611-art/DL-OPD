#!/bin/bash
# fix_v2 paper-facing Mixed-TRG (A:B:C=1:2:2).
#SBATCH --job-name=fix_v2_mixed
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/teacher_correction/fix_v2/slurm_mixed_%j.out
#SBATCH --error=logs/teacher_correction/fix_v2/slurm_mixed_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT
# shellcheck disable=SC1091
source scripts/teacher_correction/fix_v2/_env_fix_v2.sh

activate_converseg
install_sam2_editable
check_assets

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SAM_V2="${CORR_OUT}/${CORR_NAME}_final.torch"
PLM_V2="${CORR_OUT}/${CORR_NAME}_plm_final.torch"
HEAD_V2="${CORR_OUT}/${CORR_NAME}_head_final.torch"
LORA_V2="${CORR_OUT}/lora_plm_adapter_final"
for f in "$SAM_V2" "$PLM_V2" "$HEAD_V2"; do
  [[ -f "$f" ]] || { echo "Missing fix_v2 checkpoint: $f"; exit 1; }
done

OUT_DIR="${CORR_OUT}/eval_recovery_stageMIX"
FLAG_PATH="${CORR_OUT}/teacher_frozen_ready_stageMIX.flag"
mkdir -p "$OUT_DIR" "$LOG_DIR"

python -u scripts/teacher_correction/fix_v2/eval_trg_fix_v2.py \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$SAM_V2" \
  --resume-plm-path "$PLM_V2" \
  --resume-head-path "$HEAD_V2" \
  --resume-lora-path "$LORA_V2" \
  --eval-stage MIX \
  --splits sam_seeded,human_annotated \
  --out-dir "$OUT_DIR" \
  --flag-path "$FLAG_PATH" \
  2>&1 | tee -a "${LOG_DIR}/eval_mixed_${TEACHER_BACKBONE}.log"

echo "Report: $OUT_DIR/trg_report.json"
echo "Flag: $FLAG_PATH"
