#!/bin/bash
# 教师 Phase1(100k)+Phase2(90k) — Qwen2.5-VL-7B + SAM2-L（teacher/）
# 超参与 scripts/slurm_phase12_train.slurm（学生）一致
#
# 提交:
#   cd /home/hanyuxi/ConverSeg-main
#   mkdir -p logs/teacher ckpts/teacher/stage1_phase1mix ckpts/teacher/stage2_phase2mix
#   sbatch scripts/teacher/slurm_phase12_train.slurm
#
# 7B 单卡 48G 若 OOM：编辑 scripts/teacher/_train_env.sh 改为 batch-size 1 acc 6

#SBATCH --job-name=converseg-t-p12
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

mkdir -p "$LOG_DIR" "$STAGE1_OUT" "$STAGE2_OUT"
cd "$ROOT"

activate_converseg

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "JobID=${SLURM_JOB_ID} Teacher Phase12 PLM=$PLM"
python -V
nvidia-smi -L || true

install_sam2_editable
check_teacher_assets
[[ -f "${ROOT}/data/phase1/phase1_mixed.jsonl" ]] || { echo "Missing phase1 jsonl"; exit 1; }

echo "========== Teacher Phase1 100000 $(date -Is) =========="
python -u train.py \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase1/phase1_mixed.jsonl \
  "${TRAIN_COMMON[@]}" \
  --steps 100000 \
  --save-every 50000 \
  --out "$STAGE1_OUT" \
  --name "$NAME1" \
  2>&1 | tee -a "$PHASE1_LOG"

FINAL_SAM="${STAGE1_OUT}/${NAME1}_final.torch"
FINAL_PLM="${STAGE1_OUT}/${NAME1}_plm_final.torch"
FINAL_LORA="${STAGE1_OUT}/lora_plm_adapter_final"
for f in "$FINAL_SAM" "$FINAL_PLM"; do
  [[ -f "$f" ]] || { echo "Phase1 missing: $f"; exit 1; }
done
[[ -f "$FINAL_LORA" || -d "$FINAL_LORA" ]] || { echo "Phase1 missing LoRA: $FINAL_LORA"; exit 1; }

[[ -f "${ROOT}/data/phase2/phase2_mixed.jsonl" ]] || { echo "Missing phase2 jsonl"; exit 1; }

echo "========== Teacher Phase2 90000 $(date -Is) =========="
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

echo "========== Teacher base ready $(date -Is) =========="
echo "SAM:  ${STAGE2_OUT}/${NAME2}_final.torch"
echo "PLM:  ${STAGE2_OUT}/${NAME2}_plm_final.torch"
echo "LoRA: ${STAGE2_OUT}/lora_plm_adapter_final"
