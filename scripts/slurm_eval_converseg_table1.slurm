#!/bin/bash
# CONVERSEG Table-1 style eval (overall + 5 concepts, gIoU/cIoU).
#
#   sbatch --export=ALL,CKPT_DIR=ckpts/stage2_phase2mix,PLM=student/Qwen2.5-VL-3B,NAME=stage2_3b \
#     scripts/slurm_eval_converseg_table1.slurm
#
# Author Net-3B example:
#   sbatch --export=ALL,CKPT_DIR=checkpoints/ConVerSeg-Net-3B,PLM=student/Qwen2.5-VL-3B,NAME=author3b,\
#     SAM_CKPT=student/sam2_hiera_large.pt,FINAL=...,PLM_CKPT=...,LORA=... \
#     scripts/slurm_eval_converseg_table1.slurm
#
#SBATCH --job-name=convseg_t1
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_author_tables/slurm_converseg_%j.out
#SBATCH --error=logs/eval_author_tables/slurm_converseg_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
mkdir -p logs/eval_author_tables

if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -d "${ROOT}/sam2_repo" ]]; then
  pip install -e "${ROOT}/sam2_repo" -q
elif [[ -d "${ROOT}/sam2" ]]; then
  pip install -e "${ROOT}/sam2" -q
fi

NAME="${NAME:-stage2_3b}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/ckpts/stage2_phase2mix}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"
FINAL="${FINAL:-${CKPT_DIR}/ConVerSeg_stage2_final.torch}"
PLM_CKPT="${PLM_CKPT:-${CKPT_DIR}/ConVerSeg_stage2_plm_final.torch}"
LORA="${LORA:-${CKPT_DIR}/lora_plm_adapter_final}"
PRECISION="${PRECISION:-bf16}"

for f in "$FINAL" "$PLM_CKPT" "$LORA" "$SAM_CKPT" "$PLM"; do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done

REPORT="logs/eval_author_tables/converseg_${NAME}_report.jsonl"
LOG="logs/eval_author_tables/converseg_${NAME}.log"
rm -f "$REPORT"

python -u eval.py \
  --final_ckpt "$FINAL" \
  --plm_ckpt "$PLM_CKPT" \
  --lora_ckpt "$LORA" \
  --plm-model-name "$PLM" \
  --base_ckpt "$SAM_CKPT" \
  --model_cfg configs/sam2/sam2_hiera_l.yaml \
  --device cuda \
  --precision "$PRECISION" \
  --eval_splits sam_seeded,human_annotated \
  --report-out "$REPORT" \
  2>&1 | tee -a "$LOG"

echo "Report: $REPORT"
echo "Log: $LOG"
