#!/usr/bin/env bash
# Stage-2: 90,000 steps (ConVerSeg paper hyperparameters)
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAM_CFG="configs/sam2/sam2_hiera_l.yaml"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"

STAGE1_OUT="${STAGE1_OUT:-./ckpts/stage1_full}"
RESUME_TAG="${RESUME_TAG:-100000}"

DATA_DIR="./data/ConverSeg-Training-Data/converseg_stage2_data"
JSONL="stage2_mixed.jsonl"

OUT="./ckpts/stage2_full"
NAME="ConVerSeg_stage2"

RESUME_SAM="${STAGE1_OUT}/ConVerSeg_stage1_${RESUME_TAG}.torch"
RESUME_PLM="${STAGE1_OUT}/ConVerSeg_stage1_plm_${RESUME_TAG}.torch"
RESUME_LORA="${STAGE1_OUT}/lora_plm_adapter_${RESUME_TAG}"

for f in "$RESUME_SAM" "$RESUME_PLM"; do
  [[ -f "$f" ]] || { echo "Missing: $f  (set RESUME_TAG=final ?)"; exit 1; }
done
[[ -f "$RESUME_LORA" || -d "$RESUME_LORA" ]] || { echo "Missing: $RESUME_LORA"; exit 1; }

mkdir -p "$OUT" logs

"$PYTHON" -u train.py \
  --data-dir "$DATA_DIR" \
  --dataset-jsonl "$JSONL" \
  --model-cfg "$SAM_CFG" \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$RESUME_SAM" \
  --resume-plm-path "$RESUME_PLM" \
  --resume-lora-path "$RESUME_LORA" \
  --device cuda \
  --steps 90000 \
  --lr 1e-4 \
  --wd 0.05 \
  --batch-size 6 \
  --acc 1 \
  --max-dim 1024 \
  --workers 4 \
  --log-every 50 \
  --save-every 5000 \
  --val-every 5000 \
  --val-count 4 \
  --out "$OUT" \
  --name "$NAME"

echo "Stage-2 done -> ${OUT}/${NAME}_final.torch"
