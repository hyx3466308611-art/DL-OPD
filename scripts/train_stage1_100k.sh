#!/usr/bin/env bash
# Stage-1: 100,000 steps (ConVerSeg paper hyperparameters)
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAM_CFG="configs/sam2/sam2_hiera_l.yaml"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"

if [[ -f "./data/ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl" ]]; then
  DATA_DIR="./data/ConverSeg-Training-Data/converseg_stage1_data"
  JSONL="open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl"
elif [[ -f "./data/ConverSeg-Training-Data/converseg_stage2_data/open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl" ]]; then
  echo "[warn] stage1_data 缺失，改用 stage2 包内 open_vocabulary_regions_data"
  DATA_DIR="./data/ConverSeg-Training-Data/converseg_stage2_data"
  JSONL="open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl"
else
  echo "ERROR: 找不到 Stage1 jsonl，请先下载 converseg_stage1_data（见 scripts/download_stage1_data.sh）"
  exit 1
fi

OUT="./ckpts/stage1_full"
NAME="ConVerSeg_stage1"
mkdir -p "$OUT" logs

"$PYTHON" -u train.py \
  --data-dir "$DATA_DIR" \
  --dataset-jsonl "$JSONL" \
  --model-cfg "$SAM_CFG" \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --device cuda \
  --steps 100000 \
  --lr 1e-4 \
  --wd 0.05 \
  --batch-size 6 \
  --acc 1 \
  --max-dim 1024 \
  --workers 4 \
  --log-every 50 \
  --save-every 10000 \
  --val-every 5000 \
  --val-count 4 \
  --out "$OUT" \
  --name "$NAME"

echo "Stage-1 done -> ${OUT}/${NAME}_final.torch"
