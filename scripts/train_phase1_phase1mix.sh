#!/usr/bin/env bash
# Phase1: paper hyperparams (100k steps, batch 6, wd 0.05, max-dim 1024)
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-${ROOT}/ckpts/stage1_phase1mix}"
NAME="${NAME:-ConVerSeg_stage1}"
LOG="${LOG:-${ROOT}/logs/stage1_phase1mix_100k.log}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"

mkdir -p "$OUT" "$(dirname "$LOG")"

echo "[train] PYTHON=$PYTHON"
echo "[train] OUT=$OUT LOG=$LOG"
echo "[train] dataset=phase1/phase1_mixed.jsonl"
echo "[train] hyperparams: lr=1e-4 wd=0.05 batch=6 acc=1 max-dim=1024 steps=100000"

nohup "$PYTHON" -u "${ROOT}/train.py" \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase1/phase1_mixed.jsonl \
  --model-cfg configs/sam2/sam2_hiera_l.yaml \
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
  --save-every 50000 \
  --val-every 5000 \
  --val-count 4 \
  --out "$OUT" \
  --name "$NAME" \
  >> "$LOG" 2>&1 &

echo $! > "${ROOT}/logs/stage1_phase1mix.pid"
echo "PID=$(cat "${ROOT}/logs/stage1_phase1mix.pid")"
sleep 5
tail -25 "$LOG"
