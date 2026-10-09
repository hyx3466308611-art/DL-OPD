#!/usr/bin/env bash
# Stage-2: paper hyperparams (90k steps), resume Phase1 final
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE1_OUT="${STAGE1_OUT:-${ROOT}/ckpts/stage1_phase1mix}"
RESUME_TAG="${RESUME_TAG:-final}"

OUT="${OUT:-${ROOT}/ckpts/stage2_phase2mix}"
NAME="${NAME:-ConVerSeg_stage2}"
LOG="${LOG:-${ROOT}/logs/stage2_90k.log}"

SAM_CFG="configs/sam2/sam2_hiera_l.yaml"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"

RESUME_SAM="${STAGE1_OUT}/ConVerSeg_stage1_${RESUME_TAG}.torch"
RESUME_PLM="${STAGE1_OUT}/ConVerSeg_stage1_plm_${RESUME_TAG}.torch"
RESUME_LORA="${STAGE1_OUT}/lora_plm_adapter_${RESUME_TAG}"

for f in "$RESUME_SAM" "$RESUME_PLM"; do
  [[ -f "$f" ]] || { echo "Missing: $f (Phase1 done? try RESUME_TAG=100000)"; exit 1; }
done
[[ -f "$RESUME_LORA" || -d "$RESUME_LORA" ]] || { echo "Missing: $RESUME_LORA"; exit 1; }
[[ -f "${ROOT}/data/phase2/phase2_mixed.jsonl" ]] || {
  echo "Missing data/phase2/phase2_mixed.jsonl — run mix_phase2_train.py first"
  exit 1
}

mkdir -p "$OUT" "$(dirname "$LOG")"

echo "[stage2] resume SAM=$RESUME_SAM"
echo "[stage2] hyperparams: lr=1e-4 wd=0.05 batch=6 acc=1 max-dim=1024 steps=90000"

nohup "$PYTHON" -u "${ROOT}/train.py" \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase2/phase2_mixed.jsonl \
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
  --name "$NAME" \
  >> "$LOG" 2>&1 &

echo $! > "${ROOT}/logs/stage2_90k.pid"
echo "PID=$(cat "${ROOT}/logs/stage2_90k.pid")  log=$LOG"
sleep 8
tail -25 "$LOG"
