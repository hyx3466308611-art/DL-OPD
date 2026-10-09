#!/usr/bin/env bash
# Stage1 从头训练（ConVerSeg 论文超参）
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-${ROOT}/ckpts/stage1_full}"
NAME="${NAME:-ConVerSeg_stage1}"
LOG="${LOG:-${ROOT}/logs/stage1_100k.log}"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"

echo "[info] Clearing old numbered checkpoints in ${OUT} (optional broken files)..."
rm -f "${OUT}/${NAME}"_[0-9]*.torch "${OUT}/${NAME}_plm_"[0-9]*.torch 2>/dev/null || true
rm -rf "${OUT}"/lora_plm_adapter_[0-9]* 2>/dev/null || true

df -h /home/hanyuxi | tail -1

nohup "$PYTHON" -u "${ROOT}/train.py" \
  --data-dir "${ROOT}/data/ConverSeg-Training-Data/converseg_stage2_data" \
  --dataset-jsonl open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl \
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
  --save-every 10000 \
  --val-every 5000 \
  --out "${OUT}" \
  --name "${NAME}" \
  >> "${LOG}" 2>&1 &

echo $! > "${ROOT}/logs/stage1_100k.pid"
echo "PID=$(cat "${ROOT}/logs/stage1_100k.pid")  log=${LOG}"
sleep 5
tail -15 "${LOG}"
