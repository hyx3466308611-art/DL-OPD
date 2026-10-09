#!/usr/bin/env bash
# Phase1 (100k) -> Phase2 (90k) in one process chain (for Slurm / foreground).
# Paper hyperparams; single GPU (5880). Do not use nohup here.
set -euo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
source "${ROOT}/scripts/_converseg_env.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE1_OUT="${STAGE1_OUT:-${ROOT}/ckpts/stage1_phase1mix}"
STAGE2_OUT="${STAGE2_OUT:-${ROOT}/ckpts/stage2_phase2mix}"
NAME1="${NAME1:-ConVerSeg_stage1}"
NAME2="${NAME2:-ConVerSeg_stage2}"
PLM="${PLM:-${ROOT}/student/Qwen2.5-VL-3B}"
SAM_CKPT="${SAM_CKPT:-${ROOT}/student/sam2_hiera_large.pt}"

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR" "$STAGE1_OUT" "$STAGE2_OUT"

PH1_LOG="${LOG_DIR}/phase1_phase1mix.log"
PH2_LOG="${LOG_DIR}/phase2_phase2mix.log"

COMMON=(
  --model-cfg configs/sam2/sam2_hiera_l.yaml
  --checkpoint "$SAM_CKPT"
  --plm-model-name "$PLM"
  --device cuda
  --lr 1e-4
  --wd 0.05
  --batch-size 6
  --acc 1
  --max-dim 1024
  --workers 4
  --log-every 50
  --val-every 5000
  --val-count 4
)

echo "========== Phase1 start $(date -Is) =========="
echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

[[ -f "${ROOT}/data/phase1/phase1_mixed.jsonl" ]] || {
  echo "Missing ${ROOT}/data/phase1/phase1_mixed.jsonl"
  exit 1
}

"$PYTHON" -u "${ROOT}/train.py" \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase1/phase1_mixed.jsonl \
  "${COMMON[@]}" \
  --steps 100000 \
  --save-every 50000 \
  --out "$STAGE1_OUT" \
  --name "$NAME1" \
  2>&1 | tee -a "$PH1_LOG"

FINAL_SAM="${STAGE1_OUT}/${NAME1}_final.torch"
FINAL_PLM="${STAGE1_OUT}/${NAME1}_plm_final.torch"
FINAL_LORA="${STAGE1_OUT}/lora_plm_adapter_final"
for f in "$FINAL_SAM" "$FINAL_PLM"; do
  [[ -f "$f" ]] || { echo "Phase1 missing checkpoint: $f"; exit 1; }
done
[[ -f "$FINAL_LORA" || -d "$FINAL_LORA" ]] || { echo "Phase1 missing LoRA: $FINAL_LORA"; exit 1; }

echo "========== Phase2 start $(date -Is) =========="

[[ -f "${ROOT}/data/phase2/phase2_mixed.jsonl" ]] || {
  echo "Missing ${ROOT}/data/phase2/phase2_mixed.jsonl"
  exit 1
}

"$PYTHON" -u "${ROOT}/train.py" \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase2/phase2_mixed.jsonl \
  "${COMMON[@]}" \
  --steps 90000 \
  --save-every 5000 \
  --resume-sam-path "$FINAL_SAM" \
  --resume-plm-path "$FINAL_PLM" \
  --resume-lora-path "$FINAL_LORA" \
  --out "$STAGE2_OUT" \
  --name "$NAME2" \
  2>&1 | tee -a "$PH2_LOG"

echo "========== Done $(date -Is) =========="
echo "Stage1 final: $FINAL_SAM"
echo "Stage2 final: ${STAGE2_OUT}/${NAME2}_final.torch"
