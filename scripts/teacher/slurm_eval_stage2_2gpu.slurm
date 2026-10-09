#!/bin/bash
# 7B 教师 Phase2：双卡并行 ConVerSeg 测评（每卡一份完整模型，按样本分片）
#
#   cd /home/hanyuxi/ConverSeg-main
#   sbatch scripts/teacher/slurm_eval_stage2_2gpu.slurm

#SBATCH --job-name=converseg-t-eval2
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/teacher/slurm-%x-%j.out
#SBATCH --error=logs/teacher/slurm-%x-%j.err

set -euo pipefail

ROOT=/home/hanyuxi/ConverSeg-main
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher/_train_env.sh"

CKPT="$STAGE2_OUT"
EVAL_LOG="${LOG_DIR}/eval_teacher_stage2_bf16.log"
SHARD_DIR="${LOG_DIR}/eval_shards_2gpu"

mkdir -p "$LOG_DIR" "$SHARD_DIR"
cd "$ROOT"
activate_converseg

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
install_sam2_editable

# jsonl 可能在 data/ConverSeg/ 或 data/ConverSeg/data/
if [[ -f "${ROOT}/data/ConverSeg/sam_seeded.jsonl" ]]; then
  EVAL_ROOT="${ROOT}/data/ConverSeg"
elif [[ -f "${ROOT}/data/ConverSeg/data/sam_seeded.jsonl" ]]; then
  EVAL_ROOT="${ROOT}/data/ConverSeg/data"
else
  echo "Missing ConVerSeg sam_seeded.jsonl under data/ConverSeg[/data]"
  exit 1
fi
echo "EVAL_ROOT=$EVAL_ROOT"

for f in \
  "${CKPT}/${NAME2}_final.torch" \
  "${CKPT}/${NAME2}_plm_final.torch" \
  "${CKPT}/lora_plm_adapter_final" \
  "${EVAL_ROOT}/sam_seeded.jsonl" \
  "${EVAL_ROOT}/human_annotated.jsonl" \
  "${PLM}/config.json" \
  "$SAM_CKPT"
do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done

rm -f "${SHARD_DIR}/shard0.jsonl" "${SHARD_DIR}/shard1.jsonl"

COMMON=(
  --final_ckpt "${CKPT}/${NAME2}_final.torch"
  --plm_ckpt "${CKPT}/${NAME2}_plm_final.torch"
  --lora_ckpt "${CKPT}/lora_plm_adapter_final"
  --plm-model-name "$PLM"
  --base_ckpt "$SAM_CKPT"
  --model_cfg "$SAM_CFG"
  --device cuda
  --precision bf16
  --eval_splits sam_seeded,human_annotated
  --num-shards 2
)

echo "Launching 2 shards on GPU0/GPU1 ..."
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  "${COMMON[@]}" \
  --shard-id 0 \
  --metrics-out "${SHARD_DIR}/shard0.jsonl" \
  > "${SHARD_DIR}/shard0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u eval.py \
  "${COMMON[@]}" \
  --shard-id 1 \
  --metrics-out "${SHARD_DIR}/shard1.jsonl" \
  > "${SHARD_DIR}/shard1.log" 2>&1 &
PID1=$!

wait "$PID0"
EC0=$?
wait "$PID1"
EC1=$?
echo "shard0 exit=$EC0 shard1 exit=$EC1"
[[ "$EC0" -eq 0 && "$EC1" -eq 0 ]] || { echo "A shard failed; see ${SHARD_DIR}/shard*.log"; exit 1; }

python -u scripts/teacher/merge_eval_shards.py \
  "${SHARD_DIR}/shard0.jsonl" \
  "${SHARD_DIR}/shard1.jsonl" \
  | tee -a "$EVAL_LOG"

echo "Per-GPU logs: ${SHARD_DIR}/shard0.log ${SHARD_DIR}/shard1.log"
echo "Merged metrics appended to $EVAL_LOG"
