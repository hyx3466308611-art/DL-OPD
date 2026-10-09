#!/bin/bash
# ConVerSeg 官方 benchmark 评测（Phase2 学生 final 权重）
# 数据: data/ConverSeg/{sam_seeded,human_annotated}.jsonl + images/ + masks/
#
# 提交:
#   cd /home/hanyuxi/ConverSeg-main
#   mkdir -p logs
#   sbatch scripts/slurm_eval_stage2.slurm

#SBATCH --job-name=converseg-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

ROOT=/home/hanyuxi/ConverSeg-main
CKPT="${ROOT}/ckpts/stage2_phase2mix"
PLM="${ROOT}/student/Qwen2.5-VL-3B"
SAM="${ROOT}/student/sam2_hiera_large.pt"
EVAL_ROOT="${ROOT}/data/ConverSeg"

mkdir -p "${ROOT}/logs"
cd "$ROOT"

if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg
else
  echo "ERROR: conda.sh not found"
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [[ -d "${ROOT}/sam2_repo" ]]; then
  pip install -e "${ROOT}/sam2_repo" -q
elif [[ -d "${ROOT}/sam2" ]]; then
  pip install -e "${ROOT}/sam2" -q
fi

for f in \
  "${CKPT}/ConVerSeg_stage2_final.torch" \
  "${CKPT}/ConVerSeg_stage2_plm_final.torch" \
  "${CKPT}/lora_plm_adapter_final" \
  "${EVAL_ROOT}/sam_seeded.jsonl" \
  "${EVAL_ROOT}/human_annotated.jsonl" \
  "${PLM}/config.json" \
  "$SAM"
do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done

python -u eval.py \
  --final_ckpt "${CKPT}/ConVerSeg_stage2_final.torch" \
  --plm_ckpt "${CKPT}/ConVerSeg_stage2_plm_final.torch" \
  --lora_ckpt "${CKPT}/lora_plm_adapter_final" \
  --plm-model-name "$PLM" \
  --base_ckpt "$SAM" \
  --model_cfg configs/sam2/sam2_hiera_l.yaml \
  --device cuda \
  --eval_splits sam_seeded,human_annotated \
  2>&1 | tee -a "${ROOT}/logs/eval_stage2_benchmark.log"

echo "Done. giou/ciou in logs/eval_stage2_benchmark.log"
