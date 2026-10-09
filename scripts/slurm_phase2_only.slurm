#!/bin/bash
# LION normal — 仅 Phase2（Phase1 已在其它作业完成时使用）
# 依赖上一作业成功（可选）:
#   sbatch --dependency=afterok:<PHASE1_JOBID> scripts/slurm_phase2_only.slurm

#SBATCH --job-name=converseg-p2
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
ROOT=/home/hanyuxi/ConverSeg-main
STAGE1_OUT="${ROOT}/ckpts/stage1_phase1mix"
STAGE2_OUT="${ROOT}/ckpts/stage2_phase2mix"
NAME1=ConVerSeg_stage1
NAME2=ConVerSeg_stage2
PLM="${ROOT}/student/Qwen2.5-VL-3B"
SAM_CKPT="${ROOT}/student/sam2_hiera_large.pt"
SAM_CFG=configs/sam2/sam2_hiera_l.yaml

mkdir -p "${ROOT}/logs" "$STAGE2_OUT"
cd "$ROOT"

source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/hanyuxi/miniconda311/etc/profile.d/conda.sh
conda activate converseg

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

FINAL_SAM="${STAGE1_OUT}/${NAME1}_final.torch"
FINAL_PLM="${STAGE1_OUT}/${NAME1}_plm_final.torch"
FINAL_LORA="${STAGE1_OUT}/lora_plm_adapter_final"

python -u train.py \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase2/phase2_mixed.jsonl \
  --model-cfg "$SAM_CFG" \
  --checkpoint "$SAM_CKPT" \
  --plm-model-name "$PLM" \
  --resume-sam-path "$FINAL_SAM" \
  --resume-plm-path "$FINAL_PLM" \
  --resume-lora-path "$FINAL_LORA" \
  --device cuda \
  --steps 90000 \
  --lr 1e-4 --wd 0.05 --batch-size 6 --acc 1 --max-dim 1024 \
  --workers 4 --log-every 50 --val-every 5000 --val-count 4 \
  --save-every 5000 \
  --out "$STAGE2_OUT" --name "$NAME2" \
  2>&1 | tee -a "${ROOT}/logs/phase2_phase2mix.log"
