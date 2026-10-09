#!/bin/bash
# =============================================================================
# LION GPU Lab — 单卡 Phase1+Phase2 学生模型（ConVerSeg，非 torch.distributed）
# 手册: LION_GPU_学生使用手册_通用版 — normal 正式训练、必须 --gres=gpu:N
#
# 提交（在项目根目录）:
#   cd /home/hanyuxi/ConverSeg-main
#   mkdir -p logs
#   sbatch scripts/slurm_phase12_train.slurm
#
# 查看: squeue -u "$USER"     取消: scancel <JobID>
# 本机状态: lion-status / sinfo
#
# normal 分区: 默认时限 24h，最长 72h（硕士单卡最多 2 张 GPU 同时占用）
# 勿写 CUDA_VISIBLE_DEVICES；勿指定物理 GPU 编号。
#
# 超参与官方 README train.py 一致（Stage1/2 学生，非 RepairOPD）:
#   SAM2 Hiera-L | Qwen2.5-VL-3B | LoRA r=16 α=32
#   lr=1e-4 wd=0.05 等效 batch=6（单卡 48G: batch-size 2 × acc 3）max-dim=1024
#   --plm-gradient-checkpointing 省 Qwen 显存
#   Phase1 100k steps | Phase2 90k steps（从 Phase1 *_final 续训）
#   save-every=50000（仅 50k 中间档 + 结束 final；非作者默认 5000，省磁盘）
# =============================================================================

#SBATCH --job-name=converseg-p12
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
# Hydra 配置名（在 sam2 包内），不是仓库根目录 configs/
SAM_CFG=configs/sam2/sam2_hiera_l.yaml

mkdir -p "${ROOT}/logs" "$STAGE1_OUT" "$STAGE2_OUT"
cd "$ROOT"

if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg
elif [[ -f /home/hanyuxi/miniconda311/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda311/etc/profile.d/conda.sh
  conda activate converseg
else
  echo "ERROR: conda.sh not found — edit scripts/slurm_phase12_train.slurm"
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "JobID=${SLURM_JOB_ID} Node=${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -V
nvidia-smi -L || true

if [[ -d "${ROOT}/sam2_repo" ]]; then
  pip install -e "${ROOT}/sam2_repo" -q
elif [[ -d "${ROOT}/sam2" ]]; then
  pip install -e "${ROOT}/sam2" -q
else
  echo "Missing SAM2 clone: ${ROOT}/sam2_repo (git clone facebookresearch/sam2 sam2_repo)"
  exit 1
fi

SAM2_YAML="${ROOT}/sam2_repo/sam2/configs/sam2/sam2_hiera_l.yaml"
[[ -f "$SAM2_YAML" ]] || SAM2_YAML="${ROOT}/sam2/sam2/configs/sam2/sam2_hiera_l.yaml"
[[ -f "$SAM2_YAML" ]] || {
  echo "Missing SAM2 config yaml (expected under sam2/sam2/configs/sam2/): $SAM2_YAML"
  exit 1
}

for f in "$SAM_CKPT" "${PLM}/config.json" "${ROOT}/data/phase1/phase1_mixed.jsonl"; do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done

TRAIN_COMMON=(
  --model-cfg "$SAM_CFG"
  --checkpoint "$SAM_CKPT"
  --plm-model-name "$PLM"
  --device cuda
  --lr 1e-4
  --wd 0.05
  --batch-size 2
  --acc 3
  --max-dim 1024
  --lora-r 16
  --lora-alpha 32
  --plm-gradient-checkpointing
  --qwen-image-size 256
  --seed 1337
  --workers 4
  --log-every 50
  --val-every 5000
  --val-count 4
)

echo "========== Phase1 100000 steps $(date -Is) =========="
python -u train.py \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase1/phase1_mixed.jsonl \
  "${TRAIN_COMMON[@]}" \
  --steps 100000 \
  --save-every 50000 \
  --out "$STAGE1_OUT" \
  --name "$NAME1" \
  2>&1 | tee -a "${ROOT}/logs/phase1_phase1mix.log"

FINAL_SAM="${STAGE1_OUT}/${NAME1}_final.torch"
FINAL_PLM="${STAGE1_OUT}/${NAME1}_plm_final.torch"
FINAL_LORA="${STAGE1_OUT}/lora_plm_adapter_final"
for f in "$FINAL_SAM" "$FINAL_PLM"; do
  [[ -f "$f" ]] || { echo "Phase1 missing: $f"; exit 1; }
done
[[ -f "$FINAL_LORA" || -d "$FINAL_LORA" ]] || { echo "Phase1 missing LoRA: $FINAL_LORA"; exit 1; }

[[ -f "${ROOT}/data/phase2/phase2_mixed.jsonl" ]] || {
  echo "Missing ${ROOT}/data/phase2/phase2_mixed.jsonl"
  exit 1
}

echo "========== Phase2 90000 steps $(date -Is) =========="
python -u train.py \
  --data-dir "${ROOT}/data" \
  --dataset-jsonl phase2/phase2_mixed.jsonl \
  "${TRAIN_COMMON[@]}" \
  --steps 90000 \
  --save-every 50000 \
  --resume-sam-path "$FINAL_SAM" \
  --resume-plm-path "$FINAL_PLM" \
  --resume-lora-path "$FINAL_LORA" \
  --out "$STAGE2_OUT" \
  --name "$NAME2" \
  2>&1 | tee -a "${ROOT}/logs/phase2_phase2mix.log"

echo "========== Student ready $(date -Is) =========="
echo "SAM:  ${STAGE2_OUT}/${NAME2}_final.torch"
echo "PLM:  ${STAGE2_OUT}/${NAME2}_plm_final.torch"
echo "LoRA: ${STAGE2_OUT}/lora_plm_adapter_final"
