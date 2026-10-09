#!/usr/bin/env bash
# shellcheck disable=SC2034
# 被 teacher/*.slurm source；勿直接 bash 运行

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
TEACHER_DIR="${ROOT}/teacher"
PLM="${TEACHER_DIR}/Qwen2.5-VL-7B"
SAM_CKPT="${TEACHER_DIR}/sam2_hiera_large.pt"
SAM_CFG=configs/sam2/sam2_hiera_l.yaml

STAGE1_OUT="${ROOT}/ckpts/teacher/stage1_phase1mix"
STAGE2_OUT="${ROOT}/ckpts/teacher/stage2_phase2mix"
NAME1=ConVerSeg_teacher_stage1
NAME2=ConVerSeg_teacher_stage2

LOG_DIR="${ROOT}/logs/teacher"
PHASE1_LOG="${LOG_DIR}/phase1_phase1mix.log"
PHASE2_LOG="${LOG_DIR}/phase2_phase2mix.log"

# 与学生 Phase12 相同（等效 batch=6）
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

activate_converseg() {
  if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
    conda activate converseg
  elif [[ -f /home/hanyuxi/miniconda311/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/hanyuxi/miniconda311/etc/profile.d/conda.sh
    conda activate converseg
  else
    echo "ERROR: conda.sh not found"
    exit 1
  fi
}

install_sam2_editable() {
  if [[ -d "${ROOT}/sam2_repo" ]]; then
    pip install -e "${ROOT}/sam2_repo" -q
  elif [[ -d "${ROOT}/sam2" ]]; then
    pip install -e "${ROOT}/sam2" -q
  else
    echo "Missing SAM2 clone under ${ROOT}/sam2_repo or sam2"
    exit 1
  fi
}

check_teacher_assets() {
  local f
  for f in "$SAM_CKPT" "${PLM}/config.json"; do
    [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
  done
}
