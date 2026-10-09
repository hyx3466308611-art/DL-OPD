#!/usr/bin/env bash
# shellcheck disable=SC2034
# Stage 1 纠错教师：训练 + TRG 评测

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
TEACHER_DIR="${ROOT}/teacher"
STUDENT_DIR="${ROOT}/student"

# author3b（默认）| teacher7b（自训 7B Phase2 → 纠错教师）
TEACHER_BACKBONE="${TEACHER_BACKBONE:-author3b}"

AUTHOR_DIR="${ROOT}/checkpoints/ConVerSeg-Net-3B"
TEACHER_P2="${ROOT}/ckpts/teacher/stage2_phase2mix"
TEACHER7B_STAGE2_NAME="${TEACHER7B_STAGE2_NAME:-ConVerSeg_teacher_stage2}"

# 等效 batch=6。7B 默认 1×6，降低单卡 48G OOM；可用 CORR_BS / CORR_ACC 覆盖
CORR_BS="${CORR_BS:-}"
CORR_ACC="${CORR_ACC:-}"

LOG_DIR="${ROOT}/logs/teacher_correction"
TRAIN_LOG="${LOG_DIR}/train_correction.log"
EVAL_LOG="${LOG_DIR}/eval_trg.log"

# 与学生 train.py 一致：manifest 路径相对 data/ 根目录
DATA_ROOT="${ROOT}/data"
PHASE2_JSONL="${DATA_ROOT}/phase2/phase2_mixed.jsonl"
PHASE2_JSONL_REL="${PHASE2_JSONL_REL:-phase2/phase2_mixed.jsonl}"

SAM_CFG=configs/sam2/sam2_hiera_l.yaml

apply_teacher_backbone_env() {
  case "$TEACHER_BACKBONE" in
    author3b)
      PLM="${STUDENT_DIR}/Qwen2.5-VL-3B"
      SAM_CKPT="${STUDENT_DIR}/sam2_hiera_large.pt"
      CORR_NAME=CorrectionTeacher_3B
      INIT_PRESET="${INIT_PRESET:-author3b_phase2}"
      CORR_OUT="${ROOT}/ckpts/teacher_correction/author3b"
      ;;
    teacher7b)
      PLM="${TEACHER_DIR}/Qwen2.5-VL-7B"
      SAM_CKPT="${TEACHER_DIR}/sam2_hiera_large.pt"
      CORR_NAME=CorrectionTeacher_7B
      INIT_PRESET="${INIT_PRESET:-teacher7b_phase2}"
      CORR_OUT="${ROOT}/ckpts/teacher_correction/teacher7b"
      CORR_BS="${CORR_BS:-1}"
      CORR_ACC="${CORR_ACC:-6}"
      ;;
    *)
      echo "Unknown TEACHER_BACKBONE=$TEACHER_BACKBONE (use author3b or teacher7b)"
      exit 1
      ;;
  esac
}

apply_teacher_backbone_env

# 3B 保持 2×3；未在 case 里设过时在此补默认
CORR_BS="${CORR_BS:-2}"
CORR_ACC="${CORR_ACC:-3}"

# 与 CORR_OUT / TEACHER_BACKBONE 绑定，供 TRG 评测 Slurm 引用
EVAL_RECOVERY_DIR="${CORR_OUT}/eval_recovery"
TEACHER_FROZEN_FLAG="${CORR_OUT}/teacher_frozen_ready.flag"

TRAIN_COMMON=(
  --model-cfg "$SAM_CFG"
  --checkpoint "$SAM_CKPT"
  --plm-model-name "$PLM"
  --teacher-backbone "$TEACHER_BACKBONE"
  --device cuda
  --steps 100000
  --lr 5e-5
  --wd 0.05
  --batch-size "$CORR_BS"
  --acc "$CORR_ACC"
  --max-dim 1024
  --lora-r 16
  --lora-alpha 32
  --plm-gradient-checkpointing
  --qwen-image-size 256
  --seed 1337
  --workers 0
  --log-every 10
  --save-every 50000
  --out "$CORR_OUT"
  --name "$CORR_NAME"
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
    echo "Missing SAM2 clone"
    exit 1
  fi
}

check_assets() {
  [[ -f "$SAM_CKPT" ]] || { echo "Missing $SAM_CKPT"; exit 1; }
  [[ -f "${PLM}/config.json" ]] || { echo "Missing PLM $PLM"; exit 1; }
  if [[ "$TEACHER_BACKBONE" == "teacher7b" ]]; then
    local sam_fin="${TEACHER_P2}/${TEACHER7B_STAGE2_NAME}_final.torch"
    local plm_fin="${TEACHER_P2}/${TEACHER7B_STAGE2_NAME}_plm_final.torch"
    [[ -f "$sam_fin" ]] || { echo "Missing 7B Phase2 SAM: $sam_fin"; exit 1; }
    [[ -f "$plm_fin" ]] || { echo "Missing 7B Phase2 PLM: $plm_fin"; exit 1; }
  fi
}
