#!/usr/bin/env bash
# TRG fix v1 — HeadV2, full GT, negatives, dual-mode; final checkpoints only
#
# TEACHER_BACKBONE=author3b（默认）→ author3b_fix_v1，init=author3b
# TEACHER_BACKBONE=teacher7b        → teacher7b_fix_v1，init=teacher7b（需先训完 CorrectionTeacher_7B）

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher_correction/_env.sh"

TEACHER_BACKBONE="${TEACHER_BACKBONE:-author3b}"
apply_teacher_backbone_env
CORR_BS="${CORR_BS:-2}"
CORR_ACC="${CORR_ACC:-3}"

INIT_NAME=""
case "$TEACHER_BACKBONE" in
  author3b)
    CORR_OUT="${ROOT}/ckpts/teacher_correction/author3b_fix_v1"
    CORR_NAME="CorrectionTeacher_3B_fix_v1"
    INIT_CORR_DIR="${INIT_CORR_DIR:-${ROOT}/ckpts/teacher_correction/author3b}"
    INIT_NAME="CorrectionTeacher_3B"
    ;;
  teacher7b)
    CORR_OUT="${ROOT}/ckpts/teacher_correction/teacher7b_fix_v1"
    CORR_NAME="CorrectionTeacher_7B_fix_v1"
    INIT_CORR_DIR="${INIT_CORR_DIR:-${ROOT}/ckpts/teacher_correction/teacher7b}"
    INIT_NAME="CorrectionTeacher_7B"
    CORR_BS="${CORR_BS:-1}"
    CORR_ACC="${CORR_ACC:-6}"
    ;;
  *)
    echo "Unknown TEACHER_BACKBONE=$TEACHER_BACKBONE"
    exit 1
    ;;
esac

EVAL_RECOVERY_DIR="${CORR_OUT}/eval_recovery"
TEACHER_FROZEN_FLAG="${CORR_OUT}/teacher_frozen_ready.flag"
LOG_DIR="${ROOT}/logs/teacher_correction/fix_v1"
TRAIN_LOG="${LOG_DIR}/train_fix_v1_${TEACHER_BACKBONE}.log"
EVAL_LOG="${LOG_DIR}/eval_trg_fix_v1_${TEACHER_BACKBONE}.log"

CONVERSEG_EVAL_ROOT="${CONVERSEG_EVAL_ROOT:-${ROOT}/data/ConverSeg/data}"
HUMAN_JSONL_REL="${HUMAN_JSONL_REL:-human_annotated.jsonl}"
# 0：禁止把 ConverSeg 测评集 human_annotated 混进训练
HUMAN_REPEAT="${HUMAN_REPEAT:-0}"
NEGATIVE_JSONL_REL="${NEGATIVE_JSONL_REL:-ConverSeg-Training-Data/converseg_stage2_data/conversational_negative_data/conversational_negative_data.jsonl}"

SAM_FIN="${INIT_CORR_DIR}/${INIT_NAME}_final.torch"
PLM_FIN="${INIT_CORR_DIR}/${INIT_NAME}_plm_final.torch"
HEAD_FIN="${INIT_CORR_DIR}/${INIT_NAME}_head_final.torch"
LORA_FIN="${INIT_CORR_DIR}/lora_plm_adapter_final"

FIX_STEPS="${FIX_STEPS:-30000}"

TRAIN_FIX_COMMON=(
  --model-cfg "$SAM_CFG"
  --checkpoint "$SAM_CKPT"
  --plm-model-name "$PLM"
  --teacher-backbone "$TEACHER_BACKBONE"
  --device cuda
  --steps "$FIX_STEPS"
  --lr 2e-5
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
  --save-every 0
  --out "$CORR_OUT"
  --name "$CORR_NAME"
  --resume-sam-path "$SAM_FIN"
  --resume-plm-path "$PLM_FIN"
  --resume-head-path "$HEAD_FIN"
  --resume-lora-path "$LORA_FIN"
  --human-jsonl "$HUMAN_JSONL_REL"
  --human-repeat "$HUMAN_REPEAT"
  --negative-jsonl "$NEGATIVE_JSONL_REL"
  --negative-sample-prob 0.175
  --dual-mode-prob 0.5
  --lam-neg 3.0
  --lam-preserve 0.4
  --preserve-iou-thresh 0.85
  --boundary-elastic-boost 0.18
  --curriculum-c-start 0.45
)

check_fix_init() {
  for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
    [[ -f "$f" ]] || {
      echo "Missing init checkpoint: $f"
      echo "  TEACHER_BACKBONE=$TEACHER_BACKBONE INIT_CORR_DIR=$INIT_CORR_DIR"
      exit 1
    }
  done
}
