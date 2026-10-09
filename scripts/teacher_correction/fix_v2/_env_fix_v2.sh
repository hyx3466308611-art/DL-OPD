#!/usr/bin/env bash
# fix_v2 paths and hyperparameters; source from Slurm scripts.

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher_correction/_env.sh"

TEACHER_BACKBONE="${TEACHER_BACKBONE:-author3b}"
apply_teacher_backbone_env

case "$TEACHER_BACKBONE" in
  author3b)
    CORR_BS="${CORR_BS:-2}"
    CORR_ACC="${CORR_ACC:-3}"
    INIT_CORR_DIR="${INIT_CORR_DIR:-${ROOT}/ckpts/teacher_correction/author3b_fix_v1}"
    INIT_NAME="CorrectionTeacher_3B_fix_v1"
    CORR_OUT="${ROOT}/ckpts/teacher_correction/author3b_fix_v2"
    CORR_NAME="CorrectionTeacher_3B_fix_v2"
    ;;
  teacher7b)
    CORR_BS="${CORR_BS:-1}"
    CORR_ACC="${CORR_ACC:-6}"
    INIT_CORR_DIR="${INIT_CORR_DIR:-${ROOT}/ckpts/teacher_correction/teacher7b_fix_v1}"
    INIT_NAME="CorrectionTeacher_7B_fix_v1"
    CORR_OUT="${ROOT}/ckpts/teacher_correction/teacher7b_fix_v2"
    CORR_NAME="CorrectionTeacher_7B_fix_v2"
    ;;
  *)
    echo "Unknown TEACHER_BACKBONE=$TEACHER_BACKBONE"
    exit 1
    ;;
esac

SAM_FIN="${INIT_CORR_DIR}/${INIT_NAME}_final.torch"
PLM_FIN="${INIT_CORR_DIR}/${INIT_NAME}_plm_final.torch"
HEAD_FIN="${INIT_CORR_DIR}/${INIT_NAME}_head_final.torch"
LORA_FIN="${INIT_CORR_DIR}/lora_plm_adapter_final"

FIX_V2_STEPS="${FIX_V2_STEPS:-30000}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PHASE2_JSONL_REL="${PHASE2_JSONL_REL:-phase2/phase2_mixed.jsonl}"
CONVERSEG_EVAL_ROOT="${CONVERSEG_EVAL_ROOT:-${ROOT}/data/ConverSeg/data}"
HUMAN_JSONL_REL="${HUMAN_JSONL_REL:-human_annotated.jsonl}"
# 0：禁止把 ConverSeg 测评集 human_annotated 混进训练
HUMAN_REPEAT="${HUMAN_REPEAT:-0}"
LOG_DIR="${ROOT}/logs/teacher_correction/fix_v2"
TRAIN_LOG="${LOG_DIR}/train_fix_v2_${TEACHER_BACKBONE}.log"

check_fix_v2_init() {
  local f
  for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
    [[ -f "$f" ]] || {
      echo "Missing fix_v2 init: $f"
      echo "Train/copy fix_v1 for TEACHER_BACKBONE=$TEACHER_BACKBONE first."
      exit 1
    }
  done
}
