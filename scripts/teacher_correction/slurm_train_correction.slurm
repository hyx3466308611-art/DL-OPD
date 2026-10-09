#!/bin/bash
# 纠错教师 Stage-1（Head V1）。3B 默认；7B：
#   TEACHER_BACKBONE=teacher7b sbatch scripts/teacher_correction/slurm_train_correction.slurm
# 7B OOM：CORR_BS=1 CORR_ACC=6（脚本默认）；仍 OOM 则 --qwen-image-size 已是 256。
#
#SBATCH --job-name=corr_teacher
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/teacher_correction/slurm_%j.out
#SBATCH --error=logs/teacher_correction/slurm_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hanyuxi/ConverSeg-main}"
ROOT="$(pwd)"
export ROOT
mkdir -p logs/teacher_correction

# shellcheck disable=SC1091
source scripts/teacher_correction/_env.sh
mkdir -p logs/teacher_correction "$LOG_DIR" "$CORR_OUT"

activate_converseg
install_sam2_editable
check_assets

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-$DATA_ROOT}"
JSONL="${DATASET_JSONL:-$PHASE2_JSONL_REL}"
export ROOT DATA_DIR JSONL
if [[ ! -f "${DATA_DIR}/${JSONL}" ]]; then
  echo "ERROR: training manifest not found: ${DATA_DIR}/${JSONL}"
  echo "  Expected same layout as student Phase2: --data-dir ${ROOT}/data --dataset-jsonl phase2/phase2_mixed.jsonl"
  echo "  If missing, build with scripts/latet_bulid_phase1/mix_phase2_train.py or copy from your Phase12 setup."
  exit 1
fi

echo "=== Correction Teacher Train ==="
echo "ROOT=$ROOT TEACHER_BACKBONE=$TEACHER_BACKBONE INIT=$INIT_PRESET DATA=${DATA_DIR}/${JSONL}"
echo "OUT=$CORR_OUT NAME=$CORR_NAME"
head -1 "${DATA_DIR}/${JSONL}" || true

python - <<PY
import json, os, sys
root = os.environ.get("ROOT", ".")
data_dir = os.environ["DATA_DIR"]
jsonl = os.environ["JSONL"]
path = os.path.join(data_dir, jsonl)
sys.path.insert(0, root)
from converseg_paths import prepare_python_import_path
prepare_python_import_path()
from data.load_samples import load_samples_from_jsonl
n = len(load_samples_from_jsonl(data_dir, jsonl, num_samples=3, repo_root=root))
print("Precheck OK:", n, "sample(s) resolved from", path)
PY

python -u train_teacher_correction.py \
  "${TRAIN_COMMON[@]}" \
  --data-dir "$DATA_DIR" \
  --dataset-jsonl "$JSONL" \
  --init-preset "$INIT_PRESET" \
  2>&1 | tee -a "$TRAIN_LOG"

echo "Done. Checkpoints under $CORR_OUT (incl. ${CORR_NAME}_head_final.torch)"
