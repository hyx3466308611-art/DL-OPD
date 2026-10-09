#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export ROOT
# shellcheck disable=SC1091
source "$ROOT/scripts/teacher_correction/_env.sh"

echo "ROOT=$ROOT"
check_assets
[[ -f "${ROOT}/data/phase2/phase2_mixed.jsonl" ]] || {
  echo "Missing ${ROOT}/data/phase2/phase2_mixed.jsonl (same as student Phase2 training)"
  exit 1
}
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python -c "from converseg_paths import prepare_python_import_path; prepare_python_import_path(); import sam2; from data.mask_corrupter import MaskCorrupter; from models.teacher_correction_model import TeacherCorrectionHead; from train_teacher_correction import ensure_square_max_dim, resize_masks_continuous; from data.load_samples import load_samples_from_jsonl; print('imports OK')"
python -c "
from converseg_paths import prepare_python_import_path
prepare_python_import_path()
from data.load_samples import load_samples_from_jsonl
from pathlib import Path
root = Path('.').resolve()
load_samples_from_jsonl(root / 'data', 'phase2/phase2_mixed.jsonl', num_samples=1, repo_root=root)
print('phase2 jsonl sample resolve OK')
"
python -c "
from converseg_paths import prepare_python_import_path
prepare_python_import_path()
from train_teacher_correction import ensure_square_max_dim, resize_masks_continuous
import numpy as np
m = np.zeros((512, 768), np.float32)
m2 = ensure_square_max_dim(m, 1024, is_mask=True)
assert m2.shape == (1024, 1024), m2.shape
t = resize_masks_continuous([m2, m2], 1024, 1024)
assert tuple(t.shape) == (2, 1, 1024, 1024), t.shape
print('batch canvas / mask_t precheck OK')
"
echo "TEACHER_BACKBONE=${TEACHER_BACKBONE} OUT=${CORR_OUT} NAME=${CORR_NAME} BS=${CORR_BS} ACC=${CORR_ACC}"
echo "TEACHER_FROZEN_FLAG=${TEACHER_FROZEN_FLAG}"
if [[ "$TEACHER_BACKBONE" == "teacher7b" ]]; then
  echo "7B Phase2 init: ${TEACHER_P2}/${TEACHER7B_STAGE2_NAME}_final.torch"
fi
echo "Stage1 train:  sbatch --export=ALL,TEACHER_BACKBONE=${TEACHER_BACKBONE} scripts/teacher_correction/slurm_train_correction.slurm"
echo "Stage1 TRG:    sbatch --export=ALL,TEACHER_BACKBONE=${TEACHER_BACKBONE} scripts/teacher_correction/slurm_eval_trg.slurm"
echo "fix_v1 train:  sbatch --export=ALL,TEACHER_BACKBONE=${TEACHER_BACKBONE} scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm"
