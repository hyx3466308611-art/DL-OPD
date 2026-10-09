#!/usr/bin/env bash
set -uo pipefail
ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/teacher/_train_env.sh"

fail=0
check() { [[ -e "$1" ]] && echo "  [OK] $1" || { echo "  [FAIL] $1"; fail=1; }; }

echo "=== Teacher Phase12 预检 ==="
check "${PLM}/config.json"
check "$SAM_CKPT"
check "${ROOT}/data/phase1/phase1_mixed.jsonl"
check "${ROOT}/data/phase2/phase2_mixed.jsonl"
check "${ROOT}/scripts/teacher/slurm_phase12_train.slurm"

if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg 2>/dev/null || true
fi

python -c "
from converseg_paths import resolve_plm_dir, prepare_python_import_path
prepare_python_import_path()
resolve_plm_dir('${PLM}')
print('  [OK] PLM path resolves')
" || fail=1

python -c "from sam2.build_sam import build_sam2; print('  [OK] sam2 import')" || fail=1

if [[ "$fail" -ne 0 ]]; then
  echo "未通过"
  exit 1
fi
echo "通过。提交: sbatch scripts/teacher/slurm_phase12_train.slurm"
