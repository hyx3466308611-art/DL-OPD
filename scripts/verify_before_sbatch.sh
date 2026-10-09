#!/usr/bin/env bash
# 提交 sbatch 前快速检查（在 ConVerSeg-main 根目录）
set -uo pipefail
ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"
fail=0

echo "=== converseg_paths ==="
grep -n prepare_python_import_path converseg_paths.py || { echo "FAIL: 旧 converseg_paths.py — 请 scp 本机 converseg_paths.py"; fail=1; }

echo "=== train.py ==="
grep -n prepare_python_import_path train.py || { echo "FAIL: 旧 train.py"; fail=1; }

echo "=== slurm batch（应为 batch-size 2, acc 3, gradient-checkpointing）==="
grep -nE 'batch-size|--acc |gradient-checkpoint' scripts/slurm_phase12_train.slurm || fail=1

echo "=== import 试跑 ==="
if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg 2>/dev/null || true
fi
python -c "from converseg_paths import prepare_python_import_path; prepare_python_import_path(); print('import OK')" || fail=1

if [[ "$fail" -ne 0 ]]; then
  echo "=== 未通过，修完再 sbatch ==="
  exit 1
fi
echo "=== 通过，可提交: sbatch scripts/slurm_phase12_train.slurm ==="
exit 0
