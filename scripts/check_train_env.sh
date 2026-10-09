#!/usr/bin/env bash
# 训练前一键自检（在 ConVerSeg-main 根目录，或设 ROOT=...）
# 用法: bash scripts/check_train_env.sh
set -uo pipefail

ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"

ok=0
fail=0
check() {
  local msg="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  [OK]   $msg"
    ok=$((ok + 1))
  else
    echo "  [FAIL] $msg"
    fail=$((fail + 1))
  fi
}
check_file() { [[ -e "$1" ]]; }

echo "=== ConVerSeg 训练环境检查 ==="
echo "ROOT=$ROOT"
echo "date: $(date -Is 2>/dev/null || date)"
echo

echo "--- Slurm（可选）---"
if command -v lion-user-show >/dev/null 2>&1; then
  lion-user-show "${USER:-hanyuxi}" 2>&1 | head -20 || true
else
  echo "  (skip lion-user-show)"
fi
echo

echo "--- Conda / Python ---"
if [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg 2>/dev/null || true
fi
which python || true
python -V || true
echo

echo "--- 代码（需从本机同步的最新文件）---"
check "converseg_paths.py" check_file converseg_paths.py
check "models/language_adapter.py" check_file models/language_adapter.py
check "train.py" check_file train.py
check "scripts/slurm_phase12_train.slurm" check_file scripts/slurm_phase12_train.slurm
check "scripts/_converseg_env.sh" check_file scripts/_converseg_env.sh
echo

echo "--- SAM2（clone 建议名 sam2_repo/，避免与 import sam2 冲突）---"
if [[ -f sam2/setup.py && ! -f sam2_repo/setup.py ]]; then
  echo "  [提示] 可将 sam2 重命名为 sam2_repo:  mv sam2 sam2_repo && pip install -e sam2_repo"
fi
if [[ -f sam2_repo/setup.py ]]; then
  check "sam2_repo/setup.py" check_file sam2_repo/setup.py
  check "sam2_repo/sam2/configs/sam2/sam2_hiera_l.yaml" check_file sam2_repo/sam2/configs/sam2/sam2_hiera_l.yaml
elif [[ -f sam2/setup.py ]]; then
  check "sam2/setup.py" check_file sam2/setup.py
  check "sam2/sam2/configs/sam2/sam2_hiera_l.yaml" check_file sam2/sam2/configs/sam2/sam2_hiera_l.yaml
else
  echo "  [FAIL] 缺少 sam2/ 或 sam2_repo/（需 git clone + pip install -e）"
  fail=$((fail + 1))
fi
check "student/sam2_hiera_large.pt" check_file student/sam2_hiera_large.pt
echo

echo "--- Qwen 基座（本地目录，含 config.json）---"
if check_file student/Qwen2.5-VL-3B/config.json; then
  check "student/Qwen2.5-VL-3B/config.json" check_file student/Qwen2.5-VL-3B/config.json
elif check_file pretrain/Qwen2.5-VL-3B-Instruct/config.json; then
  check "pretrain/Qwen2.5-VL-3B-Instruct/config.json" check_file pretrain/Qwen2.5-VL-3B-Instruct/config.json
else
  echo "  [FAIL] 找不到 Qwen（student/Qwen2.5-VL-3B 或 pretrain/...）"
  fail=$((fail + 1))
fi
echo

echo "--- 训练数据 jsonl ---"
check "data/phase1/phase1_mixed.jsonl" check_file data/phase1/phase1_mixed.jsonl
check "data/phase2/phase2_mixed.jsonl" check_file data/phase2/phase2_mixed.jsonl
echo

echo "--- Python 依赖（huggingface-hub 须 <1.0）---"
python - <<'PY' || fail=$((fail + 1))
import sys
try:
    import huggingface_hub as h
    v = h.__version__
    major = int(v.split(".")[0])
    if major >= 1:
        print("  [FAIL] huggingface-hub==%s (需要 >=0.34,<1.0)" % v)
        print("         pip install 'huggingface-hub>=0.34.0,<1.0'")
        sys.exit(1)
    print("  [OK]   huggingface-hub==%s" % v)
except Exception as e:
    print("  [FAIL] huggingface_hub import:", e)
    sys.exit(1)
try:
    from converseg_paths import (
        resolve_plm_dir,
        default_sam_config_str,
        sam2_hiera_l_yaml_on_disk,
        prepare_python_import_path,
    )
    prepare_python_import_path()
    print("  [OK]   PLM dir:", resolve_plm_dir(None))
    print("  [OK]   --model-cfg (Hydra):", default_sam_config_str())
    y = sam2_hiera_l_yaml_on_disk()
    print("  [OK]   yaml on disk:", y if y else "(install sam2 package to use Hydra)")
except Exception as e:
    print("  [FAIL] converseg_paths:", e)
    sys.exit(1)
try:
    from sam2.build_sam import build_sam2
    print("  [OK]   import sam2.build_sam")
except Exception as e:
    print("  [FAIL] sam2 not installed? pip install -e sam2 —", e)
    sys.exit(1)
PY
echo

echo "--- 汇总 ---"
echo "  通过项: $ok  失败项: $fail"
if [[ "$fail" -gt 0 ]]; then
  echo "  请先修复 [FAIL] 再 sbatch。"
  exit 1
fi
echo "  可以提交: sbatch scripts/slurm_phase12_train.slurm"
exit 0
