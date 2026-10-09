#!/usr/bin/env bash
# 检查 converseg_stage2_data 目录与 Group4 路径是否可被构建脚本识别
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ST2="$ROOT/data/ConverSeg-Training-Data/converseg_stage2_data"
PY="${PYTHON:-python3}"
if [[ -x "$HOME/miniconda311/envs/converseg/bin/python" ]]; then
  PY="$HOME/miniconda311/envs/converseg/bin/python"
fi

echo "=== Stage2 目录 ==="
echo "ST2=$ST2"
if [[ ! -d "$ST2" ]]; then
  echo "[FAIL] 目录不存在"
  exit 1
fi
ls -la "$ST2"

echo ""
echo "=== jsonl 行数 ==="
for f in \
  "$ST2/conversational_positive_data/conversational_positive_data.jsonl" \
  "$ST2/conversational_negative_data/conversational_negative_data.jsonl" \
  "$ST2/open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl" \
  "$ST2/stage2_mixed.jsonl"; do
  if [[ -f "$f" ]]; then
    echo -n "$(basename "$(dirname "$f")")/$(basename "$f"): "
    wc -l < "$f"
  else
    echo "[..] 不存在: ${f#$ROOT/}"
  fi
done

echo ""
echo "=== 各 split 下 png 数量 ==="
for split in conversational_positive_data conversational_negative_data open_vocabulary_regions_data; do
  d="$ST2/$split"
  if [[ -d "$d" ]]; then
    ni=$(find "$d/images" -name '*.png' 2>/dev/null | wc -l)
    nm=$(find "$d/masks" -name '*.png' 2>/dev/null | wc -l)
    echo "$split: images=$ni masks=$nm"
  fi
done

echo ""
echo "=== jsonl 首行解析后的绝对路径（应用 converseg_stage2_data 前缀）==="
head -1 "$ST2/conversational_positive_data/conversational_positive_data.jsonl" | "$PY" -c "
import json, sys
from pathlib import Path
root = Path('$ROOT')
data = root / 'data'
st2 = 'ConverSeg-Training-Data/converseg_stage2_data'
item = json.loads(sys.stdin.read())
for k in ('image','mask_merged'):
    p = item[k].replace('\\\\','/')
    if not p.startswith('ConverSeg-Training-Data/'):
        p = st2 + '/' + p.lstrip('/')
    full = data / p
    print(k, '->', full)
    print('   exists:', full.is_file())
"

echo ""
echo "=== Python 统计 G4 有效行（与 preflight / build_group4 一致）==="
BD="$ROOT/scripts/latet_bulid_phase1"
"$PY" "$BD/check_stage2.py"
