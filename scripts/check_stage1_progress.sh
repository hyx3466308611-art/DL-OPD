#!/usr/bin/env bash
# 当前 Stage1 训到哪一步、磁盘、已有 checkpoint
set -euo pipefail
ROOT="${ROOT:-/home/hanyuxi/ConverSeg-main}"
cd "$ROOT"

LOG="${LOG:-logs/stage1_100k.log}"
OUT="${OUT:-./ckpts/stage1_full}"
NAME="${NAME:-ConVerSeg_stage1}"

echo "=== 训练进程 ==="
pgrep -af "train.py.*${OUT}" || echo "(未在跑)"

echo ""
echo "=== 当前 step（log 最后一行 step）==="
grep -E "^step [0-9]+" "$LOG" 2>/dev/null | tail -3 || echo "无 log"

echo ""
echo "=== 崩溃 / 存盘 ==="
grep -E "Saved checkpoints|checkpoint save failed|RuntimeError|Traceback" "$LOG" 2>/dev/null | tail -8 || true

echo ""
echo "=== 已有权重（每 step 三件套）==="
for f in "$OUT/${NAME}"_*.torch; do
  [[ -e "$f" ]] || continue
  [[ "$f" == *"_plm_"* ]] && continue
  base=$(basename "$f" .torch)
  step=${base#${NAME}_}
  [[ "$step" =~ ^[0-9]+$ ]] || continue
  sam="$OUT/${NAME}_${step}.torch"
  plm="$OUT/${NAME}_plm_${step}.torch"
  lora="$OUT/lora_plm_adapter_${step}"
  tag="complete"
  [[ -f "$sam" && -f "$plm" && ( -d "$lora" || -f "$lora" ) ]] || tag="INCOMPLETE"
  ls -lh "$sam" 2>/dev/null | awk -v st="$step" -v t="$tag" '{print "step", st, t, $5, $9}'
done | sort -n -k2

echo ""
echo "=== best（若跑过 pick_best）==="
ls -lh "$OUT/${NAME}"_best.torch "$OUT/${NAME}_plm_best.torch" 2>/dev/null || true
cat "$OUT/${NAME}_best_step.txt" 2>/dev/null || true

echo ""
echo "=== 磁盘 ==="
df -h "$OUT" /home/hanyuxi 2>/dev/null | head -5
