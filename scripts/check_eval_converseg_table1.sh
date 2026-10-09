#!/usr/bin/env bash
# 查看 CONVERSEG Table-1 评测进度 / 结果
# 用法:
#   bash scripts/check_eval_converseg_table1.sh              # 列出全部 NAME
#   bash scripts/check_eval_converseg_table1.sh stage2_3b    # 指定 NAME
#   bash scripts/check_eval_converseg_table1.sh author3b 928 # NAME + JobID

set -euo pipefail
cd "${ROOT:-/home/hanyuxi/ConverSeg-main}"
DIR=logs/eval_author_tables
mkdir -p "$DIR"

NAME="${1:-}"
JOBID="${2:-}"

echo "===== squeue ====="
squeue -u "${USER}" -o '%.10i %.12P %.20j %.8T %.10M %.4D %R' 2>/dev/null || true
echo

if [[ -z "$NAME" ]]; then
  echo "===== 已有评测日志 ====="
  ls -lt "$DIR"/converseg_*.log 2>/dev/null || echo "(无 log)"
  echo
  echo "===== 已有 TABLE 汇总 ====="
  grep -H -E 'TABLE1\[|TABLE8\[' "$DIR"/converseg_*.log 2>/dev/null || echo "(尚无 TABLE 行)"
  echo
  echo "用法: $0 <NAME> [JOBID]"
  echo "  NAME 例: stage2_3b | author3b | teacher7b_stage2"
  exit 0
fi

LOG="$DIR/converseg_${NAME}.log"
REPORT="$DIR/converseg_${NAME}_report.jsonl"

echo "===== 文件 ====="
echo "LOG:    $LOG"
echo "REPORT: $REPORT"
[[ -n "$JOBID" ]] && echo "SLURM:  $DIR/slurm_converseg_${JOBID}.out / .err"
ls -lh "$LOG" "$REPORT" 2>/dev/null || true
[[ -n "$JOBID" ]] && ls -lh "$DIR/slurm_converseg_${JOBID}".{out,err} 2>/dev/null || true
echo

if [[ -n "$JOBID" ]]; then
  echo "===== sacct $JOBID ====="
  sacct -j "$JOBID" --format=JobID,State,Elapsed,ExitCode,MaxRSS,End 2>/dev/null || true
  echo
fi

echo "===== RESULTS / TABLE1 / TABLE8 ====="
if [[ -f "$LOG" ]]; then
  grep -E 'RESULTS \[|TABLE1\[|TABLE8\[' "$LOG" || echo "(log 里还没有汇总行，任务可能仍在跑)"
else
  echo "log 不存在"
fi
echo

if [[ -f "$REPORT" ]]; then
  echo "===== report 行数 ====="
  wc -l "$REPORT"
else
  echo "report 尚未写出"
fi

echo
echo "实时跟踪:"
echo "  tail -f $LOG"
[[ -n "$JOBID" ]] && echo "  tail -f $DIR/slurm_converseg_${JOBID}.out"
