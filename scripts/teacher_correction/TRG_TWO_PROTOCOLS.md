# 纠错教师两套 TRG 测评协议

| 协议 | 用途 | 命令要点 | 产出 |
| --- | --- | --- | --- |
| **A. Mixed-TRG** | 论文主表：证明整体能纠错 | `--eval-stage MIX`，无 `--force-op` | `eval_recovery_stageMIX/` |
| **B. Per-op 诊断** | 改进模型：看每个算子是否过关 | `--eval-stage C --force-op ALL`（或单算子） | `eval_recovery_per_op/<op>/` + `per_op_summary.json` |

详细指标定义见仓库根目录 [TRG_EVAL_METRICS.md](../../TRG_EVAL_METRICS.md)。

---

## A. Mixed-TRG（主表）

每样本按 **A:B:C = 1:2:2** 抽 stage，再随机腐化。主表只报：

- mean TRG
- positive_fraction
- mean output IoU（+ mean input IoU）

```bash
# fix_v1 3B 修正教师（已跑过 Job 904）
sbatch --export=ALL,TEACHER_BACKBONE=author3b,EVAL_ERODE_GT=1 \
  scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm
```

报告：`ckpts/teacher_correction/author3b_fix_v1/eval_recovery_stageMIX/trg_report.json`  
结论文档：`ckpts/teacher_correction/author3b/TRG_EVAL_MIXED.md`

---

## B. Per-op 诊断（改进用）

对下列算子 **各跑一遍全量测评**（severity 默认 Stage **C**；**关闭**随机 dropout / compound）：

`morph, boundary, translate, drop_comp, partial, false_pos, invert_halluc, elastic, mask_dropout, empty_gt`

- `empty_gt`：只统计 **GT 为空** 的样本（n 很小，约几十）。
- 单算子诊断 **不写** `ready=true`（不作为冻结依据）。

```bash
# 一次加载，扫全部算子（推荐）
sbatch --export=ALL,TEACHER_BACKBONE=author3b,EVAL_ERODE_GT=1 \
  scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm

# 只盯 empty_gt
sbatch --export=ALL,TEACHER_BACKBONE=author3b,FORCE_OP=empty_gt,EVAL_ERODE_GT=1 \
  scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm
```

查看汇总：

```bash
tail -f logs/teacher_correction/fix_v1/slurm_per_op_<JOBID>.out

python scripts/teacher_correction/merge_per_op_reports.py \
  --root ckpts/teacher_correction/author3b_fix_v1/eval_recovery_per_op

# 或直接
cat ckpts/teacher_correction/author3b_fix_v1/eval_recovery_per_op/per_op_summary.json
```

改进目标（建议自定门槛，例如）：

| 算子类 | 期望 |
| --- | --- |
| 多数主算子 | mean TRG > 0 且 pos_frac ≥ 0.55 |
| `empty_gt` | mean TRG ≈ 0（输出保持空），勿大量 TRG=−1 |
| `boundary` / `elastic` | 高 input IoU 时勿系统性 TRG≤0；至少 pos_frac 明显 >0 |

---

## 与 Mixed 里 `by_op_family` 的区别

| | Mixed 附录 `by_op_family` | Per-op 诊断 |
| --- | --- | --- |
| 样本覆盖 | 随机抽到的才进该 family | **每个算子打满**（empty_gt 除外） |
| 复合腐化 | 可能有 `a+b` | **关闭** compound |
| 用途 | 主跑附带分层 | **改模型前后对比** |

改 `empty_gt` / boundary 后：先跑 **B** 看该算子是否修好，再跑 **A** 确认主表不掉点。
