# 纠错教师 fix_v2：空 GT、边界保真与空间门控

fix_v2 是完全隔离的第二次改进实验；不修改 baseline、fix_v1 或公共模型实现。

## 1. 四项改动

| 维度 | fix_v2 实现 |
| --- | --- |
| 双重门控 | `global_gate=-1.0`（sigmoid≈0.269）+ `3×3 Conv → SiLU → 1×1 Conv → Sigmoid` 空间门控 |
| 条件损失 | 空 GT：仅 `3×BCE`；非空：BCE+Dice+Focal；input IoU>0.85 再加 `0.5×L1(pred,input)` |
| 数据强化 | 原始完整 GT（不 erode）；10% 不相关 prompt 假负样本；human×4；B/C mask dropout=0.15/0.30 |
| 腐化熔断 | 非 Stage C 且腐化面积/GT面积<5% 时退回 severity=0.1 的轻微 morph |

初始化默认来自对应的 **fix_v1**：

- 3B：`ckpts/teacher_correction/author3b_fix_v1`
- 7B：`ckpts/teacher_correction/teacher7b_fix_v1`

只迁移旧 head 的 `mask_encoder`。旧全局/空间门控不迁移，fix_v2 全局门控严格从 `-1.0` 开始。

## 2. 文件

| 文件 | 作用 |
| --- | --- |
| `teacher_correction_head_v2.py` | fix_v2 双重置信度门控 |
| `mask_corrupt_fix_v2.py` | fix_v2 独立腐化配置与面积熔断 |
| `train_fix_v2.py` | 完整 GT、10% 假负样本、逐样本条件损失 |
| `eval_trg_fix_v2.py` | Mixed 与逐算子两套评测 |
| `_env_fix_v2.sh` | 3B/7B 路径与默认参数 |
| `slurm_train_fix_v2.slurm` | 训练 |
| `slurm_eval_mixed_fix_v2.slurm` | 论文主指标 Mixed-TRG |
| `slurm_eval_per_op_fix_v2.slurm` | 全算子/指定算子诊断 |

训练使用 fix_v2 的增强腐化分布；**评测固定使用共享 benchmark 腐化器**，避免把模型改进与测试分布变化混在一起，保证 baseline / fix_v1 / fix_v2 可直接比较。

## 3. 训练

```bash
cd /home/hanyuxi/ConverSeg-main
sed -i 's/\r$//' scripts/teacher_correction/fix_v2/*.sh \
  scripts/teacher_correction/fix_v2/*.slurm

# 先检查 fix_v1 初始化权重
ls ckpts/teacher_correction/author3b_fix_v1/CorrectionTeacher_3B_fix_v1_{final,plm_final,head_final}.torch

sbatch --export=ALL,TEACHER_BACKBONE=author3b \
  scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm
```

日志：

```bash
tail -f logs/teacher_correction/fix_v2/slurm_train_<JOBID>.out
tail -f logs/teacher_correction/fix_v2/train_fix_v2_author3b.log
```

输出：

```text
ckpts/teacher_correction/author3b_fix_v2/
  CorrectionTeacher_3B_fix_v2_final.torch
  CorrectionTeacher_3B_fix_v2_plm_final.torch
  CorrectionTeacher_3B_fix_v2_head_final.torch
  lora_plm_adapter_final/
```

7B 使用：

```bash
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b \
  scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm
```

前提：`teacher7b_fix_v1` 已训练完成。

## 4. 两套评测

### A. Mixed-TRG（论文主表）

```bash
sbatch --export=ALL,TEACHER_BACKBONE=author3b \
  scripts/teacher_correction/fix_v2/slurm_eval_mixed_fix_v2.slurm
```

结果：

```text
ckpts/teacher_correction/author3b_fix_v2/eval_recovery_stageMIX/trg_report.json
ckpts/teacher_correction/author3b_fix_v2/teacher_frozen_ready_stageMIX.flag
```

主看：`mean_trg`、`positive_fraction`、`mean_input_iou`、`mean_output_iou`。

### B. Per-op（模型改进诊断）

```bash
# 一次加载模型，扫八主算子 + mask_dropout + empty_gt
sbatch --export=ALL,TEACHER_BACKBONE=author3b \
  scripts/teacher_correction/fix_v2/slurm_eval_per_op_fix_v2.slurm

# 只测 empty_gt
sbatch --export=ALL,TEACHER_BACKBONE=author3b,FORCE_OP=empty_gt \
  scripts/teacher_correction/fix_v2/slurm_eval_per_op_fix_v2.slurm

# 只测 boundary
sbatch --export=ALL,TEACHER_BACKBONE=author3b,FORCE_OP=boundary \
  scripts/teacher_correction/fix_v2/slurm_eval_per_op_fix_v2.slurm
```

结果：

```text
ckpts/teacher_correction/author3b_fix_v2/eval_recovery_per_op/per_op_summary.json
ckpts/teacher_correction/author3b_fix_v2/eval_recovery_per_op/<op>/trg_report.json
```

逐算子诊断默认使用 Stage C severity，关闭随机 dropout 和复合算子；`empty_gt` 只统计 GT 为空的样本。

## 5. 验收关注点

1. `empty_gt`：mean TRG 应从约 −0.75 显著向 0 靠近，TRG=−1 大幅减少。
2. `boundary` / `elastic`：高 input IoU 样本不再系统性负 TRG。
3. `mask_dropout` 与 Stage C：mean output IoU 不低于 fix_v1。
4. `human_annotated`：Mixed 报告 `by_split` 中 output/TRG 不下降。
5. 最后仍以 Mixed 的 `ready=true` 为冻结依据；单算子诊断不生成冻结结论。
