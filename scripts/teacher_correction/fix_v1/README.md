# 纠错教师 fix_v1（完整缺陷修复方案）

实现文档中的 **6 项缺陷** 对策：负样本 / 空 GT 抑制、全精度 GT 监督、Identity 保真、Dual-Mode、复合腐化约束、**TeacherCorrectionHeadV2** 空间门控。

**仅保存 final 权重**

- 3B：`ckpts/teacher_correction/author3b_fix_v1/*_final.torch`
- 7B：`TEACHER_BACKBONE=teacher7b` → `ckpts/teacher_correction/teacher7b_fix_v1/`，init 来自 `teacher7b/CorrectionTeacher_7B_*_final`

---

## 代码与仓库改动

| 位置 | 内容 |
| --- | --- |
| `data/mask_corrupter.py` | 复合算子 severity×0.5、禁止连续同算子、10% 面积下限；boundary/elastic 幅度降低 |
| `models/teacher_correction_model.py` | 新增 `TeacherCorrectionHeadV2`（全局+空间门控、负样本 suppress） |
| `fix_v1/train_fix_v1.py` | 全量 GT（无 erode）、17.5% 负样本、50% 纯文本分割、损失与 V2 头 |
| `fix_v1/eval_trg_fix_v1.py` | HeadV2 TRG；默认 `--full-gt` 与训练一致 |
| `fix_v1/slurm_*.slurm` | 可执行集群脚本 |

---

## 执行（仓库根目录）

```bash
cd /home/hanyuxi/ConverSeg-main
mkdir -p logs/teacher_correction/fix_v1

# 3B（需已有 author3b CorrectionTeacher_3B_*_final）
sbatch --export=ALL,TEACHER_BACKBONE=author3b scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm

# 7B（需已有 teacher7b CorrectionTeacher_7B_*_final）
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm

# 主表 Mixed-TRG（默认 EVAL_STAGE=MIX）；与旧 B/C 对比时加 EVAL_ERODE_GT=1
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_ERODE_GT=1 scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm
# 附录：EVAL_STAGE=B 或 C
```

可选：

```bash
FIX_STEPS=50000 sbatch scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm
```

---

## 训练要点（默认 `_env_fix_v1.sh`）

- **负样本**：`conversational_negative_data.jsonl`，batch 内 **17.5%**，GT 强制全 0，`lam_neg=3`
- **Dual-Mode**：50% step `input_mask=0`（纯文本分割），50% 腐化纠错
- **保真**：`lam_preserve=0.4`，IoU(input,GT)≥**0.85** 时 L1 约束输出≈输入
- **GT**：`bin_mask(ann)`，**不做 erode**
- **Head**：V2；从 V1 head 部分加载 `mask_encoder`+`gate`，spatial gate 新初始化
- **human**：默认 **不混入**。`human_annotated` 属于测评集，`HUMAN_REPEAT` 必须保持 0

---

## 输出

| 路径 | 说明 |
| --- | --- |
| `ckpts/teacher_correction/author3b_fix_v1/CorrectionTeacher_3B_fix_v1_*_final.torch` | 权重 |
| `eval_recovery/trg_report.json` | TRG（含 `full_gt`, `head_version`） |
| `teacher_frozen_ready.flag` | 冻结门槛 |

详细 TRG 指标见 [TRG_EVAL_METRICS.md](../../../TRG_EVAL_METRICS.md)。
