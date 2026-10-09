# Stage 1：纠错教师（Correction Teacher）

## 教师骨干切换（重要）

| 环境变量 / CLI | 含义 |
|----------------|------|
| `TEACHER_BACKBONE=author3b`（**默认**） | 作者 `checkpoints/ConVerSeg-Net-3B` + `student/Qwen2.5-VL-3B` |
| `TEACHER_BACKBONE=teacher7b` | 自训 7B Phase2（`ckpts/teacher/stage2_phase2mix`）+ `teacher/Qwen2.5-VL-7B` |

产出目录：

- `ckpts/teacher_correction/author3b/` → `CorrectionTeacher_3B_*`
- `ckpts/teacher_correction/teacher7b/` → `CorrectionTeacher_7B_*`
- 修正版：`author3b_fix_v1/`、`teacher7b_fix_v1/`

TRG 按 stage 分目录：`${CORR_OUT}/eval_recovery_stage{B,C}/`。

Python 等价参数：`--teacher-backbone author3b|teacher7b`

**7B 默认 batch=1 × acc=6**（等效 batch=6，单卡 48G）。覆盖：`CORR_BS=2 CORR_ACC=3`。必须用 `sbatch --export=ALL,...`，否则集群可能丢掉环境变量。

## 保存物（Stage 2 必须齐全）

- `{name}_final.torch` — SAM
- `{name}_plm_final.torch` — PLM adapter
- `{name}_head_final.torch` — **含 `gate` + `mask_encoder`**
- `lora_plm_adapter_final/`

## 训练数据路径（与学生 Phase2 相同）

```text
--data-dir   /home/hanyuxi/ConverSeg-main/data
--dataset-jsonl phase2/phase2_mixed.jsonl
```

不要用 `ConverSeg-Training-Data/phase2_mixed.jsonl` 作为 `--data-dir` 根目录。

## 服务器命令

```bash
cd /home/hanyuxi/ConverSeg-main
sed -i 's/\r$//' scripts/teacher_correction/*.sh scripts/teacher_correction/*.slurm
bash scripts/teacher_correction/verify_before_sbatch.sh

# 作者 3B 教师
sbatch --export=ALL,TEACHER_BACKBONE=author3b scripts/teacher_correction/slurm_train_correction.slurm
sbatch --export=ALL,TEACHER_BACKBONE=author3b scripts/teacher_correction/slurm_eval_trg.slurm

# 自训 7B 纠错教师（需 ckpts/teacher/stage2_phase2mix/*_final.torch）
export TEACHER_BACKBONE=teacher7b
bash scripts/teacher_correction/verify_before_sbatch.sh
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/slurm_train_correction.slurm
# 训完 — 主表 Mixed-TRG（默认 EVAL_STAGE=MIX）：
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/slurm_eval_trg.slurm
# 附录难度扫描（可选）：
# sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_STAGE=B scripts/teacher_correction/slurm_eval_trg.slurm
# sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_STAGE=C scripts/teacher_correction/slurm_eval_trg.slurm

# 再跑 fix_v1（依赖上一步 CorrectionTeacher_7B_*_final）
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm
sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,EVAL_ERODE_GT=1 scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm
```

## 两套测评（必读）

详见 **[TRG_TWO_PROTOCOLS.md](./TRG_TWO_PROTOCOLS.md)**。

| 协议 | 用途 | 提交 |
| --- | --- | --- |
| **Mixed-TRG** | 论文主表 | `fix_v1/slurm_eval_trg_fix_v1.slurm`（默认 `EVAL_STAGE=MIX`） |
| **Per-op 诊断** | 改进 empty_gt / boundary 等 | `fix_v1/slurm_eval_trg_per_op.slurm`（`--force-op ALL`，severity=C） |

```bash
# 主表
sbatch --export=ALL,TEACHER_BACKBONE=author3b,EVAL_ERODE_GT=1 \
  scripts/teacher_correction/fix_v1/slurm_eval_trg_fix_v1.slurm

# 按算子诊断（一次加载扫 10 个算子）
sbatch --export=ALL,TEACHER_BACKBONE=author3b,EVAL_ERODE_GT=1 \
  scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm
```

Mixed 主表：mean TRG / pos_frac / output IoU；冻结门槛 mean>0.03 且 pos≥0.55。  
指标细节：[TRG_EVAL_METRICS.md](../../TRG_EVAL_METRICS.md)。

## 自检

```bash
python -c "from converseg_paths import prepare_python_import_path; prepare_python_import_path(); from models.teacher_correction_model import TeacherCorrectionHead; print('Head import and gate check OK')"
```
