# DL-OPD 代码包（统一管理）

本目录实现《DL_OPD_IMPLEMENTATION_PLAN.zh.md》中的**双层在策略蒸馏**全流程。配置在仓库 `configs/dl_opd/`。

## 目录

| 路径 | 作用 |
| --- | --- |
| `gate_teacher.py` | 样本级教师门控（空 GT / 高 IoU / 置信度 / 改进） |
| `losses.py` | `L_sup/opd/bd/rep/col/anc` + identity |
| `rollout.py` | current-state 前向辅助 |
| `metrics_ops.py` | Recovery / Empty Acc / Boundary / ECE |
| `hard_buffer.py` | Hard Buffer + TTL |
| `teacher_cache.py` | 近在线教师缓存 |
| `data_policy/` | 误差画像、价值函数、五类算子、Q_data 验证 |
| `train_mask_opd.py` | **内层** Mask-Policy OPD 训练 |
| `train_dl_opd.py` | **外层+内层** 协同调度 |
| `build_error_profile.py` 等 | Data-Policy CLI |
| `export_eval_suites.py` | Hard/Neg/Recovery 清单导出 |
| `eval_dl_opd.py` | 统一评测入口 |
| `run_ablation.py` | §13 消融卡片 |
| `tests/` | 关键单元测试 |
| `slurm_*.slurm` | 集群提交 |

## 快速自检（无需 GPU）

```bash
cd /path/to/ConverSeg-main
python scripts/dl_opd/tests/test_gate_teacher.py
python scripts/dl_opd/tests/test_losses.py
python scripts/dl_opd/tests/test_hard_buffer.py
python scripts/dl_opd/tests/test_qdata_and_cache.py
python scripts/dl_opd/tests/test_mask_state_encoder.py
```

## Data-Policy dry-run

```bash
python -u scripts/dl_opd/train_dl_opd.py --policy_only --n_outer 2
python -u scripts/dl_opd/export_eval_suites.py
```

## 内层 Mask-OPD（需 GPU + 权重）

1. 编辑 `configs/dl_opd/dl_opd.yaml` 中 student/teacher 路径。  
2. 提交：

```bash
sed -i 's/\r$//' scripts/dl_opd/*.slurm scripts/dl_opd/*.py
sbatch scripts/dl_opd/slurm_mask_opd.slurm
# 或完整 outer 调度：
sbatch scripts/dl_opd/slurm_dl_opd.slurm
```

## 实现分层对照文档

| 工程阶段 | 本包覆盖 |
| --- | --- |
| MVP-A | `train_mask_opd.py` + gate + losses + cache + 单测 |
| MVP-B | `train_dl_opd.py` + error profile + rewrite/minimal-pair + buffer + suites |
| 完整 | 五类算子钩子、双教师门控开关、`L_rep/L_col` 日程、`run_ablation.py` 矩阵 |

逐步核对时：先跑单测 → policy dry-run → 短 steps Mask-OPD → 再开 outer。
