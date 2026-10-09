# 教师模型 Phase1+2（Qwen2.5-VL-7B）

## 路径

| 用途 | 路径 |
|------|------|
| Qwen 7B | `teacher/Qwen2.5-VL-7B/` |
| SAM2-L | `teacher/sam2_hiera_large.pt` |
| Phase1 产出 | `ckpts/teacher/stage1_phase1mix/` |
| Phase2 产出 | `ckpts/teacher/stage2_phase2mix/` |
| 日志 | `logs/teacher/` |

## 提交

```bash
cd /home/hanyuxi/ConverSeg-main
mkdir -p logs/teacher ckpts/teacher/stage1_phase1mix ckpts/teacher/stage2_phase2mix
bash scripts/teacher/verify_before_sbatch.sh
sbatch scripts/teacher/slurm_phase12_train.slurm
```

72h 不够时：Phase1 完成后 `sbatch --dependency=afterok:<JID> scripts/teacher/slurm_phase2_only.slurm`

## 7B 显存

与学生相同 `batch-size 2 acc 3`；若 OOM，改 `_train_env.sh` 为 `batch-size 1 acc 6`。

## 后续

腐化掩码 / OPD 训练另开脚本（待 Phase2 final 就绪）。
