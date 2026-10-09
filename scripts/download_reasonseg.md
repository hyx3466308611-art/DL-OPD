# ReasonSeg 评测数据下载（对齐作者 Table 2）

作者在 ReasonSeg 上做 **zero-shot** 评测（未在该集上微调），报告 val / test / test(short) / test(long) 的 gIoU（附录另有 cIoU）。

## 官方来源

ReasonSeg 随 LISA 发布，请以 LISA 仓库当前说明为准：

- https://github.com/dvlab-research/LISA  
- 文档中的 ReasonSeg / reasoning segmentation dataset 链接（通常为 Google Drive）

下载后放到：

```text
/home/hanyuxi/ConverSeg-main/data/ReasonSeg/
```

建议保持官方目录结构（常见为 `train` / `val` / `test` 下的 image + json 标注）。若解压后结构不同，把实际树贴出来再写导出脚本。

## 接入计划（数据到位后）

1. `scripts/export_reasonseg_eval_manifests.py`  
   导出 `data/eval_manifests/reasonseg/{val,test,test_short,test_long}.jsonl`  
   字段与 CONVERSEG 相同：`id,image,mask,prompt`
2. 用现有 `eval.py --input_jsonl ... --report-out ...` 跑 gIoU/cIoU  
3. 数字写入对照表，对齐 `AUTHOR_PAPER_EVAL_RESULTS.md` §3 / §4

## 自检

```bash
find data/ReasonSeg -maxdepth 3 -type d | head
# 确认能找到 val/test 的图与标注后再继续写 export 脚本
```

**当前状态：仓库未包含 ReasonSeg；在下载完成前不要提交相关 sbatch。**
