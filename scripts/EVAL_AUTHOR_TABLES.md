# 对齐作者论文测评表（CONVERSEG / RefCOCO / ReasonSeg）

对照数字见仓库根目录 [`AUTHOR_PAPER_EVAL_RESULTS.md`](../AUTHOR_PAPER_EVAL_RESULTS.md)。

## 0. 顺序

1. **CONVERSEG 分概念**（Table 1 / 附录 Table 8）— 数据已有，改 `eval.py` 即可  
2. **RefCOCO/+/g**（Table 2）— 先导出 eval manifest，再跑 `eval.py`  
3. **ReasonSeg**（Table 2）— 需先下载数据，再接入  

---

## 1. CONVERSEG 分概念（立刻可跑）

本地 jsonl 已含 `concept` ∈ `{entities,spatial,relations,affordances,physics}`。

```bash
cd /home/hanyuxi/ConverSeg-main
mkdir -p logs/eval_author_tables

# 例：自训 Stage2 学生 3B
python -u eval.py \
  --final_ckpt ckpts/stage2_phase2mix/ConVerSeg_stage2_final.torch \
  --plm_ckpt ckpts/stage2_phase2mix/ConVerSeg_stage2_plm_final.torch \
  --lora_ckpt ckpts/stage2_phase2mix/lora_plm_adapter_final \
  --plm-model-name student/Qwen2.5-VL-3B \
  --base_ckpt student/sam2_hiera_large.pt \
  --eval_splits sam_seeded,human_annotated \
  --precision bf16 \
  --report-out logs/eval_author_tables/converseg_stage2_3b_report.jsonl \
  2>&1 | tee logs/eval_author_tables/converseg_stage2_3b.log
```

日志里会出现：

- `TABLE1[sam_seeded] gIoU%: ...`（All Ent. Spat. Rel. Aff. Phys.）
- `TABLE8[sam_seeded] cIoU%: ...`

作者成品 3B / 自训 7B 只需改 ckpt 与 `--plm-model-name`。

双卡分片时，`--metrics-out` 现已带 `by_concept`，合并：

```bash
python scripts/teacher/merge_eval_shards.py shard0.jsonl shard1.jsonl
```

---

## 2. RefCOCO / + / g

### 2.1 导出评测清单（一次性）

依赖：`data/raw/refer/{refcoco,refcoco+,refcocog}/` + `data/raw/coco/train2014/`。

```bash
python scripts/export_refcoco_eval_manifests.py
# 输出: data/eval_manifests/refcoco/*.jsonl + masks/
```

对应 Table 2 列：

| 文件 | 表列 |
| --- | --- |
| `refcoco_{val,testA,testB}.jsonl` | RefCOCO |
| `refcocop_{val,testA,testB}.jsonl` | RefCOCO+ |
| `refcocog_umd_{val,test}.jsonl` | RefCOCOg val(U)/test(U) |
| `refcocog_google_val.jsonl` | RefCOCOg val(G) |

### 2.2 跑评测

```bash
for f in data/eval_manifests/refcoco/refcoco_val.jsonl \
         data/eval_manifests/refcoco/refcoco_testA.jsonl \
         data/eval_manifests/refcoco/refcoco_testB.jsonl \
         data/eval_manifests/refcoco/refcocop_val.jsonl \
         data/eval_manifests/refcoco/refcocop_testA.jsonl \
         data/eval_manifests/refcoco/refcocop_testB.jsonl \
         data/eval_manifests/refcoco/refcocog_umd_val.jsonl \
         data/eval_manifests/refcoco/refcocog_umd_test.jsonl \
         data/eval_manifests/refcoco/refcocog_google_val.jsonl
do
  stem=$(basename "$f" .jsonl)
  python -u eval.py \
    --input_jsonl "$f" \
    --final_ckpt ckpts/stage2_phase2mix/ConVerSeg_stage2_final.torch \
    --plm_ckpt ckpts/stage2_phase2mix/ConVerSeg_stage2_plm_final.torch \
    --lora_ckpt ckpts/stage2_phase2mix/lora_plm_adapter_final \
    --plm-model-name student/Qwen2.5-VL-3B \
    --base_ckpt student/sam2_hiera_large.pt \
    --precision bf16 \
    --report-out "logs/eval_author_tables/ref_${stem}_report.jsonl" \
    2>&1 | tee "logs/eval_author_tables/ref_${stem}.log"
done
```

Ref 样本没有 `concept` 时只会打 overall；Table 2 只需要 overall gIoU/cIoU。

---

## 3. ReasonSeg

数据需单独下载（本仓库默认未包含）。布局约定：

```text
data/ReasonSeg/
  val/     # 或官方 val json + images
  test/
```

下载说明与接入脚本见 [`download_reasonseg.md`](./download_reasonseg.md)。  
数据就绪后再写 `scripts/export_reasonseg_eval_manifests.py`（与 Ref 同样导出 jsonl → `eval.py`）。

---

## 4. 同步到服务器时注意

- `eval.py`、`scripts/teacher/merge_eval_shards.py`、`scripts/export_refcoco_eval_manifests.py`  
- 本说明文档  
- Windows 编辑后如有 CRLF：`sed -i 's/\r$//' scripts/*.md scripts/export_refcoco_eval_manifests.py`
