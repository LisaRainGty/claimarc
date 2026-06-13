# CLAIMARC 数据流水线（src/）

把原始直播电商数据（评论 / 主播 SRT / 商品图文）处理成 pair-level 实验数据集
`data/final/dataset.jsonl`（每条 = 一个 `(product_id, attribute_id)`）。

## 运行环境
- 计算在远程 GPU 服务器（matpool A30）；LLM/VLM 走 matpool 网关（OpenAI 兼容）。
- 文本模型 `Gemini-3.5-Flash`，视觉模型 `Qwen3-VL-Plus`，嵌入 `BGE-large-zh-v1.5`，OCR `PaddleOCR`。
- 所有产出实时同步回本机 `data/processed`、`data/final`、`data/cache`。

```bash
source env.sh                 # 设置 KEY / ROOT / PATH / PYTHONPATH
python -m run_pipeline --pilot   # food_and_beverages 小样端到端
python -m run_pipeline --all     # 全量
```

## 阶段与产物
| 阶段 | 模块 | 产物 |
|---|---|---|
| A0 | `stage_a.a0_build_cas` | `stageA/CAS_<cat>.json` |
| A1 | `stage_a.a1_extract_aspects` | `stageA/raw_aspects.jsonl` |
| A2 | `stage_a.a2_aggregate_free` | `stageA/CAS+_<cat>.json` + `free_resolution_<cat>.json` |
| A3 | `stage_a.a3_resolve_labels` | `stageA/resolved_aspects.jsonl` |
| B0 | `stage_b.b0_acmt` | `stageB/acmt.json` |
| B1 | `stage_b.b1_claim_extract` | `stageB/claim_list/<pid>.jsonl` |
| B2B3 | `stage_b.b2_b3_passage` | `stageB/pair_skeleton.jsonl` |
| B4B5 | `stage_b.b4_b5_align` | `stageB/pair_records.jsonl` |
| C1 | `stage_c.c1_image_triage` | `stageC/image_index.json` |
| C2 | `stage_c.c2_params` | `stageC/evidence_params.json` |
| C3 | `stage_c.c3_ocr` | `stageC/evidence_ocr.json` (+ `ocr_text/`) |
| C4 | `stage_c.c4_vlm` | `stageC/evidence_vlm.json` |
| C5 | `stage_c.c5_fact_records` | `stageC/fact_records.jsonl` |
| labels | `labels.build_labels` | `labels.jsonl` |
| final | `final.join_split` | `final/dataset.jsonl` + `final/table1_stats.{json,md}` |

## 关键设计
- **按 product_id 聚合**：一个商品可出现在多个 clip，先合并其全部 SRT 与评论。
- **source-span grounding（B1）**：claim 必须是 SRT 原文连续子串，字符区间反查时间戳。
- **断点续跑**：LLM 调用按 `(model,prompt)` 哈希磁盘缓存（`data/cache/`）；B1/C1/C3 按商品落盘可续。
- **样本权重 c**：§2 四因子（f_sat / f_cov / f_asym / f_fake）+ φ_bonus，下限 0.05。
- **划分**：按 `room_id` 分组 70:10:20，同直播间不跨 split。

## 数据安全
服务器随时可能停租。用 `scripts/sync_back.sh loop 300` 每 5 分钟把 `processed/final/cache` 拉回本机。
