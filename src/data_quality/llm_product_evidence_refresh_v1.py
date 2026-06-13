"""Targeted LLM/VLM product-evidence refresh for proposal-faithful triplets.

Input rows are second-stage `product_evidence_refresh` tasks.  Their claim side
has already been judged usable enough for an evidence-only pass, so this tool
asks the model to search product-side material only: title, params, OCR text,
and detail images.  It preserves proposal labels and never treats SRT text as
product evidence.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import llm
from common.io_utils import read_jsonl, write_json
from data_quality.llm_verify_regeneration_queue_v2 import (
    PRODUCT_SOURCES,
    VALID_REL,
    clean,
    choose_images,
    load_done,
    ocr_context,
    product_context,
)
from common import product_index as pidx


VALID_PRODUCT_SRC = PRODUCT_SOURCES | {"none"}


def claim_for_task(row: dict[str, Any]) -> str:
    verdict = row.get("previous_triplet_verdict") or {}
    claim = clean(verdict.get("claim_text"))
    if claim:
        return claim
    preview = clean(row.get("claim_preview"))
    return preview.split("\n", 1)[0][:120]


def make_prompt(row: dict[str, Any], product_ctx: str, ocr_ctx: str, image_paths: list[str]) -> str:
    claim = claim_for_task(row)
    verdict = row.get("previous_triplet_verdict") or {}
    image_list = "\n".join(f"{i + 1}. {Path(p).name}" for i, p in enumerate(image_paths)) or "无"
    current_evidence = json.dumps(row.get("current_evidence_preview") or [], ensure_ascii=False)[:1200]
    return f"""你是直播电商商品事实证据取证员。请只找商品侧证据，不修改消费者标签，不用外部知识。

商品标题：{row.get("product_title")}
目标属性：{row.get("attribute_name")} ({row.get("attribute_id")})
属性类型：{row.get("expected_value_type")}
当前可用主播 claim：{claim or "无"}
上一轮判断：{verdict.get("relation_to_claim")} / {verdict.get("confidence")} / {verdict.get("reject_reason")}
当前旧 evidence 预览：{current_evidence or "[]"}

商品标题与参数：
{product_ctx or "【无参数】"}

OCR候选：
{ocr_ctx or "【无OCR候选】"}

输入图片顺序：
{image_list}

硬约束：
1. evidence 只能来自商品标题、商品参数、详情图 OCR 或详情图视觉观察；不能来自主播/SRT。
2. raw_text 必须是目标属性上的具体商品事实；孤立数字、泛泛促销语、无属性名的短词都不能作为证据。
3. 判断 relation_to_claim 时，只比较同一目标属性的具体值或命题；只是相关但不能蕴含/反驳时输出 insufficient。
4. 如果图片中能直接看到规格表、材质、尺码、版型、结构、功能或包装文字，请优先给出最具体的原文或客观视觉描述。
5. 不改变 current_y/current_c；若证据充分但 claim 与评论命题可能改变，后续交给 B4，而不是在这里 relabel。

请严格输出 JSON：
{{
  "evidence_found": true/false,
  "source_type": "params|product_title|detail_image_ocr|detail_image_vlm|none",
  "raw_text": "证据原文或客观视觉观察，不超过100字",
  "normalized_value": "规格化属性值；无则空",
  "path_or_image": "参数名、标题、图片文件名或图片序号；无则空",
  "relation_to_claim": "supports_claim|contradicts_claim|insufficient|claim_only",
  "confidence": "high|medium|low",
  "reject_reason": "不能进入主训练的原因；否则空",
  "curation_action": "keep_clean|keep_risk|keep_silver|rerun_more_evidence|drop"
}}
只输出 JSON。"""


def clean_result(obj: dict[str, Any], row: dict[str, Any], model: str) -> dict[str, Any]:
    src = str(obj.get("source_type") or "none")
    if src not in VALID_PRODUCT_SRC:
        src = "none"
    rel = str(obj.get("relation_to_claim") or "insufficient")
    if rel not in VALID_REL or rel == "evidence_only":
        rel = "insufficient"
    evidence_found = bool(obj.get("evidence_found")) and src in PRODUCT_SOURCES
    conf = str(obj.get("confidence") or "").strip().lower()
    if not evidence_found:
        rel = "claim_only"
        action = "rerun_more_evidence"
    elif rel == "supports_claim":
        action = "keep_clean" if conf == "high" else "keep_silver"
    elif rel == "contradicts_claim":
        action = "keep_risk" if conf == "high" else "keep_silver"
    else:
        action = "rerun_more_evidence"
    return {
        "pair_id": row.get("pair_id"),
        "second_stage_task": row.get("second_stage_task"),
        "priority": row.get("priority"),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "expected_value_type": row.get("expected_value_type"),
        "current_y": row.get("current_y"),
        "current_c": row.get("current_c"),
        "claim_text": claim_for_task(row),
        "evidence_found": evidence_found,
        "source_type": src,
        "raw_text": str(obj.get("raw_text") or "")[:220],
        "normalized_value": str(obj.get("normalized_value") or "")[:120],
        "path_or_image": str(obj.get("path_or_image") or "")[:160],
        "relation_to_claim": rel,
        "confidence": conf,
        "reject_reason": str(obj.get("reject_reason") or "")[:220],
        "curation_action": action,
        "label_policy": row.get("label_policy"),
        "previous_triplet_verdict": row.get("previous_triplet_verdict"),
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_refresh.jsonl",
    )
    ap.add_argument(
        "--out",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_refresh_llm_v1.jsonl",
    )
    ap.add_argument(
        "--report",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_refresh_llm_v1.report.json",
    )
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--max_images", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=500)
    args = ap.parse_args()

    rows = list(read_jsonl(args.queue))
    priorities = {p.strip() for p in args.priority.replace(",", " ").split() if p.strip()}
    if priorities:
        rows = [r for r in rows if str(r.get("priority")) in priorities]
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    rows = [r for r in rows if str(r.get("pair_id")) not in done]
    bundles = pidx.build_bundles()

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        product_ctx = product_context(row, bundles)
        ocr_ctx = ocr_context(row)
        images = choose_images(row, args.max_images)
        data_urls = [u for p in images if (u := llm.encode_image(p))]
        obj = llm.chat_json(
            make_prompt(row, product_ctx, ocr_ctx, images),
            system="你是严谨的商品事实证据取证员，只输出 JSON。",
            model=args.model,
            images=data_urls or None,
            temperature=0.0,
            namespace="proposal_product_evidence_refresh_v1",
            max_tokens=args.max_tokens,
        )
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not a JSON object")
        return clean_result(obj, row, args.model)

    results = llm.run_many(rows, verify, concurrency=args.concurrency, desc="product_evidence_refresh_v1")
    with out_path.open("a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    all_rows = list(read_jsonl(out_path))
    report = {
        "queue": args.queue,
        "out": args.out,
        "new": len(results),
        "total": len(all_rows),
        "action": dict(Counter(str(r.get("curation_action")) for r in all_rows)),
        "relation": dict(Counter(str(r.get("relation_to_claim")) for r in all_rows)),
        "source_type": dict(Counter(str(r.get("source_type")) for r in all_rows)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
