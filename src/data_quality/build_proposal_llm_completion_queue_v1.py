"""Build prompt-ready LLM/VLM completion queues from proposal quality audit.

The queue is designed for repairing data from raw materials before promotion
to the main supervised benchmark.  It does not contain model labels beyond the
existing proposal weak label and never asks the verifier to relabel consumer
perception directly.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl


NUMERIC_TERMS = ("价格", "尺码", "尺寸", "厚度", "重量", "容量", "净含量", "含量", "数量", "件数", "码数")
MATERIAL_TERMS = ("材质", "面料", "成分", "绒", "棉", "羊毛", "皮", "不锈钢", "塑料")
VISUAL_TERMS = ("颜色", "款式", "版型", "外观", "形状", "图案", "包装", "结构", "功能", "是否")


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    segs = claim.get("segments") or []
    parts = [str(s.get("text", "") or "") for s in segs if isinstance(s, dict)]
    text = "\n".join(p for p in parts if p).strip()
    return text or str(claim.get("passage", "") or "").strip()


def bundle_fields(bundle: pidx.ProductBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {
            "product_title": "",
            "room_id": "UNKNOWN",
            "srt_files": [],
            "comment_files": [],
            "detail_images": [],
            "raw_params": {},
        }
    return {
        "product_title": bundle.title,
        "room_id": bundle.room_id,
        "srt_files": [str(pidx.resolve(p)) for p in bundle.srt_files],
        "comment_files": [str(pidx.resolve(p)) for p in bundle.comment_files],
        "detail_images": [str(pidx.resolve(p)) for p in bundle.detail_images],
        "raw_params": bundle.params,
    }


def expected_value_type(attribute_name: str, attribute_id: str) -> str:
    text = f"{attribute_name} {attribute_id}"
    if any(t in text for t in NUMERIC_TERMS):
        return "number_or_range"
    if any(t in text for t in MATERIAL_TERMS):
        return "material_or_ingredient"
    if any(t in text for t in VISUAL_TERMS):
        return "visual_or_boolean_or_style"
    return "attribute_value"


def target_sources(attribute_name: str, attribute_id: str, issue_types: set[str]) -> list[str]:
    text = f"{attribute_name} {attribute_id}"
    sources = ["srt"]
    if "missing_product_evidence" in issue_types:
        if any(t in text for t in NUMERIC_TERMS):
            sources.extend(["params", "product_title", "detail_image_ocr"])
        elif any(t in text for t in MATERIAL_TERMS):
            sources.extend(["params", "detail_image_ocr", "detail_image_vlm", "product_title"])
        elif any(t in text for t in VISUAL_TERMS):
            sources.extend(["detail_image_ocr", "detail_image_vlm", "params"])
        else:
            sources.extend(["params", "product_title", "detail_image_ocr", "detail_image_vlm"])
    else:
        sources.extend(["params", "product_title", "detail_image_ocr"])
    out = []
    for src in sources:
        if src not in out:
            out.append(src)
    return out


def priority(row: dict[str, Any]) -> tuple[str, int]:
    pq = row.get("_proposal_quality") or {}
    issues = set(pq.get("issues") or [])
    y = int(row.get("y", 0) or 0)
    c = float(row.get("c", 0.05) or 0.05)
    refs = int(pq.get("direct_consumer_claim_reference_count", 0) or 0)
    score = 0
    if "no_claim_but_direct_consumer_claim_reference" in issues:
        score += 35 + min(20, refs * 5)
    if "missing_product_evidence" in issues and y == 1:
        score += 28
    if "claim_specificity_review" in issues and y == 1:
        score += 18
    if "missing_claim" in issues:
        score += 12
    if "weak_or_unsupported_consumer_label" not in issues:
        score += 8
    score += min(20, int(c * 40))
    if score >= 55:
        return "P0", score
    if score >= 32:
        return "P1", score
    return "P2", score


def queue_type(issues: set[str]) -> str:
    claim_issue = bool({"missing_claim", "claim_specificity_review", "no_claim_but_direct_consumer_claim_reference"} & issues)
    evidence_issue = "missing_product_evidence" in issues
    if claim_issue and evidence_issue:
        return "joint_claim_evidence_completion"
    if claim_issue:
        return "claim_completion_from_raw_srt"
    if evidence_issue:
        return "product_evidence_completion_from_raw_details"
    return "label_alignment_review"


def build_item(row: dict[str, Any], bundle: pidx.ProductBundle | None) -> dict[str, Any]:
    pq = row.get("_proposal_quality") or {}
    issues = set(pq.get("issues") or [])
    attr_name = str(row.get("attribute_name", "") or "")
    attr_id = str(row.get("attribute_id", "") or "")
    p, score = priority(row)
    claim = row.get("claim") or {}
    return {
        "queue_type": queue_type(issues),
        "priority": p,
        "priority_score": score,
        "pair_id": pair_id(row),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "attribute_id": attr_id,
        "attribute_name": attr_name,
        "attribute_objectivity": "product_attribute",
        "expected_value_type": expected_value_type(attr_name, attr_id),
        "target_sources": target_sources(attr_name, attr_id, issues),
        "current_y": row.get("y"),
        "current_c": row.get("c"),
        "label_policy": "keep_current_proposal_label; verifier only repairs claim/evidence/alignment evidence",
        "label_state": (pq.get("label") or {}).get("state"),
        "claim_state": (pq.get("claim") or {}).get("state"),
        "evidence_state": (pq.get("evidence") or {}).get("state"),
        "issues": sorted(issues),
        "claim_preview": claim_text(row)[:600],
        "claim_segments": claim.get("segments", [])[:12],
        "risk_comment_example": "；".join(pq.get("direct_consumer_claim_reference_examples", [])[:3]),
        "direct_consumer_claim_reference_examples": pq.get("direct_consumer_claim_reference_examples", []),
        "accept_rule": {
            "claim": "must be an exact livestream/SRT substring about the target attribute",
            "evidence": "must come from product title/params/OCR/VLM, not from reviews or the anchor speech",
            "label": "do not relabel; if repaired claim changes alignment, queue for B4 label rebuild",
            "promotion": "promote only when claim, product evidence, and proposal label audit are all available",
        },
        **bundle_fields(bundle),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/proposal_llm_completion_queue_v1_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/proposal_llm_completion_queue_v1_20260613.report.json")
    ap.add_argument("--include_label_alignment", action="store_true")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    bundles = pidx.build_bundles()
    items = []
    skipped = Counter()
    for row in read_jsonl(args.audit):
        pq = row.get("_proposal_quality") or {}
        issues = set(pq.get("issues") or [])
        if str(row.get("_attribute_scope", "")) != "product_attribute":
            skipped["non_product_attribute"] += 1
            continue
        if not issues:
            skipped["complete"] += 1
            continue
        if issues == {"weak_or_unsupported_consumer_label"} and not args.include_label_alignment:
            skipped["label_only"] += 1
            continue
        item = build_item(row, bundles.get(str(row.get("product_id", ""))))
        items.append(item)
    items.sort(key=lambda r: (-int(r.get("priority_score", 0)), str(r.get("category", "")), str(r.get("pair_id", ""))))
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]

    write_jsonl(args.out, items)
    report = {
        "input": args.audit,
        "output": args.out,
        "n": len(items),
        "skipped": dict(skipped),
        "priority": dict(Counter(str(r.get("priority", "")) for r in items)),
        "queue_type": dict(Counter(str(r.get("queue_type", "")) for r in items)),
        "labels": dict(Counter(str(r.get("current_y", "")) for r in items)),
        "claim_state": dict(Counter(str(r.get("claim_state", "")) for r in items)),
        "evidence_state": dict(Counter(str(r.get("evidence_state", "")) for r in items)),
        "top_examples": [
            {
                "pair_id": r.get("pair_id"),
                "queue_type": r.get("queue_type"),
                "priority": r.get("priority"),
                "attribute_name": r.get("attribute_name"),
                "claim_state": r.get("claim_state"),
                "evidence_state": r.get("evidence_state"),
                "risk_comment_example": r.get("risk_comment_example"),
                "claim_preview": r.get("claim_preview", "")[:160],
            }
            for r in items[:20]
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
