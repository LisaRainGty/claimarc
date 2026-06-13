"""Build prompt-ready repair queue for triplet-alignment failures.

Rows in `proposal_triplet_alignment_repair_queue_v2` already have claim,
evidence, and labels, but at least one side is not aligned with the target
attribute.  This script turns them into raw-material repair tasks for LLM/VLM
verification.  It preserves the existing proposal labels and never asks the
verifier to relabel consumer perception directly.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.build_proposal_llm_completion_queue_v1 import (
    bundle_fields,
    claim_text,
    expected_value_type,
    pair_id,
)


def evidence_preview(row: dict[str, Any], limit: int = 12) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for ev in row.get("evidence_params") or []:
        items.append({
            "source": "params",
            "key": str(ev.get("param_key", "") or ""),
            "text": str(ev.get("raw_text", "") or "")[:180],
        })
    for ev in row.get("evidence_ocr") or []:
        items.append({
            "source": "detail_image_ocr",
            "key": Path(str(ev.get("image_path", "") or "")).name,
            "text": str(ev.get("raw_text", "") or "")[:180],
        })
    for ev in row.get("evidence_vlm") or []:
        items.append({
            "source": "detail_image_vlm",
            "key": Path(str(ev.get("image_path", "") or "")).name,
            "text": str(ev.get("raw_quote", "") or ev.get("raw_text", "") or "")[:180],
        })
    return items[:limit]


def priority(row: dict[str, Any], audit: dict[str, Any]) -> tuple[str, int]:
    y = int(row.get("y", 0) or 0)
    c = float(row.get("c", 0.05) or 0.05)
    issues = set(audit.get("issues") or [])
    score = 0
    if y == 1:
        score += 35
    if "claim_attribute_alignment_review" in issues:
        score += 22
    if "product_evidence_alignment_review" in issues:
        score += 18
    if (audit.get("label") or {}).get("supported_by_aligned_review"):
        score += 12
    score += min(20, int(c * 40))
    if score >= 60:
        return "P0", score
    if score >= 36:
        return "P1", score
    return "P2", score


def target_sources(audit: dict[str, Any]) -> list[str]:
    issues = set(audit.get("issues") or [])
    sources: list[str] = []
    if "claim_attribute_alignment_review" in issues:
        sources.append("srt")
    if "product_evidence_alignment_review" in issues:
        sources.extend(["params", "product_title", "detail_image_ocr", "detail_image_vlm"])
    if not sources:
        sources.extend(["srt", "params", "product_title", "detail_image_ocr"])
    out = []
    for src in sources:
        if src not in out:
            out.append(src)
    return out


def queue_type(audit: dict[str, Any]) -> str:
    issues = set(audit.get("issues") or [])
    claim_bad = "claim_attribute_alignment_review" in issues
    ev_bad = "product_evidence_alignment_review" in issues
    if claim_bad and ev_bad:
        return "triplet_claim_and_evidence_realignment"
    if claim_bad:
        return "triplet_claim_attribute_realignment"
    if ev_bad:
        return "triplet_product_evidence_realignment"
    return "triplet_label_reliability_audit"


def build_item(row: dict[str, Any], bundle: pidx.ProductBundle | None) -> dict[str, Any]:
    audit = row.get("_proposal_triplet_alignment_v2") or {}
    p, score = priority(row, audit)
    attr_name = str(row.get("attribute_name", "") or "")
    attr_id = str(row.get("attribute_id", "") or "")
    claim = row.get("claim") or {}
    return {
        "queue_type": queue_type(audit),
        "priority": p,
        "priority_score": score,
        "pair_id": pair_id(row),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "attribute_id": attr_id,
        "attribute_name": attr_name,
        "attribute_family": audit.get("attribute_family"),
        "attribute_objectivity": "product_attribute",
        "expected_value_type": expected_value_type(attr_name, attr_id),
        "target_sources": target_sources(audit),
        "current_y": row.get("y"),
        "current_c": row.get("c"),
        "label_policy": "preserve proposal y/c; repair only claim/evidence/alignment materials",
        "label_state": (audit.get("label") or {}).get("state"),
        "triplet_issues": audit.get("issues") or [],
        "claim_valid": audit.get("claim_valid"),
        "evidence_valid": audit.get("evidence_valid"),
        "claim_checks": audit.get("claim_checks") or [],
        "evidence_checks": audit.get("evidence_checks") or [],
        "claim_preview": claim_text(row)[:800],
        "claim_segments": claim.get("segments", [])[:16],
        "current_evidence_preview": evidence_preview(row),
        "risk_comment_example": "",
        "accept_rule": {
            "claim": "must be an exact livestream/SRT substring about the target attribute; replace or mark invalid if current claim drifts to another attribute",
            "evidence": "must come from product title/params/OCR/VLM and refer to the target attribute; isolated OCR digits or generic promo text are insufficient",
            "relation": "judge whether repaired product evidence supports or contradicts the repaired claim, but do not create the consumer label",
            "label": "do not relabel; if repaired claim changes review alignment, queue for B4 label rebuild",
            "promotion": "promote only when claim_valid, evidence_valid, and relation_to_claim is supports_claim or contradicts_claim",
        },
        **bundle_fields(bundle),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", default="data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.report.json")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    bundles = pidx.build_bundles()
    items = [build_item(row, bundles.get(str(row.get("product_id", "")))) for row in read_jsonl(args.repair)]
    items.sort(key=lambda r: (-int(r.get("priority_score", 0)), str(r.get("category", "")), str(r.get("pair_id", ""))))
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]
    write_jsonl(args.out, items)
    report = {
        "input": args.repair,
        "output": args.out,
        "n": len(items),
        "priority": dict(Counter(str(r.get("priority", "")) for r in items)),
        "queue_type": dict(Counter(str(r.get("queue_type", "")) for r in items)),
        "labels": dict(Counter(str(r.get("current_y", "")) for r in items)),
        "triplet_issues": dict(Counter(issue for r in items for issue in (r.get("triplet_issues") or []))),
        "top_examples": [
            {
                "pair_id": r.get("pair_id"),
                "priority": r.get("priority"),
                "queue_type": r.get("queue_type"),
                "attribute_name": r.get("attribute_name"),
                "current_y": r.get("current_y"),
                "triplet_issues": r.get("triplet_issues"),
                "claim_preview": str(r.get("claim_preview", ""))[:180],
                "evidence_preview": r.get("current_evidence_preview", [])[:2],
            }
            for r in items[:20]
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
