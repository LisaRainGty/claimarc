"""Build full product-attribute reconstruction queue for proposal-faithful data.

This queue is deliberately broader than the previous repair queues.  The unit
is every product-scope `(product_id, attribute_id)` pair from the proposal
audit.  Each item asks later runners to:

1. re-extract target-attribute livestream claim spans from raw SRT;
2. refresh product-side evidence from title, params, OCR, and detail images;
3. rebuild the consumer-perception label by checking whether attribute-level
   comments support or refute the repaired claim.

The script is deterministic and does not change labels or training data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.build_proposal_llm_completion_queue_v1 import (
    bundle_fields,
    claim_text,
    expected_value_type,
    pair_id,
    target_sources,
)


def clean(value: Any) -> str:
    return str(value or "").strip()


def mention_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("product_id") or ""), str(row.get("attribute_id") or "")


def read_mentions(paths: list[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        source_name = Path(path).stem
        for row in read_jsonl(path):
            pid, aid = mention_key(row)
            span = clean(row.get("evidence_span"))
            if not pid or not aid or not span:
                continue
            key = (pid, aid, str(row.get("review_id") or ""), span)
            if key in seen:
                continue
            seen.add(key)
            by_pair[(pid, aid)].append({
                "review_id": row.get("review_id"),
                "evidence_span": span[:220],
                "polarity": row.get("polarity"),
                "review_polarity": row.get("review_polarity"),
                "mention_strength": row.get("mention_strength"),
                "explicit_fact_hit": bool(row.get("explicit_fact_hit", False)),
                "review_time": row.get("review_time", ""),
                "type": row.get("type"),
                "_was_free": row.get("_was_free", ""),
                "_mention_source": source_name,
            })
    return by_pair


def mention_score(row: dict[str, Any]) -> int:
    score = 0
    if row.get("polarity") == "neg":
        score += 20
    if row.get("explicit_fact_hit"):
        score += 16
    if row.get("mention_strength") == "strong":
        score += 8
    if row.get("review_polarity") == "neg":
        score += 4
    text = clean(row.get("evidence_span"))
    for cue in ("宣传", "主播", "直播", "说是", "说的", "承诺", "标", "页面", "详情", "虚标", "不符", "不是", "骗人"):
        if cue in text:
            score += 4
    return score


def compact_mentions(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: (-mention_score(r), str(r.get("review_id") or ""), clean(r.get("evidence_span"))))
    out: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for row in rows:
        text = clean(row.get("evidence_span"))
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        r = dict(row)
        r["priority_score"] = mention_score(row)
        out.append(r)
        if len(out) >= cap:
            break
    return out


def task_flags(row: dict[str, Any]) -> dict[str, Any]:
    pq = row.get("_proposal_quality") or {}
    claim_state = (pq.get("claim") or {}).get("state")
    evidence_state = (pq.get("evidence") or {}).get("state")
    label_state = (pq.get("label") or {}).get("state")
    claim_needs = claim_state in {"claim_missing", "claim_present_review_needed"}
    evidence_needs = evidence_state in {"evidence_missing", "evidence_single_source"}
    return {
        "claim_reextract_required": bool(claim_needs),
        "evidence_refresh_required": bool(evidence_needs),
        "label_rebuild_required": True,
        "claim_state": claim_state,
        "evidence_state": evidence_state,
        "label_state": label_state,
        "issues": pq.get("issues") or [],
    }


def priority(row: dict[str, Any], mentions: list[dict[str, Any]]) -> tuple[str, int]:
    flags = task_flags(row)
    y = int(row.get("y", 0) or 0)
    c = float(row.get("c", 0.05) or 0.05)
    n_neg = sum(1 for m in mentions if m.get("polarity") == "neg")
    n_explicit = sum(1 for m in mentions if m.get("explicit_fact_hit"))
    score = 0
    if y == 1:
        score += 28
    if flags["claim_reextract_required"]:
        score += 24
    if flags["evidence_refresh_required"]:
        score += 18
    if flags["label_state"] == "label_negative_no_aligned_review" and n_neg:
        score += 18
    score += min(24, n_neg * 4)
    score += min(18, n_explicit * 6)
    score += min(20, int(c * 40))
    score += min(16, max((mention_score(m) for m in mentions), default=0) // 2)
    if score >= 70:
        return "P0", score
    if score >= 45:
        return "P1", score
    if score >= 22:
        return "P2", score
    return "P3", score


def queue_type(flags: dict[str, Any]) -> str:
    if flags["claim_reextract_required"] and flags["evidence_refresh_required"]:
        return "full_claim_evidence_label_rebuild"
    if flags["claim_reextract_required"]:
        return "claim_reextract_label_rebuild"
    if flags["evidence_refresh_required"]:
        return "evidence_refresh_label_rebuild"
    return "label_rebuild_existing_triplet"


def evidence_preview(row: dict[str, Any], cap: int = 8) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for ev in row.get("evidence_params") or []:
        items.append({"source": "params", "key": clean(ev.get("param_key")), "text": clean(ev.get("raw_text"))[:180]})
    for ev in row.get("evidence_ocr") or []:
        items.append({"source": "detail_image_ocr", "key": Path(clean(ev.get("image_path"))).name, "text": clean(ev.get("raw_text"))[:180]})
    for ev in row.get("evidence_vlm") or []:
        items.append({"source": "detail_image_vlm", "key": Path(clean(ev.get("image_path"))).name, "text": clean(ev.get("raw_quote") or ev.get("raw_text"))[:180]})
    return [x for x in items if x["text"]][:cap]


def build_item(
    row: dict[str, Any],
    bundle: pidx.ProductBundle | None,
    mentions: list[dict[str, Any]],
    mention_cap: int,
) -> dict[str, Any]:
    flags = task_flags(row)
    p, score = priority(row, mentions)
    attr_name = clean(row.get("attribute_name"))
    attr_id = clean(row.get("attribute_id"))
    issues = set(flags["issues"])
    tgt = target_sources(attr_name, attr_id, issues | {"missing_product_evidence"})
    if "product_title" not in tgt:
        tgt.append("product_title")
    item = {
        "queue_type": queue_type(flags),
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
        "target_sources": tgt,
        "old_y": row.get("y"),
        "old_c": row.get("c"),
        "old_label_state": flags["label_state"],
        "old_label_audit": row.get("label_audit") or {},
        "claim_state": flags["claim_state"],
        "evidence_state": flags["evidence_state"],
        "issues": flags["issues"],
        "reconstruction_flags": flags,
        "claim_preview": claim_text(row)[:800],
        "claim_segments": (row.get("claim") or {}).get("segments", [])[:16],
        "current_evidence_preview": evidence_preview(row),
        "consumer_mentions_total": len(mentions),
        "consumer_mentions_neg": sum(1 for m in mentions if m.get("polarity") == "neg"),
        "consumer_mentions_explicit": sum(1 for m in mentions if m.get("explicit_fact_hit")),
        "consumer_mentions": compact_mentions(mentions, mention_cap),
        "label_rebuild_policy": {
            "unit": "product_attribute_pair",
            "old_label_role": "audit_only_not_final",
            "positive_rule": (
                "After claim/evidence reconstruction, set y=1 only if at least one "
                "attribute-level consumer comment is aligned to the same repaired "
                "livestream claim and refutes/contradicts it."
            ),
            "negative_rule": (
                "Set y=0 when comments support the repaired claim or discuss the "
                "attribute without contradicting the claim; keep low c when no comment "
                "can be aligned to the repaired claim."
            ),
            "c_rule": "Recompute reliability from aligned support/refute counts, mention strength, explicit fact hit, and evidence coverage.",
        },
        "accept_rule": {
            "claim": "must be a minimal continuous raw SRT substring about the target attribute",
            "product_evidence": "must come from product title, params, detail OCR, or detail-image VLM",
            "comment_label": "must compare attribute-level comments against the repaired claim, not against generic product quality",
            "no_shortcut": "do not use product evidence contradiction alone as y; consumer refutation is required for y=1",
        },
        **bundle_fields(bundle),
    }
    return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl")
    ap.add_argument(
        "--mentions",
        nargs="*",
        default=[
            "data/processed/stageA_repaired_v1/resolved_aspects_schema_clean_v1.jsonl",
            "data/processed/stageA/resolved_aspects.jsonl",
        ],
    )
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.report.json")
    ap.add_argument("--mention_cap", type=int, default=12)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    mentions_by_pair = read_mentions(args.mentions)
    bundles = pidx.build_bundles()
    items: list[dict[str, Any]] = []
    skipped = Counter()
    for row in read_jsonl(args.audit):
        if str(row.get("_attribute_scope", "")) != "product_attribute":
            skipped["non_product_attribute"] += 1
            continue
        key = (str(row.get("product_id") or ""), str(row.get("attribute_id") or ""))
        mentions = mentions_by_pair.get(key, [])
        item = build_item(row, bundles.get(str(row.get("product_id") or "")), mentions, args.mention_cap)
        items.append(item)

    items.sort(key=lambda r: (-int(r.get("priority_score", 0)), str(r.get("category", "")), str(r.get("pair_id", ""))))
    if args.max_items > 0:
        items = items[:args.max_items]
    write_jsonl(args.out, items)
    report = {
        "audit": args.audit,
        "mentions": args.mentions,
        "out": args.out,
        "n": len(items),
        "skipped": dict(skipped),
        "priority": dict(Counter(str(r.get("priority")) for r in items)),
        "queue_type": dict(Counter(str(r.get("queue_type")) for r in items)),
        "old_label": dict(Counter(str(r.get("old_y")) for r in items)),
        "claim_state": dict(Counter(str(r.get("claim_state")) for r in items)),
        "evidence_state": dict(Counter(str(r.get("evidence_state")) for r in items)),
        "old_label_state": dict(Counter(str(r.get("old_label_state")) for r in items)),
        "mention_coverage": {
            "pairs_with_mentions": sum(1 for r in items if int(r.get("consumer_mentions_total", 0)) > 0),
            "pairs_with_neg_mentions": sum(1 for r in items if int(r.get("consumer_mentions_neg", 0)) > 0),
            "pairs_with_explicit_mentions": sum(1 for r in items if int(r.get("consumer_mentions_explicit", 0)) > 0),
        },
        "top_examples": [
            {
                "pair_id": r.get("pair_id"),
                "priority": r.get("priority"),
                "queue_type": r.get("queue_type"),
                "attribute_id": r.get("attribute_id"),
                "old_y": r.get("old_y"),
                "claim_state": r.get("claim_state"),
                "evidence_state": r.get("evidence_state"),
                "mentions": r.get("consumer_mentions", [])[:3],
                "claim_preview": clean(r.get("claim_preview"))[:160],
            }
            for r in items[:20]
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
