"""Build a Stage-B claim repair queue from full-pair LLM reviews.

Rows in this queue failed to recover a livestream claim in the joint
reconstruction pass.  They are routed to the stricter pair-targeted
`llm_pair_claim_reextract_v1` runner, which scans the product's SRT chunks and
locally verifies exact substring grounding.

This script does not write labels or promote rows.  It preserves hard cases,
including comment-reported live claims, as source-recovery tasks.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def read_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid:
            out[pid] = row
    return out


CLAIM_TRIGGER_TERMS = (
    "主播", "直播", "直播间", "宣传", "说", "说的", "说是", "承诺", "介绍",
    "标注", "写着", "写的", "页面", "详情", "虚标", "不符", "不一致",
    "缩水", "骗人", "假", "没有宣传", "没说的", "根本没有",
)


def trigger_comments(row: dict[str, Any], max_comments: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in row.get("consumer_mentions") or []:
        text = clean(item.get("evidence_span"))
        hits = [t for t in CLAIM_TRIGGER_TERMS if t in text]
        if item.get("explicit_fact_hit") or hits or clean(item.get("polarity")) == "neg":
            out.append({
                "text": text[:240],
                "polarity": item.get("polarity"),
                "mention_strength": item.get("mention_strength"),
                "explicit_fact_hit": bool(item.get("explicit_fact_hit")),
                "hits": hits[:8],
            })
        if len(out) >= max_comments:
            break
    return out


def evidence_hint(row: dict[str, Any]) -> dict[str, Any]:
    previews = row.get("current_evidence_preview") or []
    if isinstance(previews, list) and previews:
        first = previews[0] or {}
        return {
            "source_type": clean(first.get("source") or first.get("_source_type") or "preview"),
            "raw_text": clean(first.get("text") or first.get("raw_text") or first.get("raw_quote")),
            "normalized_value": clean(first.get("key") or first.get("param_key") or first.get("source")),
        }
    params = row.get("raw_params") or {}
    attr = clean(row.get("attribute_name"))
    if isinstance(params, dict):
        for key, val in params.items():
            if attr and (attr in clean(key) or clean(key) in attr):
                return {
                    "source_type": "params",
                    "raw_text": clean(val),
                    "normalized_value": clean(key),
                }
    return {}


def should_select(queue_row: dict[str, Any], review: dict[str, Any] | None, min_trigger_comments: int) -> bool:
    if review is None:
        return False
    if review.get("__error__"):
        return True
    if review.get("claim_found"):
        return False
    pref = queue_row.get("srt_prefilter") or {}
    if clean(pref.get("prefilter_state")) == "no_srt_candidate":
        return len(trigger_comments(queue_row, 12)) >= min_trigger_comments
    return True


def build_item(
    queue_row: dict[str, Any],
    review: dict[str, Any],
    source_rank: int,
    max_comments: int,
    max_srt_candidates: int,
) -> dict[str, Any]:
    item = dict(queue_row)
    pid = pair_id(queue_row)
    triggers = trigger_comments(queue_row, max_comments)
    pref = item.get("srt_prefilter") or {}
    if isinstance(pref.get("claim_candidates"), list):
        pref = dict(pref)
        pref["claim_candidates"] = pref["claim_candidates"][:max_srt_candidates]
        item["srt_prefilter"] = pref
    item["queue_type"] = "claim_srt_reextract_from_llm_review"
    item["risk_comment_example"] = "；".join(t["text"] for t in triggers[:4])
    item["direct_consumer_claim_reference_examples"] = [t["text"] for t in triggers[:12]]
    item["aliases"] = [
        clean(item.get("attribute_name")),
        clean(item.get("attribute_id")),
    ] + [
        clean(k) for k in (item.get("raw_params") or {}).keys()
        if clean(item.get("attribute_name")) and clean(item.get("attribute_name")) in clean(k)
    ][:8]
    item["_verify_context"] = evidence_hint(item)
    item["_claim_repair"] = {
        "repair_queue_version": "full_pair_claim_repair_v1_20260614",
        "source_rank": source_rank,
        "source_review_action": clean(review.get("action")),
        "source_review_error": clean(review.get("__error__")),
        "source_claim_found": bool(review.get("claim_found")),
        "srt_prefilter_state": clean(pref.get("prefilter_state")),
        "srt_prefilter_top_score": pref.get("top_score", 0),
        "trigger_comment_count": len(triggers),
        "old_label_role": "audit_only_not_target",
        "purpose": "recover exact streamer SRT claim before any label rebuild",
    }
    if not item.get("pair_id"):
        item["pair_id"] = pid
    return item


def selection_key(row: dict[str, Any]) -> tuple:
    meta = row.get("_claim_repair") or {}
    state_rank = {
        "strong_srt_candidate": 0,
        "weak_srt_candidate": 1,
        "very_weak_srt_candidate": 2,
        "no_srt_candidate": 3,
    }
    return (
        state_rank.get(clean(meta.get("srt_prefilter_state")), 9),
        -int(meta.get("trigger_comment_count", 0) or 0),
        -int(row.get("consumer_mentions_explicit", 0) or 0),
        clean(row.get("category")),
        pair_id(row),
    )


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair Claim Repair Queue v1",
        "",
        "This queue routes claim-missing full-pair reviews to exact SRT re-extraction.",
        "It preserves hard/source-missing rows and does not change labels.",
        "",
        "## Inputs",
        "",
        f"- queue: `{report['queue']}`",
        f"- reviews: `{report['reviews']}`",
        "",
        "## Outputs",
        "",
        f"- claim repair queue: `{report['out']}`",
        f"- report: `{report['report']}`",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "review_rows",
        "selected_rows",
        "limit",
        "missing_queue_rows",
        "prefilter_state",
        "source_review_action",
        "trigger_comment_bucket",
        "category",
        "old_label_state",
    ]:
        lines.append(f"- `{key}`: `{report.get(key)}`")
    lines.extend(["", "## Examples", ""])
    for ex in report.get("examples", []):
        lines.append(
            f"- `{ex['pair_id']}` cat={ex['category']} attr={ex['attribute_name']} "
            f"pref={ex['prefilter_state']} triggers={ex['trigger_comment_count']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def trigger_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    return "5+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_CLAIM_REPAIR_QUEUE_20260614.md")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--min_trigger_comments_for_no_srt", type=int, default=1)
    ap.add_argument("--max_comments", type=int, default=12)
    ap.add_argument("--max_srt_candidates", type=int, default=8)
    args = ap.parse_args()

    queue_by_pair = read_by_pair(args.queue)
    selected: list[dict[str, Any]] = []
    missing_queue_rows = 0
    review_rows = 0

    for rank, review in enumerate(read_jsonl(args.reviews), 1):
        review_rows += 1
        pid = pair_id(review)
        queue_row = queue_by_pair.get(pid)
        if not queue_row:
            missing_queue_rows += 1
            continue
        if not should_select(queue_row, review, args.min_trigger_comments_for_no_srt):
            continue
        selected.append(build_item(queue_row, review, rank, args.max_comments, args.max_srt_candidates))

    selected.sort(key=selection_key)
    if args.limit and args.limit > 0:
        selected = selected[:args.limit]
    write_jsonl(args.out, selected)

    report = {
        "queue": args.queue,
        "reviews": args.reviews,
        "out": args.out,
        "report": args.report,
        "markdown": args.markdown,
        "review_rows": review_rows,
        "selected_rows": len(selected),
        "limit": args.limit,
        "missing_queue_rows": missing_queue_rows,
        "prefilter_state": dict(Counter(clean((r.get("_claim_repair") or {}).get("srt_prefilter_state")) for r in selected)),
        "source_review_action": dict(Counter(clean((r.get("_claim_repair") or {}).get("source_review_action")) for r in selected)),
        "trigger_comment_bucket": dict(Counter(trigger_bucket(int((r.get("_claim_repair") or {}).get("trigger_comment_count", 0) or 0)) for r in selected)),
        "category": dict(Counter(clean(r.get("category")) for r in selected)),
        "old_label_state": dict(Counter(clean(r.get("old_label_state")) for r in selected)),
        "examples": [
            {
                "pair_id": pair_id(r),
                "category": r.get("category"),
                "attribute_name": r.get("attribute_name"),
                "prefilter_state": (r.get("_claim_repair") or {}).get("srt_prefilter_state"),
                "trigger_comment_count": (r.get("_claim_repair") or {}).get("trigger_comment_count"),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
