"""Audit full-pair LLM reconstruction reviews before dataset promotion.

This script is a local quality gate.  It checks whether each reconstructed row
obeys the proposal-level label definition:

1. a recoverable livestream claim,
2. product-side evidence from title/params/OCR/VLM, and
3. a consumer comment relation judged against that same claim.

It does not drop rows or optimize benchmark difficulty.  Rows with problems are
flagged for rerun, silver handling, or manual inspection before promotion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


VALID_ACTION = {
    "promote_candidate",
    "silver_review",
    "rerun_claim",
    "rerun_evidence",
    "rerun_joint",
    "drop_no_reconstructable_claim",
}
VALID_CONFIDENCE = {"high", "medium", "low", ""}
VALID_EVIDENCE_SOURCE = {
    "product_title",
    "params",
    "detail_image_ocr",
    "detail_image_vlm",
    "none",
    "",
}
VALID_CLAIM_EVIDENCE_REL = {
    "supports_claim",
    "contradicts_claim",
    "insufficient",
    "",
}
VALID_COMMENT_REL = {"support", "refute", "mixed", "unclear", "not_aligned", ""}


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def compact(text: Any) -> str:
    return "".join(clean(text).split()).lower()


def boolish(value: Any) -> bool:
    return bool(value)


def int01(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        return 0


def read_reviews(path: str | Path) -> tuple[dict[str, dict[str, Any]], Counter]:
    path = Path(path)
    if not path.exists():
        return {}, Counter()
    out: dict[str, dict[str, Any]] = {}
    duplicates: Counter = Counter()
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid in out:
            duplicates[pid] += 1
        out[pid] = row
    return out, duplicates


def srt_candidate_texts(queue_row: dict[str, Any]) -> list[str]:
    pref = queue_row.get("srt_prefilter") or {}
    return [clean(c.get("text")) for c in (pref.get("claim_candidates") or []) if clean(c.get("text"))]


def claim_in_prefilter(queue_row: dict[str, Any], claim: str) -> bool | None:
    claim_norm = compact(claim)
    if len(claim_norm) < 4:
        return None
    candidates = [compact(x) for x in srt_candidate_texts(queue_row)]
    if not candidates:
        return None
    return any(claim_norm in cand or cand in claim_norm for cand in candidates)


def add_flag(flags: list[dict[str, str]], severity: str, code: str, detail: str = "") -> None:
    flags.append({"severity": severity, "code": code, "detail": detail})


def comment_stats(review: dict[str, Any], max_comments: int, flags: list[dict[str, str]]) -> Counter:
    stats: Counter = Counter()
    judgments = review.get("comment_judgments")
    if not isinstance(judgments, list):
        add_flag(flags, "high", "comment_judgments_not_list")
        return stats
    seen: Counter = Counter()
    for item in judgments:
        if not isinstance(item, dict):
            add_flag(flags, "high", "comment_judgment_not_object")
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        if cid < 1 or cid > max_comments:
            add_flag(flags, "high", "comment_cid_out_of_range", str(cid))
            continue
        seen[cid] += 1
        rel = clean(item.get("relation"))
        aligned = boolish(item.get("aligned_to_claim"))
        if rel not in VALID_COMMENT_REL:
            add_flag(flags, "high", "invalid_comment_relation", rel)
        if aligned and rel not in {"support", "refute", "mixed"}:
            add_flag(flags, "high", "aligned_comment_without_valid_relation", f"cid={cid}, rel={rel}")
        if (not aligned) and rel in {"support", "refute", "mixed"}:
            add_flag(flags, "medium", "relation_present_but_not_aligned", f"cid={cid}, rel={rel}")
        if aligned:
            stats[rel] += 1
    for cid, count in seen.items():
        if count > 1:
            add_flag(flags, "low", "duplicate_comment_judgment", str(cid))
    return stats


def promotion_state(review: dict[str, Any], rel: Counter) -> str:
    if review.get("__error__"):
        return "llm_error"
    claim_found = boolish(review.get("claim_found"))
    evidence_found = boolish(review.get("product_evidence_found"))
    claim_evidence_relation = clean(review.get("claim_evidence_relation"))
    if not claim_found:
        return "repair_missing_claim"
    if not evidence_found:
        if rel.get("refute", 0) > 0:
            return "silver_refute_missing_product_evidence"
        return "repair_missing_evidence"
    if claim_evidence_relation in {"", "insufficient"}:
        if rel.get("refute", 0) > 0:
            return "silver_refute_insufficient_product_evidence"
        return "repair_insufficient_product_evidence"
    if rel.get("refute", 0) > 0:
        return "main_positive_refute"
    if rel.get("support", 0) > 0 and rel.get("mixed", 0) == 0:
        return "main_negative_support"
    if rel.get("mixed", 0) > 0:
        return "silver_mixed_comment_relation"
    return "lowinfo_no_aligned_comment"


def audit_one(queue_row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    flags: list[dict[str, str]] = []
    if review is None:
        add_flag(flags, "missing", "missing_review")
        return {
            "pair_id": pair_id(queue_row),
            "priority": queue_row.get("priority"),
            "queue_type": queue_row.get("queue_type"),
            "category": queue_row.get("category"),
            "attribute_name": queue_row.get("attribute_name"),
            "new_y": None,
            "promotion_state": "missing_review",
            "flags": flags,
        }

    max_comments = len(queue_row.get("consumer_mentions") or [])
    rel = comment_stats(review, max_comments, flags)
    state = promotion_state(review, rel)

    claim_found = boolish(review.get("claim_found"))
    claim_text = clean(review.get("claim_text"))
    evidence_found = boolish(review.get("product_evidence_found"))
    evidence_text = clean(review.get("evidence_text"))
    evidence_source_type = clean(review.get("evidence_source_type"))
    claim_evidence_relation = clean(review.get("claim_evidence_relation"))
    confidence = clean(review.get("confidence")).lower()
    action = clean(review.get("action"))
    new_y = int01(review.get("new_y"))
    raw_new_y = review.get("raw_new_y")

    if review.get("__error__"):
        add_flag(flags, "high", "llm_error", clean(review.get("__error__"))[:80])
    if claim_found and not claim_text:
        add_flag(flags, "high", "claim_found_empty_text")
    if (not claim_found) and claim_text:
        add_flag(flags, "low", "claim_text_present_but_claim_found_false")
    hit = claim_in_prefilter(queue_row, claim_text)
    if hit is False:
        add_flag(flags, "medium", "claim_not_in_top_srt_prefilter", "manual source check needed")

    if evidence_found and evidence_source_type in {"none", ""}:
        add_flag(flags, "high", "evidence_found_without_source_type")
    if evidence_found and not evidence_text:
        add_flag(flags, "high", "evidence_found_empty_text")
    if (not evidence_found) and evidence_text:
        add_flag(flags, "low", "evidence_text_present_but_evidence_found_false")
    if evidence_source_type not in VALID_EVIDENCE_SOURCE:
        add_flag(flags, "high", "invalid_evidence_source_type", evidence_source_type)
    if claim_evidence_relation not in VALID_CLAIM_EVIDENCE_REL:
        add_flag(flags, "high", "invalid_claim_evidence_relation", claim_evidence_relation)
    if confidence not in VALID_CONFIDENCE:
        add_flag(flags, "medium", "invalid_confidence", confidence)
    if action not in VALID_ACTION:
        add_flag(flags, "medium", "invalid_action", action)
    if action == "promote_candidate" and claim_evidence_relation in {"", "insufficient"}:
        add_flag(flags, "high", "promote_with_insufficient_claim_evidence_relation", claim_evidence_relation or "empty")

    expected_y = 1 if claim_found and rel.get("refute", 0) > 0 else 0
    if new_y != expected_y:
        add_flag(flags, "high", "new_y_inconsistent_with_claim_comment_relation", f"new_y={new_y}, expected={expected_y}")
    if raw_new_y not in {None, ""} and int01(raw_new_y) != expected_y:
        add_flag(flags, "low", "raw_llm_y_overridden_by_clean_rule", f"raw_new_y={raw_new_y}, expected={expected_y}")
    if new_y == 1 and not evidence_found:
        add_flag(flags, "medium", "positive_label_missing_product_evidence_for_main")
    if claim_evidence_relation == "contradicts_claim" and rel.get("refute", 0) == 0:
        add_flag(flags, "info", "mechanism_contradiction_without_consumer_refute")
    if action == "promote_candidate" and state not in {"main_positive_refute", "main_negative_support"}:
        add_flag(flags, "medium", "promote_action_but_not_main_ready", state)

    return {
        "pair_id": pair_id(queue_row),
        "priority": queue_row.get("priority"),
        "queue_type": queue_row.get("queue_type"),
        "category": queue_row.get("category"),
        "attribute_name": queue_row.get("attribute_name"),
        "new_y": new_y,
        "confidence": confidence,
        "promotion_state": state,
        "comment_relation_counts": dict(rel),
        "flags": flags,
    }


def summarize(audits: list[dict[str, Any]], review_rows: int, duplicates: Counter) -> dict[str, Any]:
    flag_code: Counter = Counter()
    flag_severity: Counter = Counter()
    flagged_rows = 0
    high_rows = 0
    medium_or_high_rows = 0
    for row in audits:
        flags = row.get("flags") or []
        if flags:
            flagged_rows += 1
        severities = {f.get("severity") for f in flags}
        if "high" in severities:
            high_rows += 1
        if "high" in severities or "medium" in severities:
            medium_or_high_rows += 1
        for f in flags:
            flag_code[clean(f.get("code"))] += 1
            flag_severity[clean(f.get("severity"))] += 1
    return {
        "queue_rows": len(audits),
        "review_rows": review_rows,
        "matched_reviews": sum(1 for r in audits if r.get("promotion_state") != "missing_review"),
        "missing_reviews": sum(1 for r in audits if r.get("promotion_state") == "missing_review"),
        "duplicate_review_pairs": len(duplicates),
        "duplicate_review_events": sum(duplicates.values()),
        "flagged_rows": flagged_rows,
        "high_flag_rows": high_rows,
        "medium_or_high_flag_rows": medium_or_high_rows,
        "flag_severity": dict(flag_severity),
        "flag_code": dict(flag_code),
        "promotion_state": dict(Counter(clean(r.get("promotion_state")) for r in audits)),
        "new_y": dict(Counter(str(r.get("new_y")) for r in audits)),
        "confidence": dict(Counter(clean(r.get("confidence")) for r in audits if r.get("confidence") is not None)),
        "category": dict(Counter(clean(r.get("category")) for r in audits)),
    }


def write_markdown(path: str | Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Full Pair LLM Review Audit v1",
        "",
        "This report audits LLM/VLM reconstruction reviews before promotion.",
        "It checks label-definition consistency rather than benchmark separability.",
        "",
        "## Inputs",
        "",
        f"- queue: `{args.queue}`",
        f"- reviews: `{args.reviews}`",
        "",
        "## Outputs",
        "",
        f"- report json: `{args.out}`",
        f"- flagged rows: `{args.flagged}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Gate Interpretation",
        "",
        "- `high` flags block main promotion until rerun or manual repair.",
        "- `medium` flags require manual sampling or silver routing.",
        "- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.",
        "- Missing reviews are expected before the LLM pilot has been executed.",
        "",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json")
    ap.add_argument("--flagged", default="data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_LLM_REVIEW_AUDIT_20260614.md")
    args = ap.parse_args()

    queue_rows = list(read_jsonl(args.queue))
    reviews, duplicates = read_reviews(args.reviews)
    audits = [audit_one(row, reviews.get(pair_id(row))) for row in queue_rows]
    flagged = [row for row in audits if row.get("flags")]
    report = summarize(audits, len(reviews), duplicates)
    report.update({
        "queue": args.queue,
        "reviews": args.reviews,
        "flagged": args.flagged,
        "out": args.out,
    })
    write_json(args.out, report)
    write_jsonl(args.flagged, flagged)
    write_markdown(args.markdown, report, args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
