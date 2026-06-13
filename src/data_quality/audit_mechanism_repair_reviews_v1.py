"""Audit LLM/VLM reviews before applying mechanism repairs.

This is a gatekeeper between ``llm_review_mechanism_repair_queue_v1.py`` and
``apply_mechanism_repair_reviews_v1.py``.  It validates schema, coverage,
duplicates, invalid enum values, relation/value consistency, and distribution
coverage relative to the pilot queue.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_RELATION = {"supports", "contradicts", "insufficient", "not_verifiable"}
VALID_VALUE = {
    "exact_match",
    "compatible",
    "contradiction",
    "ambiguous",
    "not_applicable",
}
VALID_SOURCE = {"params", "ocr", "vlm", "detail_image", "mixed", "none"}
VALID_ACTION = {
    "keep_relation",
    "recover_more_evidence",
    "drop_bad_claim",
    "review_consumer_signal",
}
VALID_CLAIM_QUALITY = {"clear", "mixed", "garbled", "no_claim"}
VALID_CONF = {"high", "medium", "low"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                rows.append({"__error__": f"json_error_line_{line_no}:{repr(exc)[:120]}"})
                continue
            rows.append(obj)
    return rows


def strval(obj: dict[str, Any], key: str) -> str:
    return str(obj.get(key, "") or "").strip()


def load_queue(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("pair_id")): row
        for row in read_jsonl(path)
        if row.get("pair_id")
    }


def audit_review(
    review: dict[str, Any],
    queue_ids: set[str],
) -> list[str]:
    issues: list[str] = []
    if review.get("__error__"):
        issues.append("review_error")
        return issues
    pair_id = strval(review, "pair_id")
    if not pair_id:
        issues.append("missing_pair_id")
    elif pair_id not in queue_ids:
        issues.append("pair_id_not_in_queue")

    checks = [
        ("claim_quality", VALID_CLAIM_QUALITY),
        ("relation_to_claim", VALID_RELATION),
        ("value_alignment", VALID_VALUE),
        ("evidence_source", VALID_SOURCE),
        ("repair_action", VALID_ACTION),
        ("confidence", VALID_CONF),
    ]
    for field, valid in checks:
        if strval(review, field) not in valid:
            issues.append(f"invalid_{field}")

    rel = strval(review, "relation_to_claim")
    val = strval(review, "value_alignment")
    evidence_found = bool(review.get("evidence_found", False))
    source = strval(review, "evidence_source")
    key_evidence = strval(review, "key_evidence")
    claim_quality = strval(review, "claim_quality")

    if rel == "supports" and val not in {"exact_match", "compatible"}:
        issues.append("support_without_compatible_value")
    if rel == "contradicts" and val != "contradiction":
        issues.append("contradiction_without_value_contradiction")
    if rel in {"supports", "contradicts"} and not evidence_found:
        issues.append("relation_requires_evidence")
    if rel in {"supports", "contradicts"} and source == "none":
        issues.append("relation_requires_source")
    if evidence_found and not key_evidence:
        issues.append("evidence_found_without_key_evidence")
    if claim_quality in {"no_claim", "garbled"} and rel in {"supports", "contradicts"}:
        issues.append("bad_claim_with_strong_relation")
    return issues


def distribution(rows: list[dict[str, Any]], queue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cats: Counter[str] = Counter()
    combos: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for review in rows:
        pair_id = strval(review, "pair_id")
        q = queue.get(pair_id)
        if not q:
            continue
        cats[str(q.get("category", ""))] += 1
        combos[str(q.get("evidence_combo", ""))] += 1
        labels[str(q.get("y_current", ""))] += 1
        reason_counts.update(str(x) for x in q.get("reasons", []) or [])
        if int(q.get("y_current", 0) or 0) == 1:
            reason_counts["positive_current_label"] += 1
    return {
        "category": dict(cats.most_common()),
        "evidence_combo": dict(combos.most_common()),
        "y_current": dict(labels.most_common()),
        "reasons": dict(reason_counts.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_reviews", type=int, default=1)
    ap.add_argument("--min_coverage", type=float, default=0.90)
    ap.add_argument("--max_issue_rate", type=float, default=0.10)
    ap.add_argument("--require_all_review_ids_in_queue", action="store_true")
    args = ap.parse_args()

    queue = load_queue(args.queue)
    queue_ids = set(queue)
    reviews = read_jsonl(args.review)

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in reviews:
        pair_id = strval(obj, "pair_id")
        if pair_id:
            by_pair[pair_id].append(obj)

    duplicate_ids = sorted(pid for pid, vals in by_pair.items() if len(vals) > 1)
    reviewed_ids = set(by_pair)
    missing_ids = sorted(queue_ids - reviewed_ids)
    extra_ids = sorted(reviewed_ids - queue_ids)

    issue_counts: Counter[str] = Counter()
    review_issues: dict[str, list[str]] = {}
    valid_reviews = []
    for i, obj in enumerate(reviews):
        pid = strval(obj, "pair_id") or f"__row_{i}"
        issues = audit_review(obj, queue_ids)
        if issues:
            review_issues[pid] = issues
            issue_counts.update(issues)
        else:
            valid_reviews.append(obj)

    n_reviews = len(reviews)
    n_queue = len(queue)
    coverage = (len(reviewed_ids & queue_ids) / n_queue) if n_queue else 0.0
    issue_rate = (sum(issue_counts.values()) / max(1, n_reviews))
    status = "pass"
    fail_reasons = []
    if n_reviews < args.min_reviews:
        status = "fail"
        fail_reasons.append("too_few_reviews")
    if coverage < args.min_coverage:
        status = "fail"
        fail_reasons.append("coverage_below_min")
    if issue_rate > args.max_issue_rate:
        status = "fail"
        fail_reasons.append("issue_rate_above_max")
    if duplicate_ids:
        status = "fail"
        fail_reasons.append("duplicate_pair_ids")
    if args.require_all_review_ids_in_queue and extra_ids:
        status = "fail"
        fail_reasons.append("review_ids_outside_queue")

    field_counts = {
        field: dict(Counter(strval(r, field) for r in valid_reviews).most_common())
        for field in (
            "claim_quality",
            "claim_type",
            "relation_to_claim",
            "value_alignment",
            "evidence_source",
            "repair_action",
            "confidence",
            "likely_issue",
        )
    }
    out = {
        "status": status,
        "fail_reasons": fail_reasons,
        "queue": args.queue,
        "review": args.review,
        "n_queue": n_queue,
        "n_reviews": n_reviews,
        "n_valid_reviews": len(valid_reviews),
        "coverage": round(float(coverage), 4),
        "issue_rate": round(float(issue_rate), 4),
        "issue_counts": dict(issue_counts.most_common()),
        "duplicate_pair_ids": duplicate_ids[:50],
        "missing_pair_ids": missing_ids[:50],
        "extra_pair_ids": extra_ids[:50],
        "field_counts": field_counts,
        "reviewed_queue_distribution": distribution(valid_reviews, queue),
        "review_issues_sample": dict(list(review_issues.items())[:50]),
        "thresholds": {
            "min_reviews": args.min_reviews,
            "min_coverage": args.min_coverage,
            "max_issue_rate": args.max_issue_rate,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
