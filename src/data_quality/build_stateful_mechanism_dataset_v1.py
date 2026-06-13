"""Build a no-drop mechanism-state dataset from pair-level review results.

This builder is deliberately different from the earlier repair applier.  It
does not relabel or remove hard examples to make the benchmark easier.  Instead
it preserves the consumer-perception label and attaches explicit claim/evidence
utility states that downstream training can use as reliability weights,
contrastive masks, or robustness slices.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BAD_CLAIM = {"no_claim", "garbled"}
UNCERTAIN_REL = {"insufficient", "not_verifiable"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x: Any, default: float = 0.05) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_reviews(path: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        if row.get("__error__"):
            continue
        pair_id = str(row.get("pair_id", "") or "")
        if pair_id:
            out[pair_id] = row
    return out


def load_queue(path: str | Path) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {
        str(row.get("pair_id")): row
        for row in read_jsonl(path)
        if row.get("pair_id")
    }


def confidence_from_c(c: float) -> str:
    if c < 0.20:
        return "absent"
    if c < 0.40:
        return "low"
    if c < 0.70:
        return "medium"
    return "high"


def classify_state(row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    y = int(row.get("y", 0) or 0)
    c0 = safe_float(row.get("c", 0.05), 0.05)
    if not review:
        return {
            "measurement_state": "unreviewed",
            "consumer_evidence_state": "original_consumer_label",
            "objective_evidence_state": "not_reviewed",
            "contrastive_utility": "use_original",
            "ce_weight_multiplier": 1.0,
            "cl_anchor": True,
            "cl_negative": True,
            "needs_reextraction": False,
            "needs_evidence_rerun": False,
            "state_reason": ["no_mechanism_review"],
            "c_after": c0,
        }

    claim_quality = str(review.get("claim_quality", "")).lower()
    relation = str(review.get("relation_to_claim", "")).lower()
    value_alignment = str(review.get("value_alignment", "")).lower()
    repair_action = str(review.get("repair_action", "")).lower()
    confidence = str(review.get("confidence", "")).lower()
    likely_issue = str(review.get("likely_issue", "")).lower()
    reasons: list[str] = []
    low_review_conf = confidence == "low"

    if claim_quality in BAD_CLAIM or repair_action == "drop_bad_claim":
        state = "invalid_or_weak_claim_span"
        consumer_state = "label_unreliable_until_claim_reextraction"
        objective_state = relation or "unknown"
        utility = "mask_from_contrastive"
        mult = 0.25
        cl_anchor = False
        cl_negative = False
        needs_reextraction = True
        needs_evidence_rerun = False
        reasons.append("claim_span_not_sufficiently_grounded")
    elif relation in UNCERTAIN_REL:
        state = "valid_claim_but_product_evidence_insufficient"
        consumer_state = "consumer_label_retained_low_reliability"
        objective_state = relation
        utility = "ignore_for_hard_contrastive"
        mult = 0.45 if c0 > 0.15 else 0.35
        cl_anchor = False
        cl_negative = False
        needs_reextraction = False
        needs_evidence_rerun = True
        reasons.append("product_evidence_not_sufficient_for_objective_check")
    elif relation == "supports":
        state = "claim_supported_by_product_evidence"
        objective_state = "supports"
        if y == 1:
            consumer_state = "perception_gap_despite_supported_claim"
            utility = "perception_gap_positive"
            mult = 0.80
            reasons.append("do_not_relabel_consumer_risk_as_clean")
        else:
            consumer_state = "low_risk_supported_claim"
            utility = "reliable_clean_support"
            mult = 1.10
        cl_anchor = True
        cl_negative = True
        needs_reextraction = False
        needs_evidence_rerun = False
    elif relation == "contradicts":
        state = "claim_contradicted_by_product_evidence"
        objective_state = "contradicts"
        if y == 1:
            consumer_state = "objective_refutation_with_consumer_risk"
            utility = "reliable_risk_contradiction"
            mult = 1.15
        else:
            consumer_state = "objective_issue_without_aligned_negative_review"
            utility = "objective_stress_without_consumer_label"
            mult = 0.75
            reasons.append("do_not_relabel_without_consumer_negative_alignment")
        cl_anchor = True
        cl_negative = True
        needs_reextraction = False
        needs_evidence_rerun = False
    else:
        state = "reviewed_unhandled_state"
        consumer_state = "consumer_label_retained"
        objective_state = relation or "unknown"
        utility = "use_with_caution"
        mult = 0.60
        cl_anchor = False
        cl_negative = False
        needs_reextraction = False
        needs_evidence_rerun = True
        reasons.append("unhandled_review_relation")

    if low_review_conf:
        mult *= 0.75
        reasons.append("low_review_confidence")
    if likely_issue in {"missing_evidence", "generic_evidence"}:
        needs_evidence_rerun = True
        reasons.append(likely_issue)
    if value_alignment in {"contradiction", "compatible", "exact_match"}:
        reasons.append(f"value_alignment:{value_alignment}")

    c_after = max(0.03, min(1.0, c0 * mult))
    return {
        "measurement_state": state,
        "consumer_evidence_state": consumer_state,
        "objective_evidence_state": objective_state,
        "contrastive_utility": utility,
        "ce_weight_multiplier": round(mult, 4),
        "cl_anchor": bool(cl_anchor),
        "cl_negative": bool(cl_negative),
        "needs_reextraction": bool(needs_reextraction),
        "needs_evidence_rerun": bool(needs_evidence_rerun),
        "state_reason": reasons,
        "c_after": round(c_after, 4),
    }


def attach_review(row: dict[str, Any], review: dict[str, Any] | None,
                  queue_row: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    c0 = safe_float(out.get("c", 0.05), 0.05)
    state = classify_state(row, review)
    out["_stateful_original_y"] = int(out.get("y", 0) or 0)
    out["_stateful_original_c"] = round(c0, 4)
    out["_mechanism_state"] = {k: v for k, v in state.items() if k != "c_after"}
    out["c"] = state["c_after"]
    out["confidence"] = confidence_from_c(out["c"])
    if review:
        out["_mechanism_repair_review"] = {
            "claim_quality": review.get("claim_quality"),
            "claim_type": review.get("claim_type"),
            "key_claim": review.get("key_claim"),
            "evidence_found": review.get("evidence_found"),
            "evidence_source": review.get("evidence_source"),
            "key_evidence": review.get("key_evidence"),
            "relation_to_claim": review.get("relation_to_claim"),
            "value_alignment": review.get("value_alignment"),
            "likely_issue": review.get("likely_issue"),
            "repair_action": review.get("repair_action"),
            "confidence": review.get("confidence"),
            "rationale": review.get("rationale"),
            "model": review.get("model"),
        }
    if queue_row:
        out["_mechanism_repair_queue"] = {
            "priority_score": queue_row.get("priority_score"),
            "reasons": queue_row.get("reasons", []),
            "claimarc_p": queue_row.get("claimarc_p"),
            "baseline_p": queue_row.get("baseline_p"),
            "evidence_combo": queue_row.get("evidence_combo"),
            "source_count": queue_row.get("source_count"),
        }
    return out


def summarize(rows: list[dict[str, Any]], out_rows: list[dict[str, Any]], reviews: dict[str, Any],
              args: argparse.Namespace) -> dict[str, Any]:
    state_counts = Counter()
    utility_counts = Counter()
    y_state = Counter()
    category_state = Counter()
    needs = Counter()
    c_delta = []
    for row in out_rows:
        st = row.get("_mechanism_state", {})
        state = str(st.get("measurement_state", ""))
        util = str(st.get("contrastive_utility", ""))
        state_counts[state] += 1
        utility_counts[util] += 1
        y_state[f"y{int(row.get('y', 0) or 0)}::{state}"] += 1
        category_state[f"{row.get('category', '')}::{state}"] += 1
        if st.get("needs_reextraction"):
            needs["claim_reextraction"] += 1
        if st.get("needs_evidence_rerun"):
            needs["evidence_rerun"] += 1
        c_delta.append(safe_float(row.get("c", 0.05)) - safe_float(row.get("_stateful_original_c", row.get("c", 0.05))))
    return {
        "dataset": args.dataset,
        "review": args.review,
        "queue": args.queue,
        "out": args.out,
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "dropped_rows": 0,
        "review_rows_loaded": len(reviews),
        "review_rows_applied": sum(1 for row in out_rows if str(row.get("pair_id", "")) in reviews),
        "state_counts": dict(state_counts.most_common()),
        "contrastive_utility_counts": dict(utility_counts.most_common()),
        "label_by_state_counts": dict(y_state.most_common()),
        "needs_counts": dict(needs.most_common()),
        "top_category_state_counts": dict(category_state.most_common(40)),
        "c_delta_mean": round(float(sum(c_delta) / len(c_delta)), 6) if c_delta else 0.0,
        "c_delta_min": round(float(min(c_delta)), 6) if c_delta else 0.0,
        "c_delta_max": round(float(max(c_delta)), 6) if c_delta else 0.0,
        "policy_note": (
            "No rows are deleted or relabeled. Consumer-perception labels are "
            "preserved; claim/evidence reviews become reliability and utility states."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--queue", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    reviews = load_reviews(args.review)
    queue = load_queue(args.queue)
    out_rows = []
    for row in rows:
        pair_id = str(row.get("pair_id", "") or "")
        out_rows.append(attach_review(row, reviews.get(pair_id), queue.get(pair_id)))
    write_jsonl(args.out, out_rows)
    report = summarize(rows, out_rows, reviews, args)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
