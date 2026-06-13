"""Apply mechanism-repair LLM/VLM reviews to a dataset candidate.

The script converts review JSONL into an auditable candidate dataset.  It does
not trust the LLM as a direct label oracle: reviews only describe claim quality
and product-evidence relation.  Deterministic rules decide whether to drop,
relabel, downweight, or simply flag each reviewed row.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORT_VALUES = {"exact_match", "compatible"}
CONTRADICT_VALUES = {"contradiction"}
RELIABLE_CONF = {"high", "medium"}
BAD_CLAIM = {"no_claim", "garbled"}
UNCERTAIN_REL = {"insufficient", "not_verifiable"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_reviews(path: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    for obj in read_jsonl(path):
        if not isinstance(obj, dict) or obj.get("__error__"):
            continue
        pair_id = str(obj.get("pair_id", "") or "")
        if pair_id:
            out[pair_id] = obj
    return out


def load_queue(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("pair_id")): row
        for row in read_jsonl(path)
        if row.get("pair_id")
    }


def is_reliable(review: dict[str, Any]) -> bool:
    return str(review.get("confidence", "")).lower() in RELIABLE_CONF


def relation(review: dict[str, Any]) -> str:
    return str(review.get("relation_to_claim", "")).lower()


def value_alignment(review: dict[str, Any]) -> str:
    return str(review.get("value_alignment", "")).lower()


def claim_quality(review: dict[str, Any]) -> str:
    return str(review.get("claim_quality", "")).lower()


def decide_action(
    row: dict[str, Any],
    review: dict[str, Any],
    *,
    mode: str,
    max_safe_relabel_c: float,
    max_uncertain_drop_c: float,
    min_repaired_c: float,
) -> dict[str, Any]:
    y_before = int(row.get("y", 0) or 0)
    c_before = safe_float(row.get("c", 0.05), 0.05)
    rel = relation(review)
    val = value_alignment(review)
    q = claim_quality(review)
    reliable = is_reliable(review)
    action = "keep_flag_only"
    y_after = y_before
    c_after = c_before
    drop = False
    reasons: list[str] = []

    if not reliable:
        reasons.append("review_low_confidence")
        if rel in UNCERTAIN_REL and c_before <= max_uncertain_drop_c and mode != "audit":
            c_after = min(c_before, 0.05)
            action = "downweight_uncertain_low_conf"
            reasons.append("uncertain_relation_low_weight")
        return {
            "action": action,
            "drop": drop,
            "y_before": y_before,
            "y_after": y_after,
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "reasons": reasons,
        }

    if q in BAD_CLAIM or str(review.get("repair_action", "")) == "drop_bad_claim":
        reasons.append("bad_claim_span")
        if mode != "audit":
            drop = True
            action = "drop_bad_claim"
        return {
            "action": action,
            "drop": drop,
            "y_before": y_before,
            "y_after": y_after,
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "reasons": reasons,
        }

    if rel == "supports" and val in SUPPORT_VALUES:
        reasons.append("product_evidence_supports_claim")
        if y_before == 1:
            if mode == "candidate" or c_before <= max_safe_relabel_c:
                y_after = 0
                c_after = max(min_repaired_c, min(1.0, c_before))
                action = "relabel_risk_to_clean_supported"
                reasons.append("consumer_label_not_strong_enough_to_override_support")
            else:
                action = "flag_supported_positive_for_consumer_review"
                reasons.append("strong_current_positive_requires_consumer_review")
        else:
            c_after = max(c_before, min_repaired_c)
            action = "strengthen_clean_supported" if mode != "audit" else "keep_flag_only"
        return {
            "action": action,
            "drop": drop,
            "y_before": y_before,
            "y_after": y_after,
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "reasons": reasons,
        }

    if rel == "contradicts" and val in CONTRADICT_VALUES:
        reasons.append("product_evidence_contradicts_claim")
        if y_before == 0:
            if mode == "candidate" or c_before <= max_safe_relabel_c:
                y_after = 1
                c_after = max(min_repaired_c, min(1.0, c_before))
                action = "relabel_clean_to_risk_contradicted"
                reasons.append("objective_contradiction_recovers_risk")
            else:
                action = "flag_contradicted_clean_for_consumer_review"
                reasons.append("strong_current_clean_requires_consumer_review")
        else:
            c_after = max(c_before, min_repaired_c)
            action = "strengthen_risk_contradicted" if mode != "audit" else "keep_flag_only"
        return {
            "action": action,
            "drop": drop,
            "y_before": y_before,
            "y_after": y_after,
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "reasons": reasons,
        }

    if rel in UNCERTAIN_REL:
        reasons.append("insufficient_or_not_verifiable")
        if mode != "audit" and c_before <= max_uncertain_drop_c:
            drop = True
            action = "drop_uncertain_weak_row"
        elif mode != "audit":
            c_after = min(c_before, max(0.05, c_before * 0.5))
            action = "downweight_uncertain_row"
        return {
            "action": action,
            "drop": drop,
            "y_before": y_before,
            "y_after": y_after,
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "reasons": reasons,
        }

    reasons.append("unhandled_review_state")
    return {
        "action": action,
        "drop": drop,
        "y_before": y_before,
        "y_after": y_after,
        "c_before": round(c_before, 4),
        "c_after": round(c_after, 4),
        "reasons": reasons,
    }


def attach_review_metadata(
    row: dict[str, Any],
    review: dict[str, Any],
    queue_row: dict[str, Any] | None,
    decision: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
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
    out["_mechanism_repair_decision"] = decision
    out["y"] = int(decision["y_after"])
    out["c"] = float(decision["c_after"])
    out["confidence"] = confidence_from_c(float(decision["c_after"]))
    return out


def confidence_from_c(c: float) -> str:
    if c < 0.20:
        return "absent"
    if c < 0.40:
        return "low"
    if c < 0.70:
        return "medium"
    return "high"


def summarize(rows_out, drops, touched, mode, args) -> dict[str, Any]:
    action_counts = Counter()
    rel_counts = Counter()
    label_changes = Counter()
    category_actions = Counter()
    for row in touched:
        dec = row.get("_mechanism_repair_decision", {})
        action = str(dec.get("action", ""))
        action_counts[action] += 1
        review = row.get("_mechanism_repair_review", {})
        rel_counts[str(review.get("relation_to_claim", ""))] += 1
        label_changes[f"{dec.get('y_before')}->{dec.get('y_after')}"] += 1
        category_actions[f"{row.get('category', '')}::{action}"] += 1
    y_counts = Counter(str(row.get("y", "")) for row in rows_out)
    conf_counts = Counter(str(row.get("confidence", "")) for row in rows_out)
    return {
        "mode": mode,
        "dataset": args.dataset,
        "review": args.review,
        "queue": args.queue,
        "out": args.out,
        "input_rows": args._input_n,
        "output_rows": len(rows_out),
        "review_rows_loaded": args._review_n,
        "touched_rows": len(touched),
        "dropped_rows": len(drops),
        "max_safe_relabel_c": args.max_safe_relabel_c,
        "max_uncertain_drop_c": args.max_uncertain_drop_c,
        "min_repaired_c": args.min_repaired_c,
        "action_counts": dict(action_counts.most_common()),
        "relation_counts": dict(rel_counts.most_common()),
        "label_changes": dict(label_changes.most_common()),
        "output_y_counts": dict(y_counts.most_common()),
        "output_confidence_counts": dict(conf_counts.most_common()),
        "top_category_actions": dict(category_actions.most_common(30)),
        "drop_examples": drops[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--queue", default="")
    ap.add_argument("--mode", choices=["audit", "conservative", "candidate"],
                    default="conservative")
    ap.add_argument("--max_safe_relabel_c", type=float, default=0.35)
    ap.add_argument("--max_uncertain_drop_c", type=float, default=0.15)
    ap.add_argument("--min_repaired_c", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    reviews = load_reviews(args.review)
    queue = load_queue(args.queue) if args.queue else {}
    args._input_n = len(rows)
    args._review_n = len(reviews)

    out_rows: list[dict[str, Any]] = []
    touched_rows: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("pair_id", "") or "")
        review = reviews.get(pair_id)
        if not review:
            out_rows.append(row)
            continue
        decision = decide_action(
            row,
            review,
            mode=args.mode,
            max_safe_relabel_c=args.max_safe_relabel_c,
            max_uncertain_drop_c=args.max_uncertain_drop_c,
            min_repaired_c=args.min_repaired_c,
        )
        repaired = attach_review_metadata(row, review, queue.get(pair_id), decision)
        touched_rows.append(repaired)
        if decision["drop"]:
            drops.append({
                "pair_id": pair_id,
                "category": row.get("category"),
                "attribute_id": row.get("attribute_id"),
                "attribute_name": row.get("attribute_name"),
                "action": decision["action"],
                "reasons": decision["reasons"],
                "review_relation": relation(review),
                "review_confidence": review.get("confidence"),
            })
            continue
        out_rows.append(repaired)

    write_jsonl(args.out, out_rows)
    report = summarize(out_rows, drops, touched_rows, args.mode, args)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
