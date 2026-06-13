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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
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


def validate_queue_dataset_alignment(
    rows: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    queue: dict[str, dict[str, Any]],
    *,
    c_tol: float = 1e-6,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Ensure review/queue rows still describe the current dataset snapshot."""
    rows_by_pair: dict[str, dict[str, Any]] = {}
    duplicate_dataset_pair_ids: list[str] = []
    for i, row in enumerate(rows):
        pid = str(row.get("pair_id", "") or "")
        if not pid:
            continue
        if pid in rows_by_pair:
            duplicate_dataset_pair_ids.append(pid)
        row_with_index = dict(row)
        row_with_index["_row_index"] = i
        rows_by_pair[pid] = row_with_index

    report: dict[str, Any] = {
        "status": "pass",
        "n_reviews": len(reviews),
        "n_queue": len(queue),
        "review_ids_missing_in_dataset_count": 0,
        "review_ids_missing_in_queue_count": 0,
        "queue_row_mismatch_count": 0,
        "queue_y_mismatch_count": 0,
        "queue_c_mismatch_count": 0,
        "queue_attribute_mismatch_count": 0,
        "queue_category_mismatch_count": 0,
        "duplicate_dataset_pair_ids": duplicate_dataset_pair_ids[:sample_limit],
        "samples": {
            "missing_dataset": [],
            "missing_queue": [],
            "row": [],
            "y": [],
            "c": [],
            "attribute_id": [],
            "category": [],
        },
    }

    def add_sample(kind: str, sample: dict[str, Any]) -> None:
        if len(report["samples"][kind]) < sample_limit:
            report["samples"][kind].append(sample)

    for pid in sorted(reviews):
        row = rows_by_pair.get(pid)
        if not row:
            report["review_ids_missing_in_dataset_count"] += 1
            add_sample("missing_dataset", {"pair_id": pid})
            continue
        if not queue:
            continue
        q = queue.get(pid)
        if not q:
            report["review_ids_missing_in_queue_count"] += 1
            add_sample("missing_queue", {"pair_id": pid})
            continue

        row_index = row.get("_row_index")
        q_index = q.get("row")
        if q_index is not None and str(q_index) != "" and safe_int(q_index, -1) != int(row_index):
            report["queue_row_mismatch_count"] += 1
            add_sample("row", {"pair_id": pid, "dataset_row": row_index, "queue_row": q_index})

        row_y = safe_int(row.get("y", 0))
        q_y = safe_int(q.get("y_current", row_y))
        if row_y != q_y:
            report["queue_y_mismatch_count"] += 1
            add_sample("y", {"pair_id": pid, "dataset": row_y, "queue": q_y})

        row_c = safe_float(row.get("c", 0.0))
        q_c = safe_float(q.get("c_current", row_c))
        if abs(row_c - q_c) > c_tol:
            report["queue_c_mismatch_count"] += 1
            add_sample("c", {"pair_id": pid, "dataset": round(row_c, 6), "queue": round(q_c, 6)})

        for field, count_key, sample_key in (
            ("attribute_id", "queue_attribute_mismatch_count", "attribute_id"),
            ("category", "queue_category_mismatch_count", "category"),
        ):
            row_v = str(row.get(field, "") or "")
            q_v = str(q.get(field, "") or "")
            if q_v and row_v != q_v:
                report[count_key] += 1
                add_sample(sample_key, {"pair_id": pid, "dataset": row_v, "queue": q_v})

    fatal_keys = (
        "review_ids_missing_in_dataset_count",
        "review_ids_missing_in_queue_count",
        "queue_row_mismatch_count",
        "queue_y_mismatch_count",
        "queue_c_mismatch_count",
        "queue_attribute_mismatch_count",
    )
    if duplicate_dataset_pair_ids or any(report[k] for k in fatal_keys):
        report["status"] = "fail"
    return report


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

    def finish(action: str, drop: bool, y_after: int, c_after: float,
               reasons: list[str]) -> dict[str, Any]:
        if mode == "audit":
            return {
                "action": "audit_only",
                "would_action": action,
                "drop": False,
                "would_drop": bool(drop),
                "y_before": y_before,
                "y_after": y_before,
                "would_y_after": int(y_after),
                "c_before": round(c_before, 4),
                "c_after": round(c_before, 4),
                "would_c_after": round(c_after, 4),
                "reasons": reasons + ["audit_mode_no_mutation"],
            }
        return {
            "action": action,
            "would_action": action,
            "drop": bool(drop),
            "would_drop": bool(drop),
            "y_before": y_before,
            "y_after": int(y_after),
            "would_y_after": int(y_after),
            "c_before": round(c_before, 4),
            "c_after": round(c_after, 4),
            "would_c_after": round(c_after, 4),
            "reasons": reasons,
        }

    if not reliable:
        reasons.append("review_low_confidence")
        if rel in UNCERTAIN_REL and c_before <= max_uncertain_drop_c:
            c_after = min(c_before, 0.05)
            action = "downweight_uncertain_low_conf"
            reasons.append("uncertain_relation_low_weight")
        return finish(action, drop, y_after, c_after, reasons)

    if q in BAD_CLAIM or str(review.get("repair_action", "")) == "drop_bad_claim":
        reasons.append("bad_claim_span")
        drop = True
        action = "drop_bad_claim"
        return finish(action, drop, y_after, c_after, reasons)

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
            action = "strengthen_clean_supported"
        return finish(action, drop, y_after, c_after, reasons)

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
            action = "strengthen_risk_contradicted"
        return finish(action, drop, y_after, c_after, reasons)

    if rel in UNCERTAIN_REL:
        reasons.append("insufficient_or_not_verifiable")
        if c_before <= max_uncertain_drop_c:
            drop = True
            action = "drop_uncertain_weak_row"
        else:
            c_after = min(c_before, max(0.05, c_before * 0.5))
            action = "downweight_uncertain_row"
        return finish(action, drop, y_after, c_after, reasons)

    reasons.append("unhandled_review_state")
    return finish(action, drop, y_after, c_after, reasons)


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
    if decision.get("action") == "audit_only":
        out["confidence"] = row.get("confidence", "")
    else:
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


def summarize(rows_out, drops, touched, mode, args, queue_alignment) -> dict[str, Any]:
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
        "queue_alignment": queue_alignment,
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
    ap.add_argument("--allow_stale_queue", action="store_true",
                    help="Apply reviews even when queue metadata no longer matches the dataset.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    reviews = load_reviews(args.review)
    queue = load_queue(args.queue) if args.queue else {}
    args._input_n = len(rows)
    args._review_n = len(reviews)
    queue_alignment = validate_queue_dataset_alignment(rows, reviews, queue)
    if queue_alignment["status"] != "pass" and not args.allow_stale_queue:
        report = {
            "status": "fail",
            "fail_reasons": ["review_queue_dataset_alignment_mismatch"],
            "mode": args.mode,
            "dataset": args.dataset,
            "review": args.review,
            "queue": args.queue,
            "out": args.out,
            "input_rows": len(rows),
            "review_rows_loaded": len(reviews),
            "queue_alignment": queue_alignment,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

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
    report = summarize(out_rows, drops, touched_rows, args.mode, args, queue_alignment)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
