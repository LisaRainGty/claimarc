"""Build a high-precision merged atomic auxiliary training pool.

The v2 raw/candidate reconstruction is useful as a recall-oriented pool, but it
contains many near-duplicates of the hard-clean set and a long tail of weak
visual/color negatives.  This builder keeps the existing hard-clean pool as the
anchor and adds candidate rows only when they pass source-aware quality gates.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


PUNCT_RE = re.compile(r"[\s,，。.!！?？;；:：、\"'“”‘’（）()\[\]【】]+")


def claim_text(row: dict[str, Any]) -> str:
    claim = row.get("claim") or {}
    segs = claim.get("segments") or []
    text = " ".join(str(s.get("text") or "").strip() for s in segs if s.get("text"))
    return text or str(claim.get("passage") or "")


def norm_text(text: str) -> str:
    return PUNCT_RE.sub("", text or "").lower()


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("product_id") or ""),
        str(row.get("attribute_id") or ""),
        norm_text(claim_text(row)),
    )


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("product_id") or ""), str(row.get("attribute_id") or "")


def audit_int(row: dict[str, Any], name: str) -> int:
    val = (row.get("label_audit") or {}).get(name, 0)
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def row_score(row: dict[str, Any]) -> float:
    y = int(row.get("y", 0) or 0)
    cov = int(row.get("coverage", 0) or 0)
    c = float(row.get("c", 0.05) or 0.05)
    aligned = audit_int(row, "n_refute_aligned" if y == 1 else "n_support_aligned")
    score = c + 0.20 * cov + 0.04 * min(aligned, 5)
    family = str(row.get("source_family") or "")
    if family == "objective_name_only":
        score -= 0.50
    if y == 0 and aligned == 0:
        score -= 0.06
    return score


def candidate_decision(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    y = int(row.get("y", 0) or 0)
    cov = int(row.get("coverage", 0) or 0)
    c = float(row.get("c", 0.05) or 0.05)
    family = str(row.get("source_family") or "")
    conf = str(row.get("confidence") or "")

    if args.exclude_objective_name_only and family == "objective_name_only":
        return False, "objective_name_only"

    if y == 1:
        if cov < args.positive_min_coverage:
            return False, "pos_low_coverage"
        if c < args.positive_min_c:
            return False, "pos_low_c"
        if audit_int(row, "n_refute_aligned") < args.positive_min_refute:
            return False, "pos_no_refute_alignment"
        return True, "keep_pos"

    if cov < args.negative_min_coverage:
        return False, "neg_low_coverage"
    if args.negative_require_support and audit_int(row, "n_support_aligned") < 1:
        return False, "neg_no_support_alignment"
    if args.drop_low_visual_neg and family == "visual_or_boolean" and conf == "low":
        return False, "neg_low_visual"
    return True, "keep_neg"


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    base_pairs: set[tuple[str, str]] = set()
    counters = Counter()

    for path in args.base:
        for row in read_jsonl(path):
            out = dict(row)
            out["_merge_view"] = {"name": args.view_name, "source": "base"}
            rows.append(out)
            seen_keys.add(row_key(row))
            base_pairs.add(pair_key(row))
            counters["base_rows"] += 1

    candidates: list[dict[str, Any]] = []
    for path in args.candidate:
        for row in read_jsonl(path):
            counters["candidate_rows"] += 1
            key = row_key(row)
            pair = pair_key(row)
            if args.skip_exact_duplicates and key in seen_keys:
                counters["drop_exact_duplicate"] += 1
                continue
            if args.new_pairs_only and pair in base_pairs:
                counters["drop_seen_pair"] += 1
                continue
            keep, reason = candidate_decision(row, args)
            if not keep:
                counters[f"drop_{reason}"] += 1
                continue
            out = dict(row)
            out["_merge_view"] = {
                "name": args.view_name,
                "source": "candidate",
                "candidate_reason": reason,
                "candidate_score": round(row_score(row), 4),
            }
            candidates.append(out)

    candidates.sort(key=lambda r: (pair_key(r), -row_score(r), str(r.get("atomic_id") or "")))
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in candidates:
        pair = pair_key(row)
        key = row_key(row)
        if args.skip_exact_duplicates and key in seen_keys:
            counters["drop_duplicate_after_sort"] += 1
            continue
        if args.candidate_max_per_pair > 0 and pair_counts[pair] >= args.candidate_max_per_pair:
            counters["drop_pair_cap"] += 1
            continue
        rows.append(row)
        seen_keys.add(key)
        pair_counts[pair] += 1
        counters["candidate_added"] += 1

    source = Counter((r.get("_merge_view") or {}).get("source", "unknown") for r in rows)
    labels = Counter(str(int(r.get("y", 0) or 0)) for r in rows)
    family = Counter(str(r.get("source_family") or "") for r in rows)
    report = {
        "view_name": args.view_name,
        "base": args.base,
        "candidate": args.candidate,
        "out": args.out,
        "n": len(rows),
        "pairs": len({pair_key(r) for r in rows}),
        "products": len({str(r.get("product_id") or "") for r in rows}),
        "labels": dict(labels),
        "source": dict(source),
        "source_family": dict(family),
        "counters": dict(counters),
        "args": vars(args),
    }
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="append", required=True)
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--view_name", default="atomic_aux_merge_v2")
    ap.add_argument("--positive_min_c", type=float, default=0.10)
    ap.add_argument("--positive_min_coverage", type=int, default=1)
    ap.add_argument("--positive_min_refute", type=int, default=1)
    ap.add_argument("--negative_min_coverage", type=int, default=2)
    ap.add_argument("--negative_require_support", action="store_true")
    ap.add_argument("--exclude_objective_name_only", action="store_true")
    ap.add_argument("--drop_low_visual_neg", action="store_true")
    ap.add_argument("--skip_exact_duplicates", action="store_true")
    ap.add_argument("--new_pairs_only", action="store_true")
    ap.add_argument("--candidate_max_per_pair", type=int, default=3)
    args = ap.parse_args()

    rows, report = build(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
