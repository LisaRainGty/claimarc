"""Build a balanced pilot queue from mechanism-repair candidates.

The full mechanism queue is priority-sorted and its first rows are dominated by
high-confidence false positives.  That is useful for one failure mode, but a
first LLM/VLM pilot should cover multiple categories, evidence-source combos,
labels, and error reasons so the downstream repair rules can be stress-tested.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REASON_TARGETS = {
    "claimarc_high_conf_false_positive": 22,
    "bge_correct_claimarc_wrong": 16,
    "both_wrong_high_conf_negative": 14,
    "price_or_coupon": 8,
    "no_product_evidence": 6,
    "positive_current_label": 14,
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_targets(text: str) -> dict[str, int]:
    if not text:
        return dict(DEFAULT_REASON_TARGETS)
    out: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = int(value)
    return out


def priority_key(row: dict[str, Any]) -> tuple:
    return (
        -int(row.get("priority_score", 0) or 0),
        -float(row.get("claimarc_p", 0.0) or 0.0),
        str(row.get("pair_id", "")),
    )


def bucket_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("category", "")),
        str(row.get("evidence_combo", "")),
        int(row.get("y_current", 0) or 0),
    )


def row_reasons(row: dict[str, Any]) -> set[str]:
    reasons = set(str(x) for x in row.get("reasons", []) or [])
    if int(row.get("y_current", 0) or 0) == 1:
        reasons.add("positive_current_label")
    return reasons


def take_for_reason(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    category_counts: Counter[str],
    reason: str,
    target: int,
    max_per_category: int,
    max_per_category_total: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    reason_category_counts: Counter[str] = Counter()
    for row in rows:
        if reason not in row_reasons(row):
            continue
        buckets[bucket_key(row)].append(row)
    for vals in buckets.values():
        vals.sort(key=priority_key)

    picked: list[dict[str, Any]] = []
    while len(picked) < target:
        made_progress = False
        for key in sorted(buckets, key=lambda x: (x[0], x[1], x[2])):
            vals = buckets[key]
            while vals and str(vals[0].get("pair_id", "")) in selected_ids:
                vals.pop(0)
            if not vals:
                continue
            cand = vals[0]
            cat = str(cand.get("category", ""))
            if reason_category_counts[cat] >= max_per_category:
                vals.pop(0)
                continue
            if category_counts[cat] >= max_per_category_total:
                vals.pop(0)
                continue
            vals.pop(0)
            selected_ids.add(str(cand.get("pair_id", "")))
            reason_category_counts[cat] += 1
            category_counts[cat] += 1
            picked.append(cand)
            made_progress = True
            if len(picked) >= target:
                break
        if not made_progress:
            break
    return picked


def take_min_per_category(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    category_counts: Counter[str],
    min_per_category: int,
    max_per_category_total: int,
) -> list[dict[str, Any]]:
    if min_per_category <= 0:
        return []
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category", ""))].append(row)
    for vals in by_cat.values():
        vals.sort(key=priority_key)

    picked: list[dict[str, Any]] = []
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        while vals and category_counts[cat] < min_per_category:
            cand = vals.pop(0)
            pid = str(cand.get("pair_id", ""))
            if pid in selected_ids:
                continue
            if category_counts[cat] >= max_per_category_total:
                break
            selected_ids.add(pid)
            category_counts[cat] += 1
            picked.append(cand)
    return picked


def fill_remaining(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    category_counts: Counter[str],
    current: list[dict[str, Any]],
    total: int,
    max_per_category: int,
) -> list[dict[str, Any]]:
    for row in sorted(rows, key=priority_key):
        if len(current) >= total:
            break
        pid = str(row.get("pair_id", ""))
        if pid in selected_ids:
            continue
        cat = str(row.get("category", ""))
        if category_counts[cat] >= max_per_category:
            continue
        selected_ids.add(pid)
        category_counts[cat] += 1
        current.append(row)
    return current


def summarize(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row_reasons(row))
    return {
        "queue": args.queue,
        "out": args.out,
        "n": len(rows),
        "target_total": args.total,
        "reason_targets": args._targets,
        "max_per_category_per_reason": args.max_per_category_per_reason,
        "max_per_category_total": args.max_per_category_total,
        "category": dict(Counter(str(r.get("category", "")) for r in rows).most_common()),
        "evidence_combo": dict(Counter(str(r.get("evidence_combo", "")) for r in rows).most_common()),
        "y_current": dict(Counter(str(r.get("y_current", "")) for r in rows).most_common()),
        "reasons": dict(reason_counts.most_common()),
        "priority_score": dict(Counter(str(r.get("priority_score", "")) for r in rows).most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--total", type=int, default=80)
    ap.add_argument("--targets", default="",
                    help="Comma-separated reason=count overrides.")
    ap.add_argument("--max_per_category_per_reason", type=int, default=5)
    ap.add_argument("--max_per_category_total", type=int, default=16)
    ap.add_argument("--min_per_category", type=int, default=3)
    args = ap.parse_args()

    rows = read_jsonl(args.queue)
    rows.sort(key=priority_key)
    targets = parse_targets(args.targets)
    args._targets = targets

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    selected.extend(take_min_per_category(
        rows,
        selected_ids,
        category_counts,
        args.min_per_category,
        args.max_per_category_total,
    ))
    for reason, target in targets.items():
        selected.extend(take_for_reason(
            rows,
            selected_ids,
            category_counts,
            reason,
            target,
            args.max_per_category_per_reason,
            args.max_per_category_total,
        ))
    selected = fill_remaining(
        rows,
        selected_ids,
        category_counts,
        selected,
        args.total,
        args.max_per_category_total,
    )
    selected = sorted(selected[:args.total], key=priority_key)
    write_jsonl(args.out, selected)
    report = summarize(selected, args)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
