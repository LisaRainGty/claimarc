"""Build a deterministic high-quality CLAIMARC training dataset candidate.

The goal is not to create an easier benchmark by hiding hard examples. Instead
we separate clean/silver/weak supervision so the existing RACL model can train
on a less contradictory signal while preserving audit metadata for ablations.

Inputs:
- full pair-level dataset: broad negative pool, usually data/final/dataset.jsonl
- verified args dataset: balanced/argument-enriched pool, usually
  data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl

Output records keep the original model schema. Additional fields:
- _quality_bucket: deterministic quality tier
- _quality_weight: multiplier folded into c
- _source_dataset: full | verified_args | merged
- _base_pair_id: pair_id before any future atomic expansion
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import has_claim, quality_bucket, read_jsonl, source_count


BUCKET_KEEP = {
    "pos_core",
    "pos_silver",
    "pos_weak",
    "neg_core",
    "neg_silver_sourceful",
    "neg_silver_comment_only",
}

DEFAULT_BUCKET_WEIGHT = {
    "pos_core": 1.20,
    "pos_silver": 0.90,
    "pos_weak": 0.45,
    "neg_core": 1.10,
    "neg_silver_sourceful": 0.65,
    "neg_silver_comment_only": 0.35,
    "neg_context_sourceful": 0.20,
}


def key(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def merge_records(full_rows: list[dict[str, Any]], verified_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {key(r): dict(r) for r in full_rows}
    verified = {key(r): dict(r) for r in verified_rows}
    rows = []
    protected_fields = {"pair_id", "product_id", "attribute_id", "room_id", "split"}
    for pid, rec in full.items():
        out = dict(rec)
        out["_source_dataset"] = "full"
        if pid in verified:
            # Prefer cleaned labels/arguments/evidence ordering while retaining
            # broad-pool rows for negatives not present in verified data. Split
            # identity belongs to the full pool; letting a merged auxiliary file
            # overwrite it causes room-level leakage in downstream variants.
            aux = dict(verified[pid])
            if "split" in aux:
                out["_verified_split"] = aux.get("split")
            for field in protected_fields:
                aux.pop(field, None)
            out.update(aux)
            out["_source_dataset"] = "merged"
        rows.append(out)
    for pid, rec in verified.items():
        if pid not in full:
            out = dict(rec)
            out["_source_dataset"] = "verified_args"
            rows.append(out)
    return rows


def add_quality_fields(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    b = quality_bucket(out)
    out["_quality_bucket"] = b
    out["_base_pair_id"] = key(out)
    weight = DEFAULT_BUCKET_WEIGHT.get(b, 0.0)
    out["_quality_weight"] = round(weight, 4)
    # Keep original c in _c_original and expose the quality-adjusted value to
    # existing training code. A floor preserves hard examples without letting
    # very weak labels dominate.
    c0 = float(out.get("c", 0.05) or 0.05)
    out["_c_original"] = round(c0, 4)
    if weight > 0:
        out["c"] = round(max(0.03, min(1.0, c0 * weight)), 4)
    return out


def choose_rows(
    rows: list[dict[str, Any]],
    *,
    neg_ratio: float,
    seed: int,
    keep_weak_pos: bool,
    include_context_neg: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated = [add_quality_fields(r) for r in rows]
    keep_buckets = set(BUCKET_KEEP)
    if include_context_neg:
        keep_buckets.add("neg_context_sourceful")
    keep = [r for r in annotated if r["_quality_bucket"] in keep_buckets and has_claim(r)]
    if not keep_weak_pos:
        keep = [r for r in keep if r["_quality_bucket"] != "pos_weak"]

    pos = [r for r in keep if int(r.get("y", 0)) == 1]
    neg = [r for r in keep if int(r.get("y", 0)) == 0]
    rng = random.Random(seed)

    # Prefer clean negatives, then sourceful silver, then comment-only silver.
    neg_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in neg:
        neg_by_bucket[r["_quality_bucket"]].append(r)
    for vals in neg_by_bucket.values():
        rng.shuffle(vals)

    target_neg = int(round(max(1, len(pos)) * neg_ratio))
    selected_neg: list[dict[str, Any]] = []
    neg_order = ["neg_core", "neg_silver_sourceful", "neg_silver_comment_only"]
    if include_context_neg:
        neg_order.append("neg_context_sourceful")
    for b in neg_order:
        remaining = target_neg - len(selected_neg)
        if remaining <= 0:
            break
        selected_neg.extend(neg_by_bucket.get(b, [])[:remaining])

    selected = pos + selected_neg
    selected.sort(key=lambda r: (str(r.get("room_id", "")), key(r)))

    report = {
        "input_n": len(rows),
        "candidate_n": len(keep),
        "selected_n": len(selected),
        "selected_pos": len(pos),
        "selected_neg": len(selected_neg),
        "neg_ratio_target": neg_ratio,
        "include_context_neg": include_context_neg,
        "quality_bucket_all": dict(Counter(r["_quality_bucket"] for r in annotated)),
        "quality_bucket_selected": dict(Counter(r["_quality_bucket"] for r in selected)),
        "label_selected": dict(Counter(int(r.get("y", 0)) for r in selected)),
        "source_zero_selected": sum(1 for r in selected if source_count(r) == 0),
        "category_selected": {
            cat: {"n": len(vals), "pos": sum(int(v.get("y", 0)) for v in vals)}
            for cat, vals in sorted(group_by(selected, "category").items())
        },
    }
    return selected, report


def group_by(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(field, ""))].append(r)
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_dataset", default="data/final/dataset.jsonl")
    ap.add_argument(
        "--verified_dataset",
        default="data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl",
    )
    ap.add_argument("--out", default="data/final/dataset_hq_v1.jsonl")
    ap.add_argument("--report", default="data/final/dataset_hq_v1_report.json")
    ap.add_argument("--neg_ratio", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--drop_weak_pos", action="store_true")
    ap.add_argument("--include_context_neg", action="store_true")
    args = ap.parse_args()

    rows = merge_records(read_jsonl(args.full_dataset), read_jsonl(args.verified_dataset))
    selected, report = choose_rows(
        rows,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        keep_weak_pos=not args.drop_weak_pos,
        include_context_neg=args.include_context_neg,
    )
    write_jsonl(args.out, selected)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[build_dataset_hq_v1] wrote {args.out} n={report['selected_n']} "
        f"pos={report['selected_pos']} neg={report['selected_neg']}"
    )
    print(f"[build_dataset_hq_v1] report={args.report}")


if __name__ == "__main__":
    main()
