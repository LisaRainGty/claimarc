"""Build a broad claimful pool for adjudicated CLAIMARC data expansion.

The earlier HQ silver pool is deliberately conservative. For scaling beyond
1.6k samples, this script keeps every pair with grounded livestream claims and
attaches deterministic quality metadata. Downstream LLM adjudication then
decides whether weak negatives are clean negatives, evidence-risk positives,
or ambiguous drops.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import has_claim, quality_bucket, read_jsonl, source_count
from data_quality.build_dataset_hq_v1 import merge_records, key


BROAD_BUCKET_WEIGHT = {
    "pos_core": 1.20,
    "pos_silver": 0.95,
    "pos_weak": 0.55,
    "neg_core": 1.10,
    "neg_silver_sourceful": 0.75,
    "neg_silver_comment_only": 0.45,
    "neg_context_sourceful": 0.20,
    "neg_suspect_fake": 0.05,
    "neg_weak": 0.08,
}


def add_broad_fields(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    bucket = quality_bucket(out)
    weight = BROAD_BUCKET_WEIGHT.get(bucket, 0.0)
    out["_quality_bucket"] = bucket
    out["_quality_weight"] = round(weight, 4)
    out["_base_pair_id"] = key(out)
    c0 = float(out.get("c", 0.05) or 0.05)
    out["_c_original"] = round(c0, 4)
    if weight > 0:
        out["c"] = round(max(0.03, min(1.0, c0 * weight)), 4)
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
    ap.add_argument("--out", default="data/final/dataset_hq_broad_claimful_v1.jsonl")
    ap.add_argument("--report", default="data/final/dataset_hq_broad_claimful_v1_report.json")
    ap.add_argument("--sourceful_only", action="store_true")
    args = ap.parse_args()

    rows = merge_records(read_jsonl(args.full_dataset), read_jsonl(args.verified_dataset))
    selected = []
    for rec in rows:
        if not has_claim(rec):
            continue
        if args.sourceful_only and source_count(rec) == 0:
            continue
        selected.append(add_broad_fields(rec))
    selected.sort(key=lambda r: (str(r.get("room_id", "")), key(r)))

    report = {
        "input_n": len(rows),
        "selected_n": len(selected),
        "sourceful_only": args.sourceful_only,
        "label_selected": dict(Counter(int(r.get("y", 0)) for r in selected)),
        "quality_bucket_selected": dict(Counter(r.get("_quality_bucket", "") for r in selected)),
        "confidence_selected": dict(Counter(str(r.get("confidence", "")) for r in selected)),
        "source_zero_selected": sum(1 for r in selected if source_count(r) == 0),
        "category_selected": dict(Counter(str(r.get("category", "")) for r in selected)),
        "split_selected": dict(Counter(str(r.get("split", "")) for r in selected)),
    }

    write_jsonl(args.out, selected)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[build_dataset_hq_broad_v1] wrote {args.out} "
        f"n={report['selected_n']} labels={report['label_selected']} "
        f"source0={report['source_zero_selected']}"
    )
    print(f"[build_dataset_hq_broad_v1] report={args.report}")


if __name__ == "__main__":
    main()
