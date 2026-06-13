"""Build deterministic high-purity dataset views for diagnosis.

The views are experimental branches. They do not replace the main benchmark;
they help test whether low-evidence and weak-label rows are the current
performance bottleneck.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from data_quality.audit_dataset_quality import has_claim, quality_bucket, source_count
from data_quality.rebuild_repaired_datasets_v1 import split_leakage


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def base_ok(row: dict[str, Any]) -> bool:
    return has_claim(row) and str(row.get("_attribute_scope", "")) == "product_attribute"


def sourceful(row: dict[str, Any]) -> bool:
    return base_ok(row) and source_count(row) >= 1


def sourceful_no_weak_pos(row: dict[str, Any]) -> bool:
    return sourceful(row) and quality_bucket(row) != "pos_weak"


def sourceful_minsrc2(row: dict[str, Any]) -> bool:
    return base_ok(row) and source_count(row) >= 2


def core_silver(row: dict[str, Any]) -> bool:
    return sourceful(row) and quality_bucket(row) in {
        "pos_core",
        "pos_silver",
        "neg_core",
        "neg_silver_sourceful",
        "neg_context_sourceful",
    }


POLICIES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "sourceful": sourceful,
    "sourceful_no_weak_pos": sourceful_no_weak_pos,
    "sourceful_minsrc2": sourceful_minsrc2,
    "core_silver": core_silver,
}


def annotate(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    out = []
    for rec in rows:
        row = dict(rec)
        row["_filtered_view_policy"] = policy
        row["_source_count"] = source_count(row)
        row["_quality_bucket"] = quality_bucket(row)
        out.append(row)
    out.sort(key=lambda r: (str(r.get("room_id", "")), pair_id(r)))
    return out


def summarize(rows: list[dict[str, Any]], policy: str, source: str, out_path: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "source": source,
        "out": out_path,
        "n": len(rows),
        "labels": dict(Counter(int(r.get("y", 0)) for r in rows)),
        "split": dict(Counter(str(r.get("split", "")) for r in rows)),
        "split_leakage": split_leakage(rows),
        "source_counts": dict(Counter(min(source_count(r), 5) for r in rows)),
        "quality_bucket": dict(Counter(quality_bucket(r) for r in rows)),
        "confidence": dict(Counter(str(r.get("confidence", "")) for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0adjudicated_v1.jsonl")
    ap.add_argument("--out_dir", default="data/final/repaired_v1/filtered_views_v1")
    ap.add_argument("--report", default="data/final/repaired_v1/filtered_views_v1_report.json")
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    report = []
    for name, predicate in POLICIES.items():
        selected = annotate([r for r in rows if predicate(r)], name)
        out = Path(args.out_dir) / f"dataset_attrpol_hq_product_rawtext_llmcurated_p0adjudicated_{name}_v1.jsonl"
        write_jsonl(out, selected)
        report.append(summarize(selected, name, args.dataset, str(out)))
    write_json(args.report, {"views": report})
    for row in report:
        print(f"{row['policy']}: n={row['n']} labels={row['labels']} out={row['out']}")


if __name__ == "__main__":
    main()
