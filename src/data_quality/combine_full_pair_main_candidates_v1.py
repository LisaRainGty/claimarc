"""Combine promoted full-pair main candidates across reconstruction batches.

Each batch-level promotion already applies local gates.  This combiner applies
the same claim-family dedupe across batches so that one product-room claim does
not enter the supervised view under multiple attributes or with conflicting
labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.build_full_pair_promoted_dataset_v1 import (
    apply_claim_family_dedupe,
    clean,
    is_main,
)
from data_quality.rebuild_repaired_datasets_v1 import assign_room_splits, split_leakage


def read_inputs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            nr = dict(row)
            nr["_combined_source"] = path
            rows.append(nr)
    return rows


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def dedupe_pair_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        pid = pair_id(row)
        if pid in seen:
            dropped += 1
            continue
        seen.add(pid)
        out.append(row)
    return out, dropped


def report(rows: list[dict[str, Any]], main_rows: list[dict[str, Any]], pair_duplicates: int, dedupe: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_rows": len(rows) + pair_duplicates,
        "pair_id_duplicates_dropped": pair_duplicates,
        "stateful_rows": len(rows),
        "main_rows": len(main_rows),
        "main_labels": dict(Counter(int(r.get("y", 0) or 0) for r in main_rows)),
        "promotion_state": dict(Counter(clean((r.get("label_audit") or {}).get("promotion_state")) for r in rows)),
        "main_split": dict(Counter(clean(r.get("split")) for r in main_rows)),
        "main_split_leakage": split_leakage(main_rows) if main_rows else {},
        "category": dict(Counter(clean(r.get("category")) for r in main_rows)),
        **dedupe,
    }


def write_markdown(path: str | Path, rep: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Combined Full Pair Main Candidates v1",
        "",
        "This report combines batch-level promoted rows and reapplies cross-batch claim-family gates.",
        "",
        "## Inputs",
        "",
    ]
    for p in args.inputs:
        lines.append(f"- `{p}`")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- stateful combined rows: `{args.out_all}`")
    lines.append(f"- main combined rows: `{args.out_main}`")
    lines.append(f"- repair/silver combined rows: `{args.out_repair}`")
    lines.append(f"- report json: `{args.report}`")
    lines.extend(["", "## Summary", ""])
    for key, value in rep.items():
        lines.append(f"- `{key}`: `{value}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_all", default="data/final/repaired_v1/dataset_full_pair_combined_stateful_v1_20260614.jsonl")
    ap.add_argument("--out_main", default="data/final/repaired_v1/dataset_full_pair_combined_main_v1_20260614.jsonl")
    ap.add_argument("--out_repair", default="data/final/repaired_v1/full_pair_combined_repair_silver_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/dataset_full_pair_combined_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_COMBINED_MAIN_CANDIDATES_20260614.md")
    args = ap.parse_args()

    rows = read_inputs(args.inputs)
    rows, pair_duplicates = dedupe_pair_ids(rows)
    rows, dedupe = apply_claim_family_dedupe(rows)
    rows = assign_room_splits(rows)
    main_rows = assign_room_splits([r for r in rows if is_main(r)])
    repair_rows = [r for r in rows if not is_main(r)]

    write_jsonl(args.out_all, rows)
    write_jsonl(args.out_main, main_rows)
    write_jsonl(args.out_repair, repair_rows)
    rep = report(rows, main_rows, pair_duplicates, dedupe)
    rep.update({
        "inputs": args.inputs,
        "out_all": args.out_all,
        "out_main": args.out_main,
        "out_repair": args.out_repair,
    })
    write_json(args.report, rep)
    write_markdown(args.markdown, rep, args)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
