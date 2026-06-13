"""Build atomic claim-level skeletons from validated B1 outputs.

The legacy pipeline aggregates all claims for a `(product, attribute)` pair
into one passage.  This builder keeps each validated direct claim as its own
training/evaluation candidate so later stages can align consumer signals at
claim granularity and optionally aggregate back to pair level.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl
from common.srt import ts_to_seconds


def build(validation_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundles = pidx.build_bundles()
    rows: list[dict[str, Any]] = []
    for r in read_jsonl(validation_path):
        if r.get("validation_status") != "direct":
            continue
        pid = str(r.get("product_id") or "")
        aid = str(r.get("attribute_id") or "")
        claim_id = str(r.get("claim_id") or f"{pid}_{len(rows) + 1}")
        b = bundles.get(pid)
        atomic_id = f"p{pid}__{aid}__{claim_id}"
        rows.append({
            "atomic_id": atomic_id,
            "pair_id": f"p{pid}__{aid}",
            "product_id": pid,
            "category": b.category if b else "",
            "subcategory": b.subcategory if b else "",
            "room_id": b.room_id if b else "UNKNOWN",
            "attribute_id": aid,
            "attribute_name": r.get("attribute_name") or aid,
            "source_family": r.get("source_family", ""),
            "claim": {
                "has_claim_srt": True,
                "passage": r.get("claim_text", ""),
                "segments": [{
                    "claim_id": claim_id,
                    "clip_id": r.get("srt_file", ""),
                    "srt_path": r.get("srt_path", ""),
                    "start_ts": r.get("start_ts", ""),
                    "end_ts": r.get("end_ts", ""),
                    "t_start": ts_to_seconds(str(r.get("start_ts") or "00:00:00,000")),
                    "t_end": ts_to_seconds(str(r.get("end_ts") or "00:00:00,000")),
                    "text": r.get("claim_text", ""),
                }],
            },
            "_claim_attribute_validation": {
                "own_hits": r.get("own_hits", []),
                "fact_hits": r.get("fact_hits", []),
                "meaningful_fact_hits": r.get("meaningful_fact_hits", []),
                "number_hit": r.get("number_hit", False),
            },
        })
    report = {
        "validation_path": validation_path,
        "n_atomic_claims": len(rows),
        "n_pairs": len({r["pair_id"] for r in rows}),
        "n_products": len({r["product_id"] for r in rows}),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "by_source_family": dict(Counter(r["source_family"] for r in rows)),
    }
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows, report = build(args.validation)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
