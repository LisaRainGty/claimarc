"""Build atomic skeleton rows from validated queue re-extraction claims."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from common import product_index as pidx
from common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from common.srt import ts_to_seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True,
                    help="Validated re-extract JSONL; keeps rows with validation_status=direct.")
    ap.add_argument("--acmt", default="data/processed/stageB_product_v2/acmt_product_v2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    bundles = pidx.build_bundles()
    acmt = read_json(args.acmt, default={}) or {}
    rows: list[dict[str, Any]] = []
    for r in read_jsonl(args.claims):
        if r.get("validation_status") != "direct":
            continue
        pid = str(r.get("product_id") or "")
        aid = str(r.get("attribute_id") or "")
        if aid not in (acmt.get(pid) or {}):
            continue
        meta = acmt[pid][aid]
        claim_id = str(r.get("claim_id") or f"{pid}_re{len(rows) + 1}")
        atomic_id = f"p{pid}__{aid}__{claim_id}"
        b = bundles.get(pid)
        rows.append({
            "atomic_id": atomic_id,
            "pair_id": f"p{pid}__{aid}",
            "product_id": pid,
            "category": b.category if b else "",
            "subcategory": b.subcategory if b else "",
            "room_id": b.room_id if b else "UNKNOWN",
            "attribute_id": aid,
            "attribute_name": r.get("attribute_name") or meta.get("canonical_name") or aid,
            "aliases": list(meta.get("aliases") or [])[:30],
            "source_family": meta.get("source_family", ""),
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
            "_claim_attribute_validation": r.get("_claim_attribute_validation", {}),
            "_source": "productv2_comment_triggered_reextract",
        })
    write_jsonl(args.out, rows)
    report = {
        "claims": args.claims,
        "acmt": args.acmt,
        "out": args.out,
        "n_atomic_claims": len(rows),
        "pairs": len({r["pair_id"] for r in rows}),
        "products": len({r["product_id"] for r in rows}),
        "source_family": dict(Counter(str(r.get("source_family")) for r in rows)),
        "category": dict(Counter(str(r.get("category")) for r in rows)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
