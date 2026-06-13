"""Filter atomic datasets by a claim-attribute validation file.

This is useful after tightening `validate_claim_attribute_v2.py`: existing
atomic rows can be reduced to a high-precision subset without rerunning the
expensive consumer-alignment step.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


def claim_ids_from_validation(path: str) -> set[str]:
    keep = set()
    for r in read_jsonl(path):
        if r.get("validation_status") != "direct":
            continue
        cid = str(r.get("claim_id") or "").strip()
        if cid:
            keep.add(cid)
    return keep


def row_claim_id(row: dict[str, Any]) -> str:
    claim = row.get("claim") or {}
    for seg in claim.get("segments") or []:
        cid = str(seg.get("claim_id") or "").strip()
        if cid:
            return cid
    atomic_id = str(row.get("atomic_id") or "")
    if "__" in atomic_id:
        return atomic_id.rsplit("__", 1)[-1]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    allowed = claim_ids_from_validation(args.validation)
    rows = []
    dropped = Counter()
    for row in read_jsonl(args.dataset):
        cid = row_claim_id(row)
        if cid in allowed:
            rows.append(row)
        else:
            dropped["claim_not_direct"] += 1
    write_jsonl(args.out, rows)
    report = {
        "dataset": args.dataset,
        "validation": args.validation,
        "out": args.out,
        "allowed_claim_ids": len(allowed),
        "n_out": len(rows),
        "dropped": dict(dropped),
        "pairs": len({str(r.get("pair_id")) for r in rows}),
        "products": len({str(r.get("product_id")) for r in rows}),
        "positives": sum(int(r.get("y", 0) or 0) for r in rows),
        "coverage": dict(Counter(str(r.get("coverage")) for r in rows)),
        "source_family": dict(Counter(str(r.get("source_family")) for r in rows)),
    }
    write_json(args.report or args.out.replace(".jsonl", "_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
