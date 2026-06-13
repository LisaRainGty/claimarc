"""Validate queue-based re-extracted claims stored in a single JSONL file."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from data_quality.validate_claim_attribute_v2 import classify_claim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default="data/processed/stageB_product_v2/acmt_product_v2.json")
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direct_out", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={}) or {}
    rows = []
    direct = []
    status = Counter()
    missing_attr = 0
    for r in read_jsonl(args.claims):
        pid = str(r.get("product_id") or "")
        aid = str(r.get("attribute_id") or "")
        attrs = acmt.get(pid) or {}
        if aid not in attrs:
            missing_attr += 1
            continue
        val = classify_claim(r, aid, attrs)
        out = dict(r)
        out["_claim_attribute_validation"] = val
        out["validation_status"] = val.get("validation_status")
        rows.append(out)
        status[str(out["validation_status"])] += 1
        if out["validation_status"] == "direct":
            direct.append(out)
    write_jsonl(args.out, rows)
    if args.direct_out:
        write_jsonl(args.direct_out, direct)
    report = {
        "claims": args.claims,
        "out": args.out,
        "n_claims": len(rows),
        "n_direct": len(direct),
        "pairs_direct": len({str(r.get("pair_id")) for r in direct}),
        "products_direct": len({str(r.get("product_id")) for r in direct}),
        "status": dict(status),
        "missing_attr": missing_attr,
    }
    write_json(args.report or args.out.replace(".jsonl", "_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
