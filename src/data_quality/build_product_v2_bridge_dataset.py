"""Build diagnostic datasets from product-v2 A_cmt and validated claims.

This bridges the new product-only schema to existing labels/fact records.  It
is intended for diagnostics before a full raw-stage B/C/label rerun.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("product_id")), str(row.get("attribute_id"))


def source_count(row: dict[str, Any]) -> int:
    cnt = row.get("evidence_count") or {}
    if isinstance(cnt, dict):
        return sum(1 for v in cnt.values() if int(v or 0) > 0)
    return int(row.get("coverage", 0) or 0)


def write_view(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "labels": dict(Counter(str(r.get("y")) for r in rows)),
        "claimful": sum(1 for r in rows if (r.get("claim") or {}).get("has_claim_srt")),
        "source_count": dict(Counter(str(source_count(r)) for r in rows)),
        "confidence": dict(Counter(str(r.get("confidence")) for r in rows)),
        "categories": dict(Counter(str(r.get("category")) for r in rows).most_common(20)),
        "top_attributes": Counter(str(r.get("attribute_name")) for r in rows).most_common(30),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_skeleton", default="data/processed/stageB_product_v2/pair_skeleton_product_v2.jsonl")
    ap.add_argument("--labels", default="data/processed/labels.jsonl")
    ap.add_argument("--facts", default="data/processed/stageC/fact_records.jsonl")
    ap.add_argument("--out_dir", default="data/final/repaired_v1/product_v2_bridge")
    args = ap.parse_args()

    pair_rows = list(read_jsonl(args.pair_skeleton))
    labels = {key(r): r for r in read_jsonl(args.labels)}
    facts = {key(r): r for r in read_jsonl(args.facts)}
    bundles = pidx.build_bundles()

    rows: list[dict[str, Any]] = []
    missing_labels = 0
    for pr in pair_rows:
        k = key(pr)
        lb = labels.get(k)
        if not lb:
            missing_labels += 1
            continue
        fr = facts.get(k, {})
        pid, aid = k
        b = bundles.get(pid)
        rows.append({
            "pair_id": pr.get("pair_id"),
            "product_id": pid,
            "category": (b.category if b else fr.get("category", "")),
            "subcategory": b.subcategory if b else "",
            "room_id": b.room_id if b else "UNKNOWN",
            "attribute_id": aid,
            "attribute_name": pr.get("attribute_canonical", aid),
            "claim": {
                "has_claim_srt": bool(pr.get("has_claim_srt")),
                "passage": pr.get("passage", ""),
                "segments": pr.get("segments", []),
            },
            "evidence_params": fr.get("evidence_params", []),
            "evidence_ocr": fr.get("evidence_ocr", []),
            "evidence_vlm": fr.get("evidence_vlm", []),
            "evidence_count": fr.get("evidence_count", {"params": 0, "ocr": 0, "vlm": 0}),
            "coverage": fr.get("coverage", 0),
            "confidence": fr.get("confidence", "absent"),
            "y": lb.get("y"),
            "c": lb.get("c"),
            "label_audit": lb.get("label_audit", {}),
            "split": "diagnostic",
            "_product_v2": {
                "source_family": pr.get("source_family"),
                "selection_score": pr.get("selection_score"),
                "bridge_uses_old_labels": True,
                "bridge_uses_old_facts": True,
            },
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "all": rows,
        "claimful": [r for r in rows if (r.get("claim") or {}).get("has_claim_srt")],
        "sourceful": [r for r in rows if source_count(r) > 0],
        "claimful_sourceful": [
            r for r in rows
            if (r.get("claim") or {}).get("has_claim_srt") and source_count(r) > 0
        ],
        "claimful_sourceful_weighted": [
            r for r in rows
            if (r.get("claim") or {}).get("has_claim_srt") and source_count(r) > 0 and float(r.get("c") or 0) > 0.05
        ],
    }
    paths = {}
    for name, view_rows in views.items():
        path = out_dir / f"dataset_product_v2_bridge_{name}.jsonl"
        write_view(path, view_rows)
        paths[name] = str(path)

    report = {
        "pair_skeleton": args.pair_skeleton,
        "labels": args.labels,
        "facts": args.facts,
        "missing_labels": missing_labels,
        "paths": paths,
        "views": {name: summarize(view_rows) for name, view_rows in views.items()},
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps({name: report["views"][name] for name in views}, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
