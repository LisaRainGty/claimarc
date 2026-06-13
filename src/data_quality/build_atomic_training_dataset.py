"""Join atomic records, atomic labels, and product fact evidence.

This builder is intentionally small and deterministic.  It does not create
labels; it only combines the outputs of B4/B5 atomic alignment, atomic label
construction, and Stage-C fact records into a train-only auxiliary JSONL.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common.io_utils import read_jsonl, write_json, write_jsonl


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("product_id") or ""), str(row.get("attribute_id") or "")


def _load_labels(path: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        aid = str(row.get("atomic_id") or "")
        if aid:
            out[aid] = row
    return out


def _load_facts(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        k = _key(row)
        if k[0] and k[1]:
            out[k] = row
    return out


def _dedup_items(items: list[dict[str, Any]], text_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in items or []:
        sig = []
        for key in text_keys:
            sig.append(str(item.get(key) or ""))
        sig.append(str(item.get("image_path") or item.get("param_key") or item.get("path_or_image") or ""))
        marker = tuple(sig)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _load_labels(args.labels)
    facts = _load_facts(args.facts)
    rows = []
    skipped = Counter()
    for rec in read_jsonl(args.atomic_records):
        atomic_id = str(rec.get("atomic_id") or "")
        lab = labels.get(atomic_id)
        if not lab:
            skipped["missing_label"] += 1
            continue
        fact = facts.get(_key(rec), {})
        cnt = dict(fact.get("evidence_count") or {})
        ev_params = _dedup_items(list(fact.get("evidence_params") or []), ("raw_text", "param_key"))
        ev_ocr = _dedup_items(list(fact.get("evidence_ocr") or []), ("raw_text",))
        ev_vlm = _dedup_items(list(fact.get("evidence_vlm") or []), ("raw_quote", "raw_text"))
        if not cnt:
            cnt = {"params": len(ev_params), "ocr": len(ev_ocr), "vlm": len(ev_vlm)}
        coverage = int(fact.get("coverage", 0) or sum(1 for v in cnt.values() if int(v or 0) > 0))
        if coverage < args.min_coverage:
            skipped["low_coverage"] += 1
            continue
        rows.append({
            "atomic_id": atomic_id,
            "pair_id": rec.get("pair_id"),
            "product_id": rec.get("product_id"),
            "category": rec.get("category", ""),
            "subcategory": rec.get("subcategory", ""),
            "room_id": rec.get("room_id", "UNKNOWN"),
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": rec.get("attribute_name"),
            "source_family": rec.get("source_family", ""),
            "claim": rec.get("claim", {}),
            "evidence_params": ev_params,
            "evidence_ocr": ev_ocr,
            "evidence_vlm": ev_vlm,
            "evidence_count": cnt,
            "coverage": coverage,
            "confidence": fact.get("confidence") or config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent"),
            "y": int(lab.get("y", 0) or 0),
            "c": float(lab.get("c", config.C_FLOOR) or config.C_FLOOR),
            "label_audit": lab.get("label_audit", {}),
            "reviews": rec.get("reviews", []),
            "split": args.split,
            "_source": args.source_tag,
            "_claim_attribute_validation": rec.get("_claim_attribute_validation", {}),
        })
    report = {
        "atomic_records": args.atomic_records,
        "labels": args.labels,
        "facts": args.facts,
        "out": args.out,
        "n": len(rows),
        "skipped": dict(skipped),
        "labels_dist": dict(Counter(int(r.get("y", 0) or 0) for r in rows)),
        "coverage": dict(Counter(int(r.get("coverage", 0) or 0) for r in rows)),
        "confidence": dict(Counter(str(r.get("confidence")) for r in rows)),
        "source_family": dict(Counter(str(r.get("source_family")) for r in rows)),
        "category": dict(Counter(str(r.get("category")) for r in rows)),
        "pairs": len({str(r.get("pair_id")) for r in rows}),
        "products": len({str(r.get("product_id")) for r in rows}),
    }
    return rows, report


def write_table(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "| metric | value |",
        "|---|---:|",
        f"| rows | {report['n']} |",
        f"| products | {report['products']} |",
        f"| pairs | {report['pairs']} |",
        f"| labels | `{json.dumps(report['labels_dist'], ensure_ascii=False)}` |",
        f"| coverage | `{json.dumps(report['coverage'], ensure_ascii=False)}` |",
        f"| skipped | `{json.dumps(report['skipped'], ensure_ascii=False)}` |",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_records", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--table_md", default="")
    ap.add_argument("--min_coverage", type=int, default=1)
    ap.add_argument("--split", default="diagnostic")
    ap.add_argument("--source_tag", default="atomic_aux")
    args = ap.parse_args()

    rows, report = build(args)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    if args.table_md:
        write_table(report, args.table_md)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
