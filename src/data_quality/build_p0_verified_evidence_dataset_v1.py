"""Promote verified P0 direct product evidence into a diagnostic dataset.

The verifier output is evidence-only: it does not see original labels or split.
This script appends high-confidence product-side evidence to the dataset but
does not relabel. A follow-up adjudication step should combine the promoted
evidence with consumer comments.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from data_quality.audit_dataset_quality import read_jsonl, source_count


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def evidence_count(rec: dict[str, Any]) -> dict[str, int]:
    return {
        "params": len(rec.get("evidence_params") or []),
        "ocr": len(rec.get("evidence_ocr") or []),
        "vlm": len(rec.get("evidence_vlm") or []),
    }


def verified_to_evidence(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    item = {
        "raw_text": v.get("raw_text", ""),
        "param_key": v.get("source_type", ""),
        "match": "llm_verified_queue_v2",
        "normalized_value": v.get("normalized_value", ""),
        "relation_to_claim": v.get("relation_to_claim", ""),
        "confidence": v.get("confidence", ""),
        "path_or_clip_id": v.get("path_or_clip_id", ""),
        "timestamp_or_image": v.get("timestamp_or_image", ""),
    }
    src = str(v.get("source_type", ""))
    if src == "detail_image_ocr":
        item["image_path"] = v.get("path_or_clip_id", "")
        return "evidence_ocr", item
    if src == "detail_image_vlm":
        item = {
            "raw_quote": v.get("raw_text", ""),
            "image_path": v.get("path_or_clip_id", ""),
            "match": "llm_verified_queue_v2",
            "normalized_value": v.get("normalized_value", ""),
            "relation_to_claim": v.get("relation_to_claim", ""),
            "confidence": v.get("confidence", ""),
        }
        return "evidence_vlm", item
    return "evidence_params", item


def add_unique(items: list[dict[str, Any]], item: dict[str, Any], text_key: str) -> list[dict[str, Any]]:
    seen = {(str(x.get(text_key, "")), str(x.get("path_or_clip_id", x.get("image_path", "")))) for x in items}
    key = (str(item.get(text_key, "")), str(item.get("path_or_clip_id", item.get("image_path", ""))))
    if key not in seen:
        items.append(item)
    return items


def build(dataset_path: str, verified_path: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(dataset_path)
    verified_all = read_jsonl(verified_path)
    verified = {
        str(v.get("pair_id")): v
        for v in verified_all
        if v.get("curation_action") in {"keep_clean", "keep_risk"}
        and v.get("source_type") in {"params", "product_title", "detail_image_ocr", "detail_image_vlm"}
        and v.get("evidence_found")
        and v.get("claim_found")
    }
    out = []
    changed = []
    for rec in rows:
        pid = str(rec.get("pair_id"))
        v = verified.get(pid)
        if not v:
            out.append(rec)
            continue
        new = dict(rec)
        field, item = verified_to_evidence(v)
        text_key = "raw_quote" if field == "evidence_vlm" else "raw_text"
        new[field] = add_unique(list(new.get(field) or []), item, text_key)
        new["_queue_v2_verified_product_evidence"] = {
            "source_type": v.get("source_type"),
            "relation_to_claim": v.get("relation_to_claim"),
            "curation_action": v.get("curation_action"),
            "normalized_value": v.get("normalized_value"),
            "confidence": v.get("confidence"),
        }
        cnt = evidence_count(new)
        new["evidence_count"] = cnt
        coverage = sum(1 for n in cnt.values() if n > 0)
        new["coverage"] = coverage
        new["confidence"] = config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent")
        out.append(new)
        changed.append({
            "pair_id": pid,
            "priority": 1,
            "queue_priority": v.get("priority"),
            "actions": ["llm_claim_comment_adjudication", "verified_product_evidence_queue_v2"],
            "source_type": v.get("source_type"),
            "relation_to_claim": v.get("relation_to_claim"),
            "curation_action": v.get("curation_action"),
        })

    report = {
        "input_dataset": dataset_path,
        "verified_input": verified_path,
        "n_input": len(rows),
        "n_verified_all": len(verified_all),
        "n_promoted": len(changed),
        "promoted_actions": dict(Counter(v.get("curation_action") for v in verified.values())),
        "promoted_relations": dict(Counter(v.get("relation_to_claim") for v in verified.values())),
        "source0_before": sum(1 for r in rows if source_count(r) == 0),
        "source0_after": sum(1 for r in out if source_count(r) == 0),
        "labels": dict(Counter(int(r.get("y", 0) or 0) for r in out)),
    }
    return out, report, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_v1.jsonl")
    ap.add_argument("--verified", default="data/final/repaired_v1/regeneration_queue_v2_llm_verify_p0_direct_vlmimg.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0verified_v1.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/regeneration_queue_v2_p0verified_manifest_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/p0_verified_evidence_v1_report.json")
    args = ap.parse_args()

    rows, report, manifest = build(args.dataset, args.verified)
    write_jsonl(args.out, rows)
    write_jsonl(args.manifest, manifest)
    write_json(args.report, report)
    print(f"[build_p0_verified_evidence_dataset_v1] wrote {args.out}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
