"""Promote strict comment-triggered claim re-extraction verifications.

The verifier output is atomic.  This builder applies fixed rules:
- risk_candidate -> y=1 only when live claim and product evidence are both
  found with high/medium confidence;
- clean_candidate -> y=0 under the same evidence requirements;
- all other rows are written to an auxiliary/drop manifest.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def load_schema_pairs(path: str | None) -> set[str] | None:
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    valid: set[str] = set()
    for pid, attrs in obj.items():
        if isinstance(attrs, dict):
            attr_ids = attrs.keys()
        elif isinstance(attrs, list):
            attr_ids = [
                a.get("attribute_id") or a.get("id") or a.get("name")
                for a in attrs
                if isinstance(a, dict)
            ]
        else:
            continue
        for aid in attr_ids:
            if aid:
                valid.add(f"p{pid}__{aid}")
    return valid


def evidence_item(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    src = str(v.get("product_source_type") or "none")
    raw = str(v.get("product_evidence_text") or "")
    item = {
        "raw_text": raw,
        "param_key": src,
        "match": "claim_reextract_v4",
        "normalized_value": v.get("product_value", ""),
        "path_or_clip_id": v.get("product_path_or_image", ""),
        "product_evidence_state": v.get("product_evidence_state", ""),
        "confidence": v.get("confidence", ""),
    }
    if src == "detail_image_ocr":
        item["image_path"] = v.get("product_path_or_image", "")
        return "evidence_ocr", item
    if src == "detail_image_vlm":
        return "evidence_vlm", {
            "raw_quote": raw,
            "image_path": v.get("product_path_or_image", ""),
            "match": "claim_reextract_v4",
            "normalized_value": v.get("product_value", ""),
            "product_evidence_state": v.get("product_evidence_state", ""),
            "confidence": v.get("confidence", ""),
        }
    return "evidence_params", item


def build_row(v: dict[str, Any]) -> dict[str, Any]:
    pid = str(v.get("product_id"))
    aid = str(v.get("attribute_id"))
    b = pidx.build_bundles().get(pid)
    ev_params: list[dict[str, Any]] = []
    ev_ocr: list[dict[str, Any]] = []
    ev_vlm: list[dict[str, Any]] = []
    field, item = evidence_item(v)
    if field == "evidence_params":
        ev_params.append(item)
    elif field == "evidence_ocr":
        ev_ocr.append(item)
    else:
        ev_vlm.append(item)
    cnt = {"params": len(ev_params), "ocr": len(ev_ocr), "vlm": len(ev_vlm)}
    coverage = sum(1 for n in cnt.values() if n > 0)
    action = str(v.get("curation_action"))
    y = 1 if action == "risk_candidate" else 0
    weight = 0.9 if v.get("product_evidence_state") == "contradicted" else 0.82
    if v.get("confidence") == "medium":
        weight = min(weight, 0.78)
    return {
        "pair_id": pair_id(v),
        "product_id": pid,
        "category": v.get("category") or (b.category if b else ""),
        "subcategory": b.subcategory if b else "",
        "room_id": b.room_id if b else "UNKNOWN",
        "attribute_id": aid,
        "attribute_name": v.get("attribute_name") or aid,
        "claim": {
            "has_claim_srt": True,
            "passage": v.get("live_claim_text") or "",
            "segments": [{
                "claim_id": f"{pair_id(v)}__claim_reextract_v4",
                "clip_id": v.get("claim_clip", ""),
                "start_ts": v.get("claim_timestamp", ""),
                "end_ts": v.get("claim_timestamp", ""),
                "text": v.get("live_claim_text") or "",
                "_claim_reextract_v4": {
                    "live_claim_value": v.get("live_claim_value", ""),
                    "consumer_signal": v.get("consumer_signal", ""),
                },
            }],
        },
        "evidence_params": ev_params,
        "evidence_ocr": ev_ocr,
        "evidence_vlm": ev_vlm,
        "evidence_count": cnt,
        "coverage": coverage,
        "confidence": config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent"),
        "y": y,
        "c": weight,
        "label_audit": {
            "source": "claim_reextract_v4_fixed_rules",
            "product_evidence_state": v.get("product_evidence_state"),
            "consumer_signal": v.get("consumer_signal"),
            "confidence": v.get("confidence"),
            "risk_comment_example": v.get("risk_comment_example"),
        },
        "split": "diagnostic",
        "_claim_reextract_v4": {
            "curation_action": action,
            "product_source_type": v.get("product_source_type"),
            "product_value": v.get("product_value"),
            "consumer_anchor": v.get("consumer_anchor"),
            "reject_reason": v.get("reject_reason"),
            "model": v.get("model"),
        },
    }


def is_promotable(v: dict[str, Any]) -> bool:
    return (
        bool(v.get("live_claim_found"))
        and bool(v.get("product_evidence_found"))
        and v.get("confidence") in {"high", "medium"}
        and v.get("curation_action") in {"risk_candidate", "clean_candidate"}
        and v.get("product_source_type") in {"params", "product_title", "detail_image_ocr", "detail_image_vlm"}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl")
    ap.add_argument("--verified", default="data/final/repaired_v1/expansion_candidate_pools_v4/claim_reextract_verify_pilot20_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_claim_reextract_v4_pilot_merged.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/claim_reextract_v4_pilot_aux_manifest.jsonl")
    ap.add_argument("--promoted_out", default="",
                    help="Optional JSONL containing only promoted rows, useful as train-only auxiliary data.")
    ap.add_argument("--report", default="data/final/repaired_v1/claim_reextract_v4_pilot_report.json")
    ap.add_argument(
        "--acmt_filter",
        default=None,
        help="Optional product schema JSON. Verified rows outside this (product, attribute) set are kept auxiliary.",
    )
    args = ap.parse_args()

    base_rows = list(read_jsonl(args.base))
    existing = {pair_id(r) for r in base_rows}
    verified = list(read_jsonl(args.verified))
    schema_pairs = load_schema_pairs(args.acmt_filter)
    promoted: list[dict[str, Any]] = []
    aux: list[dict[str, Any]] = []
    for v in verified:
        pid = pair_id(v)
        in_schema = schema_pairs is None or pid in schema_pairs
        if in_schema and is_promotable(v) and pid not in existing:
            promoted.append(build_row(v))
        else:
            if not in_schema:
                reason = "outside_acmt_filter"
            elif pid in existing:
                reason = "already_in_base"
            elif not is_promotable(v):
                reason = "not_promotable"
            else:
                reason = "auxiliary"
            aux.append({
                "pair_id": pid,
                "curation_action": v.get("curation_action"),
                "live_claim_found": v.get("live_claim_found"),
                "product_evidence_found": v.get("product_evidence_found"),
                "product_evidence_state": v.get("product_evidence_state"),
                "consumer_signal": v.get("consumer_signal"),
                "confidence": v.get("confidence"),
                "reject_reason": v.get("reject_reason"),
                "aux_reason": reason,
            })
    out_rows = base_rows + promoted
    report = {
        "base": args.base,
        "verified": args.verified,
        "acmt_filter": args.acmt_filter,
        "n_acmt_pairs": len(schema_pairs) if schema_pairs is not None else None,
        "n_base": len(base_rows),
        "n_verified": len(verified),
        "n_promoted": len(promoted),
        "n_output": len(out_rows),
        "verified_actions": dict(Counter(str(v.get("curation_action")) for v in verified)),
        "promoted_labels": dict(Counter(str(r.get("y")) for r in promoted)),
        "aux_reasons": dict(Counter(str(a.get("curation_action")) for a in aux)),
        "aux_filter_reasons": dict(Counter(str(a.get("aux_reason")) for a in aux)),
    }
    write_jsonl(args.out, out_rows)
    write_jsonl(args.manifest, aux)
    if args.promoted_out:
        write_jsonl(args.promoted_out, promoted)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
