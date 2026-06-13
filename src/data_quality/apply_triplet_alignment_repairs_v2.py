"""Apply high-confidence triplet-alignment repairs as a new candidate view.

This merge is deliberately conservative:

- only high-confidence keep_clean / keep_risk verifier rows are promoted;
- labels and sample weights are copied unchanged from the proposal dataset;
- returned claim_text must map back to an existing SRT segment/passage;
- product evidence is replaced by the verifier-grounded minimal evidence item;
- product_title evidence is represented as a PARAM-style evidence item with
  param_key="product_title" so existing model input code can read it.

The action name `keep_clean` means product evidence supports the claim.  It does
not relabel the consumer-perception target to 0.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


MAIN_ACTIONS = {"keep_clean", "keep_risk"}
QUESTION_LIKE_RE = re.compile(r"[?？]|多大|帮我看|看一下|看下|有没有|是不是|可以吗|能买吗|行吗|谁要|谁拍")


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def load_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    return {pair_id(r): r for r in read_jsonl(path)}


def map_claim_segment(row: dict[str, Any], claim_text: str) -> tuple[dict[str, Any] | None, str]:
    claim_text = str(claim_text or "").strip()
    if not claim_text:
        return None, "empty_claim_text"
    claim = row.get("claim") or {}
    for seg in claim.get("segments") or []:
        text = str(seg.get("text", "") or "")
        if claim_text == text or claim_text in text:
            out = dict(seg)
            out["text"] = claim_text
            out["_triplet_repair_span_source"] = "existing_segment_substring" if claim_text != text else "existing_segment_exact"
            return out, ""
    passage = str(claim.get("passage", "") or "")
    if claim_text in passage:
        return {
            "claim_id": f"{pair_id(row)}__triplet_repair",
            "clip_id": "",
            "t_start": 0.0,
            "t_end": 0.0,
            "start_ts": "",
            "end_ts": "",
            "text": claim_text,
            "_triplet_repair_span_source": "passage_substring_without_timestamp",
        }, "claim_mapped_without_timestamp"
    return None, "claim_text_not_found_in_existing_claim"


def evidence_from_result(res: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], str]:
    src = str(res.get("source_type", "") or "")
    raw = str(res.get("raw_text", "") or "").strip()
    path = str(res.get("path_or_clip_id", "") or "").strip()
    ts = str(res.get("timestamp_or_image", "") or "").strip()
    if not raw:
        return {}, "empty_evidence_text"
    meta = {
        "_triplet_repair_source_type": src,
        "_triplet_repair_relation": res.get("relation_to_claim"),
        "_triplet_repair_confidence": res.get("confidence"),
        "_triplet_repair_normalized_value": res.get("normalized_value"),
    }
    if src == "product_title":
        return {"evidence_params": [{"param_key": "product_title", "raw_text": raw, **meta}]}, ""
    if src == "params":
        key = "llm_verified_param"
        value = raw
        for sep in (":", "："):
            if sep in raw:
                left, right = raw.split(sep, 1)
                if left.strip() and right.strip():
                    key, value = left.strip(), right.strip()
                    break
        return {"evidence_params": [{"param_key": key, "raw_text": value, **meta}]}, ""
    if src == "detail_image_ocr":
        return {"evidence_ocr": [{"raw_text": raw, "image_path": path or ts, **meta}]}, ""
    if src == "detail_image_vlm":
        return {"evidence_vlm": [{"raw_quote": raw, "image_path": path or ts, **meta}]}, ""
    return {}, f"unsupported_source_type:{src}"


def promote_row(row: dict[str, Any], res: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if str(res.get("curation_action")) not in MAIN_ACTIONS:
        return None, "not_main_action"
    if str(res.get("confidence")) != "high":
        return None, "not_high_confidence"
    if str(res.get("relation_to_claim")) not in {"supports_claim", "contradicts_claim"}:
        return None, "not_support_or_contradict"
    seg, claim_warn = map_claim_segment(row, str(res.get("claim_text", "") or ""))
    if seg is None:
        return None, claim_warn
    if QUESTION_LIKE_RE.search(str(seg.get("text", "") or "")):
        return None, "question_like_not_assertive_claim"
    evidence, ev_warn = evidence_from_result(res)
    if not evidence:
        return None, ev_warn

    out = dict(row)
    out["claim"] = {
        "has_claim_srt": True,
        "passage": seg["text"],
        "segments": [seg],
    }
    out["evidence_params"] = evidence.get("evidence_params", [])
    out["evidence_ocr"] = evidence.get("evidence_ocr", [])
    out["evidence_vlm"] = evidence.get("evidence_vlm", [])
    out["evidence_count"] = {
        "params": len(out["evidence_params"]),
        "ocr": len(out["evidence_ocr"]),
        "vlm": len(out["evidence_vlm"]),
    }
    out["coverage"] = sum(1 for k in ("params", "ocr", "vlm") if out["evidence_count"][k] > 0)
    out["confidence"] = "high"
    out["_proposal_triplet_alignment_v2"] = {
        **(out.get("_proposal_triplet_alignment_v2") or {}),
        "status": "triplet_aligned_after_llm_repair",
        "claim_valid": True,
        "evidence_valid": True,
        "repair_warning": claim_warn,
    }
    out["_triplet_repair_v2"] = {
        "source_result_pair_id": res.get("pair_id"),
        "curation_action": res.get("curation_action"),
        "relation_to_claim": res.get("relation_to_claim"),
        "source_type": res.get("source_type"),
        "normalized_value": res.get("normalized_value"),
        "model": res.get("model"),
        "label_policy": "y/c preserved from proposal; action describes claim-evidence relation only",
    }
    return out, claim_warn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_v2_20260613.jsonl")
    ap.add_argument("--repair_rows", default="data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl")
    ap.add_argument("--results", default="data/final/repaired_v1/proposal_triplet_alignment_llm_repair_p0_v3_withlabel_20260613.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_plus_p0repair_v2_20260613.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/triplet_alignment_p0repair_promoted_manifest_v2_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/triplet_alignment_p0repair_promoted_v2_20260613.report.json")
    args = ap.parse_args()

    base_rows = list(read_jsonl(args.base))
    repair_by_id = load_by_pair(args.repair_rows)
    results = list(read_jsonl(args.results))
    promoted: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped = Counter()
    warnings = Counter()

    for res in results:
        row = repair_by_id.get(pair_id(res))
        if row is None:
            skipped["missing_repair_row"] += 1
            continue
        out, reason = promote_row(row, res)
        if out is None:
            skipped[reason] += 1
            continue
        warnings[reason or "none"] += 1
        promoted.append(out)
        manifest.append({
            "pair_id": pair_id(out),
            "y": out.get("y"),
            "c": out.get("c"),
            "attribute_name": out.get("attribute_name"),
            "curation_action": res.get("curation_action"),
            "relation_to_claim": res.get("relation_to_claim"),
            "claim_text": (out.get("claim") or {}).get("passage"),
            "source_type": res.get("source_type"),
            "raw_text": res.get("raw_text"),
            "warning": reason,
        })

    base_ids = {pair_id(r) for r in base_rows}
    new_rows = [r for r in promoted if pair_id(r) not in base_ids]
    merged = base_rows + new_rows
    write_jsonl(args.out, merged)
    write_jsonl(args.manifest, manifest)
    report = {
        "base": args.base,
        "results": args.results,
        "out": args.out,
        "base_n": len(base_rows),
        "promoted_n": len(promoted),
        "new_rows_n": len(new_rows),
        "merged_n": len(merged),
        "skipped": dict(skipped),
        "warnings": dict(warnings),
        "promoted_actions": dict(Counter(str(r.get("_triplet_repair_v2", {}).get("curation_action")) for r in promoted)),
        "promoted_labels": dict(Counter(str(r.get("y")) for r in promoted)),
        "source_type": dict(Counter(str(r.get("_triplet_repair_v2", {}).get("source_type")) for r in promoted)),
        "examples": manifest[:20],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
