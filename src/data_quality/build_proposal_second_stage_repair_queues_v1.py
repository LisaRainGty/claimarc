"""Build second-stage proposal-faithful repair queues.

The first triplet verifier tells us whether a row fails because the livestream
claim is off-attribute, the product evidence is off-attribute, or both.  This
script does not relabel consumer perception and does not drop valid hard rows;
it only separates raw-material repair tasks so Stage B/C can be rerun with a
clear objective.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


REL_CLAIM_OK_EVIDENCE_BAD = {"claim_only"}
REL_EVIDENCE_OK_CLAIM_BAD = {"evidence_only"}
REL_JOINT_RESCAN = {"insufficient"}


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def compact_verdict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": row.get("model"),
        "relation_to_claim": row.get("relation_to_claim"),
        "confidence": row.get("confidence"),
        "curation_action": row.get("curation_action"),
        "claim_found": row.get("claim_found"),
        "claim_text": row.get("claim_text"),
        "evidence_found": row.get("evidence_found"),
        "source_type": row.get("source_type"),
        "raw_text": row.get("raw_text"),
        "normalized_value": row.get("normalized_value"),
        "path_or_clip_id": row.get("path_or_clip_id"),
        "timestamp_or_image": row.get("timestamp_or_image"),
        "reject_reason": row.get("reject_reason"),
    }


def base_task(original: dict[str, Any], verdict: dict[str, Any], task_type: str) -> dict[str, Any]:
    out = dict(original)
    out["second_stage_task"] = task_type
    out["previous_triplet_verdict"] = compact_verdict(verdict)
    out["label_policy"] = (
        "preserve proposal current_y/current_c; repair claim/evidence provenance only; "
        "if repaired claim changes the reviewed proposition, send to B4 label rebuild "
        "instead of silently relabeling"
    )
    return out


def classify(verdict: dict[str, Any]) -> str:
    rel = str(verdict.get("relation_to_claim") or "")
    claim_found = bool(verdict.get("claim_found"))
    evidence_found = bool(verdict.get("evidence_found"))
    if rel in REL_CLAIM_OK_EVIDENCE_BAD or (claim_found and not evidence_found):
        return "product_evidence_refresh"
    if rel in REL_EVIDENCE_OK_CLAIM_BAD or (evidence_found and not claim_found):
        return "full_srt_claim_reextract"
    if rel in REL_JOINT_RESCAN:
        return "joint_raw_rescan"
    return "manual_or_silver_review"


def priority_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    priority = str(row.get("priority") or "P9")
    pnum = {"P0": 0, "P1": 1, "P2": 2}.get(priority, 9)
    return pnum, -int(float(row.get("priority_score") or 0)), pair_id(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repair_queue",
        default="data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.jsonl",
    )
    ap.add_argument(
        "--verifier",
        default="data/final/repaired_v1/proposal_triplet_alignment_llm_repair_p0_v3_withlabel_20260613.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613",
    )
    args = ap.parse_args()

    originals = {pair_id(r): r for r in read_jsonl(args.repair_queue)}
    verifier_rows = list(read_jsonl(args.verifier))

    buckets: dict[str, list[dict[str, Any]]] = {
        "product_evidence_refresh": [],
        "full_srt_claim_reextract": [],
        "joint_raw_rescan": [],
        "manual_or_silver_review": [],
    }
    missing_original: list[str] = []
    for verdict in verifier_rows:
        pid = pair_id(verdict)
        original = originals.get(pid)
        if original is None:
            missing_original.append(pid)
            continue
        task_type = classify(verdict)
        buckets[task_type].append(base_task(original, verdict, task_type))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_type, rows in buckets.items():
        rows.sort(key=priority_rank)
        write_jsonl(out_dir / f"{task_type}.jsonl", rows)

    report = {
        "repair_queue": args.repair_queue,
        "verifier": args.verifier,
        "out_dir": str(out_dir),
        "n_verifier": len(verifier_rows),
        "missing_original": missing_original[:20],
        "buckets": {k: len(v) for k, v in buckets.items()},
        "priority_by_bucket": {
            k: dict(Counter(str(r.get("priority")) for r in v)) for k, v in buckets.items()
        },
        "label_by_bucket": {
            k: dict(Counter(str(r.get("current_y")) for r in v)) for k, v in buckets.items()
        },
        "queue_type_by_bucket": {
            k: dict(Counter(str(r.get("queue_type")) for r in v)) for k, v in buckets.items()
        },
        "next_steps": {
            "product_evidence_refresh": (
                "Rerun target-attribute product evidence extraction from title/params/OCR/detail images; "
                "then verify relation to the existing SRT claim."
            ),
            "full_srt_claim_reextract": (
                "Scan full raw SRT for a minimal continuous claim span about the target attribute; "
                "then send changed propositions to B4 review-label rebuilding."
            ),
            "joint_raw_rescan": (
                "Rerun both Stage B claim extraction and Stage C product evidence extraction from raw material."
            ),
            "manual_or_silver_review": (
                "Keep out of main training unless a later high-confidence provenance check promotes it."
            ),
        },
        "top_examples": {
            k: [
                {
                    "pair_id": r.get("pair_id"),
                    "priority": r.get("priority"),
                    "attribute_id": r.get("attribute_id"),
                    "current_y": r.get("current_y"),
                    "relation_to_claim": (r.get("previous_triplet_verdict") or {}).get("relation_to_claim"),
                    "claim_preview": str(r.get("claim_preview") or "")[:120],
                    "evidence_preview": r.get("current_evidence_preview", [])[:2],
                }
                for r in v[:8]
            ]
            for k, v in buckets.items()
        },
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
