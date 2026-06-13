"""Build a reproducible Stage-C/VLM evidence repair queue.

This queue is for rows where the LLM reconstruction has recovered a streamer
claim and claim-aligned refuting consumer comments, but product-side evidence is
missing or still insufficient.  These rows are not removed, relabeled, or
promoted by this script; they are routed back to product evidence extraction so
the final dataset can keep the proposal's full claim-evidence-comment chain.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def read_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid:
            out[pid] = row
    return out


def aligned_refute_count(review: dict[str, Any]) -> int:
    n = 0
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if item.get("aligned_to_claim") and clean(item.get("relation")) == "refute":
            n += 1
    return n


def needs_evidence_repair(review: dict[str, Any]) -> bool:
    if review.get("__error__"):
        return False
    if not review.get("claim_found"):
        return False
    if aligned_refute_count(review) <= 0:
        return False
    if int(review.get("new_y", 0) or 0) != 1:
        return False
    if not review.get("product_evidence_found"):
        return True
    return clean(review.get("claim_evidence_relation")) in {"", "insufficient"}


def image_count(row: dict[str, Any]) -> int:
    images = row.get("detail_images") or []
    if isinstance(images, list):
        return len(images)
    return 0


def evidence_preview_from_review(review: dict[str, Any]) -> list[dict[str, str]]:
    text = clean(review.get("evidence_text"))
    if not text:
        return []
    return [{
        "source": clean(review.get("evidence_source_type")) or "review_seed",
        "key": clean(review.get("evidence_source")),
        "text": text,
        "_seed_from_llm_review": "true",
    }]


def build_item(queue_row: dict[str, Any], review: dict[str, Any], source_rank: int, max_srt_candidates: int) -> dict[str, Any]:
    item = dict(queue_row)
    pid = pair_id(queue_row)
    pref = item.get("srt_prefilter") or {}
    if isinstance(pref.get("claim_candidates"), list):
        pref = dict(pref)
        pref["claim_candidates"] = pref["claim_candidates"][:max_srt_candidates]
        item["srt_prefilter"] = pref

    claim_text = clean(review.get("claim_text"))
    if claim_text:
        item["claim_preview"] = claim_text
        item["claim_state"] = "claim_seeded_from_llm_review"
        item["claim_segments"] = [{
            "claim_id": f"{pid}__evidence_repair_seed",
            "claim_text": claim_text,
            "clip_id": clean(review.get("claim_source")),
            "start_ts": clean(review.get("claim_timestamp")),
            "end_ts": "",
            "_seed_from_llm_review": True,
        }]

    seed_evidence = evidence_preview_from_review(review)
    if seed_evidence:
        item["current_evidence_preview"] = seed_evidence
    item["queue_type"] = "evidence_vlm_repair_from_llm_review"
    item["target_sources"] = ["detail_image_vlm", "detail_image_ocr", "params", "product_title"]
    item["_evidence_repair"] = {
        "repair_queue_version": "full_pair_evidence_repair_v1_20260614",
        "source_rank": source_rank,
        "source_review_action": clean(review.get("action")),
        "source_claim_evidence_relation": clean(review.get("claim_evidence_relation")),
        "source_product_evidence_found": bool(review.get("product_evidence_found")),
        "source_evidence_text": clean(review.get("evidence_text")),
        "aligned_refute_count": aligned_refute_count(review),
        "old_label_role": "audit_only_not_target",
        "purpose": "repair product-side evidence while preserving recovered claim and comment-refute label audit",
    }
    return item


def selection_key(item: dict[str, Any]) -> tuple:
    repair = item.get("_evidence_repair") or {}
    missing_first = 0 if not repair.get("source_product_evidence_found") else 1
    return (
        missing_first,
        -int(repair.get("aligned_refute_count", 0) or 0),
        -int(item.get("consumer_mentions_explicit", 0) or 0),
        clean(item.get("category")),
        pair_id(item),
    )


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair Evidence Repair Queue v1",
        "",
        "This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.",
        "It does not drop hard rows, alter labels, or promote rows into the main benchmark.",
        "",
        "## Inputs",
        "",
        f"- queue: `{report['queue']}`",
        f"- reviews: `{report['reviews']}`",
        "",
        "## Outputs",
        "",
        f"- repair queue: `{report['out']}`",
        f"- report: `{report['report']}`",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "review_rows",
        "selected_rows",
        "limit",
        "missing_queue_rows",
        "skipped_no_detail_images",
        "source_claim_evidence_relation",
        "source_product_evidence_found",
        "category",
        "attribute_objectivity",
        "image_count_bucket",
    ]:
        lines.append(f"- `{key}`: `{report.get(key)}`")
    lines.extend(["", "## Examples", ""])
    for ex in report.get("examples", []):
        lines.append(
            f"- `{ex['pair_id']}` cat={ex['category']} attr={ex['attribute_name']} "
            f"relation={ex['relation']} evidence_found={ex['evidence_found']} images={ex['image_count']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def bucket_image_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 4:
        return "1-4"
    if n <= 12:
        return "5-12"
    return "13+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_EVIDENCE_REPAIR_QUEUE_20260614.md")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--require_detail_images", action="store_true")
    ap.add_argument("--max_srt_candidates", type=int, default=8)
    args = ap.parse_args()

    queue_by_pair = read_by_pair(args.queue)
    selected: list[dict[str, Any]] = []
    missing_queue_rows = 0
    skipped_no_detail_images = 0
    review_rows = 0

    for rank, review in enumerate(read_jsonl(args.reviews), 1):
        review_rows += 1
        if not needs_evidence_repair(review):
            continue
        pid = pair_id(review)
        queue_row = queue_by_pair.get(pid)
        if not queue_row:
            missing_queue_rows += 1
            continue
        if args.require_detail_images and image_count(queue_row) == 0:
            skipped_no_detail_images += 1
            continue
        selected.append(build_item(queue_row, review, rank, args.max_srt_candidates))

    selected.sort(key=selection_key)
    if args.limit and args.limit > 0:
        selected = selected[:args.limit]

    write_jsonl(args.out, selected)
    report = {
        "queue": args.queue,
        "reviews": args.reviews,
        "out": args.out,
        "report": args.report,
        "markdown": args.markdown,
        "review_rows": review_rows,
        "selected_rows": len(selected),
        "limit": args.limit,
        "missing_queue_rows": missing_queue_rows,
        "skipped_no_detail_images": skipped_no_detail_images,
        "source_claim_evidence_relation": dict(Counter(clean((r.get("_evidence_repair") or {}).get("source_claim_evidence_relation")) for r in selected)),
        "source_product_evidence_found": dict(Counter(str(bool((r.get("_evidence_repair") or {}).get("source_product_evidence_found"))) for r in selected)),
        "category": dict(Counter(clean(r.get("category")) for r in selected)),
        "attribute_objectivity": dict(Counter(clean(r.get("attribute_objectivity")) for r in selected)),
        "image_count_bucket": dict(Counter(bucket_image_count(image_count(r)) for r in selected)),
        "examples": [
            {
                "pair_id": pair_id(r),
                "category": r.get("category"),
                "attribute_name": r.get("attribute_name"),
                "relation": (r.get("_evidence_repair") or {}).get("source_claim_evidence_relation"),
                "evidence_found": (r.get("_evidence_repair") or {}).get("source_product_evidence_found"),
                "image_count": image_count(r),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
