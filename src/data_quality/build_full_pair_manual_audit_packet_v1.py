"""Build a CSV packet for manual audit of full-pair reconstruction rows.

The packet is designed for the proposal-level question:

Does this product-attribute row contain a recoverable streamer claim, product
evidence, and consumer comments that can be judged against the same claim?

It can be used before LLM reconstruction (queue-only inspection) or after LLM
reconstruction (with review and automatic audit flags joined in).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def read_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    return {pair_id(row): row for row in read_jsonl(path)}


def read_flags_by_pair(path: str | Path) -> dict[str, list[dict[str, str]]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for row in read_jsonl(path):
        out[pair_id(row)] = row.get("flags") or []
    return out


def compact_json(value: Any, max_chars: int = 1400) -> str:
    text = json.dumps(value or [], ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def format_srt_candidates(row: dict[str, Any], max_candidates: int) -> str:
    pref = row.get("srt_prefilter") or {}
    lines = []
    for i, cand in enumerate((pref.get("claim_candidates") or [])[:max_candidates], 1):
        why = cand.get("why") or {}
        hits = []
        for key in ("attribute_hits", "value_hits", "comment_hits"):
            vals = [clean(x) for x in (why.get(key) or []) if clean(x)]
            if vals:
                hits.append(f"{key}={','.join(vals[:5])}")
        lines.append(
            f"{i}. score={cand.get('score')} "
            f"{Path(clean(cand.get('srt_file'))).name} "
            f"{clean(cand.get('start_ts'))}-{clean(cand.get('end_ts'))} "
            f"{'; '.join(hits)} :: {clean(cand.get('text'))}"
        )
    return "\n".join(lines)


def format_comments(row: dict[str, Any], max_comments: int) -> str:
    lines = []
    for i, item in enumerate((row.get("consumer_mentions") or [])[:max_comments], 1):
        lines.append(
            f"{i}. [{item.get('polarity')}/{item.get('mention_strength')}/"
            f"{'explicit' if item.get('explicit_fact_hit') else 'implicit'}] "
            f"{clean(item.get('evidence_span'))}"
        )
    return "\n".join(lines)


def flag_codes(flags: list[dict[str, str]]) -> str:
    return "; ".join(
        f"{clean(f.get('severity'))}:{clean(f.get('code'))}"
        + (f"({clean(f.get('detail'))})" if clean(f.get("detail")) else "")
        for f in flags
    )


def audit_questions() -> str:
    return "\n".join([
        "1. claim 是否为主播/SRT 中针对该属性的最小连续原话？",
        "2. product evidence 是否来自标题/参数/详情图 OCR/VLM，而非评论或主播话术？",
        "3. 评论是否讨论同一具体命题，并支持/反驳该 claim？",
        "4. 若商品证据与 claim 矛盾但评论未反驳，是否只记录机制状态而不置正类？",
    ])


def build_row(
    queue_row: dict[str, Any],
    review: dict[str, Any] | None,
    flags: list[dict[str, str]],
    max_srt_candidates: int,
    max_comments: int,
) -> dict[str, Any]:
    pref = queue_row.get("srt_prefilter") or {}
    review = review or {}
    return {
        "pair_id": pair_id(queue_row),
        "priority": queue_row.get("priority"),
        "queue_type": queue_row.get("queue_type"),
        "prefilter_state": pref.get("prefilter_state"),
        "prefilter_top_score": pref.get("top_score"),
        "category": queue_row.get("category"),
        "subcategory": queue_row.get("subcategory"),
        "product_id": queue_row.get("product_id"),
        "room_id": queue_row.get("room_id"),
        "attribute_id": queue_row.get("attribute_id"),
        "attribute_name": queue_row.get("attribute_name"),
        "product_title": queue_row.get("product_title"),
        "claim_state": queue_row.get("claim_state"),
        "evidence_state": queue_row.get("evidence_state"),
        "old_label_state": queue_row.get("old_label_state"),
        "current_claim_preview": clean(queue_row.get("claim_preview")),
        "current_evidence_preview": compact_json(queue_row.get("current_evidence_preview"), 1000),
        "srt_candidates": format_srt_candidates(queue_row, max_srt_candidates),
        "consumer_mentions": format_comments(queue_row, max_comments),
        "llm_claim_found": review.get("claim_found", ""),
        "llm_claim_text": review.get("claim_text", ""),
        "llm_claim_source": review.get("claim_source", ""),
        "llm_product_evidence_found": review.get("product_evidence_found", ""),
        "llm_evidence_source_type": review.get("evidence_source_type", ""),
        "llm_evidence_text": review.get("evidence_text", ""),
        "llm_claim_evidence_relation": review.get("claim_evidence_relation", ""),
        "llm_comment_judgments": compact_json(review.get("comment_judgments"), 1400),
        "llm_new_y": review.get("new_y", ""),
        "llm_confidence": review.get("confidence", ""),
        "llm_action": review.get("action", ""),
        "audit_flags": flag_codes(flags),
        "audit_questions": audit_questions(),
        "manual_claim_source_valid": "",
        "manual_claim_attribute_specific": "",
        "manual_product_evidence_valid": "",
        "manual_comments_same_claim": "",
        "manual_new_y": "",
        "manual_decision": "",
        "manual_notes": "",
    }


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair Manual Audit Packet v1",
        "",
        "This packet is for manual inspection of the full-pair reconstruction pilot.",
        "It is not a training dataset and should not be used to select only easy rows.",
        "",
        "## Outputs",
        "",
        f"- csv: `{report['out']}`",
        f"- report: `{report['report']}`",
        "",
        "## Summary",
        "",
        f"- rows: `{report['rows']}`",
        f"- reviews joined: `{report['reviews_joined']}`",
        f"- rows with audit flags: `{report['rows_with_audit_flags']}`",
        f"- prefilter state: `{report['prefilter_state']}`",
        f"- queue type: `{report['queue_type']}`",
        f"- claim state: `{report['claim_state']}`",
        f"- category: `{report['category']}`",
        "",
        "## Manual Columns",
        "",
        "- `manual_claim_source_valid`: whether the claim is traceable to SRT.",
        "- `manual_claim_attribute_specific`: whether the claim is about the target attribute.",
        "- `manual_product_evidence_valid`: whether evidence comes from product-side material.",
        "- `manual_comments_same_claim`: whether comments discuss the same atomic claim.",
        "- `manual_new_y`: final human label under the consumer-perception definition.",
        "- `manual_decision`: promote, silver, rerun_claim, rerun_evidence, rerun_joint, or reject_out_of_scope.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl")
    ap.add_argument("--audit_flags", default="data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_manual_audit_packet_v1_20260614.csv")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_manual_audit_packet_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_MANUAL_AUDIT_PACKET_20260614.md")
    ap.add_argument("--max_srt_candidates", type=int, default=5)
    ap.add_argument("--max_comments", type=int, default=8)
    args = ap.parse_args()

    reviews = read_by_pair(args.reviews)
    flags = read_flags_by_pair(args.audit_flags)
    rows = []
    for queue_row in read_jsonl(args.queue):
        rows.append(
            build_row(
                queue_row,
                reviews.get(pair_id(queue_row)),
                flags.get(pair_id(queue_row), []),
                args.max_srt_candidates,
                args.max_comments,
            )
        )

    rows.sort(key=lambda r: (
        clean(r.get("prefilter_state")),
        clean(r.get("category")),
        clean(r.get("attribute_name")),
        clean(r.get("pair_id")),
    ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "queue": args.queue,
        "reviews": args.reviews,
        "audit_flags": args.audit_flags,
        "out": args.out,
        "report": args.report,
        "rows": len(rows),
        "reviews_joined": sum(1 for r in rows if clean(r.get("llm_claim_text")) or clean(r.get("llm_action"))),
        "rows_with_audit_flags": sum(1 for r in rows if clean(r.get("audit_flags"))),
        "prefilter_state": dict(Counter(clean(r.get("prefilter_state")) for r in rows)),
        "queue_type": dict(Counter(clean(r.get("queue_type")) for r in rows)),
        "claim_state": dict(Counter(clean(r.get("claim_state")) for r in rows)),
        "category": dict(Counter(clean(r.get("category")) for r in rows)),
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
