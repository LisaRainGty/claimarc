"""Audit the full-pair reconstruction queue without filtering rows.

This is a measurement-audit companion for `build_full_pair_reconstruction_queue_v1`.
It summarizes coverage, missingness combinations, comment provenance, and
service-like commercial-promise boundaries before any LLM/VLM relabeling.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json


TRANSACTION_TERMS = (
    "保价",
    "价格",
    "赠品",
    "赠送",
    "发货",
    "物流",
    "售后",
    "客服",
    "退货",
    "退款",
    "换货",
    "质保",
    "保修",
    "承诺",
)

COMMENT_CLAIM_CUES = ("主播", "直播", "宣传", "广告", "说是", "说的", "承诺", "页面", "详情")


def clean(value: Any) -> str:
    return str(value or "").strip()


def top(counter: Counter, n: int = 20) -> dict[str, int]:
    return dict(counter.most_common(n))


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def quality_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    mentions = row.get("consumer_mentions") or []
    attr = clean(row.get("attribute_name"))
    if not clean(row.get("claim_preview")):
        flags.append("no_claim_preview")
    if not row.get("current_evidence_preview"):
        flags.append("no_current_evidence_preview")
    if not any(m.get("polarity") == "neg" for m in mentions):
        flags.append("no_negative_compact_mention")
    if any(str(m.get("type")) == "service" for m in mentions):
        flags.append("service_comment_present")
    if mentions and all(str(m.get("type")) == "service" for m in mentions):
        flags.append("all_compact_mentions_service")
    if has_any(attr, TRANSACTION_TERMS):
        flags.append("commercial_promise_attribute")
    if any(has_any(clean(m.get("evidence_span")), COMMENT_CLAIM_CUES) for m in mentions):
        flags.append("comment_has_claim_cue")
    return flags


def exemplar(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "priority": row.get("priority"),
        "queue_type": row.get("queue_type"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "claim_state": row.get("claim_state"),
        "evidence_state": row.get("evidence_state"),
        "old_label_state": row.get("old_label_state"),
        "old_y": row.get("old_y"),
        "consumer_mentions_total": row.get("consumer_mentions_total"),
        "consumer_mentions_neg": row.get("consumer_mentions_neg"),
        "claim_preview": clean(row.get("claim_preview"))[:160],
        "mentions": [
            {
                "text": clean(m.get("evidence_span"))[:120],
                "polarity": m.get("polarity"),
                "type": m.get("type"),
                "explicit_fact_hit": m.get("explicit_fact_hit"),
            }
            for m in (row.get("consumer_mentions") or [])[:3]
        ],
    }


def write_markdown(path: str, report: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Pair Reconstruction Queue Audit v1",
        "",
        "This report audits the reconstruction queue before LLM/VLM relabeling. It does not remove rows.",
        "",
        "## Summary",
        "",
        f"- queue: `{report['queue']}`",
        f"- rows: `{report['n']}`",
        f"- output json: `{report['out_json']}`",
        "",
        "## Core Counts",
        "",
        "| field | top counts |",
        "|---|---|",
        f"| priority | `{report['priority']}` |",
        f"| queue type | `{report['queue_type']}` |",
        f"| claim state | `{report['claim_state']}` |",
        f"| evidence state | `{report['evidence_state']}` |",
        f"| old label state | `{report['old_label_state']}` |",
        f"| category | `{report['category']}` |",
        "",
        "## Comment Provenance",
        "",
        f"- compact mention type counts: `{report['compact_mention_type']}`",
        f"- rows with service comments in compact mentions: `{report['quality_flag_counts'].get('service_comment_present', 0)}`",
        f"- rows where all compact mentions are service typed: `{report['quality_flag_counts'].get('all_compact_mentions_service', 0)}`",
        f"- commercial-promise attributes: `{report['quality_flag_counts'].get('commercial_promise_attribute', 0)}`",
        f"- comments with explicit claim cues: `{report['quality_flag_counts'].get('comment_has_claim_cue', 0)}`",
        "",
        "## Missingness Combos",
        "",
        "| combo | count |",
        "|---|---:|",
    ]
    for combo, cnt in report["missingness_combo_top"].items():
        lines.append(f"| `{combo}` | {cnt} |")
    lines.extend([
        "",
        "## Important Interpretation",
        "",
        "- Large missing-claim or missing-evidence groups are reconstruction targets, not negative labels.",
        "- Old labels are audit-only until a recovered claim is compared with aligned consumer comments.",
        "- Service-like comments remain in the queue, but they cannot trigger a positive label unless they align to the same commercial-promise claim.",
        "",
        "## Example Flags",
        "",
    ])
    for name, examples in report["examples"].items():
        lines.extend([f"### {name}", ""])
        for ex in examples:
            lines.append(
                f"- `{ex['pair_id']}` old_y={ex['old_y']} claim={ex['claim_state']} "
                f"evidence={ex['evidence_state']} label={ex['old_label_state']} "
                f"claim_preview={json.dumps(ex['claim_preview'], ensure_ascii=False)}"
            )
            for m in ex.get("mentions", [])[:2]:
                lines.append(
                    f"  - [{m.get('type')}/{m.get('polarity')}] "
                    f"{json.dumps(m.get('text'), ensure_ascii=False)}"
                )
        lines.append("")
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--out_json", default="data/final/repaired_v1/full_pair_reconstruction_queue_audit_v1_20260614.json")
    ap.add_argument("--out_md", default="docs/FULL_PAIR_RECONSTRUCTION_QUEUE_AUDIT_20260614.md")
    ap.add_argument("--example_cap", type=int, default=8)
    args = ap.parse_args()

    rows = list(read_jsonl(args.queue))
    counters: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        counters["priority"][clean(row.get("priority"))] += 1
        counters["queue_type"][clean(row.get("queue_type"))] += 1
        counters["claim_state"][clean(row.get("claim_state"))] += 1
        counters["evidence_state"][clean(row.get("evidence_state"))] += 1
        counters["old_label_state"][clean(row.get("old_label_state"))] += 1
        counters["category"][clean(row.get("category"))] += 1
        counters["expected_value_type"][clean(row.get("expected_value_type"))] += 1
        combo = "|".join([
            clean(row.get("claim_state")),
            clean(row.get("evidence_state")),
            clean(row.get("old_label_state")),
        ])
        counters["missingness_combo"] [combo] += 1
        for src in row.get("target_sources") or []:
            counters["target_sources"][clean(src)] += 1
        for m in row.get("consumer_mentions") or []:
            counters["compact_mention_type"][clean(m.get("type")) or "unknown"] += 1
        flags = quality_flags(row)
        for flag in flags:
            counters["quality_flag_counts"][flag] += 1

        if (
            row.get("claim_state") == "claim_missing"
            and row.get("old_label_state") == "label_negative_no_aligned_review"
            and int(row.get("consumer_mentions_neg", 0) or 0) > 0
            and len(examples["missing_claim_old_negative_with_neg_comments"]) < args.example_cap
        ):
            examples["missing_claim_old_negative_with_neg_comments"].append(exemplar(row))
        if "service_comment_present" in flags and len(examples["service_comment_boundary"]) < args.example_cap:
            examples["service_comment_boundary"].append(exemplar(row))
        if "comment_has_claim_cue" in flags and len(examples["explicit_claim_cue_comments"]) < args.example_cap:
            examples["explicit_claim_cue_comments"].append(exemplar(row))
        if (
            row.get("queue_type") == "label_rebuild_existing_triplet"
            and len(examples["existing_triplet_label_rebuild"]) < args.example_cap
        ):
            examples["existing_triplet_label_rebuild"].append(exemplar(row))

    report = {
        "queue": args.queue,
        "out_json": args.out_json,
        "out_md": args.out_md,
        "n": len(rows),
        "priority": top(counters["priority"]),
        "queue_type": top(counters["queue_type"]),
        "claim_state": top(counters["claim_state"]),
        "evidence_state": top(counters["evidence_state"]),
        "old_label_state": top(counters["old_label_state"]),
        "category": top(counters["category"]),
        "expected_value_type": top(counters["expected_value_type"]),
        "target_sources": top(counters["target_sources"]),
        "compact_mention_type": top(counters["compact_mention_type"]),
        "quality_flag_counts": top(counters["quality_flag_counts"]),
        "missingness_combo_top": top(counters["missingness_combo"], 30),
        "examples": dict(examples),
    }
    write_json(args.out_json, report)
    write_markdown(args.out_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
