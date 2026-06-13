"""Build proposal-faithful negative-control review queues.

The claim-repair queues are intentionally positive-risk heavy because they
focus on consumer refutations.  For a usable perception classifier, we also
need natural negatives: rows where a livestream claim exists and consumers
either support it or discuss the attribute without contradiction.

This builder samples from the full reconstruction queue without changing
labels.  The same full-pair LLM reviewer later rebuilds the final label.
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


def read_pair_ids(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        for row in read_jsonl(p):
            pid = pair_id(row)
            if pid:
                out.add(pid)
    return out


def score(row: dict[str, Any]) -> tuple:
    claim_rank = {
        "claim_present_specific": 0,
        "claim_present_review_needed": 1,
    }
    evidence_rank = {
        "evidence_multi_source": 0,
        "evidence_single_source": 1,
        "evidence_missing": 2,
    }
    label_rank = {
        "label_negative_claim_aligned_nonneg": 0,
        "label_negative_no_aligned_review": 1,
    }
    return (
        label_rank.get(clean(row.get("old_label_state")), 9),
        evidence_rank.get(clean(row.get("evidence_state")), 9),
        claim_rank.get(clean(row.get("claim_state")), 9),
        -int(row.get("consumer_mentions_total", 0) or 0),
        -int(row.get("consumer_mentions_explicit", 0) or 0),
        clean(row.get("category")),
        pair_id(row),
    )


def build_item(row: dict[str, Any], rank: int) -> dict[str, Any]:
    item = dict(row)
    item["queue_type"] = "negative_control_joint_review"
    item["priority"] = "N0"
    item["_negative_control"] = {
        "queue_version": "full_pair_negative_control_v1_20260614",
        "source_rank": rank,
        "old_label_state": row.get("old_label_state"),
        "purpose": (
            "recover proposal-faithful negatives where a streamer claim exists "
            "but aligned consumers do not refute the claim"
        ),
    }
    return item


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair Negative-Control Queue v1",
        "",
        "This queue samples claim-present rows likely to become natural negatives.",
        "It does not assign labels; labels are rebuilt by the full-pair reviewer.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "input_rows",
        "selected_rows",
        "excluded_pair_ids",
        "excluded_rows",
        "old_label_state",
        "claim_state",
        "evidence_state",
        "category",
    ]:
        lines.append(f"- `{key}`: `{report.get(key)}`")
    lines.extend(["", "## Examples", ""])
    for ex in report.get("examples", []):
        lines.append(
            f"- `{ex['pair_id']}` cat={ex['category']} attr={ex['attribute_name']} "
            f"old={ex['old_label_state']} claim={ex['claim_state']} evidence={ex['evidence_state']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_negative_control_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_negative_control_queue_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_NEGATIVE_CONTROL_QUEUE_20260614.md")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--old_label_states", default="label_negative_claim_aligned_nonneg")
    ap.add_argument("--claim_states", default="claim_present_specific,claim_present_review_needed")
    ap.add_argument("--evidence_states", default="evidence_multi_source,evidence_single_source")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    old_label_states = {x.strip() for x in args.old_label_states.replace(",", " ").split() if x.strip()}
    claim_states = {x.strip() for x in args.claim_states.replace(",", " ").split() if x.strip()}
    evidence_states = {x.strip() for x in args.evidence_states.replace(",", " ").split() if x.strip()}
    excluded = read_pair_ids(args.exclude)
    selected: list[dict[str, Any]] = []
    excluded_rows = 0
    input_rows = 0
    for rank, row in enumerate(read_jsonl(args.queue), 1):
        input_rows += 1
        if pair_id(row) in excluded:
            excluded_rows += 1
            continue
        if old_label_states and clean(row.get("old_label_state")) not in old_label_states:
            continue
        if claim_states and clean(row.get("claim_state")) not in claim_states:
            continue
        if evidence_states and clean(row.get("evidence_state")) not in evidence_states:
            continue
        selected.append(build_item(row, rank))

    selected.sort(key=score)
    if args.limit > 0:
        selected = selected[:args.limit]
    write_jsonl(args.out, selected)
    report = {
        "queue": args.queue,
        "out": args.out,
        "report": args.report,
        "markdown": args.markdown,
        "input_rows": input_rows,
        "selected_rows": len(selected),
        "limit": args.limit,
        "filters": {
            "old_label_states": sorted(old_label_states),
            "claim_states": sorted(claim_states),
            "evidence_states": sorted(evidence_states),
        },
        "excluded_pair_ids": len(excluded),
        "excluded_rows": excluded_rows,
        "old_label_state": dict(Counter(clean(r.get("old_label_state")) for r in selected)),
        "claim_state": dict(Counter(clean(r.get("claim_state")) for r in selected)),
        "evidence_state": dict(Counter(clean(r.get("evidence_state")) for r in selected)),
        "category": dict(Counter(clean(r.get("category")) for r in selected)),
        "examples": [
            {
                "pair_id": pair_id(r),
                "category": r.get("category"),
                "attribute_name": r.get("attribute_name"),
                "old_label_state": r.get("old_label_state"),
                "claim_state": r.get("claim_state"),
                "evidence_state": r.get("evidence_state"),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
