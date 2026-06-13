"""Build a stratified LLM/VLM pilot queue for full-pair reconstruction.

The pilot must diagnose the reconstruction protocol, not cherry-pick easy rows.
It therefore samples across SRT prefilter states, claim states, and categories.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


DEFAULT_QUOTAS = {
    "strong_srt_candidate": 24,
    "weak_srt_candidate": 24,
    "very_weak_srt_candidate": 16,
    "no_srt_candidate": 8,
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def read_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    return {pair_id(r): r for r in read_jsonl(path)}


def parse_quotas(text: str) -> dict[str, int]:
    if not text:
        return dict(DEFAULT_QUOTAS)
    out: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, val = part.split("=", 1)
        out[clean(key)] = int(val)
    return out


def priority_rank(value: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(clean(value), 9)


def selection_score(row: dict[str, Any], pref: dict[str, Any]) -> tuple:
    return (
        priority_rank(row.get("priority")),
        -int(row.get("consumer_mentions_explicit", 0) or 0),
        -int(row.get("consumer_mentions_neg", 0) or 0),
        -int(pref.get("top_score", 0) or 0),
        clean(row.get("category")),
        pair_id(row),
    )


def round_robin_by_category(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    by_cat: dict[str, deque] = defaultdict(deque)
    for row in rows:
        by_cat[clean(row.get("category"))].append(row)
    out: list[dict[str, Any]] = []
    cats = deque(sorted(by_cat))
    while cats and len(out) < cap:
        cat = cats.popleft()
        if by_cat[cat]:
            out.append(by_cat[cat].popleft())
        if by_cat[cat]:
            cats.append(cat)
    return out


def build_item(row: dict[str, Any], pref: dict[str, Any], max_candidates: int) -> dict[str, Any]:
    item = dict(row)
    item["_pilot"] = {
        "pilot_version": "full_pair_llm_pilot_v1_20260614",
        "purpose": "stratified reconstruction protocol diagnosis; not a training dataset",
        "selection_prefilter_state": pref.get("prefilter_state", "missing_prefilter"),
        "selection_top_score": pref.get("top_score", 0),
        "old_label_role": "audit_only_not_selection_target",
    }
    item["srt_prefilter"] = {
        "prefilter_state": pref.get("prefilter_state", "missing_prefilter"),
        "top_score": pref.get("top_score", 0),
        "srt_file_count": pref.get("srt_file_count", 0),
        "claim_candidates": (pref.get("claim_candidates") or [])[:max_candidates],
    }
    return item


def write_markdown(path: str, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair LLM Pilot Queue v1",
        "",
        "This queue is a stratified pilot for reconstruction protocol diagnosis, not a training dataset.",
        "",
        "## Outputs",
        "",
        f"- queue: `{report['out']}`",
        f"- report: `{report['report']}`",
        "",
        "## Summary",
        "",
        f"- rows: `{report['n']}`",
        f"- priority: `{report['priority']}`",
        f"- prefilter state: `{report['prefilter_state']}`",
        f"- claim state: `{report['claim_state']}`",
        f"- queue type: `{report['queue_type']}`",
        f"- category: `{report['category']}`",
        "",
        "## Why This Sampling Matters",
        "",
        "- Strong SRT candidates test whether the new prefilter recovers exact claim spans.",
        "- Weak and very weak candidates test whether lexical overlap is too noisy.",
        "- No-candidate rows test the boundary between true missing claims and schema/comment noise.",
        "- Old labels are retained only for audit and are not exposed as target labels.",
        "",
        "## Examples",
        "",
    ]
    for ex in report.get("examples", []):
        lines.append(
            f"- `{ex['pair_id']}` state={ex['prefilter_state']} claim={ex['claim_state']} "
            f"cat={ex['category']} attr={ex['attribute_name']} top={ex['top_score']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--srt_prefilter", default="data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_LLM_PILOT_QUEUE_20260614.md")
    ap.add_argument("--quotas", default="")
    ap.add_argument("--priorities", default="P0,P1")
    ap.add_argument("--max_srt_candidates", type=int, default=5)
    args = ap.parse_args()

    quotas = parse_quotas(args.quotas)
    priorities = {x.strip() for x in args.priorities.replace(",", " ").split() if x.strip()}
    pref_by_pair = read_by_pair(args.srt_prefilter)
    rows_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_prefilter = 0
    for row in read_jsonl(args.queue):
        if priorities and clean(row.get("priority")) not in priorities:
            continue
        pid = pair_id(row)
        pref = pref_by_pair.get(pid)
        if not pref:
            missing_prefilter += 1
            continue
        state = clean(pref.get("prefilter_state"))
        if state not in quotas:
            continue
        rows_by_state[state].append(row)

    selected: list[dict[str, Any]] = []
    quota_fill = {}
    for state, cap in quotas.items():
        rows = sorted(rows_by_state.get(state, []), key=lambda r: selection_score(r, pref_by_pair[pair_id(r)]))
        picks = round_robin_by_category(rows, cap)
        quota_fill[state] = {"available": len(rows), "selected": len(picks), "quota": cap}
        selected.extend(build_item(row, pref_by_pair[pair_id(row)], args.max_srt_candidates) for row in picks)

    selected.sort(key=lambda r: (
        clean((r.get("srt_prefilter") or {}).get("prefilter_state")),
        priority_rank(r.get("priority")),
        clean(r.get("category")),
        pair_id(r),
    ))
    write_jsonl(args.out, selected)
    report = {
        "queue": args.queue,
        "srt_prefilter": args.srt_prefilter,
        "out": args.out,
        "report": args.report,
        "n": len(selected),
        "quotas": quota_fill,
        "missing_prefilter": missing_prefilter,
        "priority": dict(Counter(clean(r.get("priority")) for r in selected)),
        "prefilter_state": dict(Counter(clean((r.get("srt_prefilter") or {}).get("prefilter_state")) for r in selected)),
        "claim_state": dict(Counter(clean(r.get("claim_state")) for r in selected)),
        "queue_type": dict(Counter(clean(r.get("queue_type")) for r in selected)),
        "category": dict(Counter(clean(r.get("category")) for r in selected)),
        "old_label_state": dict(Counter(clean(r.get("old_label_state")) for r in selected)),
        "examples": [
            {
                "pair_id": pair_id(r),
                "prefilter_state": clean((r.get("srt_prefilter") or {}).get("prefilter_state")),
                "top_score": (r.get("srt_prefilter") or {}).get("top_score"),
                "claim_state": r.get("claim_state"),
                "category": r.get("category"),
                "attribute_name": r.get("attribute_name"),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
