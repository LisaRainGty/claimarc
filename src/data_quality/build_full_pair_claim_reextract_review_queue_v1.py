"""Build a joint-review queue from pair-targeted claim re-extraction results.

`llm_pair_claim_reextract_v1` is a recall layer: it may recover many exact SRT
claim candidates for a claim-missing pair, including noisy adjacent-attribute
claims.  This builder injects those exact candidates back into the full-pair
LLM/VLM reviewer so label construction still happens through the same
claim-evidence-comment audit gate.
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
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid:
            out[pid] = row
    return out


def claim_candidates(result: dict[str, Any], max_claims: int) -> list[dict[str, Any]]:
    claims = []
    for rank, c in enumerate(result.get("claims") or [], 1):
        text = clean(c.get("claim_text"))
        if not text:
            continue
        claims.append({
            "score": max(1, 100 - rank),
            "srt_file": c.get("srt_path") or c.get("srt_file"),
            "start_ts": c.get("start_ts"),
            "end_ts": c.get("end_ts"),
            "text": text,
            "why": {
                "source": ["pair_claim_reextract_v1"],
                "claim_type": [clean(c.get("claim_type"))],
                "confidence": [clean(c.get("confidence"))],
            },
            "_claim_reextract_rank": rank,
        })
        if len(claims) >= max_claims:
            break
    return claims


def build_item(queue_row: dict[str, Any], result: dict[str, Any], max_claims: int) -> dict[str, Any]:
    item = dict(queue_row)
    pid = pair_id(item)
    cands = claim_candidates(result, max_claims)
    item["claim_preview"] = "\n---\n".join(c["text"] for c in cands[:4])
    item["claim_state"] = "claim_reextract_candidates"
    item["claim_segments"] = [
        {
            "claim_id": f"{pid}__claim_reextract_v1_{i}",
            "claim_text": c["text"],
            "clip_id": Path(clean(c.get("srt_file"))).name,
            "start_ts": clean(c.get("start_ts")),
            "end_ts": clean(c.get("end_ts")),
            "_seed_from_claim_reextract": True,
        }
        for i, c in enumerate(cands, 1)
    ]
    pref = dict(item.get("srt_prefilter") or {})
    old = pref.get("claim_candidates") or []
    pref["prefilter_state"] = "claim_reextract_seeded"
    pref["top_score"] = max([int(c.get("score", 0) or 0) for c in cands] + [int(pref.get("top_score", 0) or 0)])
    pref["claim_candidates"] = cands + old[:max(0, max_claims - len(cands))]
    item["srt_prefilter"] = pref
    item["queue_type"] = "claim_reextract_joint_review"
    item["_claim_reextract_review"] = {
        "review_queue_version": "full_pair_claim_reextract_review_v1_20260614",
        "claim_reextract_status": result.get("status"),
        "raw_claim_count": len(result.get("claims") or []),
        "seeded_claim_count": len(cands),
        "old_label_role": "audit_only_not_target",
        "purpose": "joint claim/evidence/comment reconstruction after exact SRT claim recall",
    }
    return item


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair Claim-Reextract Joint Review Queue v1",
        "",
        "This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.",
        "It is a recall-to-review bridge, not a promoted training dataset.",
        "",
        "## Inputs",
        "",
        f"- queue: `{report['queue']}`",
        f"- claim reextract: `{report['claim_reextract']}`",
        "",
        "## Outputs",
        "",
        f"- joint review queue: `{report['out']}`",
        f"- report: `{report['report']}`",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "claim_reextract_rows",
        "selected_rows",
        "claim_found_pairs",
        "no_claim_pairs",
        "seeded_claim_count_bucket",
        "category",
    ]:
        lines.append(f"- `{key}`: `{report.get(key)}`")
    lines.extend(["", "## Examples", ""])
    for ex in report.get("examples", []):
        lines.append(
            f"- `{ex['pair_id']}` cat={ex['category']} attr={ex['attribute_name']} "
            f"raw_claims={ex['raw_claim_count']} seeded={ex['seeded_claim_count']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 10:
        return "5-10"
    return "11+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.jsonl")
    ap.add_argument("--claim_reextract", default="data/final/repaired_v1/full_pair_claim_reextract_pilot44_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_CLAIM_REEXTRACT_JOINT_REVIEW_QUEUE_20260614.md")
    ap.add_argument("--max_claims", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sort_by", choices=["more_claims", "fewer_claims"], default="more_claims")
    args = ap.parse_args()

    queue_by_pair = read_by_pair(args.queue)
    results = list(read_jsonl(args.claim_reextract))
    selected = []
    no_claim_pairs = 0
    missing_queue = 0
    for result in results:
        if clean(result.get("status")) != "claim_found" or not result.get("claims"):
            no_claim_pairs += 1
            continue
        row = queue_by_pair.get(pair_id(result))
        if not row:
            missing_queue += 1
            continue
        selected.append(build_item(row, result, args.max_claims))

    if args.sort_by == "fewer_claims":
        selected.sort(key=lambda r: (
            int((r.get("_claim_reextract_review") or {}).get("seeded_claim_count", 0) or 0),
            clean(r.get("category")),
            pair_id(r),
        ))
    else:
        selected.sort(key=lambda r: (
            -int((r.get("_claim_reextract_review") or {}).get("seeded_claim_count", 0) or 0),
            clean(r.get("category")),
            pair_id(r),
        ))
    if args.limit and args.limit > 0:
        selected = selected[:args.limit]
    write_jsonl(args.out, selected)
    report = {
        "queue": args.queue,
        "claim_reextract": args.claim_reextract,
        "out": args.out,
        "report": args.report,
        "markdown": args.markdown,
        "claim_reextract_rows": len(results),
        "selected_rows": len(selected),
        "claim_found_pairs": len([r for r in results if clean(r.get("status")) == "claim_found"]),
        "no_claim_pairs": no_claim_pairs,
        "missing_queue_rows": missing_queue,
        "sort_by": args.sort_by,
        "seeded_claim_count_bucket": dict(Counter(count_bucket(int((r.get("_claim_reextract_review") or {}).get("seeded_claim_count", 0) or 0)) for r in selected)),
        "category": dict(Counter(clean(r.get("category")) for r in selected)),
        "examples": [
            {
                "pair_id": pair_id(r),
                "category": r.get("category"),
                "attribute_name": r.get("attribute_name"),
                "raw_claim_count": (r.get("_claim_reextract_review") or {}).get("raw_claim_count"),
                "seeded_claim_count": (r.get("_claim_reextract_review") or {}).get("seeded_claim_count"),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
