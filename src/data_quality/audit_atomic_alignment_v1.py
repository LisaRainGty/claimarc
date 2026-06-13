"""Audit suspicious atomic claim/comment alignments."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common.io_utils import bigram_jaccard, normalize, read_jsonl, write_json


def _overlap(claim: str, text: str) -> dict:
    cn = normalize(claim)
    tn = normalize(text)
    chars = set(cn) & set(tn)
    return {
        "char_overlap": len(chars),
        "bigram_jaccard": round(bigram_jaccard(cn, tn), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_examples", type=int, default=80)
    args = ap.parse_args()

    rows = list(read_jsonl(args.atomic_records))
    suspicious = []
    rel_counter = Counter()
    for r in rows:
        claim = (r.get("claim") or {}).get("passage", "")
        for c in r.get("reviews", []) or []:
            if int(c.get("y_supportability", 0) or 0) != 1:
                continue
            rel = str(c.get("relation") or "unclear")
            rel_counter[rel] += 1
            ov = _overlap(claim, c.get("text", ""))
            if ov["char_overlap"] == 0 or ov["bigram_jaccard"] < 0.08:
                suspicious.append({
                    "atomic_id": r.get("atomic_id"),
                    "pair_id": r.get("pair_id"),
                    "attribute_name": r.get("attribute_name"),
                    "claim": claim,
                    "comment": c.get("text", ""),
                    "relation": rel,
                    "rationale": c.get("alignment_rationale", ""),
                    **ov,
                })
    suspicious.sort(key=lambda x: (x["bigram_jaccard"], x["char_overlap"], x["atomic_id"]))
    report = {
        "atomic_records": args.atomic_records,
        "n_records": len(rows),
        "aligned_relation_counts": dict(rel_counter),
        "n_suspicious_low_overlap": len(suspicious),
        "examples": suspicious[: args.max_examples],
    }
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
