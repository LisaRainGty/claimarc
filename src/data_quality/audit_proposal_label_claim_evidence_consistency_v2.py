"""Audit proposal-faithful label/claim/evidence consistency.

This audit is designed for `build_stateful_proposal_dataset_v2` outputs.  It
checks whether labels still follow the proposal logic:

- positive labels require aligned consumer refutation of a repaired claim;
- objective product-evidence contradiction alone is not a positive label;
- missing claim/evidence rows are preserved as repair/silver states;
- dedupe or promotion state must not overwrite the perception label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def rel_counts(row: dict[str, Any]) -> Counter:
    audit = row.get("proposal_label_audit") or row.get("label_audit") or {}
    return Counter(audit.get("comment_relation_counts") or {})


def state(row: dict[str, Any]) -> str:
    audit = row.get("proposal_label_audit") or row.get("label_audit") or {}
    return clean(audit.get("promotion_state"))


def claim_evidence_relation(row: dict[str, Any]) -> str:
    audit = row.get("proposal_label_audit") or {}
    return clean(audit.get("claim_evidence_relation"))


def y_value(row: dict[str, Any]) -> int | None:
    val = row.get("y_perception", row.get("y"))
    text = "" if val is None else str(val).strip()
    if text in {"", "None", "null"}:
        return None
    try:
        return int(val)
    except Exception:
        return None


def example(row: dict[str, Any], reason: str) -> dict[str, Any]:
    claim = row.get("claim") or {}
    audit = row.get("proposal_label_audit") or row.get("label_audit") or {}
    return {
        "pair_id": pair_id(row),
        "reason": reason,
        "category": row.get("category"),
        "attribute_name": row.get("attribute_name"),
        "state": state(row),
        "y_perception": row.get("y_perception", row.get("y")),
        "sample_role": row.get("sample_role"),
        "claim": clean(claim.get("passage"))[:120],
        "relation_counts": audit.get("comment_relation_counts"),
        "claim_evidence_relation": audit.get("claim_evidence_relation"),
    }


def add_issue(issues: dict[str, list[dict[str, Any]]], key: str, row: dict[str, Any], cap: int) -> None:
    if len(issues[key]) < cap:
        issues[key].append(example(row, key))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--markdown", default="")
    ap.add_argument("--example_cap", type=int, default=25)
    args = ap.parse_args()

    rows = list(read_jsonl(args.dataset))
    issues: dict[str, list[dict[str, Any]]] = {
        "positive_without_refute": [],
        "positive_without_claim": [],
        "negative_or_unobserved_with_refute": [],
        "objective_contradiction_without_refute": [],
        "silver_refute_not_positive": [],
        "claim_family_changed_label_risk": [],
    }
    counters = Counter()
    by_state = Counter()
    by_role = Counter()
    y_counts = Counter()
    for row in rows:
        y = y_value(row)
        rel = rel_counts(row)
        st = state(row)
        review = row.get("_llm_review") or {}
        by_state[st] += 1
        by_role[clean(row.get("sample_role"))] += 1
        y_counts[str(y)] += 1
        if y == 1 and rel.get("refute", 0) <= 0:
            add_issue(issues, "positive_without_refute", row, args.example_cap)
        if y == 1 and not review.get("claim_found"):
            add_issue(issues, "positive_without_claim", row, args.example_cap)
        if y != 1 and rel.get("refute", 0) > 0:
            add_issue(issues, "negative_or_unobserved_with_refute", row, args.example_cap)
        if claim_evidence_relation(row) == "contradicts_claim" and rel.get("refute", 0) <= 0:
            add_issue(issues, "objective_contradiction_without_refute", row, args.example_cap)
        if st.startswith("silver_refute_") and y != 1:
            add_issue(issues, "silver_refute_not_positive", row, args.example_cap)
        audit = row.get("proposal_label_audit") or {}
        if audit.get("claim_family_conflicting_labels") and row.get("contrastive_mask") is False:
            counters["claim_family_conflict_masked_not_relabelled"] += 1
        if audit.get("claim_family_conflicting_labels") and y is None:
            add_issue(issues, "claim_family_changed_label_risk", row, args.example_cap)

    issue_counts = {k: len(v) for k, v in issues.items()}
    report = {
        "dataset": args.dataset,
        "rows": len(rows),
        "y_perception": dict(y_counts),
        "promotion_state": dict(by_state),
        "sample_role": dict(by_role),
        "issue_counts": issue_counts,
        "counters": dict(counters),
        "examples": issues,
    }
    write_json(args.report, report)
    if args.markdown:
        lines = [
            "# Proposal Label/Claim/Evidence Consistency Audit v2",
            "",
            f"- dataset: `{args.dataset}`",
            f"- rows: `{len(rows)}`",
            f"- y_perception: `{dict(y_counts)}`",
            f"- issue_counts: `{issue_counts}`",
            "",
            "## Issue Examples",
            "",
        ]
        for key, vals in issues.items():
            lines.append(f"### {key}")
            if not vals:
                lines.append("- none")
            for ex in vals[:10]:
                lines.append(
                    f"- `{ex['pair_id']}` attr={ex['attribute_name']} state={ex['state']} "
                    f"y={ex['y_perception']} rel={ex['relation_counts']}"
                )
            lines.append("")
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
