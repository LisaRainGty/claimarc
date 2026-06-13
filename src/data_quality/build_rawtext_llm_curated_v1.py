"""Build a strict raw-text + LLM-adjudicated clean dataset.

This script promotes deterministic raw-text evidence recovery only when a
separate LLM adjudication provides clear atomic judgments. It does not trust the
LLM's final label recommendation directly. Instead, it derives labels from:

- claim quality;
- attribute quality;
- product-evidence state;
- consumer response signal.

Rows with mixed/uncertain judgments are removed from the strict benchmark and
left for the regeneration queue.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import read_jsonl, source_count


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def adjudication_decision(adj: dict[str, Any]) -> tuple[str, int | None, float, str]:
    """Return action, derived label, confidence weight, and rule id."""
    if (
        adj.get("claim_quality") != "clear"
        or adj.get("attribute_quality") != "product_attribute"
        or adj.get("confidence") not in {"high", "medium"}
    ):
        return "drop", None, 0.0, "bad_claim_or_attribute"

    ev_state = str(adj.get("product_evidence_state", ""))
    consumer = str(adj.get("consumer_signal", ""))

    if ev_state == "contradicted" and consumer == "refutes_claim":
        return "keep", 1, 0.90, "evidence_and_consumer_refute"
    if ev_state == "contradicted":
        return "keep", 1, 0.78, "evidence_refutes_claim"
    if consumer == "refutes_claim":
        return "keep", 1, 0.72, "consumer_refutes_claim"
    if ev_state == "supported" and consumer == "supports_claim":
        return "keep", 0, 0.82, "evidence_and_consumer_support"
    if ev_state == "supported" and consumer in {"irrelevant", "insufficient"}:
        return "keep", 0, 0.62, "evidence_supports_no_consumer_refute"

    return "drop", None, 0.0, "mixed_or_insufficient"


def compact_adj(adj: dict[str, Any], action: str, label: int | None, weight: float, rule: str) -> dict[str, Any]:
    keep_keys = [
        "claim_quality",
        "attribute_quality",
        "product_evidence_state",
        "consumer_signal",
        "label_recommendation",
        "confidence",
        "keep_for_training",
        "recommended_actions",
        "key_claim",
        "key_evidence",
        "key_review",
        "rationale",
        "model",
    ]
    out = {k: adj.get(k) for k in keep_keys if k in adj}
    out.update({
        "curation_action": action,
        "derived_y": label,
        "derived_c": weight,
        "curation_rule": rule,
    })
    return out


def build(dataset_path: str, adjudication_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    adjudications = {str(r.get("pair_id")): r for r in read_jsonl(adjudication_path)}

    curated: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    changes: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    rules: Counter[str] = Counter()

    for rec in rows:
        pair_id = str(rec.get("pair_id"))
        adj = adjudications.get(pair_id)
        if not adj:
            curated.append(rec)
            continue

        action, label, weight, rule = adjudication_decision(adj)
        rules[rule] += 1
        old_y = int(rec.get("y", 0) or 0)
        if action != "keep" or label is None:
            dropped.append({
                "pair_id": pair_id,
                "product_id": rec.get("product_id"),
                "attribute_id": rec.get("attribute_id"),
                "old_y": old_y,
                "rule": rule,
                "adjudication": compact_adj(adj, action, label, weight, rule),
            })
            changes["drop"] += 1
            transitions[f"{old_y}->drop"] += 1
            continue

        new = dict(rec)
        new["_y_before_llm_rawtext_curation"] = old_y
        new["_c_before_llm_rawtext_curation"] = rec.get("c")
        new["_llm_rawtext_curation"] = compact_adj(adj, action, label, weight, rule)
        new["y"] = int(label)
        new["c"] = float(weight)
        curated.append(new)
        changes["keep_adjudicated"] += 1
        transitions[f"{old_y}->{label}"] += 1
        if old_y != label:
            changes[f"flip_{old_y}_to_{label}"] += 1

    report = {
        "input_dataset": dataset_path,
        "adjudication": adjudication_path,
        "n_input": len(rows),
        "n_adjudicated": len(adjudications),
        "n_output": len(curated),
        "n_dropped": len(dropped),
        "labels_input": dict(Counter(int(r.get("y", 0) or 0) for r in rows)),
        "labels_output": dict(Counter(int(r.get("y", 0) or 0) for r in curated)),
        "splits_output": dict(Counter(str(r.get("split", "")) for r in curated)),
        "source0_output": sum(1 for r in curated if source_count(r) == 0),
        "label_source0_output": dict(Counter(f"{int(r.get('y', 0) or 0)}:{source_count(r) == 0}" for r in curated)),
        "changes": dict(changes),
        "transitions": dict(transitions),
        "rules": dict(rules),
        "dropped_rows": dropped,
    }
    return curated, report


def write_markdown(report: dict[str, Any], path: str | Path, out_dataset: str) -> None:
    lines = [
        "# Raw-Text LLM Curated Dataset v1",
        "",
        f"- dataset: `{out_dataset}`",
        f"- input: `{report['input_dataset']}`",
        f"- adjudication: `{report['adjudication']}`",
        "",
        "## Summary",
        f"- input rows: `{report['n_input']}`",
        f"- adjudicated rows: `{report['n_adjudicated']}`",
        f"- output rows: `{report['n_output']}`",
        f"- dropped rows: `{report['n_dropped']}`",
        f"- input labels: `{report['labels_input']}`",
        f"- output labels: `{report['labels_output']}`",
        f"- source0 output: `{report['source0_output']}`",
        f"- changes: `{report['changes']}`",
        f"- transitions: `{report['transitions']}`",
        f"- rules: `{report['rules']}`",
        "",
        "## Curation Rule",
        "The LLM final label recommendation is not used directly. The script keeps",
        "only clear product-attribute rows and derives labels from atomic evidence",
        "and consumer-response states: contradicted/refuting cases become risk",
        "positives; supported and non-refuting cases become clean negatives;",
        "mixed, insufficient, service, subjective, or malformed rows are removed",
        "from this strict benchmark and remain candidates for regeneration.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_v1.jsonl")
    ap.add_argument("--adjudication", default="data/final/repaired_v1/llm_recovered_evidence_adjudication_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/rawtext_llm_curated_v1_report.json")
    ap.add_argument("--md", default="docs/RAWTEXT_LLM_CURATED_V1.md")
    args = ap.parse_args()

    rows, report = build(args.dataset, args.adjudication)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    write_markdown(report, args.md, args.out)
    print(f"[build_rawtext_llm_curated_v1] wrote {args.out}")
    print(f"[build_rawtext_llm_curated_v1] n={report['n_output']} labels={report['labels_output']} changes={report['changes']}")


if __name__ == "__main__":
    main()
