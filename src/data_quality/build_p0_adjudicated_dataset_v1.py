"""Build a strict p0-verified adjudicated dataset.

This applies fixed rules to atomic LLM adjudication fields. It never trusts the
LLM's final label recommendation directly.
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


def fixed_decision(adj: dict[str, Any]) -> tuple[str, int | None, float, str]:
    if adj.get("claim_quality") != "clear" or adj.get("attribute_quality") != "product_attribute":
        return "drop", None, 0.0, "bad_claim_or_attribute"
    if adj.get("confidence") not in {"high", "medium"}:
        return "drop", None, 0.0, "low_adjudication_confidence"

    ev = str(adj.get("product_evidence_state", ""))
    consumer = str(adj.get("consumer_signal", ""))
    if ev == "contradicted":
        return "keep", 1, 0.90, "product_evidence_contradicts"
    if ev == "supported" and consumer == "refutes_claim":
        return "keep", 1, 0.82, "consumer_refutes_supported_claim"
    if ev == "supported" and consumer in {"supports_claim", "irrelevant"}:
        return "keep", 0, 0.82, "product_and_consumer_support"
    return "drop", None, 0.0, "mixed_or_insufficient"


def compact(adj: dict[str, Any], action: str, label: int | None, weight: float, rule: str) -> dict[str, Any]:
    keys = [
        "claim_quality", "attribute_quality", "product_evidence_state",
        "consumer_signal", "label_recommendation", "confidence",
        "keep_for_training", "recommended_actions", "key_claim",
        "key_evidence", "key_review", "rationale", "model",
    ]
    out = {k: adj.get(k) for k in keys if k in adj}
    out.update({
        "fixed_action": action,
        "fixed_y": label,
        "fixed_c": weight,
        "fixed_rule": rule,
    })
    return out


def build(dataset_path: str, adjudication_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    adjs = {str(r.get("pair_id")): r for r in read_jsonl(adjudication_path)}
    out = []
    dropped = []
    changes: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    rules: Counter[str] = Counter()

    for rec in rows:
        pid = str(rec.get("pair_id"))
        adj = adjs.get(pid)
        if not adj:
            out.append(rec)
            continue
        action, label, weight, rule = fixed_decision(adj)
        rules[rule] += 1
        old_y = int(rec.get("y", 0) or 0)
        if action != "keep" or label is None:
            dropped.append({
                "pair_id": pid,
                "old_y": old_y,
                "rule": rule,
                "adjudication": compact(adj, action, label, weight, rule),
            })
            changes["drop"] += 1
            transitions[f"{old_y}->drop"] += 1
            continue
        new = dict(rec)
        new["_y_before_p0_verified_adjudication"] = old_y
        new["_c_before_p0_verified_adjudication"] = rec.get("c")
        new["_p0_verified_adjudication"] = compact(adj, action, label, weight, rule)
        new["y"] = int(label)
        new["c"] = float(weight)
        out.append(new)
        changes["keep_adjudicated"] += 1
        transitions[f"{old_y}->{label}"] += 1
        if old_y != label:
            changes[f"flip_{old_y}_to_{label}"] += 1

    report = {
        "input_dataset": dataset_path,
        "adjudication": adjudication_path,
        "n_input": len(rows),
        "n_adjudicated": len(adjs),
        "n_output": len(out),
        "n_dropped": len(dropped),
        "labels_input": dict(Counter(int(r.get("y", 0) or 0) for r in rows)),
        "labels_output": dict(Counter(int(r.get("y", 0) or 0) for r in out)),
        "source0_output": sum(1 for r in out if source_count(r) == 0),
        "changes": dict(changes),
        "transitions": dict(transitions),
        "rules": dict(rules),
        "dropped_rows": dropped,
    }
    return out, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0verified_v1.jsonl")
    ap.add_argument("--adjudication", default="data/final/repaired_v1/llm_p0_verified_evidence_adjudication_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0adjudicated_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/p0_adjudicated_v1_report.json")
    args = ap.parse_args()

    rows, report = build(args.dataset, args.adjudication)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    print(f"[build_p0_adjudicated_dataset_v1] wrote {args.out}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
