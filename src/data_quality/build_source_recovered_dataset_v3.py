"""Apply source-recovery v3 verification to a CLAIMARC dataset.

The script treats the verifier as a data-curation audit:

- high/medium confidence promote_clean/promote_risk rows receive grounded
  product evidence and updated source counts;
- high/medium confidence bad-attribute/bad-claim rows are removed from the
  main claim-evidence task;
- commercial/process rows are exported to an auxiliary manifest and removed
  from the main task;
- unresolved rows are retained unless --drop_unresolved_source0 is set.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from data_quality.audit_dataset_quality import read_jsonl, source_count


PROMOTE = {"promote_clean", "promote_risk"}
DROP_MAIN = {"drop_bad_attribute", "drop_bad_claim"}
AUX = {"keep_for_auxiliary"}
GOOD_CONF = {"high", "medium"}


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


def evidence_count(rec: dict[str, Any]) -> dict[str, int]:
    return {
        "params": len(rec.get("evidence_params") or []),
        "ocr": len(rec.get("evidence_ocr") or []),
        "vlm": len(rec.get("evidence_vlm") or []),
    }


def verified_to_evidence(v: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    src = str(v.get("product_source_type", ""))
    text = str(v.get("product_evidence_text", "") or "")
    base = {
        "match": "llm_verified_source_recovery_v3",
        "normalized_value": v.get("normalized_value", ""),
        "relation_to_claim": v.get("evidence_state", ""),
        "confidence": v.get("confidence", ""),
        "path_or_image": v.get("path_or_image", ""),
        "verifier_action": v.get("training_action", ""),
    }
    if src == "detail_image_ocr":
        item = dict(base)
        item.update({"raw_text": text, "image_path": v.get("path_or_image", "")})
        return "evidence_ocr", item
    if src == "detail_image_vlm":
        item = dict(base)
        item.update({"raw_quote": text, "image_path": v.get("path_or_image", "")})
        return "evidence_vlm", item
    item = dict(base)
    item.update({
        "raw_text": text,
        "param_key": "product_title" if src == "product_title" else v.get("path_or_image", src),
    })
    return "evidence_params", item


def add_unique(items: list[dict[str, Any]], item: dict[str, Any], text_key: str) -> list[dict[str, Any]]:
    seen = {
        (
            str(x.get(text_key, "")),
            str(x.get("image_path", x.get("param_key", x.get("path_or_image", "")))),
        )
        for x in items
    }
    key = (
        str(item.get(text_key, "")),
        str(item.get("image_path", item.get("param_key", item.get("path_or_image", "")))),
    )
    if key not in seen:
        items.append(item)
    return items


def load_verified(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for v in read_jsonl(path):
        if "__error__" in v or not v.get("pair_id"):
            continue
        out[str(v.get("pair_id"))] = v
    return out


def compact(v: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "attribute_quality", "claim_state", "product_evidence_found",
        "product_source_type", "evidence_state", "training_action",
        "confidence", "claim_text", "product_evidence_text",
        "normalized_value", "path_or_image", "rationale", "model",
    ]
    return {k: v.get(k) for k in keys if k in v}


def build(dataset_path: str, verified_path: str, drop_unresolved_source0: bool) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(dataset_path)
    verified = load_verified(verified_path)
    out: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    aux: list[dict[str, Any]] = []
    changed: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()

    for rec in rows:
        pid = str(rec.get("pair_id"))
        v = verified.get(pid)
        old_y = int(rec.get("y", 0) or 0)
        old_sc = source_count(rec)
        if not v:
            out.append(rec)
            continue

        action = str(v.get("training_action", ""))
        conf = str(v.get("confidence", ""))
        action_counts[action] += 1
        good_conf = conf in GOOD_CONF

        if action in PROMOTE and good_conf and v.get("product_evidence_found"):
            new = dict(rec)
            new["_source_recovery_v3"] = compact(v)
            field, item = verified_to_evidence(v)
            text_key = "raw_quote" if field == "evidence_vlm" else "raw_text"
            new[field] = add_unique(list(new.get(field) or []), item, text_key)
            cnt = evidence_count(new)
            new["evidence_count"] = cnt
            coverage = sum(1 for n in cnt.values() if n > 0)
            new["coverage"] = coverage
            new["confidence"] = config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent")
            new["_y_before_source_recovery_v3"] = old_y
            new["_c_before_source_recovery_v3"] = rec.get("c")
            if action == "promote_clean":
                new["y"] = 0
                new["c"] = max(0.62, float(rec.get("c", 0.0) or 0.0))
            elif action == "promote_risk":
                new["y"] = 1
                new["c"] = max(0.82, float(rec.get("c", 0.0) or 0.0))
            out.append(new)
            changed["promote"] += 1
            transitions[f"{old_y}->{new['y']}"] += 1
            continue

        if action in DROP_MAIN and good_conf:
            removed.append({
                "pair_id": pid,
                "old_y": old_y,
                "old_source_count": old_sc,
                "reason": action,
                "verification": compact(v),
            })
            changed["drop_main"] += 1
            transitions[f"{old_y}->drop"] += 1
            continue

        if action in AUX and good_conf:
            aux.append({
                "pair_id": pid,
                "old_y": old_y,
                "old_source_count": old_sc,
                "aux_task": "merchant_or_commercial_process_claim",
                "verification": compact(v),
            })
            removed.append({
                "pair_id": pid,
                "old_y": old_y,
                "old_source_count": old_sc,
                "reason": "auxiliary_not_main_task",
                "verification": compact(v),
            })
            changed["move_aux"] += 1
            transitions[f"{old_y}->aux"] += 1
            continue

        if drop_unresolved_source0 and old_sc == 0:
            removed.append({
                "pair_id": pid,
                "old_y": old_y,
                "old_source_count": old_sc,
                "reason": "unresolved_source0",
                "verification": compact(v),
            })
            changed["drop_unresolved_source0"] += 1
            transitions[f"{old_y}->drop_unresolved"] += 1
            continue

        new = dict(rec)
        new["_source_recovery_v3"] = compact(v)
        out.append(new)
        changed["retain_unresolved"] += 1
        transitions[f"{old_y}->{old_y}"] += 1

    report = {
        "input_dataset": dataset_path,
        "verified": verified_path,
        "n_input": len(rows),
        "n_verified": len(verified),
        "n_output": len(out),
        "n_removed_main": len(removed),
        "n_aux": len(aux),
        "labels_input": dict(Counter(int(r.get("y", 0) or 0) for r in rows)),
        "labels_output": dict(Counter(int(r.get("y", 0) or 0) for r in out)),
        "source0_input": sum(1 for r in rows if source_count(r) == 0),
        "source0_output": sum(1 for r in out if source_count(r) == 0),
        "action_counts": dict(action_counts),
        "changes": dict(changed),
        "transitions": dict(transitions),
        "removed_rows": removed[:500],
    }
    return out, report, aux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0p1adjudicated_v1.jsonl")
    ap.add_argument("--verified", default="data/final/repaired_v1/source_recovery_queue_v3_llm_verify_p0_v3strict.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/source_recovered_v3_report.json")
    ap.add_argument("--aux", default="data/final/repaired_v1/source_recovery_v3_auxiliary_manifest.jsonl")
    ap.add_argument("--drop_unresolved_source0", action="store_true")
    args = ap.parse_args()

    rows, report, aux = build(args.dataset, args.verified, args.drop_unresolved_source0)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    write_jsonl(args.aux, aux)
    print(f"[build_source_recovered_dataset_v3] wrote {args.out}")
    print(json.dumps({
        "n_output": report["n_output"],
        "labels_output": report["labels_output"],
        "source0_output": report["source0_output"],
        "changes": report["changes"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
