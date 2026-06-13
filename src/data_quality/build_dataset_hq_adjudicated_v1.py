"""Build an adjudicated high-quality CLAIMARC dataset candidate.

This script joins the deterministic HQ silver pool with LLM claim-evidence
adjudication. The adjudicator only sees the livestream claim and product
evidence, never the weak label. We then combine two independent signals:

1. consumer-perception weak label from aligned reviews (original y/c), and
2. objective claim-evidence risk state from the adjudicator.

The output preserves the original model schema while adding audit fields. It
does not overwrite any existing dataset.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import has_claim, read_jsonl, source_count


BAD_CLAIM_QUALITY = {"no_claim", "garbled"}
UNCERTAIN_CLAIM_QUALITY = {"mixed"}
SUPPORT_STATES = {"supported"}
RISK_STATES = {"contradicted"}
UNCERTAIN_STATES = {"insufficient", "not_verifiable"}
HIGH_RISK = {"high"}
MED_HIGH_RISK = {"medium", "high"}
LOW_NONE_RISK = {"none", "low", ""}


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def canonical_state(value: Any) -> str:
    state = norm(value)
    if state in SUPPORT_STATES | RISK_STATES | UNCERTAIN_STATES:
        return state
    # Some gateway responses use "mixed" despite the restricted schema. Treat
    # it as insufficient rather than silently keeping a novel class.
    return "insufficient"


def canonical_risk(value: Any) -> str:
    risk = norm(value)
    if risk in {"none", "low", "medium", "high"}:
        return risk
    return "medium"


def key(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def read_adjudication(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in read_jsonl(path):
        pid = str(obj.get("pair_id") or "")
        if not pid or obj.get("__error__"):
            continue
        out[pid] = obj
    return out


def base_confidence(rec: dict[str, Any]) -> float:
    return max(0.03, min(1.0, float(rec.get("c", 0.05) or 0.05)))


def risk_weight(risk: str, state: str) -> float:
    if state == "contradicted":
        return 0.80 if risk != "high" else 0.90
    if risk == "high":
        return 0.75
    if risk == "medium":
        return 0.55
    return 0.35


def classify(rec: dict[str, Any], adj: dict[str, Any]) -> tuple[str, int | None, float]:
    """Return (decision, final_y, quality_weight). final_y=None means drop."""
    y0 = int(rec.get("y", 0))
    c0 = base_confidence(rec)
    bucket = str(rec.get("_quality_bucket", ""))
    cq = norm(adj.get("claim_quality"))
    state = canonical_state(adj.get("evidence_state"))
    risk = canonical_risk(adj.get("misleading_risk"))
    sourceful = source_count(rec) > 0

    if not has_claim(rec) or cq in BAD_CLAIM_QUALITY:
        return "drop_bad_claim", None, 0.0

    uncertain_claim_scale = 0.75 if cq in UNCERTAIN_CLAIM_QUALITY else 1.0
    evidence_clean = state in SUPPORT_STATES and risk in LOW_NONE_RISK
    evidence_risky = state in RISK_STATES or risk in HIGH_RISK or (
        state in UNCERTAIN_STATES and risk in MED_HIGH_RISK
    )
    evidence_low_risk = state in UNCERTAIN_STATES and risk in LOW_NONE_RISK

    if y0 == 1:
        if evidence_risky:
            scale = 1.25 if bucket in {"pos_core", "pos_silver"} else 0.95
            return "pos_confirmed_or_unverifiable", 1, min(1.0, c0 * scale * uncertain_claim_scale)
        if evidence_clean:
            if bucket == "pos_core":
                return "pos_perception_supported", 1, min(0.45, c0 * 0.55 * uncertain_claim_scale)
            return "drop_positive_supported_conflict", None, 0.0
        # If reviews indicate perception risk and evidence is merely weak, keep
        # the sample but stop it from dominating the contrastive objective.
        scale = 0.75 if bucket in {"pos_core", "pos_silver"} else 0.45
        return "pos_perception_weak_evidence", 1, min(0.60, c0 * scale * uncertain_claim_scale)

    if evidence_risky:
        # Original negative labels are frequently "no aligned complaint", not
        # true no-risk labels. High-risk objective conflicts become silver
        # positives with explicit provenance.
        return "pos_relabel_evidence_risk", 1, risk_weight(risk, state) * uncertain_claim_scale

    if evidence_clean:
        scale = 1.20 if bucket in {"neg_core", "neg_silver_sourceful"} else 0.90
        return "neg_supported_clean", 0, min(1.0, max(c0, 0.12) * scale * uncertain_claim_scale)

    if evidence_low_risk and sourceful:
        return "neg_low_risk_sourceful", 0, min(0.35, max(c0, 0.08) * 0.75 * uncertain_claim_scale)

    return "drop_ambiguous_negative", None, 0.0


def add_adjudication_fields(rec: dict[str, Any], adj: dict[str, Any]) -> dict[str, Any]:
    decision, final_y, quality = classify(rec, adj)
    out = dict(rec)
    out["_base_pair_id"] = key(rec)
    out["_y_original"] = int(rec.get("y", 0))
    out["_c_original_before_adjudication"] = base_confidence(rec)
    out["_adjudication"] = {
        "claim_quality": adj.get("claim_quality"),
        "evidence_state": adj.get("evidence_state"),
        "misleading_risk": adj.get("misleading_risk"),
        "key_claim": adj.get("key_claim", ""),
        "key_evidence": adj.get("key_evidence", ""),
        "rationale": adj.get("rationale", ""),
        "flags": adj.get("flags", []),
        "model": adj.get("model", ""),
    }
    out["_adjudication_decision"] = decision
    out["_adjudication_weight"] = round(float(quality or 0.0), 4)
    if final_y is not None:
        out["y"] = int(final_y)
        out["c"] = round(max(0.03, min(1.0, float(quality))), 4)
    return out


def downsample_by_label(
    rows: list[dict[str, Any]],
    *,
    neg_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if neg_ratio <= 0:
        return rows
    pos = [r for r in rows if int(r.get("y", 0)) == 1]
    neg = [r for r in rows if int(r.get("y", 0)) == 0]
    target_neg = int(round(max(1, len(pos)) * neg_ratio))
    if len(neg) <= target_neg:
        return rows
    rng = random.Random(seed)
    neg_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in neg:
        neg_by_decision[str(r.get("_adjudication_decision", ""))].append(r)
    for vals in neg_by_decision.values():
        rng.shuffle(vals)

    selected_neg: list[dict[str, Any]] = []
    for decision in ("neg_supported_clean", "neg_low_risk_sourceful"):
        need = target_neg - len(selected_neg)
        if need <= 0:
            break
        selected_neg.extend(neg_by_decision.get(decision, [])[:need])
    selected = pos + selected_neg
    selected.sort(key=lambda r: (str(r.get("room_id", "")), key(r)))
    return selected


def apply_profile(rows: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile == "full":
        return rows
    if profile == "strict":
        out = []
        for r in rows:
            decision = str(r.get("_adjudication_decision", ""))
            adj = r.get("_adjudication") or {}
            state = canonical_state(adj.get("evidence_state"))
            sourceful = source_count(r) > 0
            if decision == "neg_supported_clean":
                out.append(r)
            elif decision == "pos_relabel_evidence_risk" and (sourceful or state == "contradicted"):
                out.append(r)
            elif decision == "pos_confirmed_or_unverifiable" and (sourceful or state == "contradicted"):
                out.append(r)
        return out
    if profile == "sourceful":
        return [r for r in rows if source_count(r) > 0]
    raise ValueError(f"unknown profile: {profile}")


def summarize(rows_all: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    transition = Counter(
        f"{int(r.get('_y_original', r.get('y', 0)))}->{int(r.get('y', 0))}"
        for r in selected
    )
    report: dict[str, Any] = {
        "input_n": len(rows_all),
        "selected_n": len(selected),
        "label_selected": dict(Counter(int(r.get("y", 0)) for r in selected)),
        "original_label_selected": dict(Counter(int(r.get("_y_original", r.get("y", 0))) for r in selected)),
        "label_transition_selected": dict(transition),
        "decision_all": dict(Counter(str(r.get("_adjudication_decision", "")) for r in rows_all)),
        "decision_selected": dict(Counter(str(r.get("_adjudication_decision", "")) for r in selected)),
        "quality_bucket_selected": dict(Counter(str(r.get("_quality_bucket", "")) for r in selected)),
        "claim_quality_selected": dict(Counter(str((r.get("_adjudication") or {}).get("claim_quality", "")) for r in selected)),
        "evidence_state_selected": dict(Counter(str((r.get("_adjudication") or {}).get("evidence_state", "")) for r in selected)),
        "risk_selected": dict(Counter(str((r.get("_adjudication") or {}).get("misleading_risk", "")) for r in selected)),
        "source_zero_selected": sum(1 for r in selected if source_count(r) == 0),
        "split_selected": dict(Counter(str(r.get("split", "")) for r in selected)),
    }
    by_decision: dict[str, dict[str, int]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        groups[str(r.get("_adjudication_decision", ""))].append(r)
    for decision, vals in sorted(groups.items()):
        by_decision[decision] = {
            "n": len(vals),
            "pos": sum(int(v.get("y", 0)) for v in vals),
            "orig_pos": sum(int(v.get("_y_original", v.get("y", 0))) for v in vals),
            "source_zero": sum(1 for v in vals if source_count(v) == 0),
        }
    report["by_decision_selected"] = by_decision
    return report


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_hq_silver_v1.jsonl")
    ap.add_argument(
        "--adjudication",
        default="data/final/claim_evidence_adjudication_hq_silver_v1.jsonl",
    )
    ap.add_argument("--out", default="data/final/dataset_hq_adjudicated_v1.jsonl")
    ap.add_argument("--report", default="data/final/dataset_hq_adjudicated_v1_report.json")
    ap.add_argument("--profile", default="full", choices=["full", "strict", "sourceful"])
    ap.add_argument("--neg_ratio", type=float, default=0.0, help="0 keeps all selected negatives")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    adjudication = read_adjudication(args.adjudication)
    annotated = []
    missing = 0
    for rec in rows:
        adj = adjudication.get(key(rec))
        if not adj:
            missing += 1
            continue
        annotated.append(add_adjudication_fields(rec, adj))

    selected = [r for r in annotated if r.get("_adjudication_weight", 0.0) > 0]
    selected = apply_profile(selected, args.profile)
    selected = downsample_by_label(selected, neg_ratio=args.neg_ratio, seed=args.seed)
    report = summarize(annotated, selected)
    report["adjudicated_missing"] = missing
    report["adjudication_path"] = str(args.adjudication)
    report["dataset_path"] = str(args.dataset)
    report["profile"] = args.profile

    write_jsonl(args.out, selected)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[build_dataset_hq_adjudicated_v1] wrote {args.out} "
        f"n={report['selected_n']} labels={report['label_selected']} missing={missing}"
    )
    print(f"[build_dataset_hq_adjudicated_v1] report={args.report}")


if __name__ == "__main__":
    main()
