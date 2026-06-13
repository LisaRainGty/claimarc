"""Mechanism diagnostics for saved grouped-CV OOF predictions.

This utility is intentionally post-hoc: it reads saved out-of-fold
probabilities/decisions plus optional JSONL records and produces subgroup
tables, paired error counts, and representative correction/failure examples.
It does not fit thresholds or retrain models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def _to_list(x: np.ndarray) -> list[Any]:
    return [v.item() if hasattr(v, "item") else v for v in x]


def _metric_or_none(fn, y, score):
    if len(np.unique(y)) < 2:
        return None
    return round(float(fn(y, score)), 4)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.sum() / total) * abs(conf - acc)
    return round(float(ece), 4)


def metric_row(y, p, yhat, c) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    yhat = np.asarray(yhat, int)
    c = np.asarray(c, float)
    return {
        "n": int(len(y)),
        "pos_rate": round(float(y.mean()), 4) if len(y) else None,
        "pred_rate": round(float(yhat.mean()), 4) if len(yhat) else None,
        "auprc": _metric_or_none(average_precision_score, y, p),
        "auroc": _metric_or_none(roc_auc_score, y, p),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
        "wF1": round(float(f1_score(
            y, yhat, average="macro", sample_weight=np.clip(c, 0.05, None),
            zero_division=0,
        )), 4),
        "brier": round(float(brier_score_loss(y, np.clip(p, 0.0, 1.0))), 4),
        "ece10": ece_score(y, np.clip(p, 0.0, 1.0)),
    }


def load_oof(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["y"] = np.asarray(out["y"], int)
    out["c"] = np.asarray(out.get("c", np.ones_like(out["y"])), float)
    return out


def get_method(oof: dict[str, Any], method: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        (f"p__{method}", f"yhat__{method}"),
        (f"{method}__p", f"{method}__yhat"),
    ]
    for p_key, yhat_key in candidates:
        if p_key in oof and yhat_key in oof:
            return np.asarray(oof[p_key], float), np.asarray(oof[yhat_key], int)
    raise KeyError(f"method {method!r} not found in OOF keys")


def source_count_bin(values: np.ndarray) -> np.ndarray:
    out = []
    for v in np.asarray(values, float):
        if v <= 0:
            out.append("0")
        elif v == 1:
            out.append("1")
        elif v == 2:
            out.append("2")
        else:
            out.append("3+")
    return np.asarray(out, dtype=object)


def confidence_quantile(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    qs = np.quantile(c, [0.25, 0.5, 0.75])
    labels = []
    for v in c:
        if v <= qs[0]:
            labels.append("q1_lowest")
        elif v <= qs[1]:
            labels.append("q2")
        elif v <= qs[2]:
            labels.append("q3")
        else:
            labels.append("q4_highest")
    return np.asarray(labels, dtype=object)


def attribute_family(attrs: np.ndarray) -> np.ndarray:
    vals = []
    for raw in attrs:
        text = str(raw)
        vals.append(text.split("_", 1)[0] if "_" in text else text)
    return np.asarray(vals, dtype=object)


def safe_array(oof: dict[str, Any], key: str, default: str = "") -> np.ndarray:
    if key in oof:
        return np.asarray(oof[key], dtype=object)
    return np.asarray([default] * len(oof["y"]), dtype=object)


def subgroup_table(
    y: np.ndarray,
    c: np.ndarray,
    p_by_method: dict[str, np.ndarray],
    yhat_by_method: dict[str, np.ndarray],
    group_values: np.ndarray,
    min_n: int,
) -> dict[str, Any]:
    out = {}
    for group in sorted(set(_to_list(group_values)), key=lambda x: str(x)):
        mask = group_values == group
        if int(mask.sum()) < min_n:
            continue
        out[str(group)] = {
            method: metric_row(y[mask], p[mask], yhat_by_method[method][mask], c[mask])
            for method, p in p_by_method.items()
        }
    return out


def delta_table(
    strata: dict[str, dict[str, Any]],
    method: str,
    baseline: str,
) -> dict[str, Any]:
    out = {}
    for group, rows in strata.items():
        if method not in rows or baseline not in rows:
            continue
        mr, br = rows[method], rows[baseline]
        out[group] = {
            "n": mr["n"],
            "d_auprc": None if mr["auprc"] is None or br["auprc"] is None else round(mr["auprc"] - br["auprc"], 4),
            "d_auroc": None if mr["auroc"] is None or br["auroc"] is None else round(mr["auroc"] - br["auroc"], 4),
            "d_macro_f1": round(mr["macro_f1"] - br["macro_f1"], 4),
            "d_wF1": round(mr["wF1"] - br["wF1"], 4),
            "d_ece10": round(mr["ece10"] - br["ece10"], 4),
        }
    return out


def load_records(path: str, n_expected: int) -> list[dict[str, Any]]:
    if not path:
        return []
    records = [json.loads(line) for line in open(path) if line.strip()]
    if len(records) != n_expected:
        raise ValueError(
            f"dataset length mismatch: {len(records)} records vs {n_expected} OOF rows")
    return records


def evidence_snippets(record: dict[str, Any], max_len: int = 500) -> dict[str, str]:
    def join_items(key: str, field: str) -> str:
        text = " | ".join(
            str(item.get(field, "") or "").strip()
            for item in record.get(key, []) or []
            if str(item.get(field, "") or "").strip()
        )
        return text[:max_len]

    return {
        "params": join_items("evidence_params", "raw_text"),
        "ocr": join_items("evidence_ocr", "raw_text"),
        "vlm": join_items("evidence_vlm", "raw_quote"),
    }


def make_examples(
    records: list[dict[str, Any]],
    y: np.ndarray,
    c: np.ndarray,
    p_method: np.ndarray,
    yhat_method: np.ndarray,
    p_base: np.ndarray,
    yhat_base: np.ndarray,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    if not records:
        return {}

    def item(i: int) -> dict[str, Any]:
        rec = records[i]
        claim = rec.get("claim", {}) or {}
        return {
            "row": int(i),
            "pair_id": rec.get("pair_id", ""),
            "room_id": rec.get("room_id", ""),
            "category": rec.get("category", ""),
            "attribute_id": rec.get("attribute_id", ""),
            "attribute_name": rec.get("attribute_name", ""),
            "y": int(y[i]),
            "c": round(float(c[i]), 4),
            "method_p": round(float(p_method[i]), 4),
            "baseline_p": round(float(p_base[i]), 4),
            "method_yhat": int(yhat_method[i]),
            "baseline_yhat": int(yhat_base[i]),
            "claim": str(claim.get("passage", ""))[:600],
            "evidence": evidence_snippets(rec),
            "confidence": rec.get("confidence", ""),
            "evidence_count": rec.get("evidence_count", {}),
            "label_audit": rec.get("label_audit", {}),
        }

    correct_m = yhat_method == y
    correct_b = yhat_base == y
    method_fix = np.flatnonzero(correct_m & ~correct_b)
    baseline_fix = np.flatnonzero(~correct_m & correct_b)
    both_wrong = np.flatnonzero(~correct_m & ~correct_b)

    def error_margin(i: int) -> float:
        target = float(y[i])
        return abs(float(p_method[i]) - target) - abs(float(p_base[i]) - target)

    method_fix = sorted(method_fix.tolist(), key=lambda i: (error_margin(i), -c[i]))
    baseline_fix = sorted(baseline_fix.tolist(), key=lambda i: (-error_margin(i), -c[i]))
    both_wrong = sorted(
        both_wrong.tolist(),
        key=lambda i: (min(abs(p_method[i] - y[i]), abs(p_base[i] - y[i])), -c[i]),
        reverse=True,
    )
    return {
        "method_correct_baseline_wrong": [item(i) for i in method_fix[:limit]],
        "baseline_correct_method_wrong": [item(i) for i in baseline_fix[:limit]],
        "both_wrong": [item(i) for i in both_wrong[:limit]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--method", default="CLAIMARC_pcls")
    ap.add_argument("--baseline", default="bge_lr")
    ap.add_argument("--min_n", type=int, default=30)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oof = load_oof(Path(args.oof))
    y, c = oof["y"], oof["c"]
    p_m, yhat_m = get_method(oof, args.method)
    p_b, yhat_b = get_method(oof, args.baseline)
    methods = {
        args.method: (p_m, yhat_m),
        args.baseline: (p_b, yhat_b),
    }
    p_by_method = {name: p for name, (p, _) in methods.items()}
    yhat_by_method = {name: yhat for name, (_, yhat) in methods.items()}

    groups = {
        "source_count": source_count_bin(np.asarray(oof.get("source_count", np.zeros(len(y))), float)),
        "evidence_combo": safe_array(oof, "evidence_combo", "unknown"),
        "confidence_label": safe_array(oof, "confidence", "unknown"),
        "confidence_quantile": confidence_quantile(c),
        "category": safe_array(oof, "category", "unknown"),
        "attribute_family": attribute_family(safe_array(oof, "attribute_id", "unknown")),
        "fold_id": np.asarray(oof.get("fold_id", np.full(len(y), -1)), dtype=object),
    }
    strata = {
        name: subgroup_table(y, c, p_by_method, yhat_by_method, values, args.min_n)
        for name, values in groups.items()
    }
    deltas = {
        name: delta_table(table, args.method, args.baseline)
        for name, table in strata.items()
    }
    records = load_records(args.dataset, len(y))
    pair_counts = {
        "method_correct_baseline_wrong": int(((yhat_m == y) & (yhat_b != y)).sum()),
        "baseline_correct_method_wrong": int(((yhat_m != y) & (yhat_b == y)).sum()),
        "both_correct": int(((yhat_m == y) & (yhat_b == y)).sum()),
        "both_wrong": int(((yhat_m != y) & (yhat_b != y)).sum()),
    }
    out = {
        "oof": args.oof,
        "dataset": args.dataset,
        "method": args.method,
        "baseline": args.baseline,
        "overall": {
            args.method: metric_row(y, p_m, yhat_m, c),
            args.baseline: metric_row(y, p_b, yhat_b, c),
        },
        "pair_counts": pair_counts,
        "strata": strata,
        "deltas_method_minus_baseline": deltas,
        "examples": make_examples(
            records, y, c, p_m, yhat_m, p_b, yhat_b, args.examples),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[diagnose_oof_mechanisms] -> {out_path}", flush=True)
    print(json.dumps({
        "overall": out["overall"],
        "pair_counts": pair_counts,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
