"""Diagnose saved OOF probabilities and threshold decisions.

The script is deliberately post-hoc: it reads probabilities and saved
decisions already emitted by a CV evaluator and reports how much of the gap is
ranking quality versus threshold calibration. Oracle thresholds are diagnostic
upper bounds on the same OOF scores, not deployable test-tuned numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def macro_f1(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(f1_score(y, yhat, average="macro", sample_weight=w, zero_division=0))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def metric_row(
    y: np.ndarray,
    p: np.ndarray,
    yhat: np.ndarray,
    c: np.ndarray,
) -> dict[str, float | int | None]:
    return {
        "auprc": round_nullable(safe_ap(y, p)),
        "auroc": round_nullable(safe_auc(y, p)),
        "macro_f1": round(float(macro_f1(y, yhat)), 4),
        "wF1": round(float(macro_f1(y, yhat, np.clip(c, 0.05, None))), 4),
        "pred_rate": round(float(np.mean(yhat)), 4),
        "n": int(len(y)),
    }


def round_nullable(x: float | None, ndigits: int = 4) -> float | None:
    return None if x is None else round(float(x), ndigits)


def best_threshold(
    y: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    weighted: bool,
) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    weights = np.clip(c, 0.05, None) if weighted else None
    for t in np.linspace(0.01, 0.99, 99):
        score = macro_f1(y, (p >= t).astype(int), weights)
        if score > best_s:
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def discover_methods(z: np.lib.npyio.NpzFile) -> list[str]:
    methods = []
    keys = set(z.files)
    for key in z.files:
        if key.startswith("p__"):
            method = key[3:]
            if f"yhat__{method}" in keys:
                methods.append(method)
        elif key.endswith("__p"):
            method = key[:-3]
            if f"{method}__yhat" in keys:
                methods.append(method)
    return sorted(dict.fromkeys(methods))


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    p_key, yhat_key = f"p__{method}", f"yhat__{method}"
    if p_key not in z.files:
        p_key, yhat_key = f"{method}__p", f"{method}__yhat"
    if p_key not in z.files or yhat_key not in z.files:
        raise KeyError(f"{method}: missing {p_key} or {yhat_key}")
    return np.asarray(z[p_key], float), np.asarray(z[yhat_key], float)


def summarize_method(
    z: np.lib.npyio.NpzFile,
    method: str,
    group_keys: list[str],
    min_group_n: int,
) -> dict[str, object]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
    fold_id = np.asarray(z["fold_id"], int) if "fold_id" in z.files else np.zeros_like(y)
    p, yhat = get_method(z, method)
    ok = (~np.isnan(p)) & (~np.isnan(yhat)) & (fold_id >= 0)
    y, c, p, yhat = y[ok], c[ok], p[ok], yhat[ok].astype(int)
    t_macro, s_macro = best_threshold(y, p, c, weighted=False)
    t_wf1, s_wf1 = best_threshold(y, p, c, weighted=True)
    out: dict[str, object] = {
        "saved": metric_row(y, p, yhat, c),
        "fixed_0.5": metric_row(y, p, (p >= 0.5).astype(int), c),
        "oracle_macro": {
            "threshold": round(t_macro, 3),
            **metric_row(y, p, (p >= t_macro).astype(int), c),
        },
        "oracle_wF1": {
            "threshold": round(t_wf1, 3),
            **metric_row(y, p, (p >= t_wf1).astype(int), c),
        },
        "calibration_gap": {
            "oracle_macro_minus_saved": round(float(s_macro - macro_f1(y, yhat)), 4),
            "oracle_wF1_minus_saved": round(
                float(s_wf1 - macro_f1(y, yhat, np.clip(c, 0.05, None))), 4),
        },
        "score_summary": {
            "mean": round(float(np.mean(p)), 4),
            "std": round(float(np.std(p)), 4),
            "q05": round(float(np.quantile(p, 0.05)), 4),
            "q50": round(float(np.quantile(p, 0.50)), 4),
            "q95": round(float(np.quantile(p, 0.95)), 4),
        },
    }
    groups = {}
    for key in group_keys:
        if key not in z.files:
            continue
        vals_all = np.asarray(z[key], dtype=object)[ok]
        key_rows = {}
        for val in sorted(set(vals_all.tolist()), key=lambda x: str(x)):
            idx = vals_all == val
            if int(idx.sum()) < min_group_n:
                continue
            key_rows[str(val)] = metric_row(y[idx], p[idx], yhat[idx], c[idx])
        groups[key] = key_rows
    if groups:
        out["groups"] = groups
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--group_key", action="append", default=[])
    ap.add_argument("--min_group_n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    methods = args.method or discover_methods(z)
    out = {
        "oof": str(Path(args.oof)),
        "methods": {
            method: summarize_method(z, method, args.group_key, args.min_group_n)
            for method in methods
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print(f"[diagnose_oof_thresholds] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
