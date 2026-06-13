"""Aggregate repeated OOF runs with pair-grouped paired bootstrap.

Repeated grouped-CV runs often contain predictions for the same product-
attribute pair under different fold seeds.  This utility reports pooled
metrics while bootstrapping by `pair_id`, so repeated predictions for the same
row stay together inside resamples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _macro_f1(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(f1_score(y, yhat, average="macro", sample_weight=w, zero_division=0))


def _metrics(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    two_class = len(set(y.astype(int).tolist())) > 1
    return {
        "auprc": round(float(average_precision_score(y, p)), 4) if two_class else None,
        "auroc": round(float(roc_auc_score(y, p)), 4) if two_class else None,
        "macro_f1": round(_macro_f1(y, yhat), 4),
        "wF1": round(_macro_f1(y, yhat, np.clip(c, 0.05, None)), 4),
        "n": int(len(y)),
        "pos": int(np.sum(y)),
    }


def _load_oof(path: Path, methods: list[str]) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {
        "y": np.asarray(z["y"], int),
        "c": np.asarray(z["c"], float),
        "pair_id": np.asarray(z["pair_id"]).astype(str),
        "run": np.full(len(z["y"]), path.name, dtype=object),
    }
    for method in methods:
        p_key = f"p__{method}"
        yhat_key = f"yhat__{method}"
        if p_key not in z.files:
            raise KeyError(f"{path}: missing {p_key}")
        out[p_key] = np.asarray(z[p_key], float)
        if yhat_key in z.files:
            out[yhat_key] = np.asarray(z[yhat_key], int)
        else:
            out[yhat_key] = (out[p_key] >= 0.5).astype(int)
    return out


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def _paired_bootstrap_grouped(
    y: np.ndarray,
    c: np.ndarray,
    pair_id: np.ndarray,
    p_a: np.ndarray,
    yhat_a: np.ndarray,
    p_b: np.ndarray,
    yhat_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    groups = np.array(sorted(set(pair_id.tolist())), dtype=object)
    group_indices = {g: np.flatnonzero(pair_id == g) for g in groups}
    deltas: dict[str, list[float]] = {k: [] for k in ("dAP", "dAUROC", "dMacroF1", "dWF1")}
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        yy = y[idx]
        if len(set(yy.astype(int).tolist())) < 2:
            continue
        ww = np.clip(c[idx], 0.05, None)
        deltas["dAP"].append(float(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx])))
        deltas["dAUROC"].append(float(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx])))
        deltas["dMacroF1"].append(float(_macro_f1(yy, yhat_a[idx]) - _macro_f1(yy, yhat_b[idx])))
        deltas["dWF1"].append(float(_macro_f1(yy, yhat_a[idx], ww) - _macro_f1(yy, yhat_b[idx], ww)))

    out: dict[str, Any] = {}
    for key, vals in deltas.items():
        arr = np.asarray(vals, float)
        out[key] = {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oofs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["RACL_U_C", "bge_lr", "CLAIMARC_selectiveRKC"])
    ap.add_argument("--primary", default="RACL_U_C")
    ap.add_argument("--baseline", default="bge_lr")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [Path(p) for p in args.oofs]
    parts = [_load_oof(p, args.methods) for p in paths]
    data = _concat(parts)

    rows = {}
    for method in args.methods:
        rows[method] = _metrics(
            data["y"],
            data[f"p__{method}"],
            data[f"yhat__{method}"],
            data["c"],
        )

    per_run = []
    for part, path in zip(parts, paths):
        run_rows = {}
        for method in args.methods:
            run_rows[method] = _metrics(
                part["y"],
                part[f"p__{method}"],
                part[f"yhat__{method}"],
                part["c"],
            )
        per_run.append({"oof": str(path), "rows": run_rows})

    significance = {
        f"{args.primary}_vs_{args.baseline}": _paired_bootstrap_grouped(
            data["y"],
            data["c"],
            data["pair_id"],
            data[f"p__{args.primary}"],
            data[f"yhat__{args.primary}"],
            data[f"p__{args.baseline}"],
            data[f"yhat__{args.baseline}"],
            n_boot=args.n_boot,
            seed=args.seed,
        )
    }
    if "CLAIMARC_selectiveRKC" in args.methods and args.primary != "CLAIMARC_selectiveRKC":
        significance[f"{args.primary}_vs_CLAIMARC_selectiveRKC"] = _paired_bootstrap_grouped(
            data["y"],
            data["c"],
            data["pair_id"],
            data[f"p__{args.primary}"],
            data[f"yhat__{args.primary}"],
            data["p__CLAIMARC_selectiveRKC"],
            data["yhat__CLAIMARC_selectiveRKC"],
            n_boot=args.n_boot,
            seed=args.seed + 1,
        )

    out = {
        "oofs": [str(p) for p in paths],
        "methods": args.methods,
        "primary": args.primary,
        "baseline": args.baseline,
        "rows": rows,
        "per_run": per_run,
        "unique_pair_id": int(len(set(data["pair_id"].tolist()))),
        "repeated_rows": int(len(data["y"])),
        "bootstrap": {
            "unit": "pair_id",
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "significance": significance,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
