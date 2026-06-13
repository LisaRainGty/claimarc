"""Fold-safe RACL-U+C calibration over saved OOF predictions.

The diagnostic treats the trained CLAIMARC/RACL-U model and the BGE-LR baseline
as frozen scorers.  It then learns a compact evidence-conditioned calibration
layer on out-of-fold predictions from all *other* folds and applies it to the
held-out fold.  This is stacked generalization, not an in-fold oracle.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def macro_f1(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(f1_score(y, yhat, average="macro", sample_weight=w, zero_division=0))


def row_metrics(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    two_class = len(set(y.astype(int).tolist())) > 1
    return {
        "auprc": round(float(average_precision_score(y, p)), 4) if two_class else None,
        "auroc": round(float(roc_auc_score(y, p)), 4) if two_class else None,
        "macro_f1": round(macro_f1(y, yhat), 4),
        "wF1": round(macro_f1(y, yhat, np.clip(c, 0.05, None)), 4),
        "n": int(len(y)),
        "pos": int(np.sum(y)),
    }


def best_threshold(
    y: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    *,
    wf1_weight: float,
    prior_penalty: float,
) -> tuple[float, dict[str, float]]:
    pos_prior = float(np.mean(y))
    best_t = 0.5
    best_score = -1e9
    best_meta: dict[str, float] = {}
    for t in np.linspace(0.02, 0.98, 97):
        yhat = (p >= t).astype(int)
        mf1 = macro_f1(y, yhat)
        wf1 = macro_f1(y, yhat, np.clip(c, 0.05, None))
        pred_prior = float(np.mean(yhat))
        score = mf1 + wf1_weight * wf1 - prior_penalty * abs(pred_prior - pos_prior)
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_meta = {
                "score": float(score),
                "macro_f1": float(mf1),
                "wF1": float(wf1),
                "pred_prior": pred_prior,
                "true_prior": pos_prior,
            }
    return best_t, best_meta


def _as_str(z: np.lib.npyio.NpzFile, key: str, n: int, default: str = "") -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=object)
    return np.asarray(z[key]).astype(str)


def build_features_simple(
    z: np.lib.npyio.NpzFile,
    p_cm: np.ndarray,
    p_bge: np.ndarray,
    train_idx: np.ndarray,
    apply_idx: np.ndarray,
    *,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Wrapper with explicit categorical stacking for readability."""
    n = len(p_cm)
    c = np.asarray(z["c"], float)
    source_count = np.asarray(z["source_count"], float) if "source_count" in z.files else np.zeros(n)
    diff = p_cm - p_bge
    X_all = np.column_stack([
        _logit(p_cm),
        _logit(p_bge),
        diff,
        np.abs(diff),
        np.minimum(p_cm, p_bge),
        np.maximum(p_cm, p_bge),
        c,
        np.log1p(np.nan_to_num(source_count, nan=0.0)),
        _logit(p_cm) * np.clip(c, 0.05, 1.0),
        _logit(p_bge) * np.clip(c, 0.05, 1.0),
    ])
    meta: dict[str, Any] = {
        "feature_set": feature_set,
        "continuous": [
            "logit_cm",
            "logit_bge",
            "diff",
            "absdiff",
            "min_p",
            "max_p",
            "c",
            "log_source_count",
            "logit_cm_x_c",
            "logit_bge_x_c",
        ],
        "categorical": [],
    }
    if feature_set == "score":
        return X_all[train_idx], X_all[apply_idx], meta

    fields = [
        ("category", _as_str(z, "category", n, "__UNK__")),
        ("evidence_combo", _as_str(z, "evidence_combo", n, "__UNK__")),
        ("confidence", _as_str(z, "confidence", n, "__UNK__")),
    ]
    if feature_set == "full":
        attr = _as_str(z, "attribute_id", n, "__UNK__")
        fields.append(("attr_prefix", np.array([a.split("_", 1)[0] if "_" in a else a for a in attr])))

    X_train = X_all[train_idx]
    X_apply = X_all[apply_idx]
    for name, vals in fields:
        vocab = sorted(set(vals[train_idx].tolist()))
        if "__UNK__" not in vocab:
            vocab.append("__UNK__")
        X_train = np.column_stack([X_train] + [(vals[train_idx] == v).astype(float) for v in vocab])
        X_apply = np.column_stack([X_apply] + [(vals[apply_idx] == v).astype(float) for v in vocab])
        meta["categorical"].append({"field": name, "vocab_size": len(vocab)})
    return X_train, X_apply, meta


def fit_predict_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float,
    class_weight: str | None,
) -> np.ndarray:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000, class_weight=class_weight)
    clf.fit((X_train - mu) / sd, y_train, sample_weight=np.clip(c_train, 0.05, None))
    return clf.predict_proba((X_test - mu) / sd)[:, 1]


def select_config(
    z: np.lib.npyio.NpzFile,
    p_cm: np.ndarray,
    p_bge: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    fold_id: np.ndarray,
    outer_train: np.ndarray,
    *,
    feature_sets: list[str],
    c_grid: list[float],
    class_weights: list[str | None],
    wf1_weight: float,
    prior_penalty: float,
) -> dict[str, Any]:
    inner_folds = sorted(set(fold_id[outer_train].astype(int).tolist()))
    best: dict[str, Any] | None = None
    for fs in feature_sets:
        for C in c_grid:
            for cw in class_weights:
                scores = []
                for held in inner_folds:
                    tr = outer_train[fold_id[outer_train] != held]
                    va = outer_train[fold_id[outer_train] == held]
                    if len(va) == 0 or len(set(y[tr].astype(int).tolist())) < 2:
                        continue
                    Xtr, Xva, _ = build_features_simple(z, p_cm, p_bge, tr, va, feature_set=fs)
                    pv = fit_predict_lr(Xtr, y[tr], c[tr], Xva, C=C, class_weight=cw)
                    thr, _ = best_threshold(
                        y[tr],
                        fit_predict_lr(Xtr, y[tr], c[tr], Xtr, C=C, class_weight=cw),
                        c[tr],
                        wf1_weight=wf1_weight,
                        prior_penalty=prior_penalty,
                    )
                    yhat = (pv >= thr).astype(int)
                    ap = average_precision_score(y[va], pv) if len(set(y[va].astype(int).tolist())) > 1 else 0.0
                    au = roc_auc_score(y[va], pv) if len(set(y[va].astype(int).tolist())) > 1 else 0.5
                    mf1 = macro_f1(y[va], yhat)
                    wf1 = macro_f1(y[va], yhat, np.clip(c[va], 0.05, None))
                    scores.append(mf1 + wf1_weight * wf1 + 0.10 * ap + 0.03 * au)
                if not scores:
                    continue
                cand = {
                    "feature_set": fs,
                    "C": C,
                    "class_weight": cw,
                    "inner_score": float(np.mean(scores)),
                    "inner_score_std": float(np.std(scores)),
                }
                if best is None or cand["inner_score"] > best["inner_score"]:
                    best = cand
    if best is None:
        best = {
            "feature_set": "score",
            "C": 0.3,
            "class_weight": "balanced",
            "inner_score": None,
            "inner_score_std": None,
        }
    return best


def paired_bootstrap_fixed(
    y: np.ndarray,
    p_a: np.ndarray,
    yhat_a: np.ndarray,
    p_b: np.ndarray,
    yhat_b: np.ndarray,
    c: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    deltas: dict[str, list[float]] = {k: [] for k in ("dAP", "dAUROC", "dMacroF1", "dWF1")}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.astype(int).tolist())) < 2:
            continue
        ww = np.clip(c[idx], 0.05, None)
        deltas["dAP"].append(float(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx])))
        deltas["dAUROC"].append(float(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx])))
        deltas["dMacroF1"].append(float(macro_f1(yy, yhat_a[idx]) - macro_f1(yy, yhat_b[idx])))
        deltas["dWF1"].append(float(macro_f1(yy, yhat_a[idx], ww) - macro_f1(yy, yhat_b[idx], ww)))
    out = {}
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
    ap.add_argument("--oof", required=True)
    ap.add_argument("--cm_method", default="CLAIMARC_selectiveRKC")
    ap.add_argument("--baseline", default="bge_lr")
    ap.add_argument("--feature_sets", nargs="+", default=["score", "source", "full"])
    ap.add_argument("--c_grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--class_weights", nargs="+", default=["balanced", "none"])
    ap.add_argument("--wf1_weight", type=float, default=0.5)
    ap.add_argument("--prior_penalty", type=float, default=0.10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    fold_id = np.asarray(z["fold_id"], int)
    p_cm = np.asarray(z[f"p__{args.cm_method}"], float)
    p_bge = np.asarray(z[f"p__{args.baseline}"], float)
    yhat_cm = np.asarray(z[f"yhat__{args.cm_method}"], int)
    yhat_bge = np.asarray(z[f"yhat__{args.baseline}"], int)
    class_weights = [None if x == "none" else x for x in args.class_weights]

    p_meta = np.full(len(y), np.nan)
    yhat_meta = np.full(len(y), -1, dtype=int)
    fold_meta = []
    for held in sorted(set(fold_id.tolist())):
        tr = np.where(fold_id != held)[0]
        te = np.where(fold_id == held)[0]
        cfg = select_config(
            z,
            p_cm,
            p_bge,
            y,
            c,
            fold_id,
            tr,
            feature_sets=args.feature_sets,
            c_grid=args.c_grid,
            class_weights=class_weights,
            wf1_weight=args.wf1_weight,
            prior_penalty=args.prior_penalty,
        )
        Xtr, Xte, feat_meta = build_features_simple(z, p_cm, p_bge, tr, te, feature_set=cfg["feature_set"])
        ptr = fit_predict_lr(
            Xtr,
            y[tr],
            c[tr],
            Xtr,
            C=float(cfg["C"]),
            class_weight=cfg["class_weight"],
        )
        pte = fit_predict_lr(
            Xtr,
            y[tr],
            c[tr],
            Xte,
            C=float(cfg["C"]),
            class_weight=cfg["class_weight"],
        )
        thr, thr_meta = best_threshold(
            y[tr],
            ptr,
            c[tr],
            wf1_weight=args.wf1_weight,
            prior_penalty=args.prior_penalty,
        )
        p_meta[te] = pte
        yhat_meta[te] = (pte >= thr).astype(int)
        fold_meta.append({
            "fold": int(held),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "threshold": round(float(thr), 4),
            "selected": cfg,
            "threshold_meta": {k: round(float(v), 4) for k, v in thr_meta.items()},
            "feature_meta": feat_meta,
            "test_pred_prior": round(float(np.mean(yhat_meta[te])), 4),
            "test_true_prior": round(float(np.mean(y[te])), 4),
        })

    rows = {
        args.cm_method: row_metrics(y, p_cm, yhat_cm, c),
        args.baseline: row_metrics(y, p_bge, yhat_bge, c),
        "RACL_U_C": row_metrics(y, p_meta, yhat_meta, c),
    }
    significance = {
        "RACL_U_C_vs_baseline": paired_bootstrap_fixed(
            y, p_meta, yhat_meta, p_bge, yhat_bge, c, n_boot=args.n_boot, seed=args.seed
        ),
        "RACL_U_C_vs_cm": paired_bootstrap_fixed(
            y, p_meta, yhat_meta, p_cm, yhat_cm, c, n_boot=args.n_boot, seed=args.seed + 1
        ),
    }
    selected_counts = Counter(
        f"{m['selected']['feature_set']}|C={m['selected']['C']}|cw={m['selected']['class_weight']}"
        for m in fold_meta
    )
    out = {
        "oof": args.oof,
        "cm_method": args.cm_method,
        "baseline": args.baseline,
        "rows": rows,
        "fold_meta": fold_meta,
        "selected_counts": dict(selected_counts.most_common()),
        "significance": significance,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "selected_counts": out["selected_counts"]}, ensure_ascii=False, indent=2))
    print(f"[RACL-U+C] -> {args.out}")


if __name__ == "__main__":
    main()
