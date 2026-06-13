"""Reconstruct CV OOF predictions and test CLAIMARC+BGE-LR hybrid fusion.

This script does not train models. It reuses `cv_eval.py` temporary prediction
files and selects a scalar blend per fold on that fold's validation split.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import (load_bundles, build_split_features, best_thr,
                                macro, paired_bootstrap)


def _rank01(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def _fit_alpha(p_cm, p_bge, y):
    best = (0.0, 1.0)
    for a in np.linspace(0.0, 1.0, 21):
        p = a * p_cm + (1.0 - a) * p_bge
        thr = best_thr(y, p)
        score = macro(y, (p >= thr).astype(int))
        if len(set(y.tolist())) > 1:
            score += 0.15 * average_precision_score(y, p)
        if score > best[0]:
            best = (float(score), float(a))
    return best[1]


def _row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(macro(y, yhat), 4),
        "wF1": round(macro(y, yhat, w=np.clip(c, 0.05, None)), 4),
        "n": int(len(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--tmpdir", default="data/final/cleancl/cv_tmp_small_e3_c10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="data/final/cleancl/cv_small_e3_c10_hybrid.json")
    args = ap.parse_args()

    full = load_split(args.dataset)
    recs = full["train"] + full["val"] + full["test"]
    folds, y_all, g_all = make_folds(recs, args.folds)
    y_oof = np.array([int(r["y"]) for r in recs], float)
    c_oof = np.array([float(r.get("c", 0.05)) for r in recs], float)

    methods = ["CLAIMARC_pcls", "bge_lr", "hybrid_valblend", "hybrid_rankavg"]
    oof = {m: {"p": np.full(len(recs), np.nan), "yhat": np.full(len(recs), np.nan)}
           for m in methods}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=fi)
        paths = [f"{args.tmpdir}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        missing = [p for p in paths if not os.path.exists(p)]
        bge_path = f"{args.tmpdir}/cv_bge_lr_f{fi}.pt"
        if missing or not os.path.exists(bge_path):
            raise FileNotFoundError(f"Missing fold {fi} files: {missing + [bge_path]}")

        bundles = load_bundles(paths)
        _, p_cm_v, yv, _, _ = build_split_features(bundles, "val")
        _, p_cm_t, yt, ct, _ = build_split_features(bundles, "test")
        bge = torch.load(bge_path, weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)

        alpha = _fit_alpha(p_cm_v, p_bge_v, yv)
        p_h_v = alpha * p_cm_v + (1.0 - alpha) * p_bge_v
        p_h_t = alpha * p_cm_t + (1.0 - alpha) * p_bge_t
        p_r_v = 0.5 * _rank01(p_cm_v) + 0.5 * _rank01(p_bge_v)
        p_r_t = 0.5 * _rank01(p_cm_t) + 0.5 * _rank01(p_bge_t)
        fold_meta.append({"fold": fi, "alpha_cm": round(alpha, 2),
                          "n_val": len(va_idx), "n_test": len(te_idx)})

        fold_probs = {
            "CLAIMARC_pcls": (p_cm_v, p_cm_t),
            "bge_lr": (p_bge_v, p_bge_t),
            "hybrid_valblend": (p_h_v, p_h_t),
            "hybrid_rankavg": (p_r_v, p_r_t),
        }
        for name, (pv, pt) in fold_probs.items():
            thr = best_thr(yv, pv)
            oof[name]["p"][te_idx] = pt
            oof[name]["yhat"][te_idx] = (pt >= thr).astype(int)

    rows = {}
    for m in methods:
        ok = ~np.isnan(oof[m]["p"])
        rows[m] = _row(y_oof[ok], oof[m]["p"][ok], oof[m]["yhat"][ok], c_oof[ok])
        print(f"  {m:16s} AP={rows[m]['auprc']} AUROC={rows[m]['auroc']} "
              f"mF1={rows[m]['macro_f1']} wF1={rows[m]['wF1']} n={rows[m]['n']}",
              flush=True)

    sig = {}
    for m in ("CLAIMARC_pcls", "hybrid_valblend", "hybrid_rankavg"):
        ok = (~np.isnan(oof[m]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{m}_vs_bge_lr"] = paired_bootstrap(
            y_oof[ok], oof[m]["p"][ok], oof["bge_lr"]["p"][ok], c_oof[ok])

    json.dump({"rows": rows, "fold_meta": fold_meta, "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_blend] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
