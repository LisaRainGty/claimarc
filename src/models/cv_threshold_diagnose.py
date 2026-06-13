"""Threshold diagnostics for repeated grouped-CV OOF predictions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import (apply_arf, best_thr, build_split_features, fit_arf,
                                load_bundles, macro)


def rank01(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def run_one(dataset, args_tmp, bge_tmp, fold_seed, folds, cm_seeds):
    recs_by = load_split(dataset)
    recs = recs_by["train"] + recs_by["val"] + recs_by["test"]
    fold_list, _, g_all = make_folds(recs, folds, seed=fold_seed)
    y_all = np.array([int(r["y"]) for r in recs], int)
    methods = ["args_pcls", "bge_lr", "rank_pcls_bge", "rank_arf_bge"]
    probs = {m: np.full(len(recs), np.nan) for m in methods}
    yhats = {m: np.full(len(recs), np.nan) for m in methods}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(fold_list):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=fold_seed * 100 + fi)
        paths = [f"{args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in cm_seeds]
        bundles = load_bundles(paths)
        Xv, pcls_v, yv, cv, _ = build_split_features(bundles, "val")
        Xt, pcls_t, yt, _, _ = build_split_features(bundles, "test")
        arf = fit_arf(Xv, yv, cv)
        arf_v, arf_t = apply_arf(arf, Xv), apply_arf(arf, Xt)
        bge = torch.load(f"{bge_tmp}/cv_bge_lr_f{fi}.pt", weights_only=False)
        bge_v, bge_t = np.asarray(bge["val"]["p"], float), np.asarray(bge["test"]["p"], float)
        vals = {
            "args_pcls": (pcls_v, pcls_t),
            "bge_lr": (bge_v, bge_t),
            "rank_pcls_bge": (
                0.5 * rank01(pcls_v) + 0.5 * rank01(bge_v),
                0.5 * rank01(pcls_t) + 0.5 * rank01(bge_t),
            ),
            "rank_arf_bge": (
                0.5 * rank01(arf_v) + 0.5 * rank01(bge_v),
                0.5 * rank01(arf_t) + 0.5 * rank01(bge_t),
            ),
        }
        meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx),
                "val_pos": round(float(np.mean(yv)), 4), "test_pos": round(float(np.mean(yt)), 4)}
        for name, (pv, pt) in vals.items():
            thr = best_thr(yv, pv)
            probs[name][te_idx] = pt
            yhats[name][te_idx] = (pt >= thr).astype(int)
            meta[f"{name}_thr"] = round(float(thr), 3)
            meta[f"{name}_val_mf1"] = round(macro(yv, (pv >= thr).astype(int)), 4)
        fold_meta.append(meta)

    rows = {}
    for name in methods:
        ok = ~np.isnan(probs[name])
        y, p, yhat = y_all[ok], probs[name][ok], yhats[name][ok]
        oracle_thr = best_thr(y, p)
        rows[name] = {
            "auprc": round(float(average_precision_score(y, p)), 4),
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "val_threshold_macro_f1": round(macro(y, yhat), 4),
            "oracle_threshold": round(float(oracle_thr), 3),
            "oracle_macro_f1": round(macro(y, (p >= oracle_thr).astype(int)), 4),
        }
    return {"fold_seed": fold_seed, "rows": rows, "fold_meta": fold_meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = run_one(args.dataset, args.args_tmp, args.bge_tmp, args.fold_seed, args.folds, args.cm_seeds)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"[cv_threshold_diagnose] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
