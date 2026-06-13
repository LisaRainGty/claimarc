"""OOF fusion between no-argument CLAIMARC and argument-augmented retrieval experts."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import (load_bundles, build_split_features, fit_arf, apply_arf,
                                best_thr, macro, paired_bootstrap)


def rank01(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def fit_alpha(y, p_left, p_right, objective):
    best = (-1e9, 1.0)
    for a in np.linspace(0, 1, 21):
        p = a * p_left + (1 - a) * p_right
        thr = best_thr(y, p)
        mf = macro(y, (p >= thr).astype(int))
        ap = average_precision_score(y, p)
        au = roc_auc_score(y, p)
        score = (mf + 0.15 * ap) if objective == "macro" else (ap + 0.5 * au)
        if score > best[0]:
            best = (float(score), float(a))
    return best[1]


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(macro(y, yhat), 4),
        "wF1": round(macro(y, yhat, w=np.clip(c, 0.05, None)), 4),
        "n": int(len(y)),
    }


def put(oof, name, idx, yv, pv, pt):
    thr = best_thr(yv, pv)
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--noargs_tmp", default="data/final/cleancl/cv_tmp_small_e3_c10")
    ap.add_argument("--args_tmp", default="data/final/cleancl/cv_tmp_args_small_e3_c10")
    ap.add_argument("--bge_tmp", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="data/final/cleancl/cv_args_expert_fusion.json")
    args = ap.parse_args()
    bge_tmp = args.bge_tmp or args.noargs_tmp

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, y_all, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_oof = np.array([int(r["y"]) for r in recs], float)
    c_oof = np.array([float(r.get("c", 0.05)) for r in recs], float)

    methods = [
        "noargs_pcls", "args_pcls", "args_knn", "args_arf",
        "prob_macro_noargs+args_arf", "prob_rank_noargs+args_arf",
        "rankavg_noargs+args_arf", "rankavg_noargs+args_knn", "bge_lr",
    ]
    oof = {m: {"p": np.full(len(recs), np.nan), "yhat": np.full(len(recs), np.nan)}
           for m in methods}
    meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in no_paths + ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing fold {fi}: {missing}")

        no_b = load_bundles(no_paths)
        ar_b = load_bundles(ar_paths)
        _, p_no_v, yv, cv, _ = build_split_features(no_b, "val")
        _, p_no_t, yt, ct, _ = build_split_features(no_b, "test")
        Xv_ar, p_ar_v, yv2, _, _ = build_split_features(ar_b, "val")
        Xt_ar, p_ar_t, yt2, _, _ = build_split_features(ar_b, "test")
        if not (np.all(yv == yv2) and np.all(yt == yt2)):
            raise ValueError(f"Fold {fi} noargs/args labels differ")
        p_knn_v, p_knn_t = Xv_ar[:, 1], Xt_ar[:, 1]
        arf = fit_arf(Xv_ar, yv, cv)
        p_arf_v, p_arf_t = apply_arf(arf, Xv_ar), apply_arf(arf, Xt_ar)
        bge = torch.load(bge_path, weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)

        a_macro = fit_alpha(yv, p_no_v, p_arf_v, "macro")
        a_rank = fit_alpha(yv, p_no_v, p_arf_v, "rank")
        fold_probs = {
            "noargs_pcls": (p_no_v, p_no_t),
            "args_pcls": (p_ar_v, p_ar_t),
            "args_knn": (p_knn_v, p_knn_t),
            "args_arf": (p_arf_v, p_arf_t),
            "prob_macro_noargs+args_arf": (
                a_macro * p_no_v + (1 - a_macro) * p_arf_v,
                a_macro * p_no_t + (1 - a_macro) * p_arf_t,
            ),
            "prob_rank_noargs+args_arf": (
                a_rank * p_no_v + (1 - a_rank) * p_arf_v,
                a_rank * p_no_t + (1 - a_rank) * p_arf_t,
            ),
            "rankavg_noargs+args_arf": (
                0.5 * rank01(p_no_v) + 0.5 * rank01(p_arf_v),
                0.5 * rank01(p_no_t) + 0.5 * rank01(p_arf_t),
            ),
            "rankavg_noargs+args_knn": (
                0.5 * rank01(p_no_v) + 0.5 * rank01(p_knn_v),
                0.5 * rank01(p_no_t) + 0.5 * rank01(p_knn_t),
            ),
            "bge_lr": (p_bge_v, p_bge_t),
        }
        for name, (pv, pt) in fold_probs.items():
            put(oof, name, te_idx, yv, pv, pt)
        meta.append({"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx),
                     "alpha_macro": round(a_macro, 2), "alpha_rank": round(a_rank, 2)})

    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_oof[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_oof[ok])
        print(f"  {name:28s} AP={rows[name]['auprc']} AUROC={rows[name]['auroc']} "
              f"mF1={rows[name]['macro_f1']} wF1={rows[name]['wF1']} n={rows[name]['n']}",
              flush=True)

    sig = {}
    for name in methods:
        if name == "bge_lr":
            continue
        ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{name}_vs_bge_lr"] = paired_bootstrap(
            y_oof[ok], oof[name]["p"][ok], oof["bge_lr"]["p"][ok], c_oof[ok])

    json.dump({"fold_seed": args.fold_seed, "rows": rows, "fold_meta": meta,
               "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_arg_expert] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
