"""Evaluate the main CLAIMARC-Hybrid on saved grouped-CV bundles.

Main hybrid = rank-average(argument-ARF retrieval expert, fair BGE+LR).
It needs only argument CLAIMARC fold bundles and argument-aware BGE+LR fold
predictions, so it is cheap to run across repeated grouped splits.
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
from models.fusion_eval import (apply_arf, best_thr, build_split_features, fit_arf,
                                load_bundles, macro, paired_bootstrap)


def rank01(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def fit_pair(y, pv_a, pv_b, objective):
    best = (-1e9, 1.0)
    for a in np.linspace(0, 1, 21):
        p = a * pv_a + (1 - a) * pv_b
        thr = best_thr(y, p)
        mf = macro(y, (p >= thr).astype(int))
        ap = average_precision_score(y, p)
        au = roc_auc_score(y, p)
        score = mf + 0.15 * ap if objective == "macro" else ap + 0.5 * au
        if score > best[0]:
            best = (float(score), float(a))
    return best[1]


def score_candidate(y, p, objective):
    thr = best_thr(y, p)
    mf = macro(y, (p >= thr).astype(int))
    ap = average_precision_score(y, p)
    au = roc_auc_score(y, p)
    if objective == "macro":
        return mf + 0.15 * ap
    if objective == "rank":
        return ap + 0.5 * au
    return mf + 0.05 * ap + 0.05 * au


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
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--out", default="data/final/cleancl/cv_main_hybrid.json")
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    n = len(recs)
    y_oof = np.array([int(r["y"]) for r in recs], float)
    c_oof = np.array([float(r.get("c", 0.05)) for r in recs], float)
    methods = [
        "args_pcls", "args_arf", "bge_lr",
        "rankavg_args_pcls+bge_lr",
        "rankavg_args_arf+bge_lr",
        "rankavg_args_pcls+args_arf",
        "rankavg_args_pcls+args_arf+bge_lr",
        "prob_macro_args_pcls+bge_lr",
        "prob_rank_args_pcls+bge_lr",
        "prob_macro_args_arf+bge_lr",
        "prob_rank_args_arf+bge_lr",
        "val_select_macro",
        "val_select_rank",
        "val_select_balanced",
    ]
    oof = {m: {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)} for m in methods}
    meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing fold {fi}: {missing}")

        ar_b = load_bundles(ar_paths)
        Xv, pcls_v, yv, cv, _ = build_split_features(ar_b, "val")
        Xt, pcls_t, yt, ct, _ = build_split_features(ar_b, "test")
        arf = fit_arf(Xv, yv, cv)
        p_arf_v, p_arf_t = apply_arf(arf, Xv), apply_arf(arf, Xt)
        bge = torch.load(bge_path, weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        rank_h_v = 0.5 * rank01(p_arf_v) + 0.5 * rank01(p_bge_v)
        rank_h_t = 0.5 * rank01(p_arf_t) + 0.5 * rank01(p_bge_t)
        rank_pcls_bge_v = 0.5 * rank01(pcls_v) + 0.5 * rank01(p_bge_v)
        rank_pcls_bge_t = 0.5 * rank01(pcls_t) + 0.5 * rank01(p_bge_t)
        rank_pcls_arf_v = 0.5 * rank01(pcls_v) + 0.5 * rank01(p_arf_v)
        rank_pcls_arf_t = 0.5 * rank01(pcls_t) + 0.5 * rank01(p_arf_t)
        rank_all_v = np.mean([rank01(pcls_v), rank01(p_arf_v), rank01(p_bge_v)], axis=0)
        rank_all_t = np.mean([rank01(pcls_t), rank01(p_arf_t), rank01(p_bge_t)], axis=0)
        a_pm = fit_pair(yv, pcls_v, p_bge_v, "macro")
        a_pr = fit_pair(yv, pcls_v, p_bge_v, "rank")
        a_am = fit_pair(yv, p_arf_v, p_bge_v, "macro")
        a_ar = fit_pair(yv, p_arf_v, p_bge_v, "rank")

        fold_probs = {
            "args_pcls": (pcls_v, pcls_t),
            "args_arf": (p_arf_v, p_arf_t),
            "bge_lr": (p_bge_v, p_bge_t),
            "rankavg_args_pcls+bge_lr": (rank_pcls_bge_v, rank_pcls_bge_t),
            "rankavg_args_arf+bge_lr": (rank_h_v, rank_h_t),
            "rankavg_args_pcls+args_arf": (rank_pcls_arf_v, rank_pcls_arf_t),
            "rankavg_args_pcls+args_arf+bge_lr": (rank_all_v, rank_all_t),
            "prob_macro_args_pcls+bge_lr": (
                a_pm * pcls_v + (1 - a_pm) * p_bge_v,
                a_pm * pcls_t + (1 - a_pm) * p_bge_t,
            ),
            "prob_rank_args_pcls+bge_lr": (
                a_pr * pcls_v + (1 - a_pr) * p_bge_v,
                a_pr * pcls_t + (1 - a_pr) * p_bge_t,
            ),
            "prob_macro_args_arf+bge_lr": (
                a_am * p_arf_v + (1 - a_am) * p_bge_v,
                a_am * p_arf_t + (1 - a_am) * p_bge_t,
            ),
            "prob_rank_args_arf+bge_lr": (
                a_ar * p_arf_v + (1 - a_ar) * p_bge_v,
                a_ar * p_arf_t + (1 - a_ar) * p_bge_t,
            ),
        }
        for objective, out_name in (
            ("macro", "val_select_macro"),
            ("rank", "val_select_rank"),
            ("balanced", "val_select_balanced"),
        ):
            choices = list(fold_probs)
            best_name = max(choices, key=lambda nm: score_candidate(yv, fold_probs[nm][0], objective))
            fold_probs[out_name] = fold_probs[best_name]

        for name, (pv, pt) in fold_probs.items():
            put(oof, name, te_idx, yv, pv, pt)
        meta.append({"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx),
                     "alpha_pcls_bge_macro": round(float(a_pm), 2),
                     "alpha_pcls_bge_rank": round(float(a_pr), 2),
                     "alpha_arf_bge_macro": round(float(a_am), 2),
                     "alpha_arf_bge_rank": round(float(a_ar), 2),
                     "select_macro": max(list(fold_probs)[:-3],
                                         key=lambda nm: score_candidate(yv, fold_probs[nm][0], "macro")),
                     "select_rank": max(list(fold_probs)[:-3],
                                       key=lambda nm: score_candidate(yv, fold_probs[nm][0], "rank")),
                     "select_balanced": max(list(fold_probs)[:-3],
                                            key=lambda nm: score_candidate(yv, fold_probs[nm][0], "balanced"))})

    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_oof[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_oof[ok])
        print(f"{name:28s} AP={rows[name]['auprc']} AUROC={rows[name]['auroc']} "
              f"mF1={rows[name]['macro_f1']} wF1={rows[name]['wF1']} n={rows[name]['n']}",
              flush=True)

    sig = {}
    for main_name in methods:
        if main_name == "bge_lr":
            continue
        ok = (~np.isnan(oof[main_name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{main_name}_vs_bge_lr"] = paired_bootstrap(
            y_oof[ok], oof[main_name]["p"][ok], oof["bge_lr"]["p"][ok], c_oof[ok],
            n_boot=args.n_boot)
        s = sig[f"{main_name}_vs_bge_lr"]
        print(f"{main_name} vs bge_lr: "
              f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
              f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
              f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
              flush=True)

    json.dump({"fold_seed": args.fold_seed, "rows": rows, "fold_meta": meta,
               "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_main_hybrid] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
