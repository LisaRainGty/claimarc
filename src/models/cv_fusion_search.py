"""Leakage-safe OOF fusion search for CLAIMARC argument experiments.

This script reuses saved fold bundles. For each outer fold it fits only tiny
fusion rules on that fold's validation carve, then applies them to the held-out
test fold. It is meant for research iteration after expensive CLAIMARC training
has already finished.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
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


def _score(y, p, objective):
    thr = best_thr(y, p)
    mf = macro(y, (p >= thr).astype(int))
    ap = average_precision_score(y, p)
    au = roc_auc_score(y, p)
    if objective == "macro":
        return mf + 0.15 * ap
    if objective == "wf1":
        return mf + 0.05 * ap
    return ap + 0.5 * au


def fit_pair(y, pv_a, pv_b, objective):
    best = (-1e9, 1.0)
    for a in np.linspace(0, 1, 21):
        p = a * pv_a + (1 - a) * pv_b
        s = _score(y, p, objective)
        if s > best[0]:
            best = (float(s), float(a))
    return best[1]


def simplex(n, step=0.1):
    units = int(round(1.0 / step))
    for counts in itertools.product(range(units + 1), repeat=n):
        if sum(counts) == units:
            yield np.asarray(counts, dtype=float) / units


def fit_combo(y, val_cols, objective, step=0.1):
    best = (-1e9, None)
    V = np.column_stack(val_cols)
    for w in simplex(V.shape[1], step):
        p = V @ w
        s = _score(y, p, objective)
        if s > best[0]:
            best = (float(s), w)
    return best[1]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_stack(y, cv, val_cols, test_cols, c_value):
    Xv = np.column_stack([logit(p) for p in val_cols])
    Xt = np.column_stack([logit(p) for p in test_cols])
    clf = LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced")
    clf.fit(Xv, y, sample_weight=np.clip(cv, 0.05, None))
    return clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1]


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
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--noargs_tmp", required=True)
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--compare_baselines", nargs="*", default=["bge_lr"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sig_top", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--force_sig", nargs="*", default=[])
    ap.add_argument("--force_sig_only", action="store_true")
    ap.add_argument("--out", default="data/final/cleancl/cv_fusion_search.json")
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_oof = np.array([int(r["y"]) for r in recs], float)
    c_oof = np.array([float(r.get("c", 0.05)) for r in recs], float)

    oof = {"_n": len(recs)}
    meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
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
        arf = fit_arf(Xv_ar, yv, cv)
        p_arf_v, p_arf_t = apply_arf(arf, Xv_ar), apply_arf(arf, Xt_ar)
        p_knn_v, p_knn_t = Xv_ar[:, 1], Xt_ar[:, 1]
        bge = torch.load(bge_path, weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)

        base = {
            "noargs_pcls": (p_no_v, p_no_t),
            "args_pcls": (p_ar_v, p_ar_t),
            "args_arf": (p_arf_v, p_arf_t),
            "args_knn": (p_knn_v, p_knn_t),
            "bge_lr": (p_bge_v, p_bge_t),
        }
        for kind in args.compare_baselines:
            if kind == "bge_lr":
                continue
            p = f"{args.bge_tmp}/cv_{kind}_f{fi}.pt"
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing compare baseline fold {fi}: {p}")
            d = torch.load(p, weights_only=False)
            base[kind] = (np.asarray(d["val"]["p"], float),
                          np.asarray(d["test"]["p"], float))
        for name, (pv, pt) in base.items():
            put(oof, name, te_idx, yv, pv, pt)

        fold_meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx)}
        pairs = [
            ("args_pcls", "bge_lr"),
            ("args_pcls", "args_arf"),
            ("args_pcls", "noargs_pcls"),
            ("noargs_pcls", "args_arf"),
            ("args_arf", "bge_lr"),
        ]
        for a, b in pairs:
            for obj in ("macro", "rank"):
                w = fit_pair(yv, base[a][0], base[b][0], obj)
                name = f"prob_{obj}_{a}+{b}"
                put(oof, name, te_idx, yv,
                    w * base[a][0] + (1 - w) * base[b][0],
                    w * base[a][1] + (1 - w) * base[b][1])
                fold_meta[f"{name}_w_{a}"] = round(float(w), 2)
            name = f"rankavg_{a}+{b}"
            put(oof, name, te_idx, yv,
                0.5 * rank01(base[a][0]) + 0.5 * rank01(base[b][0]),
                0.5 * rank01(base[a][1]) + 0.5 * rank01(base[b][1]))

        combos = [
            ("args_pcls", "noargs_pcls", "args_arf"),
            ("args_pcls", "args_arf", "bge_lr"),
            ("args_pcls", "noargs_pcls", "bge_lr"),
            ("args_pcls", "noargs_pcls", "args_arf", "bge_lr"),
        ]
        for combo in combos:
            for obj in ("macro", "rank"):
                w = fit_combo(yv, [base[k][0] for k in combo], obj, step=0.1)
                name = f"combo_{obj}_{'+'.join(combo)}"
                put(oof, name, te_idx, yv,
                    np.column_stack([base[k][0] for k in combo]) @ w,
                    np.column_stack([base[k][1] for k in combo]) @ w)
                fold_meta[f"{name}_w"] = [round(float(x), 2) for x in w]
            name = f"rankavg_{'+'.join(combo)}"
            put(oof, name, te_idx, yv,
                np.mean([rank01(base[k][0]) for k in combo], axis=0),
                np.mean([rank01(base[k][1]) for k in combo], axis=0))

        stack_cols = ("args_pcls", "noargs_pcls", "args_arf", "bge_lr")
        for c_value in (0.05, 0.1, 0.3, 1.0):
            pv, pt = fit_stack(yv, cv, [base[k][0] for k in stack_cols],
                               [base[k][1] for k in stack_cols], c_value)
            put(oof, f"stack_c{c_value:g}_{'+'.join(stack_cols)}", te_idx, yv, pv, pt)
        meta.append(fold_meta)

    methods = [m for m in oof if m != "_n"]
    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_oof[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_oof[ok])

    ranked = sorted(rows.items(), key=lambda kv: (kv[1]["macro_f1"], kv[1]["auprc"]), reverse=True)
    print("=== Top candidates by row metrics ===", flush=True)
    for name, r in ranked[:20]:
        print(f"{name:70s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    compare_baselines = [b for b in args.compare_baselines if b in rows]
    if "bge_lr" not in compare_baselines and "bge_lr" in rows:
        compare_baselines.insert(0, "bge_lr")

    sig_names = set()
    if not args.force_sig_only:
        for key in ("macro_f1", "auprc", "auroc", "wF1"):
            top = sorted((m for m in methods if m not in compare_baselines),
                         key=lambda m: rows[m][key], reverse=True)[:args.sig_top]
            sig_names.update(top)
        sig_names.update(["args_pcls", "prob_macro_noargs_pcls+args_arf",
                          "rankavg_noargs_pcls+args_arf"])
    sig_names.update(args.force_sig)

    sig = {}
    print(f"=== Bootstrap for {len(sig_names)} selected candidates vs {compare_baselines} "
          f"(n_boot={args.n_boot}) ===",
          flush=True)
    for name in sorted(sig_names):
        r = rows[name]
        for base_name in compare_baselines:
            ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof[base_name]["p"]))
            key = f"{name}_vs_{base_name}"
            sig[key] = paired_bootstrap(
                y_oof[ok], oof[name]["p"][ok], oof[base_name]["p"][ok], c_oof[ok],
                n_boot=args.n_boot)
            s = sig[key]
            print(f"{name:70s} vs {base_name:12s} AP={r['auprc']:.4f} "
                  f"AUROC={r['auroc']:.4f} mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}"
                  f" dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']})"
                  f" dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']})"
                  f" dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                  flush=True)

    json.dump({"fold_seed": args.fold_seed, "rows": rows, "fold_meta": meta,
               "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_fusion_search] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
