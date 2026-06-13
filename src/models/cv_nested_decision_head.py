"""Nested source/NLI decision head for the fs1 hard split.

The decision head is trained inside each outer grouped-CV fold.  Supervised
base scores used as meta features are inner-OOF on the outer train split; outer
validation/test scores are produced by models fit only on that outer train
split.  This keeps the final Macro-F1 decision diagnostic leakage-safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.cv_dual_head_router import paired_bootstrap_dual
from models.cv_eval import make_folds, val_carve
from models.cv_reliability_gate import rank01
from models.data import load_split
from models.fusion_eval import best_thr, macro


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def source_count(rec):
    ev = rec.get("evidence_count", {}) or {}
    return int(ev.get("params", 0) or 0) + int(ev.get("ocr", 0) or 0) + int(ev.get("vlm", 0) or 0)


def source_len(rec):
    total = 0
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for item in rec.get(key, []) or []:
            total += len(str(item.get(field, "") or ""))
    return total


def arg_len(rec):
    args = rec.get("arguments", {}) or {}
    return sum(len(str(args.get(k, "") or "")) for k in
               ("supporting_argument", "refuting_argument", "evidence_gap"))


def record_features(recs, cats, confs):
    rows = []
    for rec in recs:
        claim = rec.get("claim", {}) or {}
        vals = [
            float(rec.get("coverage", 0.0) or 0.0),
            float(source_count(rec)),
            np.log1p(source_len(rec)),
            np.log1p(arg_len(rec)),
            float(bool(claim.get("has_claim_srt", False))),
            np.log1p(len(claim.get("segments", []) or [])),
            np.log1p(len(str(claim.get("passage", "") or ""))),
        ]
        cat = str(rec.get("category", ""))
        vals.extend([float(cat == c) for c in cats])
        conf = str(rec.get("confidence", ""))
        vals.extend([float(conf == c) for c in confs])
        rows.append(vals)
    return np.asarray(rows, float)


def fit_bge_lr(Xtr, ytr, ctr, Xq):
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
    return clf.predict_proba(Xq)[:, 1], clf


def fit_nli_hgb_fixed(Xtr, ytr, ctr, Xq):
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06, l2_regularization=0.1, max_leaf_nodes=15,
        max_iter=120, random_state=0)
    clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
    return clf.predict_proba(Xq)[:, 1], clf


def safe_inner_splits(idx, y_all, g_all, seed, n_splits):
    y = y_all[idx]
    groups = g_all[idx]
    min_class = int(np.bincount(y.astype(int), minlength=2).min())
    n = min(int(n_splits), min_class)
    if n >= 2 and len(set(groups.tolist())) >= n:
        try:
            splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
            for tr, te in splitter.split(np.zeros(len(idx)), y, groups):
                yield idx[tr], idx[te]
            return
        except Exception:
            pass
    n = min(int(n_splits), min_class)
    if n < 2:
        raise ValueError("not enough labels for inner OOF")
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, te in splitter.split(np.zeros(len(idx)), y):
        yield idx[tr], idx[te]


def inner_oof_scores(pair_x, nli_x, y_all, c_all, g_all, train_idx, inner_folds, seed):
    p_bge = np.full(len(train_idx), np.nan, float)
    p_nli = np.full(len(train_idx), np.nan, float)
    local = {int(v): i for i, v in enumerate(train_idx)}
    for inner_tr, inner_te in safe_inner_splits(np.asarray(train_idx), y_all, g_all,
                                                seed=seed, n_splits=inner_folds):
        p, _ = fit_bge_lr(pair_x[inner_tr], y_all[inner_tr], c_all[inner_tr], pair_x[inner_te])
        pn, _ = fit_nli_hgb_fixed(nli_x[inner_tr], y_all[inner_tr],
                                  c_all[inner_tr], nli_x[inner_te])
        for j, val in zip(inner_te, p):
            p_bge[local[int(j)]] = val
        for j, val in zip(inner_te, pn):
            p_nli[local[int(j)]] = val
    if np.isnan(p_bge).any() or np.isnan(p_nli).any():
        raise RuntimeError("inner OOF left missing scores")
    return p_bge, p_nli


def meta_features(nli_x, rec_x, p_bge, p_nli):
    rankmix25 = 0.25 * rank01(p_nli) + 0.75 * rank01(p_bge)
    rankmix50 = 0.50 * rank01(p_nli) + 0.50 * rank01(p_bge)
    cols = [
        p_bge,
        p_nli,
        rankmix25,
        rankmix50,
        p_nli - p_bge,
        np.abs(p_nli - p_bge),
        np.abs(p_bge - 0.5),
        np.abs(p_nli - 0.5),
    ]
    return np.column_stack(cols + [rec_x, nli_x])


def fit_meta_lr(Xtr, ytr, ctr, Xv, yv, cv, Xt):
    best = None
    for class_weight in (None, "balanced"):
        for c_value in (0.003, 0.01, 0.03, 0.1, 0.3):
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xvs = scaler.transform(Xv)
            Xts = scaler.transform(Xt)
            clf = LogisticRegression(
                C=c_value, max_iter=3000, class_weight=class_weight, solver="liblinear")
            clf.fit(Xtrs, ytr, sample_weight=np.clip(ctr, 0.05, None))
            pv = clf.predict_proba(Xvs)[:, 1]
            pt = clf.predict_proba(Xts)[:, 1]
            thr = best_thr(yv, pv)
            score = macro(yv, (pv >= thr).astype(int)) + 0.05 * average_precision_score(yv, pv)
            if best is None or score > best[0]:
                best = (score, pv, pt, thr, c_value, class_weight or "none")
    return best


def fit_meta_hgb(Xtr, ytr, ctr, Xv, yv, Xt):
    best = None
    for lr in (0.03, 0.06):
        for l2 in (0.01, 0.1, 1.0):
            for leaves in (7, 15):
                clf = HistGradientBoostingClassifier(
                    learning_rate=lr, l2_regularization=l2, max_leaf_nodes=leaves,
                    max_iter=140, random_state=0)
                clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
                pv = clf.predict_proba(Xv)[:, 1]
                pt = clf.predict_proba(Xt)[:, 1]
                thr = best_thr(yv, pv)
                score = macro(yv, (pv >= thr).astype(int)) + 0.05 * average_precision_score(yv, pv)
                if best is None or score > best[0]:
                    best = (score, pv, pt, thr, lr, l2, leaves)
    return best


def put(oof, name, idx, yv, pv, pt):
    thr = best_thr(yv, pv)
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)
    return float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nli_cache", required=True)
    ap.add_argument("--pair_cache", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs_by = load_split(args.dataset)
    recs = recs_by["train"] + recs_by["val"] + recs_by["test"]
    nli_x = np.load(args.nli_cache)["X"]
    pair_npz = np.load(args.pair_cache)
    pair_x = pair_npz["pair"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    cats = sorted({str(r.get("category", "")) for r in recs})
    confs = sorted({str(r.get("confidence", "")) for r in recs})
    rec_x = record_features(recs, cats, confs)
    oof = {"_n": len(recs)}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        tr_idx = np.asarray(tr_idx)
        va_idx = np.asarray(va_idx)
        te_idx = np.asarray(te_idx)
        p_bge_tr, p_nli_tr = inner_oof_scores(
            pair_x, nli_x, y_all, c_all, g_all, tr_idx,
            inner_folds=args.inner_folds, seed=args.fold_seed * 1000 + fi)
        p_bge_v, bge_model = fit_bge_lr(pair_x[tr_idx], y_all[tr_idx], c_all[tr_idx], pair_x[va_idx])
        p_bge_t = bge_model.predict_proba(pair_x[te_idx])[:, 1]
        p_nli_v, nli_model = fit_nli_hgb_fixed(
            nli_x[tr_idx], y_all[tr_idx], c_all[tr_idx], nli_x[va_idx])
        p_nli_t = nli_model.predict_proba(nli_x[te_idx])[:, 1]
        Xtr = meta_features(nli_x[tr_idx], rec_x[tr_idx], p_bge_tr, p_nli_tr)
        Xv = meta_features(nli_x[va_idx], rec_x[va_idx], p_bge_v, p_nli_v)
        Xt = meta_features(nli_x[te_idx], rec_x[te_idx], p_bge_t, p_nli_t)

        ytr, yv = y_all[tr_idx], y_all[va_idx]
        ctr, cv = c_all[tr_idx], c_all[va_idx]
        lr_fit = fit_meta_lr(Xtr, ytr, ctr, Xv, yv, cv, Xt)
        hgb_fit = fit_meta_hgb(Xtr, ytr, ctr, Xv, yv, Xt)
        for name, fit in (("nested_decision_lr", lr_fit), ("nested_decision_hgb", hgb_fit)):
            oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                                  "yhat": np.full(oof["_n"], np.nan)})
            oof[name]["p"][te_idx] = fit[2]
            oof[name]["yhat"][te_idx] = (fit[2] >= fit[3]).astype(int)
        import torch
        bge_saved = torch.load(Path(args.bge_tmp) / f"cv_bge_lr_f{fi}.pt",
                               map_location="cpu", weights_only=False)
        put(oof, "bge_lr", te_idx, np.asarray(bge_saved["val"]["y"], int),
            np.asarray(bge_saved["val"]["p"], float), np.asarray(bge_saved["test"]["p"], float))
        score25_v = 0.25 * rank01(p_nli_v) + 0.75 * rank01(p_bge_v)
        score25_t = 0.25 * rank01(p_nli_t) + 0.75 * rank01(p_bge_t)
        for name, fit in (("nested_decision_lr", lr_fit), ("nested_decision_hgb", hgb_fit)):
            method = f"dual_score=rankmix_nli25_hgb_bge__decision={name}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = score25_t
            oof[method]["yhat"][te_idx] = (fit[2] >= fit[3]).astype(int)
        put(oof, "rankmix_nli25_hgb_bge", te_idx, yv, score25_v, score25_t)
        fold_meta.append({
            "fold": fi,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "lr": {"thr": round(float(lr_fit[3]), 3), "C": lr_fit[4],
                   "class_weight": lr_fit[5]},
            "hgb": {"thr": round(float(hgb_fit[3]), 3), "lr": hgb_fit[4],
                    "l2": hgb_fit[5], "leaves": hgb_fit[6]},
            "nli_hgb_fixed": {"lr": 0.06, "l2": 0.1, "leaves": 15},
        })
        print(f"[fold {fi}] done", flush=True)

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Nested decision candidates ===", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:72s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig = {}
    if args.n_boot > 0:
        for name in ranked:
            if name == "bge_lr":
                continue
            ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
            sig[f"{name}_vs_bge_lr"] = paired_bootstrap_dual(
                y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
                oof["bge_lr"]["p"][ok], oof["bge_lr"]["yhat"][ok],
                n_boot=args.n_boot)
            s = sig[f"{name}_vs_bge_lr"]
            r = rows[name]
            print(f"{name:72s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
                  f"mF1={r['macro_f1']:.4f} "
                  f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                  f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                  f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                  flush=True)

    out = {"fold_seed": args.fold_seed, "inner_folds": args.inner_folds,
           "rows": rows, "fold_meta": fold_meta, "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_nested_decision_head] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
