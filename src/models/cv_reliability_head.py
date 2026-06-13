"""Outer-train reliability head for CLAIMARC.

This is a stricter follow-up to ``cv_reliability_gate.py``.  The existing gate
fits a tiny rule only on each outer fold's validation carve.  Here we keep the
same grouped-CV outer folds, but train a small regularized reliability head on
the outer training split, select hyperparameters and thresholds on that fold's
validation carve, then evaluate only on the held-out test fold.

The head uses frozen BGE+LR probabilities, no-args CLAIMARC probabilities,
argument-aware CLAIMARC probabilities, retrieval features, and label-free
record metadata.  It does not use the outer test labels for fitting, threshold
selection, or feature scaling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.baselines import claim_text, evidence_text
from models.cv_eval import make_folds, val_carve
from models.data import load_split, resolve_bge_path
from models.fusion_eval import (apply_arf, best_thr, build_split_features, fit_arf,
                                load_bundles, macro, paired_bootstrap)
from models.cv_reliability_gate import load_llm_scores, meta_features, rank01, rec_features


def metric_row(y, p, yhat, c):
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
    return float(thr)


def pair_ids_for(recs, idx):
    return [recs[i].get("pair_id", "") for i in idx]


def assert_bundle_order(bundle, split, recs, idx, label):
    got = list(bundle[split].get("pair_id", []))
    exp = pair_ids_for(recs, idx)
    if got and got != exp:
        bad = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), None)
        raise ValueError(
            f"{label} {split} pair_id order mismatch at {bad}: "
            f"bundle={got[bad] if bad is not None else None}, "
            f"recs={exp[bad] if bad is not None else None}"
        )


def encode_all_bge(recs, batch_size=64):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(resolve_bge_path(), device=device)
    claims = model.encode([claim_text(r) for r in recs], normalize_embeddings=True,
                          batch_size=batch_size, show_progress_bar=False)
    evidence = model.encode([evidence_text(r) for r in recs], normalize_embeddings=True,
                            batch_size=batch_size, show_progress_bar=False)
    claims = np.asarray(claims, float)
    evidence = np.asarray(evidence, float)
    return np.concatenate([claims, evidence, claims - evidence, claims * evidence], axis=1)


def fit_bge_lr(X_all, y_all, c_all, train_idx, val_idx, test_idx):
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(X_all[train_idx], y_all[train_idx],
            sample_weight=np.clip(c_all[train_idx], 0.05, None))
    return {
        "train": clf.predict_proba(X_all[train_idx])[:, 1],
        "val": clf.predict_proba(X_all[val_idx])[:, 1],
        "test": clf.predict_proba(X_all[test_idx])[:, 1],
    }


def rankavg_scores(base):
    return {
        "rankavg_args_no_bge": np.mean(
            [rank01(base[k]) for k in ("args_pcls", "noargs_pcls", "bge_lr")], axis=0),
        "rankavg_no_bge": 0.5 * rank01(base["noargs_pcls"]) + 0.5 * rank01(base["bge_lr"]),
        "rankavg_args_bge": 0.5 * rank01(base["args_pcls"]) + 0.5 * rank01(base["bge_lr"]),
    }


def fit_meta_head(ytr, ctr, yv, cv, Xtr, Xv, Xt, objective):
    best = None
    for class_weight in (None, "balanced"):
        for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xvs = scaler.transform(Xv)
            Xts = scaler.transform(Xt)
            clf = LogisticRegression(
                C=c_value,
                max_iter=4000,
                class_weight=class_weight,
                solver="liblinear",
            )
            clf.fit(Xtrs, ytr, sample_weight=np.clip(ctr, 0.05, None))
            pv = clf.predict_proba(Xvs)[:, 1]
            pt = clf.predict_proba(Xts)[:, 1]
            thr = best_thr(yv, pv)
            mf = macro(yv, (pv >= thr).astype(int), w=None)
            wmf = macro(yv, (pv >= thr).astype(int), w=np.clip(cv, 0.05, None))
            ap = average_precision_score(yv, pv)
            au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
            if objective == "macro":
                score = mf + 0.10 * ap + 0.05 * au
            elif objective == "weighted":
                score = wmf + 0.08 * ap + 0.04 * au
            else:
                score = ap + 0.50 * au + 0.05 * mf
            if best is None or score > best[0]:
                best = (score, c_value, class_weight, pv, pt, thr)
    return {
        "C": best[1],
        "class_weight": best[2] or "none",
        "pv": best[3],
        "pt": best[4],
        "thr": float(best[5]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--noargs_tmp", required=True)
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--llm_pred", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=2)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="data/final/cleancl/cv_reliability_head.json")
    ap.add_argument("--bge_feature_cache", default="")
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], float)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    cats = sorted({str(r.get("category", "")) for r in recs})
    confs = sorted({str(r.get("confidence", "")) for r in recs})
    llm_scores = load_llm_scores(args.llm_pred)

    cache = Path(args.bge_feature_cache) if args.bge_feature_cache else None
    if cache and cache.exists():
        X_bge = np.load(cache)
        print(f"[bge_cache] loaded {cache} {X_bge.shape}", flush=True)
    else:
        print("[bge_cache] encoding all records once", flush=True)
        X_bge = encode_all_bge(recs)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, X_bge)
            print(f"[bge_cache] saved {cache} {X_bge.shape}", flush=True)

    oof = {"_n": len(recs)}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        missing = [p for p in no_paths + ar_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing fold {fi}: {missing}")

        no_b = load_bundles(no_paths)
        ar_b = load_bundles(ar_paths)
        assert_bundle_order(no_b[0], "train", recs, tr_idx, f"noargs fold {fi}")
        assert_bundle_order(no_b[0], "val", recs, va_idx, f"noargs fold {fi}")
        assert_bundle_order(no_b[0], "test", recs, te_idx, f"noargs fold {fi}")
        assert_bundle_order(ar_b[0], "train", recs, tr_idx, f"args fold {fi}")
        assert_bundle_order(ar_b[0], "val", recs, va_idx, f"args fold {fi}")
        assert_bundle_order(ar_b[0], "test", recs, te_idx, f"args fold {fi}")

        _, p_no_tr, ytr, ctr, _ = build_split_features(no_b, "train")
        _, p_no_v, yv, cv, _ = build_split_features(no_b, "val")
        _, p_no_t, yt, ct, _ = build_split_features(no_b, "test")
        Xtr_ar, p_ar_tr, ytr2, _, _ = build_split_features(ar_b, "train")
        Xv_ar, p_ar_v, yv2, _, _ = build_split_features(ar_b, "val")
        Xt_ar, p_ar_t, yt2, _, _ = build_split_features(ar_b, "test")
        if not (np.all(ytr == ytr2) and np.all(yv == yv2) and np.all(yt == yt2)):
            raise ValueError(f"Fold {fi} noargs/args labels differ")

        arf = fit_arf(Xtr_ar, ytr, ctr)
        p_arf_tr = apply_arf(arf, Xtr_ar)
        p_arf_v = apply_arf(arf, Xv_ar)
        p_arf_t = apply_arf(arf, Xt_ar)
        bge = fit_bge_lr(X_bge, y_all, c_all, np.asarray(tr_idx), np.asarray(va_idx), np.asarray(te_idx))

        train_recs = [recs[i] for i in tr_idx]
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        p_llm_tr = np.asarray([llm_scores.get(r.get("pair_id", ""), 0.5) for r in train_recs], float)
        p_llm_v = np.asarray([llm_scores.get(r.get("pair_id", ""), 0.5) for r in val_recs], float)
        p_llm_t = np.asarray([llm_scores.get(r.get("pair_id", ""), 0.5) for r in test_recs], float)

        base_tr = {
            "bge_lr": bge["train"],
            "noargs_pcls": p_no_tr,
            "args_pcls": p_ar_tr,
            "args_arf": p_arf_tr,
            "args_knn": Xtr_ar[:, 1],
            "llm": p_llm_tr,
        }
        base_v = {
            "bge_lr": bge["val"],
            "noargs_pcls": p_no_v,
            "args_pcls": p_ar_v,
            "args_arf": p_arf_v,
            "args_knn": Xv_ar[:, 1],
            "llm": p_llm_v,
        }
        base_t = {
            "bge_lr": bge["test"],
            "noargs_pcls": p_no_t,
            "args_pcls": p_ar_t,
            "args_arf": p_arf_t,
            "args_knn": Xt_ar[:, 1],
            "llm": p_llm_t,
        }

        for name in ("bge_lr", "noargs_pcls", "args_pcls", "args_arf", "args_knn", "llm"):
            put(oof, name, te_idx, yv, base_v[name], base_t[name])
        ranks_v = rankavg_scores(base_v)
        ranks_t = rankavg_scores(base_t)
        for name in ranks_v:
            put(oof, name, te_idx, yv, ranks_v[name], ranks_t[name])

        rec_Xtr = rec_features(train_recs, cats, confs, llm_scores)
        rec_Xv = rec_features(val_recs, cats, confs, llm_scores)
        rec_Xt = rec_features(test_recs, cats, confs, llm_scores)
        Xtr = meta_features(base_tr, rec_Xtr)
        Xv = meta_features(base_v, rec_Xv)
        Xt = meta_features(base_t, rec_Xt)

        fm = {"fold": fi, "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx)}
        for obj in ("macro", "weighted", "rank"):
            fit = fit_meta_head(ytr, ctr, yv, cv, Xtr, Xv, Xt, obj)
            nm = f"reliability_head_{obj}"
            oof.setdefault(nm, {"p": np.full(oof["_n"], np.nan),
                                "yhat": np.full(oof["_n"], np.nan)})
            oof[nm]["p"][te_idx] = fit["pt"]
            oof[nm]["yhat"][te_idx] = (fit["pt"] >= fit["thr"]).astype(int)
            fm[f"{nm}_C"] = fit["C"]
            fm[f"{nm}_class_weight"] = fit["class_weight"]
            fm[f"{nm}_thr"] = round(float(fit["thr"]), 3)
        fold_meta.append(fm)

    methods = [m for m in oof if m != "_n"]
    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = metric_row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])

    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Top reliability-head candidates ===", flush=True)
    for name in ranked[:25]:
        r = rows[name]
        print(f"{name:36s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig_names = set(ranked[:20])
    for key in ("auprc", "auroc", "wF1"):
        sig_names.update(sorted(rows, key=lambda m: rows[m][key], reverse=True)[:10])
    sig_names.update(["bge_lr", "noargs_pcls", "args_pcls",
                      "rankavg_no_bge", "rankavg_args_no_bge", "rankavg_args_bge"])
    sig = {}
    for name in sorted(sig_names):
        if name == "bge_lr":
            continue
        ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{name}_vs_bge_lr"] = paired_bootstrap(
            y_all[ok], oof[name]["p"][ok], oof["bge_lr"]["p"][ok], c_all[ok],
            n_boot=args.n_boot)
        s = sig[f"{name}_vs_bge_lr"]
        r = rows[name]
        print(f"{name:36s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f} "
              f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
              f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
              f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
              flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows, "fold_meta": fold_meta,
           "significance": sig,
           "notes": "Head trained on each outer fold train split, selected on val carve, evaluated on held-out test."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_reliability_head] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
