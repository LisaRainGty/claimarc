"""Leakage-safe reliability/gating diagnostics for CLAIMARC.

The goal is to test whether the RACL/argument branches can be used only when
they look more reliable than the strong BGE+LR baseline. Every gate is fitted
on the validation carve inside each outer grouped-CV fold, then applied to that
fold's held-out test samples.
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

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import (apply_arf, best_thr, build_split_features, fit_arf,
                                load_bundles, macro, paired_bootstrap)


def rank01(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


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
    return float(thr)


def load_llm_scores(path):
    if not path:
        return {}
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("pair_id") and r.get("risk_score") is not None:
                scores[r["pair_id"]] = float(r["risk_score"])
    return scores


def rec_features(recs, cats, confs, llm_scores):
    rows = []
    for r in recs:
        ev = r.get("evidence_count", {}) or {}
        claim = r.get("claim", {}) or {}
        args = r.get("arguments", {}) or {}
        segs = claim.get("segments", []) or []
        passage = claim.get("passage", "") or ""
        arg_len = sum(len(str(args.get(k, "") or "")) for k in
                      ("supporting_argument", "refuting_argument", "evidence_gap"))
        risk_cues = args.get("risk_cues", []) or []
        vals = [
            float(r.get("coverage", 0.0) or 0.0),
            float(ev.get("params", 0) or 0),
            float(ev.get("ocr", 0) or 0),
            float(ev.get("vlm", 0) or 0),
            float(bool(claim.get("has_claim_srt", False))),
            np.log1p(len(segs)),
            np.log1p(len(passage)),
            np.log1p(arg_len),
            np.log1p(len(risk_cues)),
            llm_scores.get(r.get("pair_id", ""), 0.5),
        ]
        cat = str(r.get("category", ""))
        vals.extend([float(cat == c) for c in cats])
        conf = str(r.get("confidence", ""))
        vals.extend([float(conf == c) for c in confs])
        rows.append(vals)
    return np.asarray(rows, float)


def meta_features(base, rec_X):
    p_b = base["bge_lr"]
    cols = []
    for name in ("bge_lr", "noargs_pcls", "args_pcls", "args_arf", "args_knn", "llm"):
        p = base.get(name)
        if p is None:
            continue
        cols.extend([p, logit(p), np.abs(p - 0.5)])
    pairs = [("noargs_pcls", "bge_lr"), ("args_pcls", "bge_lr"),
             ("args_arf", "bge_lr"), ("llm", "bge_lr"),
             ("args_pcls", "noargs_pcls")]
    for a, b in pairs:
        if a in base and b in base:
            cols.extend([base[a] - base[b], np.abs(base[a] - base[b])])
    cols.extend([
        np.mean([rank01(base[k]) for k in ("bge_lr", "noargs_pcls", "args_pcls")], axis=0),
        np.min([np.abs(base[k] - 0.5) for k in ("bge_lr", "noargs_pcls", "args_pcls")], axis=0),
        np.max([np.abs(base[k] - 0.5) for k in ("bge_lr", "noargs_pcls", "args_pcls")], axis=0),
    ])
    return np.column_stack(cols + [rec_X])


def fit_meta_lr(yv, cv, Xv, Xt, objective):
    best = None
    for c_value in (0.01, 0.03, 0.05, 0.1, 0.3, 1.0):
        scaler = StandardScaler()
        Xvs = scaler.fit_transform(Xv)
        Xts = scaler.transform(Xt)
        clf = LogisticRegression(
            C=c_value,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )
        clf.fit(Xvs, yv, sample_weight=np.clip(cv, 0.05, None))
        pv = clf.predict_proba(Xvs)[:, 1]
        pt = clf.predict_proba(Xts)[:, 1]
        thr = best_thr(yv, pv)
        mf = macro(yv, (pv >= thr).astype(int))
        ap = average_precision_score(yv, pv)
        au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
        score = mf + 0.10 * ap if objective == "macro" else ap + 0.5 * au
        if best is None or score > best[0]:
            best = (score, c_value, pv, pt)
    return best[1], best[2], best[3]


def fit_uncertain_switch(yv, pv_bge, pt_bge, pv_alt, pt_alt, objective):
    best = None
    bconf_v = np.abs(pv_bge - 0.5)
    bconf_t = np.abs(pt_bge - 0.5)
    for t in np.linspace(0.02, 0.48, 24):
        use_alt_v = bconf_v <= t
        use_alt_t = bconf_t <= t
        pv = np.where(use_alt_v, pv_alt, pv_bge)
        pt = np.where(use_alt_t, pt_alt, pt_bge)
        thr = best_thr(yv, pv)
        mf = macro(yv, (pv >= thr).astype(int))
        ap = average_precision_score(yv, pv)
        au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
        score = mf + 0.10 * ap if objective == "macro" else ap + 0.5 * au
        if best is None or score > best[0]:
            best = (score, t, pv, pt, float(use_alt_v.mean()), float(use_alt_t.mean()))
    return best[1:]


def fit_conf_advantage_switch(yv, pv_bge, pt_bge, pv_alt, pt_alt, objective):
    best = None
    margin_v = np.abs(pv_alt - 0.5) - np.abs(pv_bge - 0.5)
    margin_t = np.abs(pt_alt - 0.5) - np.abs(pt_bge - 0.5)
    for m in np.linspace(-0.20, 0.20, 21):
        use_alt_v = margin_v >= m
        use_alt_t = margin_t >= m
        pv = np.where(use_alt_v, pv_alt, pv_bge)
        pt = np.where(use_alt_t, pt_alt, pt_bge)
        thr = best_thr(yv, pv)
        mf = macro(yv, (pv >= thr).astype(int))
        ap = average_precision_score(yv, pv)
        au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
        score = mf + 0.10 * ap if objective == "macro" else ap + 0.5 * au
        if best is None or score > best[0]:
            best = (score, m, pv, pt, float(use_alt_v.mean()), float(use_alt_t.mean()))
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--noargs_tmp", required=True)
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--llm_pred", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=1)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="data/final/cleancl/cv_reliability_gate.json")
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_oof = np.asarray([int(r["y"]) for r in recs], float)
    c_oof = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    cats = sorted({str(r.get("category", "")) for r in recs})
    confs = sorted({str(r.get("confidence", "")) for r in recs})
    llm_scores = load_llm_scores(args.llm_pred)
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
        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        p_llm_v = np.asarray([llm_scores.get(r.get("pair_id", ""), 0.5) for r in val_recs], float)
        p_llm_t = np.asarray([llm_scores.get(r.get("pair_id", ""), 0.5) for r in test_recs], float)

        base_v = {
            "bge_lr": p_bge_v,
            "noargs_pcls": p_no_v,
            "args_pcls": p_ar_v,
            "args_arf": p_arf_v,
            "args_knn": Xv_ar[:, 1],
            "llm": p_llm_v,
        }
        base_t = {
            "bge_lr": p_bge_t,
            "noargs_pcls": p_no_t,
            "args_pcls": p_ar_t,
            "args_arf": p_arf_t,
            "args_knn": Xt_ar[:, 1],
            "llm": p_llm_t,
        }
        rankavg_v = {
            "rankavg_args_no_bge": np.mean(
                [rank01(base_v[k]) for k in ("args_pcls", "noargs_pcls", "bge_lr")], axis=0),
            "rankavg_no_bge": 0.5 * rank01(p_no_v) + 0.5 * rank01(p_bge_v),
            "rankavg_args_bge": 0.5 * rank01(p_ar_v) + 0.5 * rank01(p_bge_v),
        }
        rankavg_t = {
            "rankavg_args_no_bge": np.mean(
                [rank01(base_t[k]) for k in ("args_pcls", "noargs_pcls", "bge_lr")], axis=0),
            "rankavg_no_bge": 0.5 * rank01(p_no_t) + 0.5 * rank01(p_bge_t),
            "rankavg_args_bge": 0.5 * rank01(p_ar_t) + 0.5 * rank01(p_bge_t),
        }

        for name in ("bge_lr", "noargs_pcls", "args_pcls", "args_arf", "args_knn", "llm"):
            put(oof, name, te_idx, yv, base_v[name], base_t[name])
        for name in rankavg_v:
            put(oof, name, te_idx, yv, rankavg_v[name], rankavg_t[name])

        fold_meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx)}
        alt_names = ["noargs_pcls", "args_pcls", "rankavg_args_no_bge",
                     "rankavg_no_bge", "rankavg_args_bge", "llm"]
        for alt in alt_names:
            pv_alt = rankavg_v.get(alt, base_v.get(alt))
            pt_alt = rankavg_t.get(alt, base_t.get(alt))
            for obj in ("macro", "rank"):
                t, pv, pt, rv, rt = fit_uncertain_switch(
                    yv, p_bge_v, p_bge_t, pv_alt, pt_alt, obj)
                nm = f"switch_uncertain_{obj}_{alt}"
                put(oof, nm, te_idx, yv, pv, pt)
                fold_meta[f"{nm}_t"] = round(float(t), 3)
                fold_meta[f"{nm}_val_rate"] = round(float(rv), 3)
                fold_meta[f"{nm}_test_rate"] = round(float(rt), 3)

                m, pv, pt, rv, rt = fit_conf_advantage_switch(
                    yv, p_bge_v, p_bge_t, pv_alt, pt_alt, obj)
                nm = f"switch_confadv_{obj}_{alt}"
                put(oof, nm, te_idx, yv, pv, pt)
                fold_meta[f"{nm}_m"] = round(float(m), 3)
                fold_meta[f"{nm}_val_rate"] = round(float(rv), 3)
                fold_meta[f"{nm}_test_rate"] = round(float(rt), 3)

        rec_Xv = rec_features(val_recs, cats, confs, llm_scores)
        rec_Xt = rec_features(test_recs, cats, confs, llm_scores)
        Xv = meta_features(base_v, rec_Xv)
        Xt = meta_features(base_t, rec_Xt)
        for obj in ("macro", "rank"):
            c_value, pv, pt = fit_meta_lr(yv, cv, Xv, Xt, obj)
            nm = f"meta_lr_{obj}"
            put(oof, nm, te_idx, yv, pv, pt)
            fold_meta[f"{nm}_C"] = c_value
        meta.append(fold_meta)

    methods = [m for m in oof if m != "_n"]
    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_oof[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_oof[ok])

    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Top reliability-gate candidates ===", flush=True)
    for name in ranked[:25]:
        r = rows[name]
        print(f"{name:48s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig_names = set(ranked[:20])
    for key in ("auprc", "auroc", "wF1"):
        sig_names.update(sorted(rows, key=lambda m: rows[m][key], reverse=True)[:10])
    sig_names.update(["bge_lr", "noargs_pcls", "args_pcls", "rankavg_args_no_bge"])
    sig = {}
    for name in sorted(sig_names):
        if name == "bge_lr":
            continue
        ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{name}_vs_bge_lr"] = paired_bootstrap(
            y_oof[ok], oof[name]["p"][ok], oof["bge_lr"]["p"][ok], c_oof[ok],
            n_boot=args.n_boot)
        s = sig[f"{name}_vs_bge_lr"]
        r = rows[name]
        print(f"{name:48s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f} "
              f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
              f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
              f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
              flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows, "fold_meta": meta,
           "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_reliability_gate] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
