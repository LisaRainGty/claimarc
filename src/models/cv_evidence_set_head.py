"""Evidence-set verification head under grouped OOF CV.

This diagnostic follows recent set-level evidence sufficiency work: retrieved
sources are not only concatenated, but summarized as a set of local claim-unit
relations.  For each outer fold, the head is trained on outer-train records,
the model/threshold is selected on the validation carve, and only then applied
to the held-out fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.baselines import claim_text, evidence_text
from models.cv_eval import make_folds, val_carve
from models.data import load_split, resolve_bge_path
from models.fusion_eval import best_thr, macro
from models.cv_dual_head_router import paired_bootstrap_dual


def source_units(rec, include_args=False):
    units = []
    for typ, key, field in (
        ("param", "evidence_params", "raw_text"),
        ("ocr", "evidence_ocr", "raw_text"),
        ("vlm", "evidence_vlm", "raw_quote"),
    ):
        for it in rec.get(key, []) or []:
            txt = str(it.get(field, "") or "").strip()
            if txt:
                units.append((typ, txt))
    if include_args:
        args = rec.get("arguments", {}) or {}
        for typ, key in (
            ("arg_sup", "supporting_argument"),
            ("arg_ref", "refuting_argument"),
            ("arg_gap", "evidence_gap"),
        ):
            txt = str(args.get(key, "") or "").strip()
            if txt:
                units.append((typ, txt))
    return units


def agg(vals):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return [0.0] * 11
    top = np.sort(vals)[-min(2, len(vals)):]
    return [
        float(len(vals)),
        float(vals.mean()),
        float(vals.max()),
        float(vals.min()),
        float(vals.std()),
        float(top.mean()),
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 75)),
        float((vals >= 0.20).mean()),
        float((vals >= 0.35).mean()),
        float((vals >= 0.50).mean()),
    ]


def rec_numeric_features(rec):
    ev = rec.get("evidence_count", {}) or {}
    claim = rec.get("claim", {}) or {}
    args = rec.get("arguments", {}) or {}
    risk_cues = args.get("risk_cues", []) or []
    source_len = 0
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for it in rec.get(key, []) or []:
            source_len += len(str(it.get(field, "") or ""))
    arg_len = sum(len(str(args.get(k, "") or "")) for k in
                  ("supporting_argument", "refuting_argument", "evidence_gap"))
    return [
        float(rec.get("coverage", 0.0) or 0.0),
        float(ev.get("params", 0) or 0),
        float(ev.get("ocr", 0) or 0),
        float(ev.get("vlm", 0) or 0),
        float(bool(claim.get("has_claim_srt", False))),
        np.log1p(len(claim.get("segments", []) or [])),
        np.log1p(len(str(claim.get("passage", "") or ""))),
        np.log1p(source_len),
        np.log1p(arg_len),
        np.log1p(len(risk_cues)),
    ]


def encode_features(recs, include_args=False, batch_size=64):
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(resolve_bge_path(), device=device)
    claims_txt = [claim_text(r) for r in recs]
    evid_txt = [evidence_text(r) for r in recs]
    claim_vec = np.asarray(model.encode(claims_txt, normalize_embeddings=True,
                                        batch_size=batch_size, show_progress_bar=False), float)
    evid_vec = np.asarray(model.encode(evid_txt, normalize_embeddings=True,
                                       batch_size=batch_size, show_progress_bar=False), float)
    flat_units = []
    offsets = []
    for r in recs:
        start = len(flat_units)
        flat_units.extend(source_units(r, include_args=include_args))
        offsets.append((start, len(flat_units)))
    unit_vec = np.zeros((0, claim_vec.shape[1]), dtype=float)
    if flat_units:
        unit_vec = np.asarray(model.encode([u[1] for u in flat_units],
                                           normalize_embeddings=True,
                                           batch_size=batch_size,
                                           show_progress_bar=False), float)
    type_order = ["all", "param", "ocr", "vlm", "arg_sup", "arg_ref", "arg_gap"]
    set_rows = []
    for i, r in enumerate(recs):
        start, end = offsets[i]
        by_type = {typ: [] for typ in type_order}
        lengths = []
        for j in range(start, end):
            typ, txt = flat_units[j]
            sim = float(claim_vec[i] @ unit_vec[j])
            by_type["all"].append(sim)
            by_type.setdefault(typ, []).append(sim)
            lengths.append(len(txt))
        row = []
        for typ in type_order:
            row.extend(agg(by_type.get(typ, [])))
        row.extend(agg(np.log1p(lengths)))
        row.extend(rec_numeric_features(r))
        set_rows.append(row)
    pair = np.concatenate([claim_vec, evid_vec, claim_vec - evid_vec, claim_vec * evid_vec], axis=1)
    return pair, np.asarray(set_rows, float)


def load_or_build_cache(path, recs, include_args):
    p = Path(path)
    if p.exists():
        data = np.load(p)
        return data["pair"], data["set"]
    pair, set_x = encode_features(recs, include_args=include_args)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, pair=pair, set=set_x)
    print(f"[cache] saved {p} pair={pair.shape} set={set_x.shape}", flush=True)
    return pair, set_x


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def fit_lr(Xtr, ytr, ctr, Xv, yv, cv, Xt, objective, quick=False):
    best = None
    class_weights = (None,) if quick else (None, "balanced")
    c_values = (0.03, 0.3) if quick else (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    for class_weight in class_weights:
        for c_value in c_values:
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xvs = scaler.transform(Xv)
            Xts = scaler.transform(Xt)
            clf = LogisticRegression(C=c_value, max_iter=1200, class_weight=class_weight,
                                     solver="liblinear")
            clf.fit(Xtrs, ytr, sample_weight=np.clip(ctr, 0.05, None))
            pv = clf.predict_proba(Xvs)[:, 1]
            pt = clf.predict_proba(Xts)[:, 1]
            thr = best_thr(yv, pv)
            mf = macro(yv, (pv >= thr).astype(int))
            ap = average_precision_score(yv, pv)
            au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
            score = mf + 0.10 * ap + 0.05 * au if objective == "macro" else ap + 0.50 * au
            if best is None or score > best[0]:
                best = (score, pv, pt, thr, c_value, class_weight or "none")
    return best


def fit_hgb(Xtr, ytr, ctr, Xv, yv, Xt, objective, quick=False):
    best = None
    lrs = (0.06,) if quick else (0.03, 0.06)
    l2s = (0.1,) if quick else (0.01, 0.1, 1.0)
    leaf_grid = (15,) if quick else (7, 15)
    for lr in lrs:
        for l2 in l2s:
            for leaves in leaf_grid:
                clf = HistGradientBoostingClassifier(
                    learning_rate=lr, l2_regularization=l2, max_leaf_nodes=leaves,
                    max_iter=120, random_state=0)
                clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
                pv = clf.predict_proba(Xv)[:, 1]
                pt = clf.predict_proba(Xt)[:, 1]
                thr = best_thr(yv, pv)
                mf = macro(yv, (pv >= thr).astype(int))
                ap = average_precision_score(yv, pv)
                au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
                score = mf + 0.10 * ap + 0.05 * au if objective == "macro" else ap + 0.50 * au
                if best is None or score > best[0]:
                    best = (score, pv, pt, thr, lr, l2, leaves)
    return best


def put(oof, name, idx, pv, pt, yv):
    thr = best_thr(yv, pv)
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--include_args", action="store_true")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    pair_x, set_x = load_or_build_cache(args.cache, recs, args.include_args)
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {"_n": len(recs)}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ytr, yv = y_all[tr_idx], y_all[va_idx]
        ctr, cv = c_all[tr_idx], c_all[va_idx]
        specs = {
            "bge_pair_lr": pair_x,
            "evidence_set_lr": set_x,
            "bge_pair_set_lr": np.concatenate([pair_x, set_x], axis=1),
        }
        fm = {"fold": fi, "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx)}
        for name, X in specs.items():
            for obj in ("macro", "rank"):
                fit = fit_lr(X[tr_idx], ytr, ctr, X[va_idx], yv, cv, X[te_idx], obj,
                             quick=args.quick)
                method = f"{name}_{obj}"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = fit[2]
                oof[method]["yhat"][te_idx] = (fit[2] >= fit[3]).astype(int)
                fm[method] = {"thr": round(float(fit[3]), 3), "C": fit[4],
                              "class_weight": fit[5]}
        hgb_x = np.column_stack([set_x, np.linalg.norm(pair_x[:, :pair_x.shape[1] // 4], axis=1)])
        for obj in ("macro", "rank"):
            fit = fit_hgb(hgb_x[tr_idx], ytr, ctr, hgb_x[va_idx], yv, hgb_x[te_idx],
                          obj, quick=args.quick)
            method = f"evidence_set_hgb_{obj}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = fit[2]
            oof[method]["yhat"][te_idx] = (fit[2] >= fit[3]).astype(int)
            fm[method] = {"thr": round(float(fit[3]), 3), "lr": fit[4],
                          "l2": fit[5], "leaves": fit[6]}
        fold_meta.append(fm)
        print(f"[fold {fi}] done", flush=True)

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Evidence-set head candidates ===", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:28s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig = {}
    base = "bge_pair_lr_macro"
    if args.n_boot > 0 and base in oof:
        for name in ranked:
            if name == base:
                continue
            ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof[base]["p"]))
            sig[f"{name}_vs_{base}"] = paired_bootstrap_dual(
                y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
                oof[base]["p"][ok], oof[base]["yhat"][ok], n_boot=args.n_boot)
    out = {"fold_seed": args.fold_seed, "include_args": args.include_args,
           "rows": rows, "fold_meta": fold_meta, "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_evidence_set_head] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
