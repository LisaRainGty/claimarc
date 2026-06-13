"""Atomic NLI evidence-posterior head under grouped OOF CV.

Each source/argument unit is scored by a small Chinese NLI model as
contradiction / entailment / neutral against the claim.  The per-unit posterior
distributions are aggregated into set-level features, then evaluated with the
same outer grouped-CV protocol used by the other diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from models.cv_dual_head_router import paired_bootstrap_dual
from models.cv_eval import make_folds, val_carve
from models.cv_reliability_gate import rank01
from models.data import load_split
from models.fusion_eval import best_thr, macro
from models.cv_evidence_set_head import agg, rec_numeric_features, source_units


def claim_statement(rec):
    claim = rec.get("claim", {}) or {}
    passage = str(claim.get("passage", "") or "").strip()
    if not passage:
        segs = claim.get("segments", []) or []
        passage = "；".join(str(s.get("text", "") or "").strip()
                           for s in segs if str(s.get("text", "") or "").strip())
    attr = str(rec.get("attribute_name", "") or "").strip()
    return f"{attr}：{passage}" if attr else passage


def load_tiny_nli(model_id):
    import torch
    from modelscope import snapshot_download
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    path = snapshot_download(model_id)
    tok = BertTokenizer.from_pretrained(path)
    cfg = BertConfig.from_pretrained(path, num_labels=3)
    model = BertForSequenceClassification(cfg)
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    renamed = {}
    for k, v in sd.items():
        if k == "encoder.embeddings.position_ids":
            continue
        renamed["bert." + k[len("encoder."):] if k.startswith("encoder.") else k] = v
    model.load_state_dict(renamed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device


def build_nli_features(recs, include_args=False, model_id="iic/nlp_structbert_nli_chinese-tiny",
                       batch_size=128, max_length=256):
    import torch

    tok, model, device = load_tiny_nli(model_id)
    premises, hypotheses, owners, types = [], [], [], []
    for i, rec in enumerate(recs):
        hyp = claim_statement(rec)
        for typ, txt in source_units(rec, include_args=include_args):
            premises.append(txt)
            hypotheses.append(hyp)
            owners.append(i)
            types.append(typ)
    probs = np.zeros((len(premises), 3), dtype=float)
    for start in range(0, len(premises), batch_size):
        end = min(len(premises), start + batch_size)
        batch = tok(premises[start:end], hypotheses[start:end],
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits.float()
        probs[start:end] = torch.softmax(logits, dim=-1).cpu().numpy()
        if end % 1000 < batch_size:
            print(f"[nli] {end}/{len(premises)} unit pairs", flush=True)

    type_order = ["all", "param", "ocr", "vlm", "arg_sup", "arg_ref", "arg_gap"]
    rows = []
    by_owner = [[] for _ in recs]
    for owner, typ, pr in zip(owners, types, probs):
        by_owner[owner].append((typ, pr))
    for rec, units in zip(recs, by_owner):
        by_type = {typ: [] for typ in type_order}
        for typ, pr in units:
            by_type["all"].append(pr)
            by_type.setdefault(typ, []).append(pr)
        feat = []
        for typ in type_order:
            arr = np.asarray(by_type.get(typ, []), float)
            if arr.size == 0:
                feat.extend([0.0] * (11 * 6))
                continue
            contr, entail, neutral = arr[:, 0], arr[:, 1], arr[:, 2]
            margin = contr - entail
            uncertainty = -np.sum(arr * np.log(np.clip(arr, 1e-8, 1.0)), axis=1)
            for vals in (contr, entail, neutral, margin, np.maximum(contr, entail), uncertainty):
                feat.extend(agg(vals))
        feat.extend(rec_numeric_features(rec))
        rows.append(feat)
    return np.asarray(rows, float)


def load_or_build_cache(path, recs, include_args, model_id):
    p = Path(path)
    if p.exists():
        data = np.load(p)
        return data["X"]
    X = build_nli_features(recs, include_args=include_args, model_id=model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=X)
    print(f"[cache] saved {p} X={X.shape}", flush=True)
    return X


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def fit_lr(Xtr, ytr, ctr, Xv, yv, Xt, quick=False):
    best = None
    c_values = (0.03, 0.3) if quick else (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    class_weights = (None,) if quick else (None, "balanced")
    for class_weight in class_weights:
        for c_value in c_values:
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xvs = scaler.transform(Xv)
            Xts = scaler.transform(Xt)
            clf = LogisticRegression(C=c_value, max_iter=2000, class_weight=class_weight,
                                     solver="liblinear")
            clf.fit(Xtrs, ytr, sample_weight=np.clip(ctr, 0.05, None))
            pv = clf.predict_proba(Xvs)[:, 1]
            pt = clf.predict_proba(Xts)[:, 1]
            thr = best_thr(yv, pv)
            score = macro(yv, (pv >= thr).astype(int)) + 0.10 * average_precision_score(yv, pv)
            if best is None or score > best[0]:
                best = (score, pv, pt, thr, c_value, class_weight or "none")
    return best


def fit_hgb(Xtr, ytr, ctr, Xv, yv, Xt, quick=False):
    best = None
    lrs = (0.06,) if quick else (0.03, 0.06)
    l2s = (0.1,) if quick else (0.01, 0.1, 1.0)
    leaves_grid = (15,) if quick else (7, 15)
    for lr in lrs:
        for l2 in l2s:
            for leaves in leaves_grid:
                clf = HistGradientBoostingClassifier(
                    learning_rate=lr, l2_regularization=l2, max_leaf_nodes=leaves,
                    max_iter=120, random_state=0)
                clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
                pv = clf.predict_proba(Xv)[:, 1]
                pt = clf.predict_proba(Xt)[:, 1]
                thr = best_thr(yv, pv)
                score = macro(yv, (pv >= thr).astype(int)) + 0.10 * average_precision_score(yv, pv)
                if best is None or score > best[0]:
                    best = (score, pv, pt, thr, lr, l2, leaves)
    return best


def put_pred(oof, name, idx, pv, pt, yv):
    thr = best_thr(yv, pv)
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)
    return float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--include_args", action="store_true")
    ap.add_argument("--model_id", default="iic/nlp_structbert_nli_chinese-tiny")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bge_tmp", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs_by = load_split(args.dataset)
    recs = recs_by["train"] + recs_by["val"] + recs_by["test"]
    X = load_or_build_cache(args.cache, recs, args.include_args, args.model_id)
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {"_n": len(recs)}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ytr, yv = y_all[tr_idx], y_all[va_idx]
        ctr = c_all[tr_idx]
        fm = {"fold": fi, "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx)}
        fit = fit_lr(X[tr_idx], ytr, ctr, X[va_idx], yv, X[te_idx], quick=args.quick)
        for method, idx in (("nli_set_lr", 2),):
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = fit[idx]
            oof[method]["yhat"][te_idx] = (fit[idx] >= fit[3]).astype(int)
        fm["nli_set_lr"] = {"thr": round(float(fit[3]), 3), "C": fit[4],
                            "class_weight": fit[5]}
        fitg = fit_hgb(X[tr_idx], ytr, ctr, X[va_idx], yv, X[te_idx], quick=args.quick)
        oof.setdefault("nli_set_hgb", {"p": np.full(oof["_n"], np.nan),
                                       "yhat": np.full(oof["_n"], np.nan)})
        oof["nli_set_hgb"]["p"][te_idx] = fitg[2]
        oof["nli_set_hgb"]["yhat"][te_idx] = (fitg[2] >= fitg[3]).astype(int)
        fm["nli_set_hgb"] = {"thr": round(float(fitg[3]), 3), "lr": fitg[4],
                             "l2": fitg[5], "leaves": fitg[6]}
        if args.bge_tmp:
            import torch
            bge = torch.load(Path(args.bge_tmp) / f"cv_bge_lr_f{fi}.pt",
                             map_location="cpu", weights_only=False)
            p_bge_v = np.asarray(bge["val"]["p"], float)
            p_bge_t = np.asarray(bge["test"]["p"], float)
            put_pred(oof, "bge_lr", te_idx, p_bge_v, p_bge_t, np.asarray(bge["val"]["y"], int))
            rv = 0.5 * rank01(fit[1]) + 0.5 * rank01(p_bge_v)
            rt = 0.5 * rank01(fit[2]) + 0.5 * rank01(p_bge_t)
            put_pred(oof, "rankavg_nli_lr_bge", te_idx, rv, rt, yv)
            rg_v = 0.5 * rank01(fitg[1]) + 0.5 * rank01(p_bge_v)
            rg_t = 0.5 * rank01(fitg[2]) + 0.5 * rank01(p_bge_t)
            put_pred(oof, "rankavg_nli_hgb_bge", te_idx, rg_v, rg_t, yv)
        fold_meta.append(fm)
        print(f"[fold {fi}] done", flush=True)

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== NLI evidence-posterior candidates ===", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:24s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig = {}
    if args.n_boot > 0 and "bge_lr" in oof:
        for name in ranked:
            if name == "bge_lr":
                continue
            ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
            sig[f"{name}_vs_bge_lr"] = paired_bootstrap_dual(
                y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
                oof["bge_lr"]["p"][ok], oof["bge_lr"]["yhat"][ok],
                n_boot=args.n_boot)
    out = {"fold_seed": args.fold_seed, "include_args": args.include_args,
           "model_id": args.model_id, "rows": rows,
           "fold_meta": fold_meta, "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_nli_evidence_head] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
