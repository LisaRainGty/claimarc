"""Grouped-CV cross-encoder reranker feature baseline.

This is intentionally separate from cv_eval.py: rerankers are heavy
cross-encoders, so we cache logits once and then run cheap OOF heads.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from models.baselines import claim_text, evidence_text
from models.cv_eval import make_folds, val_carve, best_thr, macro, paired_bootstrap
from models.data import load_split, resolve_bge_path, source_count, evidence_combo, confidence_bin


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def score_reranker(recs, model_name, cache, batch_size=8, max_length=512):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        pair_id = np.array([r.get("pair_id", "") for r in recs], dtype=object)
        if "pair_id" in z.files and not np.array_equal(z["pair_id"], pair_id):
            raise ValueError(f"cache pair_id mismatch: {cache}")
        print(f"[cache] loaded reranker logits from {cache}", flush=True)
        return z["logit"].astype(float)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = resolve_bge_path(model_name)
    print(f"[reranker] loading {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    logits = []
    queries = [claim_text(r) for r in recs]
    docs = [evidence_text(r) for r in recs]
    for start in range(0, len(recs), batch_size):
        end = min(len(recs), start + batch_size)
        batch = tok(
            queries[start:end],
            docs[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            out = model(**batch).logits.reshape(-1).float().cpu().numpy()
        logits.append(out)
        if (end == len(recs)) or (end % (batch_size * 20) == 0):
            print(f"[reranker] scored {end}/{len(recs)}", flush=True)
    logit = np.concatenate(logits).astype(float)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(
            cache,
            logit=logit,
            pair_id=np.array([r.get("pair_id", "") for r in recs], dtype=object),
        )
        print(f"[cache] saved reranker logits to {cache}", flush=True)
    return logit


def fit_lr(x_train, y_train, c_train):
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(x_train, y_train, sample_weight=np.clip(c_train, 0.05, None))
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--score_cache", default="")
    ap.add_argument("--compare_oof", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_oof", default="")
    args = ap.parse_args()

    full = load_split(args.dataset)
    recs = full["train"] + full["val"] + full["test"]
    y = np.array([int(r["y"]) for r in recs], dtype=int)
    c = np.array([float(r.get("c", 0.05)) for r in recs], dtype=float)
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    logit = score_reranker(
        recs, args.model_name, args.score_cache,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    prob_raw = sigmoid(logit)

    methods = {
        "reranker_direct": {"p": np.full(len(recs), np.nan), "yhat": np.full(len(recs), np.nan)},
        "reranker_lr": {"p": np.full(len(recs), np.nan), "yhat": np.full(len(recs), np.nan)},
    }
    fold_id = np.full(len(recs), -1, dtype=int)
    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        print(f"[fold {fi}] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}", flush=True)

        thr = best_thr(y[va_idx], prob_raw[va_idx])
        methods["reranker_direct"]["p"][te_idx] = prob_raw[te_idx]
        methods["reranker_direct"]["yhat"][te_idx] = (prob_raw[te_idx] >= thr).astype(int)

        x = logit.reshape(-1, 1)
        clf = fit_lr(x[tr_idx], y[tr_idx], c[tr_idx])
        pv = clf.predict_proba(x[va_idx])[:, 1]
        pt = clf.predict_proba(x[te_idx])[:, 1]
        lthr = best_thr(y[va_idx], pv)
        methods["reranker_lr"]["p"][te_idx] = pt
        methods["reranker_lr"]["yhat"][te_idx] = (pt >= lthr).astype(int)
        fold_id[te_idx] = fi

    compare = {}
    if args.compare_oof:
        z = np.load(args.compare_oof, allow_pickle=True)
        pair_id = np.array([r.get("pair_id", "") for r in recs], dtype=object)
        if not np.array_equal(z["pair_id"], pair_id):
            raise ValueError(f"compare_oof pair_id mismatch: {args.compare_oof}")
        compare["bge_lr"] = {
            "p": z["p__bge_lr"].astype(float),
            "yhat": z["yhat__bge_lr"].astype(int),
        }

    rows = {}
    all_methods = {**methods, **compare}
    for name, obj in all_methods.items():
        p = obj["p"]
        yhat = obj["yhat"].astype(int)
        ok = ~np.isnan(p)
        rows[name] = {
            "auprc": round(float(average_precision_score(y[ok], p[ok])), 4),
            "auroc": round(float(roc_auc_score(y[ok], p[ok])), 4),
            "macro_f1": round(float(macro(y[ok], yhat[ok])), 4),
            "wF1": round(float(macro(y[ok], yhat[ok], w=np.clip(c[ok], 0.05, None))), 4),
            "n": int(ok.sum()),
        }
        print(f"  {name:18s} {rows[name]}", flush=True)

    sig = {}
    if "bge_lr" in compare:
        for name in methods:
            ok = ~np.isnan(methods[name]["p"])
            sig[f"{name}_vs_bge_lr"] = paired_bootstrap(
                y[ok], methods[name]["p"][ok], compare["bge_lr"]["p"][ok], c[ok]
            )
            s = sig[f"{name}_vs_bge_lr"]
            print(
                f"  {name:18s} vs bge_lr "
                f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                f"dMacroF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                flush=True,
            )

    if args.dump_oof:
        dump = {
            "y": y.astype(float),
            "c": c,
            "fold_id": fold_id,
            "pair_id": np.array([r.get("pair_id", "") for r in recs], dtype=object),
            "room_id": np.array([r.get("room_id", "") for r in recs], dtype=object),
            "attribute_id": np.array([r.get("attribute_id", "") for r in recs], dtype=object),
            "category": np.array([r.get("category", "") for r in recs], dtype=object),
            "source_count": np.array([source_count(r) for r in recs], dtype=float),
            "evidence_combo": np.array([evidence_combo(r) for r in recs], dtype=object),
            "confidence": np.array([confidence_bin(r) for r in recs], dtype=object),
            "reranker_logit": logit,
        }
        for name, obj in all_methods.items():
            dump[f"p__{name}"] = obj["p"]
            dump[f"yhat__{name}"] = obj["yhat"]
        np.savez_compressed(args.dump_oof, **dump)
        print(f"[dump_oof] -> {args.dump_oof}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(
        {
            "model_name": args.model_name,
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "rows": rows,
            "significance": sig,
        },
        open(args.out, "w"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"[cv_reranker_feature] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
