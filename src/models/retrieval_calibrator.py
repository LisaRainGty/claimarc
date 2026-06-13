"""Leakage-safe retrieval calibration probe for CLAIMARC.

This script tests whether train-only neighbor evidence can stabilize the
claim/evidence classifier beyond the current neural ARF setting. It deliberately
does not use label_audit fields, because those are generated from the weak-label
construction process and would leak target information.

Usage:
  python -m models.retrieval_calibrator \
      --dataset ../data/final/dataset_verify_faithful.jsonl \
      --out ../data/final/v2/retrieval_calibrator.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

def load_split(path: str) -> dict[str, list[dict]]:
    out = {"train": [], "val": [], "test": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r.get("split", "train"), []).append(r)
    return {k: out.get(k, []) for k in ("train", "val", "test")}


def ycw(recs: list[dict]):
    y = np.asarray([int(r.get("y", 0)) for r in recs], dtype=int)
    c = np.asarray([float(r.get("c", 0.05)) for r in recs], dtype=float)
    w = np.clip(c, 0.05, None)
    return y, c, w


def claim_text(r: dict) -> str:
    claim = r.get("claim", {}) or {}
    segs = claim.get("segments", []) or []
    return (r.get("attribute_name", "") + " " + " ".join(s.get("text", "") for s in segs)).strip()


def evidence_text(r: dict) -> str:
    parts = [r.get("attribute_name", "")]
    for it in r.get("evidence_params", []) or []:
        parts.append(it.get("raw_text", ""))
    for it in r.get("evidence_ocr", []) or []:
        parts.append(it.get("raw_text", ""))
    for it in r.get("evidence_vlm", []) or []:
        parts.append(it.get("raw_quote", ""))
    return " ".join(p for p in parts if p).strip()


def joined_text(r: dict) -> str:
    c = claim_text(r)
    e = evidence_text(r)
    parts = [
        "类目", r.get("category", ""),
        "子类", r.get("subcategory", ""),
        "属性", r.get("attribute_name", ""),
        "主播话术", c,
        "商品证据", e,
    ]
    return " ".join(p for p in parts if p)


def struct_dict(r: dict) -> dict[str, float | str]:
    claim = r.get("claim", {}) or {}
    evidence_count = r.get("evidence_count", {}) or {}
    segs = claim.get("segments", []) or []
    ctext = claim_text(r)
    etext = evidence_text(r)
    d: dict[str, float | str] = {
        "category": "cat=" + str(r.get("category", "")),
        "subcategory": "sub=" + str(r.get("subcategory", "")),
        "attribute": "attr=" + str(r.get("attribute_id", "")),
        "coverage": float(r.get("coverage", 0) or 0),
        "has_claim": float(bool(claim.get("has_claim_srt"))),
        "n_claim_segments": float(len(segs)),
        "claim_chars": float(min(len(ctext), 2000)) / 2000.0,
        "evidence_chars": float(min(len(etext), 4000)) / 4000.0,
        "params_count": float(evidence_count.get("params", 0) or 0),
        "ocr_count": float(evidence_count.get("ocr", 0) or 0),
        "vlm_count": float(evidence_count.get("vlm", 0) or 0),
    }
    return d


def build_base_features(sp: dict[str, list[dict]], min_df: int = 2):
    texts = {k: [joined_text(r) for r in v] for k, v in sp.items()}
    dicts = {k: [struct_dict(r) for r in v] for k, v in sp.items()}

    word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=120_000,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=min_df,
        max_features=160_000,
        sublinear_tf=True,
    )
    dv = DictVectorizer()
    text_vec = FeatureUnion([("word", word), ("char", char)])

    X_text_tr = text_vec.fit_transform(texts["train"])
    X_text = {
        "train": X_text_tr,
        "val": text_vec.transform(texts["val"]),
        "test": text_vec.transform(texts["test"]),
    }
    X_struct_tr = dv.fit_transform(dicts["train"])
    X_struct = {
        "train": X_struct_tr,
        "val": dv.transform(dicts["val"]),
        "test": dv.transform(dicts["test"]),
    }
    X = {k: sparse.hstack([X_text[k], X_struct[k]], format="csr") for k in X_text}
    return X


def best_thr(y, p):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        score = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if score > best:
            best, best_t = score, t
    return float(best_t)


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p <= bins[i + 1] if i == n_bins - 1 else p < bins[i + 1])
        if not mask.any():
            continue
        acc = (y[mask] == (p[mask] >= 0.5)).mean()
        err += mask.mean() * abs(float(p[mask].mean()) - float(acc))
    return float(err)


def metric_row(y, p, c, thr):
    pred = (p >= thr).astype(int)
    two = len(set(y.tolist())) > 1
    return {
        "thr": round(float(thr), 3),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "pos_f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "wF1": round(float(f1_score(y, pred, average="macro", sample_weight=np.clip(c, 0.05, None), zero_division=0)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4) if two else None,
        "auroc": round(float(roc_auc_score(y, p)), 4) if two else None,
        "ece": round(ece(y, p), 4),
        "n": int(len(y)),
        "pos": int(y.sum()),
    }


def tune_selective_mix(yv, pv_base, knn_v, pt_base, knn_t):
    """Val-tuned sparse retrieval correction.

    Columns per source are:
      p, top1, mean, log_support for global / attr / category.
    Only confident neighbor blocks are allowed to override the base probability.
    """
    sources = {
        "global": (0, 1, 3),
        "attr": (4, 5, 7),
        "category": (8, 9, 11),
    }
    best = None
    for name, (p_col, top_col, sup_col) in sources.items():
        pv_knn = knn_v[:, p_col]
        pt_knn = knn_t[:, p_col]
        for alpha in np.linspace(0.1, 0.9, 9):
            for min_top in (-1.0, 0.03, 0.06, 0.09, 0.12):
                for min_sup in (0.0, np.log1p(2), np.log1p(5), np.log1p(10)):
                    for min_agree in (0.0, 0.1, 0.2, 0.3):
                        mv = (
                            (knn_v[:, top_col] >= min_top)
                            & (knn_v[:, sup_col] >= min_sup)
                            & (np.abs(pv_knn - 0.5) >= min_agree)
                        )
                        if mv.mean() < 0.03:
                            continue
                        pv = pv_base.copy()
                        pv[mv] = alpha * pv_base[mv] + (1 - alpha) * pv_knn[mv]
                        thr = best_thr(yv, pv)
                        score = f1_score(yv, (pv >= thr).astype(int), average="macro", zero_division=0)
                        if best is None or score > best["score"]:
                            mt = (
                                (knn_t[:, top_col] >= min_top)
                                & (knn_t[:, sup_col] >= min_sup)
                                & (np.abs(pt_knn - 0.5) >= min_agree)
                            )
                            pt = pt_base.copy()
                            pt[mt] = alpha * pt_base[mt] + (1 - alpha) * pt_knn[mt]
                            best = {
                                "source": name,
                                "alpha_base": float(alpha),
                                "min_top": float(min_top),
                                "min_support_log": float(min_sup),
                                "min_agreement": float(min_agree),
                                "val_coverage": float(mv.mean()),
                                "test_coverage": float(mt.mean()),
                                "score": float(score),
                                "thr": float(thr),
                                "p_val": pv,
                                "p_test": pt,
                            }
    return best


def fit_lr(X, y, w, C=1.0, balanced=True):
    clf = LogisticRegression(
        C=C,
        max_iter=5000,
        solver="saga",
        n_jobs=-1,
        class_weight=("balanced" if balanced else None),
        penalty="l2",
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def lr_oof(X, y, w, groups, C=1.0):
    p = np.zeros(len(y), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
    for tr, va in cv.split(X, y, groups):
        clf = fit_lr(X[tr], y[tr], w[tr], C=C)
        p[va] = clf.predict_proba(X[va])[:, 1]
    return p


def _neighbor_vote(sims, train_y, train_w, train_attr, query_attr, idx, k):
    if idx.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    kk = min(k, idx.size)
    s = sims[idx]
    top = np.argpartition(-s, kk - 1)[:kk]
    j = idx[top]
    sj = np.clip(s[top], 0, None)
    ww = train_w[j] * sj
    mass = float(ww.sum())
    if mass <= 1e-8:
        return float(train_y[idx].mean()), 0.0, 0.0, float(idx.size)
    p = float((ww * train_y[j]).sum() / mass)
    return p, float(s[top].max()), float(sj.mean()), float(idx.size)


def knn_features(X_store, y_store, w_store, attr_store, cat_store,
                 X_query, attr_query, cat_query, k=15, self_index=None):
    Xs = normalize(X_store, norm="l2", copy=True)
    Xq = normalize(X_query, norm="l2", copy=True)
    sims = (Xq @ Xs.T).toarray()
    attr_store = np.asarray(attr_store)
    cat_store = np.asarray(cat_store)
    y_store = np.asarray(y_store, dtype=float)
    w_store = np.asarray(w_store, dtype=float)
    rows = []
    all_idx = np.arange(len(y_store))
    for i in range(Xq.shape[0]):
        banned = self_index[i] if self_index is not None else -1
        base = all_idx[all_idx != banned]
        same_attr = base[attr_store[base] == attr_query[i]]
        same_cat = base[cat_store[base] == cat_query[i]]
        pg, tg, mg, sg = _neighbor_vote(sims[i], y_store, w_store, attr_store, attr_query[i], base, k)
        pa, ta, ma, sa = _neighbor_vote(sims[i], y_store, w_store, attr_store, attr_query[i], same_attr, k)
        pc, tc, mc, sc = _neighbor_vote(sims[i], y_store, w_store, attr_store, attr_query[i], same_cat, k)
        rows.append([pg, tg, mg, np.log1p(sg), pa, ta, ma, np.log1p(sa), pc, tc, mc, np.log1p(sc)])
    return np.asarray(rows, dtype=float)


def knn_oof(X, y, w, recs, k=15):
    attr = np.asarray([r.get("attribute_id", "") for r in recs])
    cat = np.asarray([r.get("category", "") for r in recs])
    groups = np.asarray([r.get("room_id", "") for r in recs])
    out = np.zeros((len(recs), 12), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=29)
    for tr, va in cv.split(X, y, groups):
        out[va] = knn_features(
            X[tr], y[tr], w[tr], attr[tr], cat[tr],
            X[va], attr[va], cat[va], k=k,
        )
    return out


def run(args):
    sp = load_split(args.dataset)
    ytr, ctr, wtr = ycw(sp["train"])
    yv, cv, _ = ycw(sp["val"])
    yt, ct, _ = ycw(sp["test"])
    groups = np.asarray([r.get("room_id", "") for r in sp["train"]])

    X = build_base_features(sp, min_df=args.min_df)

    base_oof = lr_oof(X["train"], ytr, wtr, groups, C=args.C)
    base = fit_lr(X["train"], ytr, wtr, C=args.C)
    pv_base = base.predict_proba(X["val"])[:, 1]
    pt_base = base.predict_proba(X["test"])[:, 1]

    attr_tr = np.asarray([r.get("attribute_id", "") for r in sp["train"]])
    cat_tr = np.asarray([r.get("category", "") for r in sp["train"]])
    knn_tr = knn_oof(X["train"], ytr, wtr, sp["train"], k=args.k)
    knn_v = knn_features(
        X["train"], ytr, wtr, attr_tr, cat_tr,
        X["val"],
        [r.get("attribute_id", "") for r in sp["val"]],
        [r.get("category", "") for r in sp["val"]],
        k=args.k,
    )
    knn_t = knn_features(
        X["train"], ytr, wtr, attr_tr, cat_tr,
        X["test"],
        [r.get("attribute_id", "") for r in sp["test"]],
        [r.get("category", "") for r in sp["test"]],
        k=args.k,
    )

    stack_tr = np.column_stack([base_oof, knn_tr])
    stack_v = np.column_stack([pv_base, knn_v])
    stack_t = np.column_stack([pt_base, knn_t])
    stack = LogisticRegression(C=args.stack_C, max_iter=5000, class_weight="balanced")
    stack.fit(stack_tr, ytr, sample_weight=wtr)
    pv_stack = stack.predict_proba(stack_v)[:, 1]
    pt_stack = stack.predict_proba(stack_t)[:, 1]
    selective = tune_selective_mix(yv, pv_base, knn_v, pt_base, knn_t)

    thr_base = best_thr(yv, pv_base)
    thr_stack = best_thr(yv, pv_stack)
    results = {
        "dataset": args.dataset,
        "n": {k: len(v) for k, v in sp.items()},
        "base_lr": metric_row(yt, pt_base, ct, thr_base),
        "retrieval_calibrated": metric_row(yt, pt_stack, ct, thr_stack),
        "selective_retrieval": metric_row(yt, selective["p_test"], ct, selective["thr"]) if selective else None,
        "val": {
            "base_lr": metric_row(yv, pv_base, cv, thr_base),
            "retrieval_calibrated": metric_row(yv, pv_stack, cv, thr_stack),
            "selective_retrieval": metric_row(yv, selective["p_val"], cv, selective["thr"]) if selective else None,
        },
        "selective_config": {k: v for k, v in (selective or {}).items()
                             if not k.startswith("p_") and k not in {"score"}},
        "config": {"k": args.k, "C": args.C, "stack_C": args.stack_C, "min_df": args.min_df},
    }
    print("RESULT", json.dumps(results["base_lr"], ensure_ascii=False), flush=True)
    print("RESULT", json.dumps(results["retrieval_calibrated"], ensure_ascii=False), flush=True)
    if results["selective_retrieval"]:
        print("RESULT", json.dumps(results["selective_retrieval"], ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[write] {args.out}", flush=True)
    if args.save_pred:
        try:
            import torch
            torch.save({
                "val": {"p": pv_stack, "y": yv, "base": pv_base},
                "test": {"p": pt_stack, "y": yt, "c": ct, "base": pt_base,
                         "selective": selective["p_test"] if selective else None,
                         "attr": [r.get("attribute_id", "") for r in sp["test"]]},
                "results": results,
            }, args.save_pred)
            print(f"[save_pred] {args.save_pred}", flush=True)
        except Exception as e:
            print(f"[save_pred:skip] {e}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--out", default="../data/final/v2/retrieval_calibrator.json")
    ap.add_argument("--save_pred", default="")
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--stack_C", type=float, default=0.5)
    ap.add_argument("--min_df", type=int, default=2)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
