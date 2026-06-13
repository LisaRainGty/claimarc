"""Validation-thresholded pooling of source-policy CLAIMARC experts.

This is a structural diagnostic for source-stratified multi-instance evidence
pooling. It reads saved fold bundles from cv_eval tmpdirs, rebuilds each outer
fold's validation/test predictions, applies fixed pooling rules, and chooses
only the decision threshold on the fold validation set.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split, evidence_combo, source_count, confidence_bin, arg_len
from models.fusion_eval import build_split_features, load_bundles, best_thr


def macro(y, pred):
    return f1_score(y, pred, average="macro", zero_division=0)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ranks01(p):
    order = np.argsort(np.asarray(p, float), kind="mergesort")
    out = np.empty(len(order), dtype=float)
    if len(order) <= 1:
        out[:] = 0.5
    else:
        out[order] = np.linspace(0.0, 1.0, len(order))
    return out


def parse_specs(items):
    specs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected name=tmpdir spec, got {item!r}")
        name, path = item.split("=", 1)
        specs[name.strip()] = path.strip()
    return specs


def load_policy_fold(tmpdir, fold, seeds):
    paths = [os.path.join(tmpdir, f"cv_cm_f{fold}_s{s}.pt") for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing source-policy bundle(s): {missing[:3]}")
    bundles = load_bundles(paths)
    _, pv, yv, cv, _ = build_split_features(bundles, "val")
    _, pt, yt, ct, _ = build_split_features(bundles, "test")
    return pv, pt, yv, yt, cv, ct


def load_bge_fold(tmpdir, fold):
    p = os.path.join(tmpdir, f"cv_bge_lr_f{fold}.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    return (
        np.asarray(d["val"]["p"], float),
        np.asarray(d["test"]["p"], float),
        np.asarray(d["val"]["y"], int),
        np.asarray(d["test"]["y"], int),
        np.ones(len(d["val"]["y"]), dtype=float),
        np.asarray(d["test"]["c"], float),
    )


def policy_matrix(preds, names):
    return np.column_stack([preds[n] for n in names])


def source_masked_pool(preds, recs, names):
    out = np.zeros(len(recs), dtype=float)
    for i, r in enumerate(recs):
        active = []
        combo = evidence_combo(r)
        if "noargs" in preds:
            active.append(preds["noargs"][i])
        if "sourcefirst" in preds:
            active.append(preds["sourcefirst"][i])
        if "P" in combo and "params" in preds:
            active.append(preds["params"][i])
        if "O" in combo and "ocr" in preds:
            active.append(preds["ocr"][i])
        if "V" in combo and "vlm" in preds:
            active.append(preds["vlm"][i])
        if arg_len(r) > 0 and "args" in preds:
            active.append(preds["args"][i])
        if not active:
            active = [preds[n][i] for n in names]
        out[i] = float(np.mean(active))
    return out


def decision_masks(recs):
    sc = np.asarray([source_count(r) for r in recs], dtype=float)
    conf = np.asarray([confidence_bin(r) for r in recs], dtype=object)
    combo = np.asarray([evidence_combo(r) for r in recs], dtype=object)
    lowabs = np.isin(conf, ["absent", "low"])
    src0 = sc <= 0
    return {
        "src0": src0,
        "lowabs": lowabs,
        "src0_or_lowabs": src0 | lowabs,
        "none_or_lowabs": (combo == "none") | lowabs,
    }


def build_methods(preds, recs):
    names = list(preds)
    methods = {}
    for n, p in preds.items():
        methods[n] = np.asarray(p, float)
    if len(names) >= 2:
        mat = policy_matrix(preds, names)
        methods["mean_all"] = mat.mean(axis=1)
        methods["logitmean_all"] = sigmoid(np.mean([logit(preds[n]) for n in names], axis=0))
        methods["rankavg_all"] = np.mean([ranks01(preds[n]) for n in names], axis=0)
        methods["max_all"] = mat.max(axis=1)
    core = [n for n in ("noargs", "sourcefirst", "ocr", "params", "args") if n in preds]
    if len(core) >= 2 and core != names:
        mat = policy_matrix(preds, core)
        methods["mean_core"] = mat.mean(axis=1)
        methods["logitmean_core"] = sigmoid(np.mean([logit(preds[n]) for n in core], axis=0))
        methods["rankavg_core"] = np.mean([ranks01(preds[n]) for n in core], axis=0)
    if len(names) >= 2:
        methods["source_masked_mean"] = source_masked_pool(preds, recs, names)
    return methods


def metric_row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(f1_score(y, yhat, average="macro",
                                    sample_weight=np.clip(c, 0.05, None),
                                    zero_division=0)), 4),
        "n": int(len(y)),
    }


def paired_bootstrap_yhat(y, p_a, yhat_a, p_b, yhat_b, c, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    dap, dau, df1 = [], [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        dap.append(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx]))
        dau.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))

    def summ(vals):
        arr = np.asarray(vals, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [round(float(np.percentile(arr, 2.5)), 4),
                   round(float(np.percentile(arr, 97.5)), 4)],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }
    return {"dAP": summ(dap), "dAUROC": summ(dau), "dMacroF1": summ(df1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl")
    ap.add_argument("--fold_seed", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--spec", nargs="+", required=True,
                    help="source policy tmpdir specs, e.g. noargs=... sourcefirst=... ocr=...")
    ap.add_argument("--bge_tmp", default="", help="tmpdir containing cv_bge_lr_f*.pt; defaults to first spec")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="data/final/cleancl/source_policy_pooling_fs1.json")
    ap.add_argument("--dump_oof", default="")
    args = ap.parse_args()

    specs = parse_specs(args.spec)
    full = load_split(args.dataset)
    recs = full["train"] + full["val"] + full["test"]
    folds, y_all, g_all = make_folds(recs, args.folds, seed=args.fold_seed)

    all_methods = {}
    fold_id = np.full(len(recs), -1, dtype=int)
    y_oof = np.asarray([int(r["y"]) for r in recs], int)
    c_oof = np.asarray([float(r.get("c", 0.05)) for r in recs], float)

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]

        val_preds = {}
        test_preds = {}
        yv_ref = yt_ref = cv_ref = ct_ref = None
        for name, tmpdir in specs.items():
            pv, pt, yv, yt, cv, ct = load_policy_fold(tmpdir, fi, args.cm_seeds)
            val_preds[name] = pv
            test_preds[name] = pt
            yv_ref, yt_ref, cv_ref, ct_ref = yv, yt, cv, ct
        bge_tmp = args.bge_tmp or next(iter(specs.values()))
        bge = load_bge_fold(bge_tmp, fi)
        if bge is not None:
            pv, pt, yv, yt, cv, ct = bge
            val_preds["bge_lr"] = pv
            test_preds["bge_lr"] = pt
            if yv_ref is None:
                yv_ref, yt_ref, cv_ref, ct_ref = yv, yt, cv, ct

        val_methods = build_methods(val_preds, val_recs)
        test_methods = build_methods(test_preds, test_recs)
        thresholds = {
            name: best_thr(yv_ref, pv)
            for name, pv in val_methods.items()
            if name in test_methods
        }
        val_yhat = {name: (val_methods[name] >= thr).astype(int)
                    for name, thr in thresholds.items()}
        test_yhat = {name: (test_methods[name] >= thr).astype(int)
                     for name, thr in thresholds.items()}

        def add_fold_output(name, score, yhat, thr):
            if name not in all_methods:
                all_methods[name] = {
                    "p": np.full(len(recs), np.nan),
                    "yhat": np.full(len(recs), np.nan),
                    "thresholds": [],
                }
            all_methods[name]["p"][te_idx] = score
            all_methods[name]["yhat"][te_idx] = yhat
            all_methods[name]["thresholds"].append(float(thr))

        for name, pt in test_methods.items():
            if name not in thresholds:
                continue
            add_fold_output(name, pt, test_yhat[name], thresholds[name])

        val_masks = decision_masks(val_recs)
        test_masks = decision_masks(test_recs)
        for score_name in ("rankavg_all", "logitmean_all", "mean_all"):
            if score_name not in test_methods:
                continue
            for ref_name in ("bge_lr", "noargs"):
                if ref_name not in test_yhat:
                    continue
                full_name = f"{score_name}_score_{ref_name}_decision"
                add_fold_output(full_name, test_methods[score_name], test_yhat[ref_name],
                                thresholds[score_name])
                for mask_name, tmask in test_masks.items():
                    yhat_guard = test_yhat[score_name].copy()
                    yhat_guard[tmask] = test_yhat[ref_name][tmask]
                    full_name = f"{score_name}_score_{ref_name}_{mask_name}_guard"
                    add_fold_output(full_name, test_methods[score_name], yhat_guard,
                                    thresholds[score_name])
                    yhat_neg_guard = test_yhat[score_name].copy()
                    yhat_neg_guard[tmask & (test_yhat[ref_name] == 0)] = 0
                    full_name = f"{score_name}_score_{ref_name}_{mask_name}_neg_guard"
                    add_fold_output(full_name, test_methods[score_name], yhat_neg_guard,
                                    thresholds[score_name])
        fold_id[te_idx] = fi

    rows = {}
    for name, d in sorted(all_methods.items()):
        ok = ~np.isnan(d["p"])
        rows[name] = metric_row(y_oof[ok], d["p"][ok], d["yhat"][ok].astype(int), c_oof[ok])
        rows[name]["thresholds"] = [round(x, 4) for x in d["thresholds"]]

    refs = [r for r in ("bge_lr", "noargs", "sourcefirst") if r in all_methods]
    significance = {}
    if args.n_boot > 0:
        for name, d in sorted(all_methods.items()):
            if name in refs:
                continue
            for ref in refs:
                a, b = all_methods[name], all_methods[ref]
                ok = (~np.isnan(a["p"])) & (~np.isnan(b["p"]))
                significance[f"{name}_vs_{ref}"] = paired_bootstrap_yhat(
                    y_oof[ok],
                    a["p"][ok],
                    a["yhat"][ok].astype(int),
                    b["p"][ok],
                    b["yhat"][ok].astype(int),
                    c_oof[ok],
                    n_boot=args.n_boot,
                )

    out_obj = {
        "dataset": args.dataset,
        "fold_seed": args.fold_seed,
        "cm_seeds": args.cm_seeds,
        "specs": specs,
        "rows": rows,
        "significance": significance,
        "notes": {
            "decision_threshold": "selected per outer fold on that fold's validation split",
            "pooling": "fixed formulas only; no global OOF weight search",
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"[source_policy_pooling] -> {args.out}")

    if args.dump_oof:
        dump = {
            "y": y_oof,
            "c": c_oof,
            "fold_id": fold_id,
            "pair_id": np.asarray([r.get("pair_id", "") for r in recs], dtype=object),
            "source_count": np.asarray([source_count(r) for r in recs], dtype=float),
            "evidence_combo": np.asarray([evidence_combo(r) for r in recs], dtype=object),
            "confidence": np.asarray([confidence_bin(r) for r in recs], dtype=object),
        }
        for name, d in all_methods.items():
            dump[f"p__{name}"] = d["p"]
            dump[f"yhat__{name}"] = d["yhat"]
        np.savez_compressed(args.dump_oof, **dump)
        print(f"[dump_oof] -> {args.dump_oof}")


if __name__ == "__main__":
    main()
