"""Source-conditioned rank-average diagnostics for CLAIMARC.

This script tests a specific hypothesis from source-first failure analysis:
rankavg(sourcefirst_args_pcls, BGE) helps when real product evidence exists,
but hurts evidence-absent samples where arguments are mostly speculative.

All thresholds and optional mask choices are selected inside each outer fold's
validation carve, then applied to the held-out test fold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import best_thr, build_split_features, load_bundles, macro, paired_bootstrap
from models.cv_reliability_gate import rank01


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


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def put(oof, name, idx, yv, pv, pt):
    thr = best_thr(yv, pv)
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)
    return float(thr)


def rec_arrays(recs):
    sc = np.asarray([source_count(r) for r in recs], int)
    sl = np.asarray([source_len(r) for r in recs], int)
    conf = np.asarray([str(r.get("confidence", "")) for r in recs], object)
    return sc, sl, conf


def fixed_masks(recs, p_bge):
    sc, sl, conf = rec_arrays(recs)
    return {
        "src_present": sc > 0,
        "src_ge2": sc >= 2,
        "src_2_3": (sc >= 2) & (sc <= 3),
        "src_ge4": sc >= 4,
        "src_len_ge20": sl >= 20,
        "conf_not_absent": conf != "absent",
        "conf_low_or_medium": np.isin(conf, ["low", "medium"]),
        "conf_low_medium_high": np.isin(conf, ["low", "medium", "high"]),
        "src_present_bge_uncertain16": (sc > 0) & (np.abs(p_bge - 0.5) <= 0.16),
        "src_present_bge_uncertain28": (sc > 0) & (np.abs(p_bge - 0.5) <= 0.28),
        "src_2p_bge_uncertain28": (sc >= 2) & (np.abs(p_bge - 0.5) <= 0.28),
    }


def apply_mask(p_bge, p_alt, mask):
    return np.where(mask, p_alt, p_bge)


def score_val(yv, pv, objective):
    thr = best_thr(yv, pv)
    mf = macro(yv, (pv >= thr).astype(int))
    ap = average_precision_score(yv, pv)
    au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
    return (mf + 0.10 * ap + 0.05 * au) if objective == "macro" else (ap + 0.50 * au + 0.05 * mf)


def select_fixed_mask(yv, p_bge_v, p_alt_v, p_bge_t, p_alt_t, val_recs, test_recs, objective):
    mv = fixed_masks(val_recs, p_bge_v)
    mt = fixed_masks(test_recs, p_bge_t)
    best = None
    for name in sorted(mv):
        pv = apply_mask(p_bge_v, p_alt_v, mv[name])
        pt = apply_mask(p_bge_t, p_alt_t, mt[name])
        score = score_val(yv, pv, objective)
        if best is None or score > best[0]:
            best = (score, name, pv, pt, float(mv[name].mean()), float(mt[name].mean()))
    return best[1:]


def select_uncertain_mask(yv, p_bge_v, p_alt_v, p_bge_t, p_alt_t, val_recs, test_recs, objective):
    sc_v, _, _ = rec_arrays(val_recs)
    sc_t, _, _ = rec_arrays(test_recs)
    best = None
    for min_src in (1, 2, 4):
        for t in np.linspace(0.04, 0.40, 19):
            mv = (sc_v >= min_src) & (np.abs(p_bge_v - 0.5) <= t)
            mt = (sc_t >= min_src) & (np.abs(p_bge_t - 0.5) <= t)
            pv = apply_mask(p_bge_v, p_alt_v, mv)
            pt = apply_mask(p_bge_t, p_alt_t, mt)
            score = score_val(yv, pv, objective)
            if best is None or score > best[0]:
                best = (score, min_src, t, pv, pt, float(mv.mean()), float(mt.mean()))
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl")
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {"_n": len(recs)}
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"fold {fi} missing {missing}")
        ar_b = load_bundles(ar_paths)
        _, p_args_v, yv, _, _ = build_split_features(ar_b, "val")
        _, p_args_t, _, _, _ = build_split_features(ar_b, "test")
        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        p_rank_v = 0.5 * rank01(p_args_v) + 0.5 * rank01(p_bge_v)
        p_rank_t = 0.5 * rank01(p_args_t) + 0.5 * rank01(p_bge_t)

        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx)}
        for name, pv, pt in (
            ("bge_lr", p_bge_v, p_bge_t),
            ("args_pcls", p_args_v, p_args_t),
            ("rankavg_args_bge", p_rank_v, p_rank_t),
        ):
            meta[f"{name}_thr"] = put(oof, name, te_idx, yv, pv, pt)

        mv = fixed_masks(val_recs, p_bge_v)
        mt = fixed_masks(test_recs, p_bge_t)
        for mask_name in sorted(mv):
            nm = f"mask_{mask_name}_rankavg_args_bge"
            pv = apply_mask(p_bge_v, p_rank_v, mv[mask_name])
            pt = apply_mask(p_bge_t, p_rank_t, mt[mask_name])
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
            meta[f"{nm}_val_rate"] = round(float(mv[mask_name].mean()), 4)
            meta[f"{nm}_test_rate"] = round(float(mt[mask_name].mean()), 4)

        for obj in ("macro", "rank"):
            mask_name, pv, pt, rv, rt = select_fixed_mask(
                yv, p_bge_v, p_rank_v, p_bge_t, p_rank_t, val_recs, test_recs, obj)
            nm = f"select_fixed_{obj}_rankavg_args_bge"
            meta[f"{nm}_mask"] = mask_name
            meta[f"{nm}_val_rate"] = round(float(rv), 4)
            meta[f"{nm}_test_rate"] = round(float(rt), 4)
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)

            min_src, t, pv, pt, rv, rt = select_uncertain_mask(
                yv, p_bge_v, p_rank_v, p_bge_t, p_rank_t, val_recs, test_recs, obj)
            nm = f"select_uncertain_{obj}_rankavg_args_bge"
            meta[f"{nm}_min_src"] = int(min_src)
            meta[f"{nm}_t"] = round(float(t), 3)
            meta[f"{nm}_val_rate"] = round(float(rv), 4)
            meta[f"{nm}_test_rate"] = round(float(rt), 4)
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
        fold_meta.append(meta)

    methods = [m for m in oof if m != "_n"]
    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])

    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Top source-conditioned candidates ===", flush=True)
    for name in ranked[:30]:
        r = rows[name]
        print(f"{name:48s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig_names = set(ranked[:20])
    for key in ("auprc", "auroc", "wF1"):
        sig_names.update(sorted(rows, key=lambda m: rows[m][key], reverse=True)[:10])
    sig_names.update(["args_pcls", "rankavg_args_bge"])
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
        print(f"{name:48s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f} "
              f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
              f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
              f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
              flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows, "fold_meta": fold_meta,
           "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_source_condition_gate] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
