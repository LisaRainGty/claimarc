"""Leakage-safe dual-head router diagnostics for CLAIMARC.

The source-first experiments show two complementary effects:

* rank/contrastive fusion improves AP and sometimes AUROC;
* source-conditioned routing improves the final binary decision.

This script evaluates that structure explicitly. A candidate exposes a ranking
score from one fold-fitted head and a binary decision from another fold-fitted
head. AP/AUROC are computed from the score, while Macro-F1 uses the routed
decision. All switches and thresholds are selected on the validation carve of
each outer fold, then applied to that fold's held-out test samples.
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
from models.cv_reliability_gate import (
    fit_conf_advantage_switch,
    fit_uncertain_switch,
    rank01,
)
from models.cv_source_condition_gate import (
    apply_mask,
    fixed_masks,
    select_fixed_mask,
    select_uncertain_mask,
)
from models.data import load_split
from models.fusion_eval import best_thr, build_split_features, load_bundles, macro


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


def put_score_with_decision(oof, name, score_name, decision_name):
    score = oof[score_name]
    decision = oof[decision_name]
    oof[name] = {
        "p": score["p"].copy(),
        "yhat": decision["yhat"].copy(),
        "score_head": score_name,
        "decision_head": decision_name,
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


def src_bin(sc):
    if sc <= 0:
        return "src0"
    if sc == 1:
        return "src1"
    if sc <= 3:
        return "src2_3"
    return "src4p"


def group_labels(recs, spec):
    sc = np.asarray([source_count(r) for r in recs], int)
    sl = np.asarray([source_len(r) for r in recs], int)
    conf = np.asarray([str(r.get("confidence", "")) for r in recs], object)
    bins = np.asarray([src_bin(int(x)) for x in sc], object)
    if spec == "srcbin":
        return bins
    if spec == "src_ge2":
        return np.where(sc >= 2, "src_ge2", "src_lt2")
    if spec == "src_ge4":
        return np.where(sc >= 4, "src_ge4", "src_lt4")
    if spec == "src_len20":
        return np.where(sl >= 20, "src_len20", "src_len_lt20")
    if spec == "confidence":
        return conf
    if spec == "confabs_srcbin":
        return np.asarray([f"{'conf_absent' if c == 'absent' else 'conf_seen'}:{b}"
                           for c, b in zip(conf, bins)], object)
    raise ValueError(f"unknown group spec: {spec}")


def greedy_group_thresholds(yv, pv, groups, min_n=35, min_pos=5):
    global_thr = best_thr(yv, pv)
    current = (pv >= global_thr).astype(int)
    current_score = macro(yv, current)
    by_group = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if int(m.sum()) < min_n:
            continue
        pos = int(yv[m].sum())
        neg = int(m.sum()) - pos
        if pos < min_pos or neg < min_pos:
            continue
        t = best_thr(yv[m], pv[m])
        proposal = current.copy()
        proposal[m] = (pv[m] >= t).astype(int)
        score = macro(yv, proposal)
        if score > current_score + 1e-8:
            by_group[str(g)] = float(t)
            current = proposal
            current_score = score
    return float(global_thr), by_group, float(current_score)


def apply_group_thresholds(pt, groups, global_thr, by_group):
    yhat = (pt >= global_thr).astype(int)
    for g, t in by_group.items():
        m = groups == g
        if m.any():
            yhat[m] = (pt[m] >= t).astype(int)
    return yhat


def put_groupthr(oof, name, idx, yv, pv, pt, gv, gt):
    global_thr, by_group, val_macro = greedy_group_thresholds(yv, pv, gv)
    oof.setdefault(name, {"p": np.full(oof["_n"], np.nan),
                          "yhat": np.full(oof["_n"], np.nan)})
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = apply_group_thresholds(pt, gt, global_thr, by_group)
    return {
        "global_thr": round(float(global_thr), 3),
        "group_thr": {k: round(float(v), 3) for k, v in by_group.items()},
        "val_macro": round(float(val_macro), 4),
    }


def paired_bootstrap_dual(y, p_a, yhat_a, p_b, yhat_b, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    dap_l, dau_l, df1_l = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        dap_l.append(average_precision_score(yy, p_a[idx]) -
                     average_precision_score(yy, p_b[idx]))
        dau_l.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1_l.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))

    def summ(values):
        arr = np.asarray(values, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {
        "dAP": summ(dap_l),
        "dAUROC": summ(dau_l),
        "dMacroF1": summ(df1_l),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--noargs_tmp", required=True)
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
        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in no_paths + ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"fold {fi} missing {missing}")

        no_b = load_bundles(no_paths)
        ar_b = load_bundles(ar_paths)
        _, p_no_v, yv, _, _ = build_split_features(no_b, "val")
        _, p_no_t, yt, _, _ = build_split_features(no_b, "test")
        _, p_ar_v, yv2, _, _ = build_split_features(ar_b, "val")
        _, p_ar_t, yt2, _, _ = build_split_features(ar_b, "test")
        if not (np.all(yv == yv2) and np.all(yt == yt2)):
            raise ValueError(f"fold {fi} noargs/args labels differ")

        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]

        base_v = {
            "bge_lr": p_bge_v,
            "noargs_pcls": p_no_v,
            "args_pcls": p_ar_v,
        }
        base_t = {
            "bge_lr": p_bge_t,
            "noargs_pcls": p_no_t,
            "args_pcls": p_ar_t,
        }
        rankavg_v = {
            "rankavg_args_no_bge": np.mean(
                [rank01(base_v[k]) for k in ("args_pcls", "noargs_pcls", "bge_lr")],
                axis=0,
            ),
            "rankavg_no_bge": 0.5 * rank01(p_no_v) + 0.5 * rank01(p_bge_v),
            "rankavg_args_bge": 0.5 * rank01(p_ar_v) + 0.5 * rank01(p_bge_v),
        }
        rankavg_t = {
            "rankavg_args_no_bge": np.mean(
                [rank01(base_t[k]) for k in ("args_pcls", "noargs_pcls", "bge_lr")],
                axis=0,
            ),
            "rankavg_no_bge": 0.5 * rank01(p_no_t) + 0.5 * rank01(p_bge_t),
            "rankavg_args_bge": 0.5 * rank01(p_ar_t) + 0.5 * rank01(p_bge_t),
        }

        meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx)}
        score_cache_v = {}
        score_cache_t = {}
        for name in ("bge_lr", "noargs_pcls", "args_pcls"):
            meta[f"{name}_thr"] = put(oof, name, te_idx, yv, base_v[name], base_t[name])
            score_cache_v[name] = base_v[name]
            score_cache_t[name] = base_t[name]
        for name in rankavg_v:
            meta[f"{name}_thr"] = put(oof, name, te_idx, yv, rankavg_v[name], rankavg_t[name])
            score_cache_v[name] = rankavg_v[name]
            score_cache_t[name] = rankavg_t[name]

        alt_names = ["noargs_pcls", "args_pcls", "rankavg_args_no_bge",
                     "rankavg_no_bge", "rankavg_args_bge"]
        for alt in alt_names:
            pv_alt = rankavg_v.get(alt, base_v.get(alt))
            pt_alt = rankavg_t.get(alt, base_t.get(alt))
            for obj in ("macro", "rank"):
                t, pv, pt, rv, rt = fit_uncertain_switch(
                    yv, p_bge_v, p_bge_t, pv_alt, pt_alt, obj)
                nm = f"switch_uncertain_{obj}_{alt}"
                meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
                score_cache_v[nm] = pv
                score_cache_t[nm] = pt
                meta[f"{nm}_t"] = round(float(t), 3)
                meta[f"{nm}_val_rate"] = round(float(rv), 3)
                meta[f"{nm}_test_rate"] = round(float(rt), 3)

                m, pv, pt, rv, rt = fit_conf_advantage_switch(
                    yv, p_bge_v, p_bge_t, pv_alt, pt_alt, obj)
                nm = f"switch_confadv_{obj}_{alt}"
                meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
                score_cache_v[nm] = pv
                score_cache_t[nm] = pt
                meta[f"{nm}_m"] = round(float(m), 3)
                meta[f"{nm}_val_rate"] = round(float(rv), 3)
                meta[f"{nm}_test_rate"] = round(float(rt), 3)

        mv = fixed_masks(val_recs, p_bge_v)
        mt = fixed_masks(test_recs, p_bge_t)
        for mask_name in ("src_ge2", "src_ge4", "src_len_ge20",
                          "src_present_bge_uncertain16"):
            nm = f"mask_{mask_name}_rankavg_args_bge"
            pv = apply_mask(p_bge_v, rankavg_v["rankavg_args_bge"], mv[mask_name])
            pt = apply_mask(p_bge_t, rankavg_t["rankavg_args_bge"], mt[mask_name])
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
            meta[f"{nm}_val_rate"] = round(float(mv[mask_name].mean()), 4)
            meta[f"{nm}_test_rate"] = round(float(mt[mask_name].mean()), 4)

        for obj in ("macro", "rank"):
            mask_name, pv, pt, rv, rt = select_fixed_mask(
                yv, p_bge_v, rankavg_v["rankavg_args_bge"], p_bge_t,
                rankavg_t["rankavg_args_bge"], val_recs, test_recs, obj)
            nm = f"select_fixed_{obj}_rankavg_args_bge"
            meta[f"{nm}_mask"] = mask_name
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
            meta[f"{nm}_val_rate"] = round(float(rv), 4)
            meta[f"{nm}_test_rate"] = round(float(rt), 4)

            min_src, t, pv, pt, rv, rt = select_uncertain_mask(
                yv, p_bge_v, rankavg_v["rankavg_args_bge"], p_bge_t,
                rankavg_t["rankavg_args_bge"], val_recs, test_recs, obj)
            nm = f"select_uncertain_{obj}_rankavg_args_bge"
            meta[f"{nm}_min_src"] = int(min_src)
            meta[f"{nm}_t"] = round(float(t), 3)
            meta[f"{nm}_thr"] = put(oof, nm, te_idx, yv, pv, pt)
            meta[f"{nm}_val_rate"] = round(float(rv), 4)
            meta[f"{nm}_test_rate"] = round(float(rt), 4)

        group_score_heads = [
            "bge_lr",
            "rankavg_args_bge",
            "switch_confadv_macro_rankavg_no_bge",
            "switch_confadv_rank_rankavg_args_bge",
            "switch_uncertain_macro_rankavg_no_bge",
        ]
        for score_name in group_score_heads:
            if score_name not in score_cache_v:
                continue
            for spec in ("srcbin", "src_ge2", "src_ge4", "src_len20",
                         "confidence", "confabs_srcbin"):
                nm = f"groupthr_{spec}_{score_name}"
                gv = group_labels(val_recs, spec)
                gt = group_labels(test_recs, spec)
                meta[f"{nm}_groupthr"] = put_groupthr(
                    oof, nm, te_idx, yv, score_cache_v[score_name],
                    score_cache_t[score_name], gv, gt)
        fold_meta.append(meta)

    score_heads = [
        "rankavg_args_bge",
        "switch_confadv_macro_rankavg_no_bge",
        "switch_uncertain_macro_rankavg_no_bge",
        "switch_confadv_rank_rankavg_args_bge",
        "switch_confadv_macro_rankavg_args_bge",
    ]
    decision_heads = [
        "mask_src_ge4_rankavg_args_bge",
        "mask_src_len_ge20_rankavg_args_bge",
        "select_uncertain_macro_rankavg_args_bge",
        "groupthr_srcbin_switch_confadv_macro_rankavg_no_bge",
        "groupthr_confabs_srcbin_switch_confadv_macro_rankavg_no_bge",
        "groupthr_src_ge2_switch_confadv_macro_rankavg_no_bge",
        "groupthr_srcbin_switch_confadv_rank_rankavg_args_bge",
        "groupthr_srcbin_rankavg_args_bge",
        "mask_src_ge2_rankavg_args_bge",
        "switch_confadv_rank_rankavg_no_bge",
        "switch_confadv_rank_rankavg_args_bge",
    ]
    for score_name in score_heads:
        if score_name not in oof:
            continue
        for decision_name in decision_heads:
            if decision_name not in oof:
                continue
            name = f"dual_score={score_name}__decision={decision_name}"
            put_score_with_decision(oof, name, score_name, decision_name)

    methods = [m for m in oof if m != "_n"]
    rows = {}
    for name in methods:
        ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof[name]["yhat"]))
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])

    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auroc"], rows[m]["auprc"]),
                    reverse=True)
    print("=== Top dual-head candidates ===", flush=True)
    for name in ranked[:35]:
        r = rows[name]
        print(f"{name:100s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig_names = set(ranked[:25])
    for key in ("auprc", "auroc", "wF1"):
        sig_names.update(sorted(rows, key=lambda m: rows[m][key], reverse=True)[:12])
    sig_names.update(score_heads + decision_heads + ["rankavg_args_bge"])
    sig = {}
    bge = oof["bge_lr"]
    for name in sorted(sig_names):
        if name == "bge_lr" or name not in oof:
            continue
        ok = ((~np.isnan(oof[name]["p"])) & (~np.isnan(oof[name]["yhat"])) &
              (~np.isnan(bge["p"])) & (~np.isnan(bge["yhat"])))
        sig[f"{name}_vs_bge_lr"] = paired_bootstrap_dual(
            y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
            bge["p"][ok], bge["yhat"][ok], n_boot=args.n_boot)
        s = sig[f"{name}_vs_bge_lr"]
        r = rows[name]
        print(f"{name:100s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f} "
              f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
              f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
              f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
              flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows, "fold_meta": fold_meta,
           "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv_dual_head_router] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
