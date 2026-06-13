"""Focused guarded group-threshold router for CLAIMARC OOF bundles.

The dual-head diagnostics found that source-bin thresholds can overfit a small
validation carve, e.g. a source-rich group threshold far below the fold-global
threshold. This script evaluates conservative guards that reject or clip such
thresholds before applying them to the held-out fold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_dual_head_router import (
    apply_group_thresholds,
    greedy_group_thresholds,
    group_labels,
    paired_bootstrap_dual,
)
from models.cv_eval import make_folds, val_carve
from models.cv_reliability_gate import fit_conf_advantage_switch, rank01
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


def guard_thresholds(global_thr, by_group, mode, max_drop, min_thr):
    guarded = {}
    rejected = {}
    lower = max(float(min_thr), float(global_thr) - float(max_drop))
    for g, t in by_group.items():
        t = float(t)
        if t < lower:
            if mode == "reject":
                rejected[g] = t
                continue
            if mode == "clip":
                guarded[g] = lower
                continue
            raise ValueError(f"unknown guard mode: {mode}")
        guarded[g] = t
    return guarded, rejected, lower


def shrink_thresholds(global_thr, by_group, groups, k, min_thr):
    shrunk = {}
    meta = {}
    for g, t in by_group.items():
        m = groups == g
        n = int(m.sum())
        alpha = n / (n + float(k))
        tt = float(global_thr) + alpha * (float(t) - float(global_thr))
        tt = max(float(min_thr), tt)
        shrunk[g] = float(tt)
        meta[g] = {"n": n, "alpha": round(float(alpha), 4), "raw": round(float(t), 3),
                   "shrunk": round(float(tt), 3)}
    return shrunk, meta


def _rate_at(p, thr):
    return float((p >= float(thr)).mean())


def _threshold_for_rate_cap(p, thr, max_rate, min_rate):
    p = np.asarray(p, float)
    out = float(thr)
    if max_rate is not None and _rate_at(p, out) > float(max_rate):
        for cand in sorted(set(p.tolist() + [out, 1.0])):
            if cand >= out and _rate_at(p, cand) <= float(max_rate):
                out = float(cand)
                break
    if min_rate is not None and _rate_at(p, out) < float(min_rate):
        for cand in sorted(set(p.tolist() + [out, 0.0]), reverse=True):
            if cand <= out and _rate_at(p, cand) >= float(min_rate):
                out = float(cand)
                break
    return out


def prior_cap_thresholds(global_thr, by_group, yv, pv, groups, up_margin, down_margin, smooth):
    adjusted = {}
    meta = {}
    global_prior = float(np.mean(yv))
    for g, t in by_group.items():
        m = groups == g
        yy = np.asarray(yv[m], int)
        pp = np.asarray(pv[m], float)
        n = int(m.sum())
        pos = int(yy.sum())
        prior = (pos + float(smooth) * global_prior) / (n + float(smooth))
        max_rate = min(0.95, prior + float(up_margin))
        min_rate = max(0.0, prior - float(down_margin))
        tt = _threshold_for_rate_cap(pp, float(t), max_rate, min_rate)
        adjusted[g] = float(tt)
        meta[g] = {
            "n": n,
            "pos_rate": round(float(np.mean(yy)), 4),
            "smooth_prior": round(float(prior), 4),
            "raw_thr": round(float(t), 3),
            "raw_pred_rate": round(_rate_at(pp, t), 4),
            "adj_thr": round(float(tt), 3),
            "adj_pred_rate": round(_rate_at(pp, tt), 4),
            "max_rate": round(float(max_rate), 4),
            "min_rate": round(float(min_rate), 4),
        }
    return adjusted, meta


def select_bge_veto(yv, p_score_v, p_bge_v, base_v, p_score_t, p_bge_t, base_t, bge_thr):
    best = {
        "score": float(macro(yv, base_v)),
        "delta": None,
        "high": None,
        "test": np.asarray(base_t, int).copy(),
    }
    for delta in [-0.04, 0.0, 0.04, 0.08, 0.12, 0.20, 0.30]:
        for high in [0.62, 0.70, 0.78, 0.86, 1.01]:
            keep_v = (p_bge_v >= float(bge_thr) - delta) | (p_score_v >= high)
            yy_v = (base_v.astype(bool) & keep_v).astype(int)
            score = float(macro(yv, yy_v))
            if score > best["score"] + 1e-8:
                keep_t = (p_bge_t >= float(bge_thr) - delta) | (p_score_t >= high)
                best = {
                    "score": score,
                    "delta": float(delta),
                    "high": float(high),
                    "test": (base_t.astype(bool) & keep_t).astype(int),
                }
    return best["test"], {k: v for k, v in best.items() if k != "test"}


def select_bge_veto_rescue(yv, p_score_v, p_bge_v, base_v, p_score_t, p_bge_t,
                           base_t, bge_thr, score_thr):
    best = {
        "score": float(macro(yv, base_v)),
        "delta": None,
        "high": None,
        "boost": None,
        "slack": None,
        "test": np.asarray(base_t, int).copy(),
    }
    for delta in [-0.04, 0.0, 0.04, 0.08, 0.12, 0.20]:
        for high in [0.70, 0.78, 0.86, 1.01]:
            keep_v = (p_bge_v >= float(bge_thr) - delta) | (p_score_v >= high)
            keep_t = (p_bge_t >= float(bge_thr) - delta) | (p_score_t >= high)
            veto_v = base_v.astype(bool) & keep_v
            veto_t = base_t.astype(bool) & keep_t
            for boost in [0.0, 0.04, 0.08, 0.14, 9.0]:
                for slack in [0.0, 0.04, 0.08, 0.12]:
                    if boost > 1.0:
                        rescue_v = np.zeros_like(veto_v, dtype=bool)
                        rescue_t = np.zeros_like(veto_t, dtype=bool)
                    else:
                        rescue_v = ((~base_v.astype(bool)) &
                                    (p_bge_v >= float(bge_thr) + boost) &
                                    (p_score_v >= float(score_thr) - slack))
                        rescue_t = ((~base_t.astype(bool)) &
                                    (p_bge_t >= float(bge_thr) + boost) &
                                    (p_score_t >= float(score_thr) - slack))
                    yy_v = (veto_v | rescue_v).astype(int)
                    score = float(macro(yv, yy_v))
                    if score > best["score"] + 1e-8:
                        best = {
                            "score": score,
                            "delta": float(delta),
                            "high": float(high),
                            "boost": None if boost > 1.0 else float(boost),
                            "slack": None if boost > 1.0 else float(slack),
                            "test": (veto_t | rescue_t).astype(int),
                        }
    return best["test"], {k: v for k, v in best.items() if k != "test"}


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
    ap.add_argument("--sig_top_k", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {"_n": len(recs)}
    meta = []
    configs = [
        ("guard_clip_drop00_min40", "clip", 0.00, 0.40),
        ("guard_clip_drop04_min40", "clip", 0.04, 0.40),
        ("guard_clip_drop08_min40", "clip", 0.08, 0.40),
        ("guard_reject_drop12_min35", "reject", 0.12, 0.35),
        ("guard_clip_drop12_min35", "clip", 0.12, 0.35),
        ("guard_clip_drop12_min45", "clip", 0.12, 0.45),
        ("guard_clip_drop16_min40", "clip", 0.16, 0.40),
        ("guard_reject_drop20_min30", "reject", 0.20, 0.30),
        ("guard_clip_drop20_min30", "clip", 0.20, 0.30),
        ("guard_clip_drop20_min45", "clip", 0.20, 0.45),
        ("guard_reject_drop28_min25", "reject", 0.28, 0.25),
        ("guard_clip_drop28_min25", "clip", 0.28, 0.25),
    ]
    shrink_configs = [
        ("shrink_k30_min30", 30, 0.30),
        ("shrink_k60_min30", 60, 0.30),
        ("shrink_k100_min30", 100, 0.30),
        ("shrink_k60_min35", 60, 0.35),
        ("shrink_k100_min35", 100, 0.35),
        ("shrink_k160_min35", 160, 0.35),
    ]
    prior_configs = [
        ("priorcap_u10_d20_s20", 0.10, 0.20, 20),
        ("priorcap_u15_d20_s20", 0.15, 0.20, 20),
        ("priorcap_u20_d20_s20", 0.20, 0.20, 20),
        ("priorcap_u10_d15_s50", 0.10, 0.15, 50),
        ("priorcap_u15_d15_s50", 0.15, 0.15, 50),
        ("priorcap_u20_d15_s50", 0.20, 0.15, 50),
        ("priorcap_u15_d10_s80", 0.15, 0.10, 80),
    ]

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
        _, p_no_t, _, _, _ = build_split_features(no_b, "test")
        _, p_args_v, yv2, _, _ = build_split_features(ar_b, "val")
        _, p_args_t, _, _, _ = build_split_features(ar_b, "test")
        if not np.all(yv == yv2):
            raise ValueError(f"fold {fi} noargs/args val labels differ")
        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        p_rank_args_v = 0.5 * rank01(p_args_v) + 0.5 * rank01(p_bge_v)
        p_rank_args_t = 0.5 * rank01(p_args_t) + 0.5 * rank01(p_bge_t)
        p_rank_no_v = 0.5 * rank01(p_no_v) + 0.5 * rank01(p_bge_v)
        p_rank_no_t = 0.5 * rank01(p_no_t) + 0.5 * rank01(p_bge_t)

        bge_thr = put(oof, "bge_lr", te_idx, yv, p_bge_v, p_bge_t)
        put(oof, "rankavg_args_bge", te_idx, yv, p_rank_args_v, p_rank_args_t)

        _, p_score_v, p_score_t, rv, rt = fit_conf_advantage_switch(
            yv, p_bge_v, p_bge_t, p_rank_no_v, p_rank_no_t, "macro")
        put(oof, "score_switch_confadv_macro_rankavg_no_bge", te_idx, yv,
            p_score_v, p_score_t)

        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        gv = group_labels(val_recs, "srcbin")
        gt = group_labels(test_recs, "srcbin")
        global_thr, by_group, val_macro = greedy_group_thresholds(yv, p_score_v, gv)
        fold_meta = {
            "fold": fi,
            "n_val": len(va_idx),
            "n_test": len(te_idx),
            "score_switch_rate_val": round(float(rv), 4),
            "score_switch_rate_test": round(float(rt), 4),
            "raw_global_thr": round(float(global_thr), 3),
            "raw_group_thr": {k: round(float(v), 3) for k, v in by_group.items()},
            "raw_val_macro": round(float(val_macro), 4),
        }
        for name, mode, max_drop, min_thr in configs:
            guarded, rejected, lower = guard_thresholds(global_thr, by_group, mode, max_drop, min_thr)
            method = f"{name}_switchmacro_srcbin"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = p_score_t
            oof[method]["yhat"][te_idx] = apply_group_thresholds(
                p_score_t, gt, global_thr, guarded)
            fold_meta[method] = {
                "lower": round(float(lower), 3),
                "group_thr": {k: round(float(v), 3) for k, v in guarded.items()},
                "rejected": {k: round(float(v), 3) for k, v in rejected.items()},
            }
            if name in {"guard_clip_drop12_min35", "guard_clip_drop20_min30"}:
                base_v = apply_group_thresholds(p_score_v, gv, global_thr, guarded)
                base_t = apply_group_thresholds(p_score_t, gt, global_thr, guarded)
                vt, vm = select_bge_veto(
                    yv, p_score_v, p_bge_v, base_v, p_score_t, p_bge_t, base_t, bge_thr)
                vname = f"{name}_bgeveto_switchmacro_srcbin"
                oof.setdefault(vname, {"p": np.full(oof["_n"], np.nan),
                                       "yhat": np.full(oof["_n"], np.nan)})
                oof[vname]["p"][te_idx] = p_score_t
                oof[vname]["yhat"][te_idx] = vt
                fold_meta[vname] = {"base": method, "selected": vm}
                vrt, vrm = select_bge_veto_rescue(
                    yv, p_score_v, p_bge_v, base_v, p_score_t, p_bge_t,
                    base_t, bge_thr, global_thr)
                vrname = f"{name}_bgeveto_rescue_switchmacro_srcbin"
                oof.setdefault(vrname, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[vrname]["p"][te_idx] = p_score_t
                oof[vrname]["yhat"][te_idx] = vrt
                fold_meta[vrname] = {"base": method, "selected": vrm}
        for name, k, min_thr in shrink_configs:
            guarded, info = shrink_thresholds(global_thr, by_group, gv, k, min_thr)
            method = f"{name}_switchmacro_srcbin"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = p_score_t
            oof[method]["yhat"][te_idx] = apply_group_thresholds(
                p_score_t, gt, global_thr, guarded)
            fold_meta[method] = {
                "group_thr": {k: round(float(v), 3) for k, v in guarded.items()},
                "shrink": info,
            }
        for name, up, down, smooth in prior_configs:
            guarded, info = prior_cap_thresholds(
                global_thr, by_group, yv, p_score_v, gv, up, down, smooth)
            method = f"{name}_switchmacro_srcbin"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = p_score_t
            oof[method]["yhat"][te_idx] = apply_group_thresholds(
                p_score_t, gt, global_thr, guarded)
            fold_meta[method] = {
                "group_thr": {k: round(float(v), 3) for k, v in guarded.items()},
                "prior_cap": info,
            }
        meta.append(fold_meta)

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== Guarded group router candidates ===", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:48s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig = {}
    sig_names = ranked[:args.sig_top_k] if args.sig_top_k > 0 else ranked
    for name in sig_names:
        if name == "bge_lr":
            continue
        if args.n_boot <= 0:
            continue
        ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
        sig[f"{name}_vs_bge_lr"] = paired_bootstrap_dual(
            y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
            oof["bge_lr"]["p"][ok], oof["bge_lr"]["yhat"][ok],
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
    print(f"[cv_guarded_group_router] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
