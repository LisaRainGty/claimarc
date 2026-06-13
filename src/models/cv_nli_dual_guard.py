"""Dual-head diagnostic: NLI evidence-posterior ranking + guarded RACL decision."""
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
from models.cv_guarded_group_router import guard_thresholds
from models.cv_nli_evidence_head import fit_hgb
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


def rank_mix(p_nli, p_bge, alpha):
    return float(alpha) * rank01(p_nli) + (1.0 - float(alpha)) * rank01(p_bge)


def select_alpha(yv, p_nli_v, p_bge_v):
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        pv = rank_mix(p_nli_v, p_bge_v, alpha)
        ap = average_precision_score(yv, pv)
        au = roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.5
        score = ap + 0.50 * au
        if best is None or score > best[0]:
            best = (score, float(alpha), pv)
    return best[1], best[2]


def val_utility(yv, pv, yhat_v, mode="balanced"):
    ap = float(average_precision_score(yv, pv))
    au = float(roc_auc_score(yv, pv)) if len(set(yv.tolist())) > 1 else 0.5
    mf = float(macro(yv, yhat_v))
    if mode == "macro":
        score = mf
    else:
        score = mf + 0.50 * ap + 0.25 * au
    return score, {"ap": ap, "auroc": au, "macro_f1": mf}


def select_score_edit(yv, base_v, score_v, base_t, score_t):
    best = {
        "score": float(macro(yv, base_v)),
        "low": None,
        "high": None,
        "test": np.asarray(base_t, int).copy(),
    }
    lows = [None, 0.15, 0.25, 0.35, 0.45]
    highs = [None, 0.55, 0.65, 0.75, 0.85]
    for low in lows:
        for high in highs:
            yy_v = np.asarray(base_v, int).copy()
            yy_t = np.asarray(base_t, int).copy()
            if low is not None:
                yy_v[(base_v == 1) & (score_v < low)] = 0
                yy_t[(base_t == 1) & (score_t < low)] = 0
            if high is not None:
                yy_v[(base_v == 0) & (score_v >= high)] = 1
                yy_t[(base_t == 0) & (score_t >= high)] = 1
            score = float(macro(yv, yy_v))
            if score > best["score"] + 1e-8:
                best = {"score": score, "low": low, "high": high, "test": yy_t}
    return best["test"], {k: v for k, v in best.items() if k != "test"}


def select_group_score_edit(yv, base_v, score_v, groups_v, base_t, score_t, groups_t,
                            min_n=25, min_pos=5, min_gain=1e-8):
    cur_v = np.asarray(base_v, int).copy()
    cur_t = np.asarray(base_t, int).copy()
    cur_score = float(macro(yv, cur_v))
    selected = {}
    lows = [None, 0.15, 0.25, 0.35, 0.45]
    highs = [None, 0.55, 0.65, 0.75, 0.85]
    for g in sorted(set(groups_v.tolist())):
        m = groups_v == g
        if int(m.sum()) < min_n:
            continue
        pos = int(yv[m].sum())
        neg = int(m.sum()) - pos
        if pos < min_pos or neg < min_pos:
            continue
        best = None
        for low in lows:
            for high in highs:
                prop = cur_v.copy()
                if low is not None:
                    prop[m & (cur_v == 1) & (score_v < low)] = 0
                if high is not None:
                    prop[m & (cur_v == 0) & (score_v >= high)] = 1
                sc = float(macro(yv, prop))
                if best is None or sc > best[0] + 1e-8:
                    best = (sc, low, high, prop)
        if best and best[0] > cur_score + min_gain:
            cur_v = best[3]
            mt = groups_t == g
            if best[1] is not None:
                cur_t[mt & (cur_t == 1) & (score_t < best[1])] = 0
            if best[2] is not None:
                cur_t[mt & (cur_t == 0) & (score_t >= best[2])] = 1
            cur_score = best[0]
            selected[str(g)] = {
                "low": best[1],
                "high": best[2],
                "val_macro": round(float(cur_score), 4),
            }
    return cur_v, cur_t, selected


def risky_source_group(g):
    parts = str(g).split(":")
    src = parts[0] if parts else str(g)
    if src == "src0":
        return True
    if src == "src2_3":
        return True
    return False


def select_source_veto(yv, base_v, signal_v, groups_v, base_t, signal_t, groups_t,
                       min_n=25, min_pos=5, min_neg=5, min_gain=0.001):
    cur_v = np.asarray(base_v, int).copy()
    cur_t = np.asarray(base_t, int).copy()
    cur_score = float(macro(yv, cur_v))
    selected = {}
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for g in sorted(set(groups_v.tolist())):
        if not risky_source_group(g):
            continue
        m = groups_v == g
        if int(m.sum()) < min_n:
            continue
        pos = int(yv[m].sum())
        neg = int(m.sum()) - pos
        if pos < min_pos or neg < min_neg:
            continue
        best = None
        for thr in thresholds:
            prop = cur_v.copy()
            prop[m & (cur_v == 1) & (signal_v < thr)] = 0
            sc = float(macro(yv, prop))
            if best is None or sc > best[0] + 1e-8:
                best = (sc, thr, prop)
        if best and best[0] > cur_score + min_gain:
            cur_v = best[2]
            mt = groups_t == g
            cur_t[mt & (cur_t == 1) & (signal_t < best[1])] = 0
            cur_score = best[0]
            selected[str(g)] = {
                "min_signal": best[1],
                "val_macro": round(float(cur_score), 4),
            }
    return cur_v, cur_t, selected


def source_mode_mask(groups, mode):
    g = np.asarray(groups).astype(str)
    if mode == "src0":
        return g == "src0"
    if mode == "src2_3":
        return g == "src2_3"
    if mode == "src0_src2_3":
        return (g == "src0") | (g == "src2_3")
    if mode == "src0_src2_3_lowabs":
        return np.asarray([
            str(x) == "src0" or str(x).startswith("src0:")
            or str(x).startswith("src2_3:low")
            or str(x).startswith("src2_3:absent")
            for x in groups
        ], bool)
    raise ValueError(f"unknown source fallback mode: {mode}")


def apply_bge_fallback(base_v, bge_v, groups_v, base_t, bge_t, groups_t, mode):
    cur_v = np.asarray(base_v, int).copy()
    cur_t = np.asarray(base_t, int).copy()

    mv = source_mode_mask(groups_v, mode)
    mt = source_mode_mask(groups_t, mode)
    cur_v[mv] = np.asarray(bge_v, int)[mv]
    cur_t[mt] = np.asarray(bge_t, int)[mt]
    return cur_v, cur_t


def select_bge_advantage_fallback(yv, base_v, bge_v, groups_v, base_t, bge_t,
                                  groups_t, target_fn, min_n=35, min_pos=5,
                                  min_neg=5, min_gain=0.01,
                                  rate_slack=0.08):
    cur_v = np.asarray(base_v, int).copy()
    cur_t = np.asarray(base_t, int).copy()
    selected = {}
    for g in sorted(set(np.asarray(groups_v).astype(str).tolist())):
        if not target_fn(str(g)):
            continue
        mv = np.asarray(groups_v).astype(str) == str(g)
        n = int(mv.sum())
        if n < min_n:
            continue
        pos = int(np.asarray(yv, int)[mv].sum())
        neg = n - pos
        if pos < min_pos or neg < min_neg:
            continue
        yy = np.asarray(yv, int)[mv]
        base_group = np.asarray(base_v, int)[mv]
        bge_group = np.asarray(bge_v, int)[mv]
        base_macro = float(macro(yy, base_group))
        bge_macro = float(macro(yy, bge_group))
        pos_rate = float(yy.mean())
        base_rate_gap = abs(float(base_group.mean()) - pos_rate)
        bge_rate_gap = abs(float(bge_group.mean()) - pos_rate)
        if bge_macro > base_macro + float(min_gain) and (
                bge_rate_gap <= base_rate_gap + float(rate_slack)):
            cur_v[mv] = bge_group
            mt = np.asarray(groups_t).astype(str) == str(g)
            cur_t[mt] = np.asarray(bge_t, int)[mt]
            selected[str(g)] = {
                "n": n,
                "pos": pos,
                "base_macro": round(base_macro, 4),
                "bge_macro": round(bge_macro, 4),
                "base_rate_gap": round(base_rate_gap, 4),
                "bge_rate_gap": round(bge_rate_gap, 4),
            }
    return cur_v, cur_t, selected


def select_bge_rate_guard_fallback(yv, base_v, bge_v, groups_v, base_t, bge_t,
                                   groups_t, target_fn, min_n=30, min_pos=5,
                                   min_neg=5, over_margin=0.05,
                                   min_rate_drop=0.02, gap_slack=0.03):
    cur_v = np.asarray(base_v, int).copy()
    cur_t = np.asarray(base_t, int).copy()
    selected = {}
    gv = np.asarray(groups_v).astype(str)
    gt = np.asarray(groups_t).astype(str)
    yv = np.asarray(yv, int)
    for g in sorted(set(gv.tolist())):
        if not target_fn(str(g)):
            continue
        mv = gv == str(g)
        n = int(mv.sum())
        if n < min_n:
            continue
        pos = int(yv[mv].sum())
        neg = n - pos
        if pos < min_pos or neg < min_neg:
            continue
        label_rate = float(yv[mv].mean())
        base_group = np.asarray(base_v, int)[mv]
        bge_group = np.asarray(bge_v, int)[mv]
        base_rate = float(base_group.mean())
        bge_rate = float(bge_group.mean())
        base_gap = abs(base_rate - label_rate)
        bge_gap = abs(bge_rate - label_rate)
        over = base_rate - label_rate
        rate_drop = base_rate - bge_rate
        if (over >= float(over_margin)
                and rate_drop >= float(min_rate_drop)
                and bge_gap <= base_gap + float(gap_slack)):
            cur_v[mv] = bge_group
            mt = gt == str(g)
            cur_t[mt] = np.asarray(bge_t, int)[mt]
            selected[str(g)] = {
                "n": n,
                "pos": pos,
                "label_rate": round(label_rate, 4),
                "base_rate": round(base_rate, 4),
                "bge_rate": round(bge_rate, 4),
                "base_gap": round(base_gap, 4),
                "bge_gap": round(bge_gap, 4),
                "over": round(over, 4),
                "rate_drop": round(rate_drop, 4),
            }
    return cur_v, cur_t, selected


def apply_score_fallback(base_v, bge_v, groups_v, base_t, bge_t, groups_t,
                         mode, bge_weight):
    cur_v = np.asarray(base_v, float).copy()
    cur_t = np.asarray(base_t, float).copy()
    bge_rank_v = rank01(bge_v)
    bge_rank_t = rank01(bge_t)
    w = float(bge_weight)
    mv = source_mode_mask(groups_v, mode)
    mt = source_mode_mask(groups_t, mode)
    cur_v[mv] = (1.0 - w) * cur_v[mv] + w * bge_rank_v[mv]
    cur_t[mt] = (1.0 - w) * cur_t[mt] + w * bge_rank_t[mt]
    return cur_v, cur_t


NLI_TYPES = ("all", "param", "ocr", "vlm", "arg_sup", "arg_ref", "arg_gap")
NLI_VALUES = ("contr", "entail", "neutral", "margin", "max_ce", "uncertainty")
NLI_STATS = ("count", "mean", "max", "min", "std", "top2mean",
             "p25", "p75", "share20", "share35", "share50")


def nli_feature_idx(typ, value, stat):
    return (
        NLI_TYPES.index(typ) * len(NLI_VALUES) * len(NLI_STATS)
        + NLI_VALUES.index(value) * len(NLI_STATS)
        + NLI_STATS.index(stat)
    )


def nli_evidence_signals(X_part):
    X_part = np.asarray(X_part, float)
    phys_types = ("param", "ocr", "vlm")
    phys_contr = np.maximum.reduce([
        X_part[:, nli_feature_idx(t, "contr", "max")] for t in phys_types
    ])
    phys_entail = np.maximum.reduce([
        X_part[:, nli_feature_idx(t, "entail", "max")] for t in phys_types
    ])
    phys_margin = np.maximum.reduce([
        X_part[:, nli_feature_idx(t, "margin", "max")] for t in phys_types
    ])
    arg_contr = np.maximum(
        X_part[:, nli_feature_idx("arg_ref", "contr", "max")],
        X_part[:, nli_feature_idx("arg_gap", "contr", "max")],
    )
    n_phys = X_part[:, 463] + X_part[:, 464] + X_part[:, 465]
    return {
        "phys_contr": phys_contr,
        "phys_entail": phys_entail,
        "phys_margin": phys_margin,
        "all_contr": X_part[:, nli_feature_idx("all", "contr", "max")],
        "all_margin": X_part[:, nli_feature_idx("all", "margin", "max")],
        "all_uncertainty": X_part[:, nli_feature_idx("all", "uncertainty", "top2mean")],
        "all_neutral": X_part[:, nli_feature_idx("all", "neutral", "mean")],
        "arg_contr": arg_contr,
        "arg_dom": arg_contr - phys_contr,
        "contr_minus_entail": phys_contr - phys_entail,
        "nphys": n_phys,
        "source_len": X_part[:, 469],
    }


def nli_veto_target_masks(recs, X_part):
    conf = edit_group_labels(recs, "confidence").astype(str)
    src = group_labels(recs, "srcbin").astype(str)
    src_conf = edit_group_labels(recs, "srcbin_conf")
    sig = nli_evidence_signals(X_part)
    return {
        "lowconf": conf == "low",
        "source_rich_low": (conf == "low") & (sig["nphys"] >= 2.0),
        "src2_3_low": (src == "src2_3") & (conf == "low"),
        "src0_src2_3_lowabs": source_mode_mask(src_conf, "src0_src2_3_lowabs"),
        "nonzero_source": sig["nphys"] > 0.0,
    }


def select_nli_evidence_veto(yv, base_v, bge_v, signals_v, masks_v,
                             base_t, bge_t, signals_t, masks_t,
                             min_flip=6, max_flip_frac=0.30,
                             min_gain=0.001):
    yv = np.asarray(yv, int)
    base_v = np.asarray(base_v, int)
    base_t = np.asarray(base_t, int)
    bge_v = np.asarray(bge_v, int)
    bge_t = np.asarray(bge_t, int)
    base_score = float(macro(yv, base_v))
    best = {
        "score": base_score,
        "test": base_t.copy(),
        "val": base_v.copy(),
        "rule": None,
    }
    quantiles = (0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
    for target_name, target_mask_v in masks_v.items():
        target_mask_v = np.asarray(target_mask_v, bool)
        target_mask_t = np.asarray(masks_t[target_name], bool)
        if int(target_mask_v.sum()) < min_flip:
            continue
        max_flip = max(min_flip, int(round(float(max_flip_frac) * target_mask_v.sum())))
        for signal_name, sig_v in signals_v.items():
            sig_v = np.asarray(sig_v, float)
            sig_t = np.asarray(signals_t[signal_name], float)
            vals = sig_v[target_mask_v]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0 or float(vals.max()) <= float(vals.min()):
                continue
            thresholds = sorted(set(float(np.quantile(vals, q)) for q in quantiles))
            for op in ("lt", "gt"):
                for thr in thresholds:
                    cond_v = target_mask_v & (base_v == 1)
                    cond_t = target_mask_t & (base_t == 1)
                    if op == "lt":
                        cond_v &= sig_v < thr
                        cond_t &= sig_t < thr
                    else:
                        cond_v &= sig_v > thr
                        cond_t &= sig_t > thr
                    n_flip = int(cond_v.sum())
                    if n_flip < min_flip or n_flip > max_flip:
                        continue
                    fp_flip = int(((base_v == 1) & (yv == 0) & cond_v).sum())
                    tp_flip = int(((base_v == 1) & (yv == 1) & cond_v).sum())
                    if fp_flip <= tp_flip:
                        continue
                    for action in ("zero", "bge"):
                        prop_v = base_v.copy()
                        prop_t = base_t.copy()
                        if action == "zero":
                            prop_v[cond_v] = 0
                            prop_t[cond_t] = 0
                        else:
                            prop_v[cond_v] = bge_v[cond_v]
                            prop_t[cond_t] = bge_t[cond_t]
                        score = float(macro(yv, prop_v))
                        if score > best["score"] + float(min_gain):
                            best = {
                                "score": score,
                                "test": prop_t,
                                "val": prop_v,
                                "rule": {
                                    "target": target_name,
                                    "signal": signal_name,
                                    "op": op,
                                    "thr": round(float(thr), 4),
                                    "action": action,
                                    "n_flip_val": n_flip,
                                    "fp_flip_val": fp_flip,
                                    "tp_flip_val": tp_flip,
                                    "val_macro": round(score, 4),
                                },
                            }
    meta = {"base_macro": round(base_score, 4)}
    if best["rule"] is not None:
        meta.update(best["rule"])
    return best["val"], best["test"], meta


def select_group_head_mix(yv, groups_v, groups_t, heads_v, heads_t, base_name,
                          candidate_names, min_n=35, min_pos=5, min_gain=0.001):
    cur_v = np.asarray(heads_v[base_name], int).copy()
    cur_t = np.asarray(heads_t[base_name], int).copy()
    cur_score = float(macro(yv, cur_v))
    selected = {"base": base_name, "groups": {}}
    for g in sorted(set(groups_v.tolist())):
        m = groups_v == g
        if int(m.sum()) < min_n:
            continue
        pos = int(yv[m].sum())
        neg = int(m.sum()) - pos
        if pos < min_pos or neg < min_pos:
            continue
        best = None
        for cand in candidate_names:
            if cand not in heads_v:
                continue
            prop = cur_v.copy()
            prop[m] = np.asarray(heads_v[cand], int)[m]
            score = float(macro(yv, prop))
            if best is None or score > best[0] + 1e-8:
                best = (score, cand, prop)
        if best and best[0] > cur_score + min_gain:
            cur_v = best[2]
            mt = groups_t == g
            cur_t[mt] = np.asarray(heads_t[best[1]], int)[mt]
            cur_score = best[0]
            selected["groups"][str(g)] = {
                "head": best[1],
                "val_macro": round(float(cur_score), 4),
            }
    return cur_v, cur_t, selected


def split_router_validation(yv, seed, fit_frac=0.5):
    rng = np.random.default_rng(seed)
    yv = np.asarray(yv, int)
    fit = np.zeros(len(yv), bool)
    for cls in (0, 1):
        idx = np.where(yv == cls)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_fit = int(round(len(idx) * float(fit_frac)))
        n_fit = max(1, min(len(idx) - 1, n_fit)) if len(idx) > 1 else len(idx)
        fit[idx[:n_fit]] = True
    sel = ~fit
    if int(fit.sum()) == 0 or int(sel.sum()) == 0:
        fit = np.zeros(len(yv), bool)
        fit[: max(1, len(yv) // 2)] = True
        sel = ~fit
    return fit, sel


def threshold_from_fit(y_fit, p_fit, p_apply):
    thr = best_thr(y_fit, p_fit)
    return (np.asarray(p_apply, float) >= thr).astype(int), float(thr)


def scoreguard_from_fit(y_fit, p_fit, g_fit, p_apply, g_apply,
                        max_drop=0.20, min_thr=0.30):
    global_thr, by_group, val_macro = greedy_group_thresholds(y_fit, p_fit, g_fit)
    guarded, rejected, lower = guard_thresholds(
        global_thr, by_group, "clip", max_drop, min_thr)
    yhat = apply_group_thresholds(p_apply, g_apply, global_thr, guarded)
    meta = {
        "global_thr": round(float(global_thr), 4),
        "raw_val_macro": round(float(val_macro), 4),
        "lower": round(float(lower), 4),
        "group_thr": {k: round(float(v), 4) for k, v in guarded.items()},
        "rejected": {k: round(float(v), 4) for k, v in rejected.items()},
    }
    return yhat, meta


def select_group_head_mix_nested(y_fit, groups_fit, groups_sel, groups_t,
                                 heads_fit, heads_sel, heads_t, base_name,
                                 candidate_names, min_n=35, min_pos=5,
                                 min_gain=0.001):
    cur_fit = np.asarray(heads_fit[base_name], int).copy()
    cur_sel = np.asarray(heads_sel[base_name], int).copy()
    cur_t = np.asarray(heads_t[base_name], int).copy()
    cur_score = float(macro(y_fit, cur_fit))
    selected = {"base": base_name, "groups": {}}
    groups_fit = np.asarray(groups_fit)
    for g in sorted(set(groups_fit.tolist())):
        m = groups_fit == g
        if int(m.sum()) < min_n:
            continue
        pos = int(np.asarray(y_fit, int)[m].sum())
        neg = int(m.sum()) - pos
        if pos < min_pos or neg < min_pos:
            continue
        best = None
        for cand in candidate_names:
            if cand not in heads_fit:
                continue
            prop = cur_fit.copy()
            prop[m] = np.asarray(heads_fit[cand], int)[m]
            score = float(macro(y_fit, prop))
            if best is None or score > best[0] + 1e-8:
                best = (score, cand, prop)
        if best and best[0] > cur_score + min_gain:
            cur_fit = best[2]
            ms = np.asarray(groups_sel) == g
            mt = np.asarray(groups_t) == g
            cur_sel[ms] = np.asarray(heads_sel[best[1]], int)[ms]
            cur_t[mt] = np.asarray(heads_t[best[1]], int)[mt]
            cur_score = best[0]
            selected["groups"][str(g)] = {
                "head": best[1],
                "fit_macro": round(float(cur_score), 4),
            }
    return cur_fit, cur_sel, cur_t, selected


def edit_group_labels(recs, spec):
    base = group_labels(recs, "srcbin")
    conf = np.asarray([str(r.get("confidence", "") or "unknown") for r in recs], object)
    cat = np.asarray([str(r.get("category", "") or "unknown") for r in recs], object)
    if spec == "srcbin":
        return base
    if spec == "confidence":
        return conf
    if spec == "srcbin_conf":
        return np.asarray([f"{b}:{c}" for b, c in zip(base, conf)], object)
    if spec == "category":
        return cat
    if spec == "category_srcbin":
        return np.asarray([f"{c}:{b}" for c, b in zip(cat, base)], object)
    raise ValueError(f"unknown edit group spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--noargs_tmp", required=True)
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bootstrap_top_k", type=int, default=0)
    ap.add_argument("--dump_oof", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X = np.load(args.cache)["X"]
    recs_by = load_split(args.dataset)
    recs = recs_by["train"] + recs_by["val"] + recs_by["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {"_n": len(recs)}
    fold_by_idx = np.full(len(recs), -1, int)
    for fi, (_, te_idx) in enumerate(folds):
        fold_by_idx[np.asarray(te_idx, int)] = fi
    fold_meta = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ytr, yv = y_all[tr_idx], y_all[va_idx]
        ctr = c_all[tr_idx]
        fitg = fit_hgb(X[tr_idx], ytr, ctr, X[va_idx], yv, X[te_idx], quick=args.quick)
        p_nli_v, p_nli_t = fitg[1], fitg[2]

        bge = torch.load(f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt",
                         map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        put(oof, "bge_lr", te_idx, np.asarray(bge["val"]["y"], int), p_bge_v, p_bge_t)
        decision_heads_v = {}
        decision_heads_t = {}
        bge_thr = best_thr(yv, p_bge_v)
        decision_heads_v["bge_thr"] = (p_bge_v >= bge_thr).astype(int)
        decision_heads_t["bge_thr"] = (p_bge_t >= bge_thr).astype(int)

        score_variants = {}
        for alpha in (0.25, 0.50, 0.75):
            name = f"rankmix_nli{int(alpha * 100):02d}_hgb_bge"
            score_variants[name] = (
                rank_mix(p_nli_v, p_bge_v, alpha),
                rank_mix(p_nli_t, p_bge_t, alpha),
                alpha,
            )
        alpha_sel, pv_sel = select_alpha(yv, p_nli_v, p_bge_v)
        score_variants["rankmix_nli_valselect_hgb_bge"] = (
            pv_sel,
            rank_mix(p_nli_t, p_bge_t, alpha_sel),
            alpha_sel,
        )
        for name, (pv, pt, _) in score_variants.items():
            put(oof, name, te_idx, yv, pv, pt)
            thr = best_thr(yv, pv)
            decision_heads_v[f"{name}_thr"] = (pv >= thr).astype(int)
            decision_heads_t[f"{name}_thr"] = (pt >= thr).astype(int)

        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        missing = [p for p in no_paths + ar_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"fold {fi} missing {missing}")
        no_b = load_bundles(no_paths)
        ar_b = load_bundles(ar_paths)
        _, p_no_v, yv2, _, _ = build_split_features(no_b, "val")
        _, p_no_t, _, _, _ = build_split_features(no_b, "test")
        if not np.all(yv == yv2):
            raise ValueError(f"fold {fi} NLI/CLAIMARC val labels differ")
        p_rank_no_v = 0.5 * rank01(p_no_v) + 0.5 * rank01(p_bge_v)
        p_rank_no_t = 0.5 * rank01(p_no_t) + 0.5 * rank01(p_bge_t)
        _, p_score_v, p_score_t, rv, rt = fit_conf_advantage_switch(
            yv, p_bge_v, p_bge_t, p_rank_no_v, p_rank_no_t, "macro")
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        gv = group_labels(val_recs, "srcbin")
        gt = group_labels(test_recs, "srcbin")
        edit_specs = {
            "srcbin": (gv, gt, 25, 5, 1e-8),
            "confidence": (edit_group_labels(val_recs, "confidence"),
                           edit_group_labels(test_recs, "confidence"), 35, 5, 0.001),
            "srcbin_conf": (edit_group_labels(val_recs, "srcbin_conf"),
                            edit_group_labels(test_recs, "srcbin_conf"), 35, 5, 0.001),
            "category": (edit_group_labels(val_recs, "category"),
                         edit_group_labels(test_recs, "category"), 45, 5, 0.001),
            "category_srcbin": (edit_group_labels(val_recs, "category_srcbin"),
                                edit_group_labels(test_recs, "category_srcbin"), 60, 5, 0.001),
        }
        global_thr, by_group, val_macro = greedy_group_thresholds(yv, p_score_v, gv)
        fm = {
            "fold": fi,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "n_test": len(te_idx),
            "nli_hgb": {"thr": round(float(fitg[3]), 3), "lr": fitg[4],
                        "l2": fitg[5], "leaves": fitg[6]},
            "switch_rate_val": round(float(rv), 4),
            "switch_rate_test": round(float(rt), 4),
            "raw_global_thr": round(float(global_thr), 3),
            "raw_group_thr": {k: round(float(v), 3) for k, v in by_group.items()},
            "raw_val_macro": round(float(val_macro), 4),
            "rankmix_alpha": {k: round(float(v[2]), 3) for k, v in score_variants.items()},
        }
        for name, max_drop, min_thr in (
            ("guard_clip_drop12_min35", 0.12, 0.35),
            ("guard_clip_drop20_min30", 0.20, 0.30),
        ):
            guarded, rejected, lower = guard_thresholds(global_thr, by_group, "clip",
                                                        max_drop, min_thr)
            base_v = apply_group_thresholds(p_score_v, gv, global_thr, guarded)
            yhat = apply_group_thresholds(p_score_t, gt, global_thr, guarded)
            decision_heads_v[f"{name}_switchmacro_srcbin"] = base_v
            decision_heads_t[f"{name}_switchmacro_srcbin"] = yhat
            for score_name, (_, p_rank_t, _) in score_variants.items():
                method = f"dual_score={score_name}__decision={name}_switchmacro_srcbin"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = p_rank_t
                oof[method]["yhat"][te_idx] = yhat
            if name == "guard_clip_drop12_min35":
                for score_name, (p_rank_v, p_rank_t, _) in score_variants.items():
                    edited, edit_meta = select_score_edit(
                        yv, base_v, p_rank_v, yhat, p_rank_t)
                    method = f"dual_score={score_name}__decision={name}_scoreedit_srcbin"
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = p_rank_t
                    oof[method]["yhat"][te_idx] = edited
                    fm[method] = {"base": name, "selected": edit_meta}
                    for spec, (edit_gv, edit_gt, min_n, min_pos, min_gain) in edit_specs.items():
                        group_edited_v, group_edited, group_edit_meta = select_group_score_edit(
                            yv, base_v, p_rank_v, edit_gv, yhat, p_rank_t, edit_gt,
                            min_n=min_n, min_pos=min_pos, min_gain=min_gain)
                        head_name = f"{score_name}_groupscoreedit_{spec}"
                        decision_heads_v[head_name] = group_edited_v
                        decision_heads_t[head_name] = group_edited
                        method = f"dual_score={score_name}__decision={name}_groupscoreedit_{spec}"
                        oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                                "yhat": np.full(oof["_n"], np.nan)})
                        oof[method]["p"][te_idx] = p_rank_t
                        oof[method]["yhat"][te_idx] = group_edited
                        fm[method] = {
                            "base": name,
                            "group_spec": spec,
                            "min_n": min_n,
                            "min_pos": min_pos,
                            "min_gain": min_gain,
                            "selected": group_edit_meta,
                        }
            fm[f"{name}_switchmacro_srcbin"] = {
                "lower": round(float(lower), 3),
                "group_thr": {k: round(float(v), 3) for k, v in guarded.items()},
                "rejected": {k: round(float(v), 3) for k, v in rejected.items()},
            }
        for score_name, (p_rank_v, p_rank_t, _) in score_variants.items():
            bge_edited, bge_edit_meta = select_score_edit(
                yv,
                decision_heads_v["bge_thr"],
                p_rank_v,
                decision_heads_t["bge_thr"],
                p_rank_t,
            )
            method = f"dual_score={score_name}__decision=bgeedit_global"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = p_rank_t
            oof[method]["yhat"][te_idx] = bge_edited
            decision_heads_v[f"{score_name}_bgeedit_global"] = select_score_edit(
                yv,
                decision_heads_v["bge_thr"],
                p_rank_v,
                decision_heads_v["bge_thr"],
                p_rank_v,
            )[0]
            decision_heads_t[f"{score_name}_bgeedit_global"] = bge_edited
            fm[method] = {"base": "bge_thr", "selected": bge_edit_meta}
            for spec, (edit_gv, edit_gt, min_n, min_pos, min_gain) in edit_specs.items():
                bge_group_v, bge_group_t, bge_group_meta = select_group_score_edit(
                    yv,
                    decision_heads_v["bge_thr"],
                    p_rank_v,
                    edit_gv,
                    decision_heads_t["bge_thr"],
                    p_rank_t,
                    edit_gt,
                    min_n=min_n,
                    min_pos=min_pos,
                    min_gain=min_gain,
                )
                head_name = f"{score_name}_bgeedit_{spec}"
                decision_heads_v[head_name] = bge_group_v
                decision_heads_t[head_name] = bge_group_t
                method = f"dual_score={score_name}__decision=bgeedit_{spec}"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = p_rank_t
                oof[method]["yhat"][te_idx] = bge_group_t
                fm[method] = {
                    "base": "bge_thr",
                    "group_spec": spec,
                    "min_n": min_n,
                    "min_pos": min_pos,
                    "min_gain": min_gain,
                    "selected": bge_group_meta,
                }
        score_group_specs = {
            "srcbin": (gv, gt),
            "confidence": edit_specs["confidence"][:2],
            "srcbin_conf": edit_specs["srcbin_conf"][:2],
        }
        scorefallback_scores = {}
        for score_name, (pv_rank, pt_rank, _) in score_variants.items():
            for spec, (score_gv, score_gt) in score_group_specs.items():
                score_global, score_by_group, score_val_macro = greedy_group_thresholds(
                    yv, pv_rank, score_gv)
                raw_v = apply_group_thresholds(pv_rank, score_gv, score_global, score_by_group)
                raw_t = apply_group_thresholds(pt_rank, score_gt, score_global, score_by_group)
                head_name = f"{score_name}_scoregroup_{spec}"
                decision_heads_v[head_name] = raw_v
                decision_heads_t[head_name] = raw_t
                method = f"dual_score={score_name}__decision=scoregroup_{spec}"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = pt_rank
                oof[method]["yhat"][te_idx] = raw_t
                fm[method] = {
                    "raw_global_thr": round(float(score_global), 3),
                    "raw_group_thr": {k: round(float(v), 3)
                                      for k, v in score_by_group.items()},
                    "raw_val_macro": round(float(score_val_macro), 4),
                }
                for guard_name, max_drop, min_thr in (
                    ("scoreguard_clip_drop12_min35", 0.12, 0.35),
                    ("scoreguard_clip_drop20_min30", 0.20, 0.30),
                ):
                    guarded, rejected, lower = guard_thresholds(
                        score_global, score_by_group, "clip", max_drop, min_thr)
                    yhat_v = apply_group_thresholds(pv_rank, score_gv, score_global, guarded)
                    yhat = apply_group_thresholds(pt_rank, score_gt, score_global, guarded)
                    head_name = f"{score_name}_{guard_name}_{spec}"
                    decision_heads_v[head_name] = yhat_v
                    decision_heads_t[head_name] = yhat
                    method = f"dual_score={score_name}__decision={guard_name}_{spec}"
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = pt_rank
                    oof[method]["yhat"][te_idx] = yhat
                    fm[method] = {
                        "raw_global_thr": round(float(score_global), 3),
                        "raw_group_thr": {k: round(float(v), 3)
                                          for k, v in score_by_group.items()},
                        "raw_val_macro": round(float(score_val_macro), 4),
                        "lower": round(float(lower), 3),
                        "group_thr": {k: round(float(v), 3) for k, v in guarded.items()},
                        "rejected": {k: round(float(v), 3) for k, v in rejected.items()},
                    }
        sourceveto_names = []
        veto_group_specs = {
            "srcbin": (gv, gt, 25, 5, 5, 0.001),
            "srcbin_conf": (edit_group_labels(val_recs, "srcbin_conf"),
                            edit_group_labels(test_recs, "srcbin_conf"), 30, 5, 5, 0.001),
        }
        signal_variants = {
            "rank": None,
            "nli": (rank01(p_nli_v), rank01(p_nli_t)),
            "joint": (np.minimum(rank01(p_nli_v), rank01(p_bge_v)),
                      np.minimum(rank01(p_nli_t), rank01(p_bge_t))),
        }
        for score_name, (pv_rank, pt_rank, _) in score_variants.items():
            signal_variants["rank"] = (pv_rank, pt_rank)
            for base_head in (
                f"{score_name}_scoreguard_clip_drop12_min35_confidence",
                f"{score_name}_scoreguard_clip_drop20_min30_confidence",
                f"{score_name}_scoreguard_clip_drop12_min35_srcbin_conf",
                f"{score_name}_scoreguard_clip_drop20_min30_srcbin_conf",
            ):
                if base_head not in decision_heads_v:
                    continue
                for spec, (veto_gv, veto_gt, min_n, min_pos, min_neg, min_gain) in veto_group_specs.items():
                    for sig_name, (sig_v, sig_t) in signal_variants.items():
                        veto_v, veto_t, veto_meta = select_source_veto(
                            yv,
                            decision_heads_v[base_head],
                            sig_v,
                            veto_gv,
                            decision_heads_t[base_head],
                            sig_t,
                            veto_gt,
                            min_n=min_n,
                            min_pos=min_pos,
                            min_neg=min_neg,
                            min_gain=min_gain,
                        )
                        head_name = f"{base_head}_sourceveto_{sig_name}_{spec}"
                        decision_heads_v[head_name] = veto_v
                        decision_heads_t[head_name] = veto_t
                        sourceveto_names.append(head_name)
                        method = f"dual_score={score_name}__decision={head_name}"
                        oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                                "yhat": np.full(oof["_n"], np.nan)})
                        oof[method]["p"][te_idx] = pt_rank
                        oof[method]["yhat"][te_idx] = veto_t
                        fm[method] = {
                            "base": base_head,
                            "group_spec": spec,
                            "signal": sig_name,
                            "min_n": min_n,
                            "min_pos": min_pos,
                            "min_neg": min_neg,
                            "min_gain": min_gain,
                            "selected": veto_meta,
                        }
        bgefallback_names = []
        bgeadvfallback_names = []
        bgerateguard_names = []
        nlievidenceveto_names = []
        for score_name, (_, pt_rank, _) in score_variants.items():
            for base_head in (
                f"{score_name}_scoreguard_clip_drop12_min35_confidence",
                f"{score_name}_scoreguard_clip_drop20_min30_confidence",
                f"{score_name}_scoreguard_clip_drop12_min35_srcbin_conf",
                f"{score_name}_scoreguard_clip_drop20_min30_srcbin_conf",
                f"{score_name}_scoregroup_confidence",
                f"{score_name}_scoregroup_srcbin_conf",
            ):
                if base_head not in decision_heads_v:
                    continue
                for mode, groups_v_fb, groups_t_fb in (
                    ("src0", gv, gt),
                    ("src2_3", gv, gt),
                    ("src0_src2_3", gv, gt),
                    ("src0_src2_3_lowabs",
                     edit_group_labels(val_recs, "srcbin_conf"),
                     edit_group_labels(test_recs, "srcbin_conf")),
                ):
                    fb_v, fb_t = apply_bge_fallback(
                        decision_heads_v[base_head],
                        decision_heads_v["bge_thr"],
                        groups_v_fb,
                        decision_heads_t[base_head],
                        decision_heads_t["bge_thr"],
                        groups_t_fb,
                        mode,
                    )
                    head_name = f"{base_head}_bgefallback_{mode}"
                    decision_heads_v[head_name] = fb_v
                    decision_heads_t[head_name] = fb_t
                    bgefallback_names.append(head_name)
                    method = f"dual_score={score_name}__decision={head_name}"
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = pt_rank
                    oof[method]["yhat"][te_idx] = fb_t
                    fm[method] = {
                        "base": base_head,
                        "fallback": "bge_thr",
                        "mode": mode,
                    }
                if score_name not in ("rankmix_nli25_hgb_bge",
                                      "rankmix_nli50_hgb_bge"):
                    continue
                if not (
                    base_head.endswith("scoreguard_clip_drop20_min30_confidence")
                    or base_head.endswith("scoreguard_clip_drop20_min30_srcbin_conf")
                ):
                    continue
                adv_specs = (
                    ("srcbin_src0_src2_3", gv, gt,
                     lambda x: x in {"src0", "src2_3"}, 35),
                    ("srcbinconf_src0_src2_3_lowabs",
                     edit_group_labels(val_recs, "srcbin_conf"),
                     edit_group_labels(test_recs, "srcbin_conf"),
                     lambda x: x.startswith("src0:")
                     or x.startswith("src2_3:low")
                     or x.startswith("src2_3:absent"),
                     30),
                )
                for adv_spec, groups_v_adv, groups_t_adv, target_fn, min_n in adv_specs:
                    for min_gain in (0.00, 0.01, 0.02):
                        adv_v, adv_t, adv_meta = select_bge_advantage_fallback(
                            yv,
                            decision_heads_v[base_head],
                            decision_heads_v["bge_thr"],
                            groups_v_adv,
                            decision_heads_t[base_head],
                            decision_heads_t["bge_thr"],
                            groups_t_adv,
                            target_fn,
                            min_n=min_n,
                            min_pos=5,
                            min_neg=5,
                            min_gain=min_gain,
                            rate_slack=0.08,
                        )
                        head_name = (
                            f"{base_head}_bgeadvfallback_{adv_spec}"
                            f"_gain{int(round(min_gain * 100)):02d}"
                        )
                        decision_heads_v[head_name] = adv_v
                        decision_heads_t[head_name] = adv_t
                        bgeadvfallback_names.append(head_name)
                        method = f"dual_score={score_name}__decision={head_name}"
                        oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                                "yhat": np.full(oof["_n"], np.nan)})
                        oof[method]["p"][te_idx] = pt_rank
                        oof[method]["yhat"][te_idx] = adv_t
                        fm[method] = {
                            "base": base_head,
                            "fallback": "bge_thr_if_group_advantage",
                            "adv_spec": adv_spec,
                            "min_gain": round(float(min_gain), 3),
                            "selected": adv_meta,
                        }
                if score_name not in ("rankmix_nli25_hgb_bge",
                                      "rankmix_nli50_hgb_bge"):
                    continue
                if not (
                    base_head.endswith("scoreguard_clip_drop20_min30_confidence")
                    or base_head.endswith("scoreguard_clip_drop20_min30_srcbin_conf")
                    or base_head.endswith("scoregroup_confidence")
                    or base_head.endswith("scoregroup_srcbin_conf")
                ):
                    continue
                rate_specs = (
                    ("srcbin_src0_src2_3", gv, gt,
                     lambda x: x in {"src0", "src2_3"}, 30),
                    ("srcbinconf_src0_src2_3_lowabs",
                     edit_group_labels(val_recs, "srcbin_conf"),
                     edit_group_labels(test_recs, "srcbin_conf"),
                     lambda x: x.startswith("src0:")
                     or x.startswith("src2_3:low")
                     or x.startswith("src2_3:absent"),
                     25),
                )
                for rate_spec, groups_v_rate, groups_t_rate, target_fn, min_n in rate_specs:
                    for over_margin in (0.03, 0.05, 0.08):
                        rate_v, rate_t, rate_meta = select_bge_rate_guard_fallback(
                            yv,
                            decision_heads_v[base_head],
                            decision_heads_v["bge_thr"],
                            groups_v_rate,
                            decision_heads_t[base_head],
                            decision_heads_t["bge_thr"],
                            groups_t_rate,
                            target_fn,
                            min_n=min_n,
                            min_pos=5,
                            min_neg=5,
                            over_margin=over_margin,
                            min_rate_drop=0.02,
                            gap_slack=0.03,
                        )
                        head_name = (
                            f"{base_head}_bgerateguard_{rate_spec}"
                            f"_over{int(round(over_margin * 100)):02d}"
                        )
                        decision_heads_v[head_name] = rate_v
                        decision_heads_t[head_name] = rate_t
                        bgerateguard_names.append(head_name)
                        method = f"dual_score={score_name}__decision={head_name}"
                        oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                                "yhat": np.full(oof["_n"], np.nan)})
                        oof[method]["p"][te_idx] = pt_rank
                        oof[method]["yhat"][te_idx] = rate_t
                        fm[method] = {
                            "base": base_head,
                            "fallback": "bge_thr_if_group_overpredicts",
                            "rate_spec": rate_spec,
                            "over_margin": round(float(over_margin), 3),
                            "selected": rate_meta,
                        }
        for score_name, (pv_rank, pt_rank, _) in score_variants.items():
            if score_name not in ("rankmix_nli25_hgb_bge",
                                  "rankmix_nli50_hgb_bge",
                                  "rankmix_nli_valselect_hgb_bge"):
                continue
            decision_candidates = [
                f"{score_name}_thr",
                f"{score_name}_scoreguard_clip_drop12_min35_confidence",
                f"{score_name}_scoreguard_clip_drop20_min30_confidence",
                f"{score_name}_scoreguard_clip_drop20_min30_srcbin_conf",
                f"{score_name}_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
                f"{score_name}_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3_lowabs",
            ]
            decision_candidates.extend(
                h for h in bgerateguard_names if h.startswith(score_name)
            )
            for mode, groups_v_sf, groups_t_sf in (
                ("src0", gv, gt),
                ("src2_3", gv, gt),
                ("src0_src2_3", gv, gt),
                ("src0_src2_3_lowabs",
                 edit_group_labels(val_recs, "srcbin_conf"),
                 edit_group_labels(test_recs, "srcbin_conf")),
            ):
                for bge_weight in (0.25, 0.50, 0.75, 1.00):
                    sf_v, sf_t = apply_score_fallback(
                        pv_rank,
                        p_bge_v,
                        groups_v_sf,
                        pt_rank,
                        p_bge_t,
                        groups_t_sf,
                        mode,
                        bge_weight,
                    )
                    sf_name = (
                        f"{score_name}_scorefallback_bge"
                        f"{int(round(bge_weight * 100)):03d}_{mode}"
                    )
                    scorefallback_scores[sf_name] = (sf_v, sf_t)
                    sf_thr = put(oof, sf_name, te_idx, yv, sf_v, sf_t)
                    decision_heads_v[f"{sf_name}_thr"] = (sf_v >= sf_thr).astype(int)
                    decision_heads_t[f"{sf_name}_thr"] = (sf_t >= sf_thr).astype(int)
                    fm[sf_name] = {
                        "base_score": score_name,
                        "fallback": "bge_rank",
                        "mode": mode,
                        "bge_weight": round(float(bge_weight), 2),
                        "thr": round(float(sf_thr), 4),
                    }
                    for dec_name in decision_candidates:
                        if dec_name not in decision_heads_t:
                            continue
                        method = f"dual_score={sf_name}__decision={dec_name}"
                        oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                                "yhat": np.full(oof["_n"], np.nan)})
                        oof[method]["p"][te_idx] = sf_t
                        oof[method]["yhat"][te_idx] = decision_heads_t[dec_name]
                        fm[method] = {
                            "score_head": sf_name,
                            "decision_head": dec_name,
                        }

        def score_pair(score_key):
            if score_key in scorefallback_scores:
                return scorefallback_scores[score_key]
            if score_key in score_variants:
                pv_key, pt_key, _ = score_variants[score_key]
                return pv_key, pt_key
            return None

        predef_lowabs_specs = (
            (
                "predef_lowabs_r25_scorefallback_thr",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs_thr",
            ),
            (
                "predef_lowabs_r25_scorefallback_srcconf_bgefallback",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs",
            ),
            (
                "predef_lowabs_r25_scorefallback_conf_bgefallback",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
            ),
            (
                "predef_lowabs_r25_srcconf_bgefallback",
                "rankmix_nli25_hgb_bge",
                "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs",
            ),
            (
                "predef_lowabs_r25_conf_bgefallback",
                "rankmix_nli25_hgb_bge",
                "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
            ),
            (
                "predef_lowabs_r50_scorefallback_srcconf_bgefallback",
                "rankmix_nli50_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli50_hgb_bge_scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs",
            ),
            (
                "predef_lowabs_r50_scorefallback_conf_bgefallback",
                "rankmix_nli50_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli50_hgb_bge_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
            ),
        )
        for proto_name, score_key, decision_key in predef_lowabs_specs:
            pair = score_pair(score_key)
            if pair is None or decision_key not in decision_heads_t:
                continue
            pv_proto, pt_proto = pair
            method = f"predef_protocol={proto_name}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = pt_proto
            oof[method]["yhat"][te_idx] = decision_heads_t[decision_key]
            fm[method] = {
                "score_head": score_key,
                "decision_head": decision_key,
                "protocol": "source_confidence_lowabs_fallback",
            }
        signals_v = nli_evidence_signals(X[va_idx])
        signals_t = nli_evidence_signals(X[te_idx])
        masks_v = nli_veto_target_masks(val_recs, X[va_idx])
        masks_t = nli_veto_target_masks(test_recs, X[te_idx])
        evidence_veto_bases = []
        for score_name, (pv_rank, pt_rank, _) in score_variants.items():
            if score_name not in ("rankmix_nli25_hgb_bge",
                                  "rankmix_nli50_hgb_bge"):
                continue
            for base_head in (
                f"{score_name}_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
                f"{score_name}_scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3",
                f"{score_name}_scoreguard_clip_drop12_min35_srcbin_conf_bgefallback_src0_src2_3",
            ):
                if base_head in decision_heads_v:
                    evidence_veto_bases.append((score_name, pv_rank, pt_rank, base_head))
        for sf_name in (
            "rankmix_nli50_hgb_bge_scorefallback_bge100_src0_src2_3",
            "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
        ):
            if sf_name in scorefallback_scores and f"{sf_name}_thr" in decision_heads_v:
                sf_v, sf_t = scorefallback_scores[sf_name]
                evidence_veto_bases.append((sf_name, sf_v, sf_t, f"{sf_name}_thr"))
        seen_evidence_bases = set()
        for score_head, score_v, score_t, base_head in evidence_veto_bases:
            key = (score_head, base_head)
            if key in seen_evidence_bases:
                continue
            seen_evidence_bases.add(key)
            veto_v, veto_t, veto_meta = select_nli_evidence_veto(
                yv,
                decision_heads_v[base_head],
                decision_heads_v["bge_thr"],
                signals_v,
                masks_v,
                decision_heads_t[base_head],
                decision_heads_t["bge_thr"],
                signals_t,
                masks_t,
                min_flip=6,
                max_flip_frac=0.30,
                min_gain=0.001,
            )
            head_name = f"{base_head}_nlievidenceveto"
            decision_heads_v[head_name] = veto_v
            decision_heads_t[head_name] = veto_t
            nlievidenceveto_names.append(head_name)
            method = f"dual_score={score_head}__decision={head_name}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = score_t
            oof[method]["yhat"][te_idx] = veto_t
            fm[method] = {
                "score_head": score_head,
                "base": base_head,
                "selected": veto_meta,
            }
        mix_candidates = [
            "bge_thr",
            "rankmix_nli25_hgb_bge_thr",
            "rankmix_nli50_hgb_bge_thr",
            "rankmix_nli_valselect_hgb_bge_thr",
            "guard_clip_drop12_min35_switchmacro_srcbin",
            "guard_clip_drop20_min30_switchmacro_srcbin",
            "rankmix_nli25_hgb_bge_groupscoreedit_srcbin",
            "rankmix_nli25_hgb_bge_groupscoreedit_confidence",
            "rankmix_nli25_hgb_bge_groupscoreedit_srcbin_conf",
            "rankmix_nli50_hgb_bge_groupscoreedit_srcbin",
            "rankmix_nli50_hgb_bge_groupscoreedit_confidence",
            "rankmix_nli50_hgb_bge_groupscoreedit_srcbin_conf",
            "rankmix_nli_valselect_hgb_bge_groupscoreedit_srcbin",
            "rankmix_nli_valselect_hgb_bge_groupscoreedit_confidence",
            "rankmix_nli_valselect_hgb_bge_groupscoreedit_srcbin_conf",
            "rankmix_nli25_hgb_bge_bgeedit_srcbin",
            "rankmix_nli25_hgb_bge_bgeedit_confidence",
            "rankmix_nli25_hgb_bge_bgeedit_srcbin_conf",
            "rankmix_nli50_hgb_bge_bgeedit_srcbin",
            "rankmix_nli50_hgb_bge_bgeedit_confidence",
            "rankmix_nli50_hgb_bge_bgeedit_srcbin_conf",
            "rankmix_nli_valselect_hgb_bge_bgeedit_srcbin",
            "rankmix_nli_valselect_hgb_bge_bgeedit_confidence",
            "rankmix_nli_valselect_hgb_bge_bgeedit_srcbin_conf",
            "rankmix_nli25_hgb_bge_scoregroup_confidence",
            "rankmix_nli25_hgb_bge_scoregroup_srcbin_conf",
            "rankmix_nli50_hgb_bge_scoregroup_confidence",
            "rankmix_nli50_hgb_bge_scoregroup_srcbin_conf",
            "rankmix_nli_valselect_hgb_bge_scoregroup_confidence",
            "rankmix_nli_valselect_hgb_bge_scoregroup_srcbin_conf",
        ]
        mix_candidates.extend(sourceveto_names)
        mix_candidates.extend(bgefallback_names)
        mix_candidates.extend(bgeadvfallback_names)
        mix_candidates.extend(bgerateguard_names)
        mix_candidates.extend(nlievidenceveto_names)
        base_mix = "rankmix_nli_valselect_hgb_bge_groupscoreedit_srcbin"
        if base_mix in decision_heads_v:
            for spec, mix_gv, mix_gt, min_n, min_pos, min_gain in (
                ("srcbin", gv, gt, 35, 5, 0.001),
                ("confidence", edit_group_labels(val_recs, "confidence"),
                 edit_group_labels(test_recs, "confidence"), 40, 5, 0.001),
                ("srcbin_conf", edit_group_labels(val_recs, "srcbin_conf"),
                 edit_group_labels(test_recs, "srcbin_conf"), 40, 5, 0.001),
            ):
                mixed_v, mixed_t, mix_meta = select_group_head_mix(
                    yv, mix_gv, mix_gt, decision_heads_v, decision_heads_t,
                    base_mix, mix_candidates, min_n=min_n, min_pos=min_pos,
                    min_gain=min_gain)
                mix_head_name = f"headmix_{spec}_from_valselect_srcbin"
                decision_heads_v[mix_head_name] = mixed_v
                decision_heads_t[mix_head_name] = mixed_t
                fm[mix_head_name] = {
                    "group_spec": spec,
                    "min_n": min_n,
                    "min_pos": min_pos,
                    "min_gain": min_gain,
                    "selected": mix_meta,
                }
                for score_name, (_, p_rank_t, _) in score_variants.items():
                    if score_name not in ("rankmix_nli25_hgb_bge",
                                          "rankmix_nli50_hgb_bge",
                                          "rankmix_nli_valselect_hgb_bge"):
                        continue
                    method = f"dual_score={score_name}__decision={mix_head_name}"
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = p_rank_t
                    oof[method]["yhat"][te_idx] = mixed_t
        router_candidates = {}

        def add_router_candidate(name, pv, pt, yhat_v, yhat_t):
            router_candidates[name] = (
                np.asarray(pv, float),
                np.asarray(pt, float),
                np.asarray(yhat_v, int),
                np.asarray(yhat_t, int),
            )

        p25_v, p25_t, _ = score_variants["rankmix_nli25_hgb_bge"]
        add_router_candidate(
            "rankmix25_thr",
            p25_v,
            p25_t,
            decision_heads_v["rankmix_nli25_hgb_bge_thr"],
            decision_heads_t["rankmix_nli25_hgb_bge_thr"],
        )
        sf25 = "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs"
        if sf25 in scorefallback_scores and f"{sf25}_thr" in decision_heads_v:
            sf_v, sf_t = scorefallback_scores[sf25]
            add_router_candidate(
                "scorefallback25_lowabs_thr",
                sf_v,
                sf_t,
                decision_heads_v[f"{sf25}_thr"],
                decision_heads_t[f"{sf25}_thr"],
            )
            add_router_candidate(
                "scorefallback25_lowabs_rankmix25_decision",
                sf_v,
                sf_t,
                decision_heads_v["rankmix_nli25_hgb_bge_thr"],
                decision_heads_t["rankmix_nli25_hgb_bge_thr"],
            )
        bgfb = (
            "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence"
            "_bgefallback_src0_src2_3"
        )
        if bgfb in decision_heads_v:
            add_router_candidate(
                "rankmix25_bgefallback_src0_src2_3",
                p25_v,
                p25_t,
                decision_heads_v[bgfb],
                decision_heads_t[bgfb],
            )
        headmix = "headmix_confidence_from_valselect_srcbin"
        if headmix in decision_heads_v:
            add_router_candidate(
                "rankmix25_headmix_confidence",
                p25_v,
                p25_t,
                decision_heads_v[headmix],
                decision_heads_t[headmix],
            )
        for mode in ("balanced", "macro", "balanced_nohead", "macro_nohead"):
            if mode.endswith("_nohead"):
                mode_candidates = {
                    k: v for k, v in router_candidates.items()
                    if "headmix" not in k
                }
                utility_mode = mode.replace("_nohead", "")
            else:
                mode_candidates = router_candidates
                utility_mode = mode
            best = None
            for cand_name, (pv_c, pt_c, yh_v_c, yh_t_c) in mode_candidates.items():
                score, val_metrics = val_utility(yv, pv_c, yh_v_c, mode=utility_mode)
                if best is None or score > best[0] + 1e-12:
                    best = (score, cand_name, val_metrics, pt_c, yh_t_c)
            if best is None:
                continue
            method = f"compact_router_{mode}_valselect"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = best[3]
            oof[method]["yhat"][te_idx] = best[4]
            fm.setdefault("compact_router", {})[mode] = {
                "selected": best[1],
                "utility": round(float(best[0]), 4),
                "val": {k: round(float(v), 4) for k, v in best[2].items()},
                "candidates": sorted(mode_candidates.keys()),
            }

        fit_m, sel_m = split_router_validation(
            yv, seed=args.fold_seed * 1000 + fi)
        y_fit = yv[fit_m]
        y_sel = yv[sel_m]
        nested_heads_fit = {}
        nested_heads_sel = {}
        nested_heads_t = {}
        nested_candidates = {}

        def add_nested(name, pv_full, pt, yh_fit, yh_sel, yh_t):
            nested_heads_fit[name] = np.asarray(yh_fit, int)
            nested_heads_sel[name] = np.asarray(yh_sel, int)
            nested_heads_t[name] = np.asarray(yh_t, int)
            nested_candidates[name] = (
                np.asarray(pv_full, float)[sel_m],
                np.asarray(pt, float),
                np.asarray(yh_sel, int),
                np.asarray(yh_t, int),
            )

        g_src_v = gv
        g_src_fit = g_src_v[fit_m]
        g_src_sel = g_src_v[sel_m]
        g_conf_v = edit_group_labels(val_recs, "confidence")
        g_conf_t = edit_group_labels(test_recs, "confidence")
        g_conf_fit = g_conf_v[fit_m]
        g_conf_sel = g_conf_v[sel_m]
        g_srcconf_v = edit_group_labels(val_recs, "srcbin_conf")
        g_srcconf_t = edit_group_labels(test_recs, "srcbin_conf")
        g_srcconf_fit = g_srcconf_v[fit_m]
        g_srcconf_sel = g_srcconf_v[sel_m]

        bge_fit, bge_thr_nested = threshold_from_fit(
            y_fit, p_bge_v[fit_m], p_bge_v[fit_m])
        bge_sel, _ = threshold_from_fit(y_fit, p_bge_v[fit_m], p_bge_v[sel_m])
        bge_t, _ = threshold_from_fit(y_fit, p_bge_v[fit_m], p_bge_t)
        add_nested("nested_bge_thr", p_bge_v, p_bge_t, bge_fit, bge_sel, bge_t)
        fm.setdefault("compact_router_nested", {})["bge_thr"] = {
            "fit_n": int(fit_m.sum()),
            "select_n": int(sel_m.sum()),
            "thr": round(float(bge_thr_nested), 4),
        }

        for short, score_name in (
            ("rankmix25", "rankmix_nli25_hgb_bge"),
            ("rankmix50", "rankmix_nli50_hgb_bge"),
        ):
            pv_full, pt_rank, _ = score_variants[score_name]
            yh_fit, thr_nested = threshold_from_fit(
                y_fit, pv_full[fit_m], pv_full[fit_m])
            yh_sel, _ = threshold_from_fit(y_fit, pv_full[fit_m], pv_full[sel_m])
            yh_t, _ = threshold_from_fit(y_fit, pv_full[fit_m], pt_rank)
            add_nested(f"nested_{short}_thr", pv_full, pt_rank,
                       yh_fit, yh_sel, yh_t)
            fm["compact_router_nested"][f"{short}_thr"] = {
                "thr": round(float(thr_nested), 4),
            }
            for spec, gv_full, gt_full, g_fit, g_sel in (
                ("confidence", g_conf_v, g_conf_t, g_conf_fit, g_conf_sel),
                ("srcbin_conf", g_srcconf_v, g_srcconf_t,
                 g_srcconf_fit, g_srcconf_sel),
            ):
                yh_fit, sg_meta = scoreguard_from_fit(
                    y_fit, pv_full[fit_m], g_fit, pv_full[fit_m], g_fit,
                    max_drop=0.20, min_thr=0.30)
                yh_sel, _ = scoreguard_from_fit(
                    y_fit, pv_full[fit_m], g_fit, pv_full[sel_m], g_sel,
                    max_drop=0.20, min_thr=0.30)
                yh_t, _ = scoreguard_from_fit(
                    y_fit, pv_full[fit_m], g_fit, pt_rank, gt_full,
                    max_drop=0.20, min_thr=0.30)
                head = f"nested_{short}_scoreguard_{spec}"
                add_nested(head, pv_full, pt_rank, yh_fit, yh_sel, yh_t)
                fm["compact_router_nested"][head] = sg_meta

        p25_full, p25_t_nested, _ = score_variants["rankmix_nli25_hgb_bge"]
        for base_head, mode, gv_full, gt_full, label in (
            ("nested_rankmix25_scoreguard_confidence", "src0_src2_3",
             g_src_v, gt, "src0_src2_3"),
            ("nested_rankmix25_scoreguard_confidence", "src0_src2_3_lowabs",
             g_srcconf_v, g_srcconf_t, "src0_src2_3_lowabs"),
            ("nested_rankmix25_scoreguard_srcbin_conf", "src0_src2_3_lowabs",
             g_srcconf_v, g_srcconf_t, "srcbinconf_lowabs"),
        ):
            if base_head not in nested_heads_fit:
                continue
            fb_fit, fb_t = apply_bge_fallback(
                nested_heads_fit[base_head], bge_fit, gv_full[fit_m],
                nested_heads_t[base_head], bge_t, gt_full, mode)
            fb_sel, _ = apply_bge_fallback(
                nested_heads_sel[base_head], bge_sel, gv_full[sel_m],
                nested_heads_t[base_head], bge_t, gt_full, mode)
            head = f"{base_head}_bgefallback_{label}"
            add_nested(head, p25_full, p25_t_nested, fb_fit, fb_sel, fb_t)
            fm["compact_router_nested"][head] = {
                "base": base_head,
                "mode": mode,
            }

        for sf_name, mode, gv_full, gt_full, weight in (
            ("nested_scorefallback25_bge025_lowabs", "src0_src2_3_lowabs",
             g_srcconf_v, g_srcconf_t, 0.25),
            ("nested_scorefallback25_bge100_src0_src2_3", "src0_src2_3",
             g_src_v, gt, 1.00),
            ("nested_scorefallback25_bge050_src2_3", "src2_3",
             g_src_v, gt, 0.50),
        ):
            sf_v, sf_t = apply_score_fallback(
                p25_full, p_bge_v, gv_full, p25_t_nested, p_bge_t,
                gt_full, mode, weight)
            yh_fit, sf_thr = threshold_from_fit(y_fit, sf_v[fit_m], sf_v[fit_m])
            yh_sel, _ = threshold_from_fit(y_fit, sf_v[fit_m], sf_v[sel_m])
            yh_t, _ = threshold_from_fit(y_fit, sf_v[fit_m], sf_t)
            add_nested(f"{sf_name}_thr", sf_v, sf_t, yh_fit, yh_sel, yh_t)
            fm["compact_router_nested"][f"{sf_name}_thr"] = {
                "mode": mode,
                "bge_weight": round(float(weight), 2),
                "thr": round(float(sf_thr), 4),
            }
            if "nested_rankmix25_thr" in nested_heads_fit:
                add_nested(
                    f"{sf_name}_rankmix25_decision",
                    sf_v,
                    sf_t,
                    nested_heads_fit["nested_rankmix25_thr"],
                    nested_heads_sel["nested_rankmix25_thr"],
                    nested_heads_t["nested_rankmix25_thr"],
                )

        mix_base = (
            "nested_rankmix25_scoreguard_confidence_bgefallback_src0_src2_3"
            if "nested_rankmix25_scoreguard_confidence_bgefallback_src0_src2_3"
            in nested_heads_fit
            else "nested_rankmix25_scoreguard_confidence"
        )
        if mix_base in nested_heads_fit:
            mix_names = sorted(nested_heads_fit.keys())
            for spec, g_fit, g_sel, g_t, min_n in (
                ("confidence", g_conf_fit, g_conf_sel, g_conf_t, 20),
                ("srcbin", g_src_fit, g_src_sel, gt, 20),
            ):
                mix_fit, mix_sel, mix_t, mix_meta = select_group_head_mix_nested(
                    y_fit, g_fit, g_sel, g_t, nested_heads_fit,
                    nested_heads_sel, nested_heads_t, mix_base, mix_names,
                    min_n=min_n, min_pos=4, min_gain=0.001)
                head = f"nested_headmix_{spec}"
                add_nested(head, p25_full, p25_t_nested, mix_fit, mix_sel, mix_t)
                fm["compact_router_nested"][head] = {
                    "group_spec": spec,
                    "min_n": min_n,
                    "min_pos": 4,
                    "selected": mix_meta,
                }

        for mode in ("balanced", "macro", "balanced_nohead", "macro_nohead"):
            if mode.endswith("_nohead"):
                cand_pool = {
                    k: v for k, v in nested_candidates.items()
                    if "headmix" not in k
                }
                utility_mode = mode.replace("_nohead", "")
            else:
                cand_pool = nested_candidates
                utility_mode = mode
            best = None
            for cand_name, (pv_sel, pt_c, yh_sel, yh_t_c) in cand_pool.items():
                score, val_metrics = val_utility(
                    y_sel, pv_sel, yh_sel, mode=utility_mode)
                if best is None or score > best[0] + 1e-12:
                    best = (score, cand_name, val_metrics, pt_c, yh_t_c)
            if best is None:
                continue
            method = f"compact_router_nested_{mode}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = best[3]
            oof[method]["yhat"][te_idx] = best[4]
            fm["compact_router_nested"][mode] = {
                "selected": best[1],
                "utility": round(float(best[0]), 4),
                "select": {k: round(float(v), 4) for k, v in best[2].items()},
                "candidates": sorted(cand_pool.keys()),
            }
        fold_meta.append(fm)
        print(f"[fold {fi}] done", flush=True)

    for spec in ("srcbin", "confidence", "srcbin_conf", "category", "category_srcbin"):
        dec = (
            "dual_score=rankmix_nli_valselect_hgb_bge"
            f"__decision=guard_clip_drop12_min35_groupscoreedit_{spec}"
        )
        if dec not in oof:
            continue
        for score_name in ("rankmix_nli25_hgb_bge", "rankmix_nli50_hgb_bge"):
            src = (
                f"dual_score={score_name}"
                f"__decision=guard_clip_drop12_min35_groupscoreedit_{spec}"
            )
            if src not in oof:
                continue
            method = (
                f"dual_score={score_name}"
                f"__decision=guard_clip_drop12_min35_groupscoreedit_valselect_{spec}"
            )
            oof[method] = {
                "p": oof[src]["p"].copy(),
                "yhat": oof[dec]["yhat"].copy(),
                "score_head": src,
                "decision_head": dec,
            }

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]), reverse=True)
    print("=== NLI dual guarded candidates ===", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:80s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)

    sig = {}
    if args.n_boot > 0:
        boot_ranked = ranked[:args.bootstrap_top_k] if args.bootstrap_top_k > 0 else ranked
        for name in boot_ranked:
            if name == "bge_lr":
                continue
            ok = (~np.isnan(oof[name]["p"])) & (~np.isnan(oof["bge_lr"]["p"]))
            sig[f"{name}_vs_bge_lr"] = paired_bootstrap_dual(
                y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok],
                oof["bge_lr"]["p"][ok], oof["bge_lr"]["yhat"][ok],
                n_boot=args.n_boot)
            s = sig[f"{name}_vs_bge_lr"]
            r = rows[name]
            print(f"{name:80s} vs bge_lr AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
                  f"mF1={r['macro_f1']:.4f} "
                  f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                  f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                  f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                  flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows,
           "fold_meta": fold_meta, "significance": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    if args.dump_oof:
        sc = np.asarray([sum(int((r.get("evidence_count", {}) or {}).get(k, 0) or 0)
                            for k in ("params", "ocr", "vlm")) for r in recs], int)
        srcbin = np.asarray([
            "src0" if x <= 0 else ("src1" if x == 1 else ("src2_3" if x <= 3 else "src4p"))
            for x in sc
        ], object)
        dump = {
            "y": y_all,
            "c": c_all,
            "fold": fold_by_idx,
            "source_count": sc,
            "source_bin": srcbin,
            "category": np.asarray([str(r.get("category", "")) for r in recs], object),
            "confidence": np.asarray([str(r.get("confidence", "")) for r in recs], object),
            "pair_id": np.asarray([str(r.get("pair_id", "")) for r in recs], object),
        }
        for name, vals in oof.items():
            if name == "_n":
                continue
            key = name.replace("=", "_").replace("+", "_").replace(",", "_").replace("/", "_")
            dump[f"{key}__p"] = vals["p"]
            dump[f"{key}__yhat"] = vals["yhat"]
        Path(args.dump_oof).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_oof, **dump)
        print(f"[cv_nli_dual_guard] oof -> {args.dump_oof}", flush=True)
    print(f"[cv_nli_dual_guard] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
