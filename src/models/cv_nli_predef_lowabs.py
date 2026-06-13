"""Predefined NLI+BGE lowabs fallback protocol without RACL head dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.cv_dual_head_router import (
    apply_group_thresholds,
    greedy_group_thresholds,
    group_labels,
    paired_bootstrap_dual,
)
from models.cv_eval import make_folds, val_carve
from models.cv_guarded_group_router import guard_thresholds
from models.cv_nli_dual_guard import (
    apply_bge_fallback,
    apply_score_fallback,
    edit_group_labels,
    put,
    rank_mix,
    row,
    select_alpha,
    val_utility,
)
from models.cv_nli_evidence_head import fit_hgb
from models.data import load_split
from models.fusion_eval import best_thr


def rec_source_count(rec):
    ev = rec.get("evidence_count", {}) or {}
    return sum(int(ev.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))


def rec_source_bin(rec):
    sc = rec_source_count(rec)
    if sc <= 0:
        return "src0"
    if sc == 1:
        return "src1"
    if sc <= 3:
        return "src2_3"
    return "src4p"


def rec_nonempty(value):
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def rec_evidence_combo(rec):
    parts = []
    if rec_nonempty(rec.get("evidence_params")):
        parts.append("P")
    if rec_nonempty(rec.get("evidence_ocr")):
        parts.append("O")
    if rec_nonempty(rec.get("evidence_vlm")):
        parts.append("V")
    return "".join(parts) if parts else "none"


NLI_STATS = [
    "n", "mean", "max", "min", "std", "top2", "p25", "p75",
    "rate20", "rate35", "rate50",
]
NLI_VARS = ["contr", "entail", "neutral", "margin", "maxce", "uncert"]
NLI_TYPES = ["all", "param", "ocr", "vlm", "arg_sup", "arg_ref", "arg_gap"]


def nli_cache_feature(Xmat, typ, var, stat):
    idx = (
        NLI_TYPES.index(typ) * 66
        + NLI_VARS.index(var) * 11
        + NLI_STATS.index(stat)
    )
    return np.asarray(Xmat[:, idx], float)


def records_mask(recs, predicate):
    return np.asarray([bool(predicate(r)) for r in recs], bool)


def fp_guard_masks(recs):
    conf = np.asarray([str(r.get("confidence", "")) for r in recs], object)
    sc = np.asarray([rec_source_count(r) for r in recs], int)
    src = np.asarray([rec_source_bin(r) for r in recs], object)
    return {
        "confidence_medium": conf == "medium",
        "source_count_ge2": sc >= 2,
        "src2_3_medium": (src == "src2_3") & (conf == "medium"),
    }


def fp_guard_stats(y, cand_yhat, ref_yhat, recs, min_neg=10):
    y = np.asarray(y, int)
    cand = np.asarray(cand_yhat, int)
    ref = np.asarray(ref_yhat, int)
    worst = 0.0
    total_created = 0
    details = {}
    for name, mask in fp_guard_masks(recs).items():
        neg = np.asarray(mask, bool) & (y == 0)
        n_neg = int(neg.sum())
        if n_neg < int(min_neg):
            continue
        cand_fp = float(cand[neg].mean())
        ref_fp = float(ref[neg].mean())
        inflation = cand_fp - ref_fp
        created = int(((cand == 1) & (ref == 0) & neg).sum())
        fixed = int(((cand == 0) & (ref == 1) & neg).sum())
        worst = max(worst, inflation)
        total_created += created
        details[name] = {
            "n_neg": n_neg,
            "cand_fp_rate": round(cand_fp, 4),
            "ref_fp_rate": round(ref_fp, 4),
            "fp_inflation": round(float(inflation), 4),
            "created_fp_vs_ref": created,
            "fixed_fp_vs_ref": fixed,
        }
    return {
        "worst_fp_inflation": round(float(worst), 4),
        "created_fp_vs_ref": int(total_created),
        "details": details,
    }


def decision_metrics(y, yhat):
    y = np.asarray(y, int)
    yhat = np.asarray(yhat, int)
    pos = y == 1
    neg = y == 0
    pos_recall = float((yhat[pos] == 1).mean()) if int(pos.sum()) else 0.0
    neg_recall = float((yhat[neg] == 0).mean()) if int(neg.sum()) else 0.0
    pred_pos_rate = float(yhat.mean()) if len(yhat) else 0.0
    return {
        "pos_recall": round(pos_recall, 4),
        "neg_recall": round(neg_recall, 4),
        "pred_pos_rate": round(pred_pos_rate, 4),
    }


def rank_mean(*scores):
    ranks = [np.asarray(rank_mix(s, s, 1.0), float) for s in scores]
    return np.mean(np.vstack(ranks), axis=0)


def rank_weighted(*weighted_scores):
    ranks = []
    weights = []
    for score, weight in weighted_scores:
        ranks.append(np.asarray(rank_mix(score, score, 1.0), float))
        weights.append(float(weight))
    weight_arr = np.asarray(weights, float)
    return np.average(np.vstack(ranks), axis=0, weights=weight_arr)


def choose_binary_switch(yv, val_recs, bge_yhat_v, base, alt, fp_slack,
                         override_gain=None):
    base_name, base_pv, base_pt, base_yv, base_yt = base
    alt_name, alt_pv, alt_pt, alt_yv, alt_yt = alt
    base_score, base_metrics = val_utility(yv, base_pv, base_yv, mode="macro")
    alt_score, alt_metrics = val_utility(yv, alt_pv, alt_yv, mode="macro")
    base_vs_bge = fp_guard_stats(yv, base_yv, bge_yhat_v, val_recs)
    alt_vs_bge = fp_guard_stats(yv, alt_yv, bge_yhat_v, val_recs)
    alt_vs_base = fp_guard_stats(yv, alt_yv, base_yv, val_recs)
    gain = alt_score - base_score
    fp_ok = alt_vs_base["worst_fp_inflation"] <= float(fp_slack)
    gain_override = (
        override_gain is not None
        and gain >= float(override_gain)
    )
    use_alt = gain > 1e-12 and (fp_ok or gain_override)
    chosen = alt if use_alt else base
    return chosen, {
        "selected": chosen[0],
        "base": base_name,
        "alt": alt_name,
        "fp_slack": round(float(fp_slack), 4),
        "base_utility": round(float(base_score), 4),
        "alt_utility": round(float(alt_score), 4),
        "alt_minus_base_utility": round(float(gain), 4),
        "base_metrics": {k: round(float(v), 4) for k, v in base_metrics.items()},
        "alt_metrics": {k: round(float(v), 4) for k, v in alt_metrics.items()},
        "base_fp_vs_bge": base_vs_bge,
        "alt_fp_vs_bge": alt_vs_bge,
        "alt_fp_vs_base": alt_vs_base,
        "override_gain": None if override_gain is None else round(float(override_gain), 4),
        "rule": "select alt if val macro improves and protected decision does not add guarded-group FP beyond slack vs self-threshold; optional override permits large macro gains",
    }


def choose_protected_default_switch(
    yv,
    val_recs,
    bge_yhat_v,
    base,
    protected,
    fp_hard=0.08,
    macro_margin=0.0,
    recall_slack=0.03,
    protected_gain_rescue=0.008,
):
    base_name, base_pv, base_pt, base_yv, base_yt = base
    prot_name, prot_pv, prot_pt, prot_yv, prot_yt = protected
    base_score, base_metrics = val_utility(yv, base_pv, base_yv, mode="macro")
    prot_score, prot_metrics = val_utility(yv, prot_pv, prot_yv, mode="macro")
    base_decision = decision_metrics(yv, base_yv)
    prot_decision = decision_metrics(yv, prot_yv)
    prot_vs_base = fp_guard_stats(yv, prot_yv, base_yv, val_recs)
    base_vs_prot = fp_guard_stats(yv, base_yv, prot_yv, val_recs)
    prot_gain = prot_score - base_score
    base_gain = base_score - prot_score
    base_recall_ok = (
        base_decision["pos_recall"]
        >= prot_decision["pos_recall"] - float(recall_slack)
    )
    self_clearly_better = (
        base_gain >= float(macro_margin)
        and base_recall_ok
    )
    protected_fp_bad = (
        prot_vs_base["worst_fp_inflation"] > float(fp_hard)
        and prot_gain < float(protected_gain_rescue)
    )
    use_base = self_clearly_better or protected_fp_bad
    chosen = base if use_base else protected
    return chosen, {
        "selected": chosen[0],
        "base": base_name,
        "protected": prot_name,
        "fp_hard": round(float(fp_hard), 4),
        "macro_margin": round(float(macro_margin), 4),
        "recall_slack": round(float(recall_slack), 4),
        "protected_gain_rescue": round(float(protected_gain_rescue), 4),
        "base_utility": round(float(base_score), 4),
        "protected_utility": round(float(prot_score), 4),
        "protected_minus_base_utility": round(float(prot_gain), 4),
        "base_metrics": {k: round(float(v), 4) for k, v in base_metrics.items()},
        "protected_metrics": {k: round(float(v), 4) for k, v in prot_metrics.items()},
        "base_decision": base_decision,
        "protected_decision": prot_decision,
        "protected_fp_vs_base": prot_vs_base,
        "base_fp_vs_protected": base_vs_prot,
        "self_clearly_better": bool(self_clearly_better),
        "protected_fp_bad": bool(protected_fp_bad),
        "rule": "default protected; use self-threshold only if validation macro is better without positive-recall loss, or protected creates guarded FP without enough macro gain",
    }


def add_scoreguard_heads(
    yv,
    score_variants,
    p_bge_v,
    p_bge_t,
    bge_yhat_v,
    bge_yhat_t,
    val_recs,
    test_recs,
    oof,
    te_idx,
    fold_meta,
):
    decision_heads_v = {
        "bge_thr": bge_yhat_v,
    }
    decision_heads_t = {
        "bge_thr": bge_yhat_t,
    }
    scorefallback_scores = {}
    group_specs = {
        "srcbin": (group_labels(val_recs, "srcbin"),
                   group_labels(test_recs, "srcbin")),
        "confidence": (edit_group_labels(val_recs, "confidence"),
                       edit_group_labels(test_recs, "confidence")),
        "srcbin_conf": (edit_group_labels(val_recs, "srcbin_conf"),
                        edit_group_labels(test_recs, "srcbin_conf")),
    }
    for score_name, (pv_rank, pt_rank, _) in score_variants.items():
        thr = put(oof, score_name, te_idx, yv, pv_rank, pt_rank)
        decision_heads_v[f"{score_name}_thr"] = (pv_rank >= thr).astype(int)
        decision_heads_t[f"{score_name}_thr"] = (pt_rank >= thr).astype(int)
        fold_meta[score_name] = {"thr": round(float(thr), 4)}
        for spec, (gv, gt) in group_specs.items():
            score_global, score_by_group, score_val_macro = greedy_group_thresholds(
                yv, pv_rank, gv)
            raw_v = apply_group_thresholds(pv_rank, gv, score_global, score_by_group)
            raw_t = apply_group_thresholds(pt_rank, gt, score_global, score_by_group)
            head_name = f"{score_name}_scoregroup_{spec}"
            decision_heads_v[head_name] = raw_v
            decision_heads_t[head_name] = raw_t
            method = f"dual_score={score_name}__decision=scoregroup_{spec}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = pt_rank
            oof[method]["yhat"][te_idx] = raw_t
            fold_meta[method] = {
                "raw_global_thr": round(float(score_global), 4),
                "raw_group_thr": {str(k): round(float(v), 4)
                                  for k, v in score_by_group.items()},
                "raw_val_macro": round(float(score_val_macro), 4),
            }
            for guard_name, max_drop, min_thr in (
                ("scoreguard_clip_drop12_min35", 0.12, 0.35),
                ("scoreguard_clip_drop20_min30", 0.20, 0.30),
            ):
                guarded, rejected, lower = guard_thresholds(
                    score_global, score_by_group, "clip", max_drop, min_thr)
                yhat_v = apply_group_thresholds(pv_rank, gv, score_global, guarded)
                yhat_t = apply_group_thresholds(pt_rank, gt, score_global, guarded)
                head_name = f"{score_name}_{guard_name}_{spec}"
                decision_heads_v[head_name] = yhat_v
                decision_heads_t[head_name] = yhat_t
                method = f"dual_score={score_name}__decision={guard_name}_{spec}"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = pt_rank
                oof[method]["yhat"][te_idx] = yhat_t
                fold_meta[method] = {
                    "raw_global_thr": round(float(score_global), 4),
                    "raw_val_macro": round(float(score_val_macro), 4),
                    "lower": round(float(lower), 4),
                    "group_thr": {str(k): round(float(v), 4)
                                  for k, v in guarded.items()},
                    "rejected": {str(k): round(float(v), 4)
                                 for k, v in rejected.items()},
                }
    for score_name, (_, pt_rank, _) in score_variants.items():
        for base_head in (
            f"{score_name}_scoreguard_clip_drop20_min30_confidence",
            f"{score_name}_scoreguard_clip_drop20_min30_srcbin_conf",
            f"{score_name}_scoregroup_confidence",
            f"{score_name}_scoregroup_srcbin_conf",
        ):
            if base_head not in decision_heads_v:
                continue
            for mode, groups_v, groups_t in (
                ("src0", group_specs["srcbin"][0], group_specs["srcbin"][1]),
                ("src2_3", group_specs["srcbin"][0], group_specs["srcbin"][1]),
                ("src0_src2_3", group_specs["srcbin"][0], group_specs["srcbin"][1]),
                ("src0_src2_3_lowabs",
                 group_specs["srcbin_conf"][0], group_specs["srcbin_conf"][1]),
            ):
                fb_v, fb_t = apply_bge_fallback(
                    decision_heads_v[base_head],
                    decision_heads_v["bge_thr"],
                    groups_v,
                    decision_heads_t[base_head],
                    decision_heads_t["bge_thr"],
                    groups_t,
                    mode,
                )
                head_name = f"{base_head}_bgefallback_{mode}"
                decision_heads_v[head_name] = fb_v
                decision_heads_t[head_name] = fb_t
                method = f"dual_score={score_name}__decision={head_name}"
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = pt_rank
                oof[method]["yhat"][te_idx] = fb_t
                fold_meta[method] = {
                    "base": base_head,
                    "fallback": "bge_thr",
                    "mode": mode,
                }
    for score_name, (pv_rank, pt_rank, _) in score_variants.items():
        if score_name not in (
            "rankmix_nli25_hgb_bge",
            "rankmix_nli50_hgb_bge",
            "rankmix_nli_valselect_hgb_bge",
        ):
            continue
        for mode, groups_v, groups_t in (
            ("src0", group_specs["srcbin"][0], group_specs["srcbin"][1]),
            ("src2_3", group_specs["srcbin"][0], group_specs["srcbin"][1]),
            ("src0_src2_3", group_specs["srcbin"][0], group_specs["srcbin"][1]),
            ("src0_src2_3_lowabs",
             group_specs["srcbin_conf"][0], group_specs["srcbin_conf"][1]),
        ):
            for bge_weight in (0.25, 0.50, 0.75, 1.00):
                sf_v, sf_t = apply_score_fallback(
                    pv_rank,
                    p_bge_v,
                    groups_v,
                    pt_rank,
                    p_bge_t,
                    groups_t,
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
                fold_meta[sf_name] = {
                    "base_score": score_name,
                    "fallback": "bge_rank",
                    "mode": mode,
                    "bge_weight": round(float(bge_weight), 2),
                    "thr": round(float(sf_thr), 4),
                }
    return decision_heads_v, decision_heads_t, scorefallback_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--cm_tmp", default="")
    ap.add_argument("--cm_seed", type=int, default=0)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_boot", type=int, default=0)
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
        fitg = fit_hgb(X[tr_idx], ytr, ctr, X[va_idx], yv, X[te_idx],
                       quick=args.quick)
        p_nli_v, p_nli_t = fitg[1], fitg[2]

        bge = torch.load(f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt",
                         map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        bge_thr = best_thr(yv, p_bge_v)
        bge_v = (p_bge_v >= bge_thr).astype(int)
        bge_t = (p_bge_t >= bge_thr).astype(int)
        put(oof, "bge_lr", te_idx, yv, p_bge_v, p_bge_t)

        p_cm_v = p_cm_t = None
        if args.cm_tmp:
            cm = torch.load(
                f"{args.cm_tmp}/cv_cm_f{fi}_s{args.cm_seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (not np.array_equal(np.asarray(cm["val"]["y"], int), yv)
                    or not np.array_equal(
                        np.asarray(cm["test"]["y"], int), y_all[te_idx])):
                raise ValueError(
                    f"CM bundle labels do not align with fold {fi}: "
                    f"{args.cm_tmp}"
                )
            p_cm_v = np.asarray(cm["val"]["p"], float)
            p_cm_t = np.asarray(cm["test"]["p"], float)
            put(oof, "sourcefirst_cm_pcls", te_idx, yv, p_cm_v, p_cm_t)

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

        fm = {
            "fold": fi,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "n_test": len(te_idx),
            "bge_thr": round(float(bge_thr), 4),
            "nli_hgb": {"thr": round(float(fitg[3]), 3), "lr": fitg[4],
                        "l2": fitg[5], "leaves": fitg[6]},
            "rankmix_alpha": {k: round(float(v[2]), 3)
                              for k, v in score_variants.items()},
        }
        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        decision_heads_v, decision_heads_t, scorefallback_scores = add_scoreguard_heads(
            yv, score_variants, p_bge_v, p_bge_t, bge_v, bge_t,
            val_recs, test_recs,
            oof, te_idx, fm)

        def score_pair(score_key):
            if score_key in scorefallback_scores:
                return scorefallback_scores[score_key]
            if score_key in score_variants:
                pv_key, pt_key, _ = score_variants[score_key]
                return pv_key, pt_key
            return None

        pcls_combo_scores = {}
        if p_cm_v is not None and p_cm_t is not None:
            pcls_combo_specs = {
                "rankavg_sourcefirst_cm_pcls_bge": (
                    rank_mean(p_cm_v, p_bge_v),
                    rank_mean(p_cm_t, p_bge_t),
                    ["sourcefirst_cm_pcls", "bge_lr"],
                ),
            }
            for tag, score_key in (
                (
                    "nli075",
                    "rankmix_nli25_hgb_bge_scorefallback_bge075_src0_src2_3_lowabs",
                ),
                (
                    "nli100",
                    "rankmix_nli25_hgb_bge_scorefallback_bge100_src0_src2_3_lowabs",
                ),
            ):
                pair = score_pair(score_key)
                if pair is None:
                    continue
                nli_v, nli_t = pair
                pcls_combo_specs[f"rankavg_sourcefirst_cm_pcls_bge_{tag}"] = (
                    rank_mean(p_cm_v, p_bge_v, nli_v),
                    rank_mean(p_cm_t, p_bge_t, nli_t),
                    ["sourcefirst_cm_pcls", "bge_lr", score_key],
                )
                if tag == "nli075":
                    pcls_combo_specs[f"rankavg_sourcefirst_cm_pcls_{tag}"] = (
                        rank_mean(p_cm_v, nli_v),
                        rank_mean(p_cm_t, nli_t),
                        ["sourcefirst_cm_pcls", score_key],
                    )
                    for cm_weight, nli_weight in (
                        (0.25, 0.75),
                        (0.33, 0.67),
                        (0.40, 0.60),
                    ):
                        weight_tag = (
                            f"cm{int(round(cm_weight * 100)):03d}_"
                            f"nli{int(round(nli_weight * 100)):03d}"
                        )
                        pcls_combo_specs[
                            f"rankw_sourcefirst_{weight_tag}"
                        ] = (
                            rank_weighted(
                                (p_cm_v, cm_weight), (nli_v, nli_weight)),
                            rank_weighted(
                                (p_cm_t, cm_weight), (nli_t, nli_weight)),
                            [
                                f"sourcefirst_cm_pcls:{cm_weight:.2f}",
                                f"{score_key}:{nli_weight:.2f}",
                            ],
                        )
            for combo_name, (pv_combo, pt_combo, components) in pcls_combo_specs.items():
                pcls_combo_scores[combo_name] = (pv_combo, pt_combo)
                thr = put(oof, combo_name, te_idx, yv, pv_combo, pt_combo)
                decision_heads_v[f"{combo_name}_thr"] = (
                    pv_combo >= thr).astype(int)
                decision_heads_t[f"{combo_name}_thr"] = (
                    pt_combo >= thr).astype(int)
                fm[combo_name] = {
                    "components": components,
                    "thr": round(float(thr), 4),
                    "protocol": "sourcefirst_cm_bge_nli_rankavg",
                }
            decoupled_score_names = [
                name for name in (
                    "rankavg_sourcefirst_cm_pcls_nli075",
                    "rankw_sourcefirst_cm025_nli075",
                    "rankw_sourcefirst_cm033_nli067",
                    "rankw_sourcefirst_cm040_nli060",
                )
                if name in pcls_combo_scores
            ]
            if decoupled_score_names:
                decision_pool = {"bge_thr": (bge_v, bge_t)}
                for decision_key in (
                    "rankavg_sourcefirst_cm_pcls_bge_thr",
                    "rankavg_sourcefirst_cm_pcls_bge_nli075_thr",
                    (
                        "rankmix_nli25_hgb_bge_scorefallback_bge075_"
                        "src0_src2_3_lowabs_thr"
                    ),
                ):
                    if (decision_key in decision_heads_v
                            and decision_key in decision_heads_t):
                        decision_pool[decision_key] = (
                            decision_heads_v[decision_key],
                            decision_heads_t[decision_key],
                        )

                def put_pcls_decoupled(score_name, rank_pv, rank_pt, method,
                                       decision_key, yhat_v, yhat_t, meta):
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = rank_pt
                    oof[method]["yhat"][te_idx] = yhat_t
                    score, val_metrics = val_utility(
                        yv, rank_pv, yhat_v, mode="macro")
                    fm[method] = {
                        "score_head": score_name,
                        "decision_head": decision_key,
                        "utility": round(float(score), 4),
                        "select": {k: round(float(v), 4)
                                   for k, v in val_metrics.items()},
                        "protocol": "sourcefirst_cm_nli_rank_decoupled_decision",
                        **meta,
                    }

                for score_name in decoupled_score_names:
                    rank_pv, rank_pt = pcls_combo_scores[score_name]
                    for suffix, decision_key in (
                        (
                            "cmbge_nli075",
                            "rankavg_sourcefirst_cm_pcls_bge_nli075_thr",
                        ),
                        (
                            "nli075_lowabs",
                            (
                                "rankmix_nli25_hgb_bge_scorefallback_bge075_"
                                "src0_src2_3_lowabs_thr"
                            ),
                        ),
                    ):
                        if decision_key in decision_pool:
                            yhat_v, yhat_t = decision_pool[decision_key]
                            put_pcls_decoupled(
                                score_name,
                                rank_pv,
                                rank_pt,
                                f"{score_name}_decision_{suffix}",
                                decision_key,
                                yhat_v,
                                yhat_t,
                                {"selection": "fixed"},
                            )

                    if score_name != "rankavg_sourcefirst_cm_pcls_nli075":
                        continue
                    best = None
                    for decision_key, (yhat_v, yhat_t) in decision_pool.items():
                        score, val_metrics = val_utility(
                            yv, rank_pv, yhat_v, mode="macro")
                        item = (score, decision_key, val_metrics, yhat_v, yhat_t)
                        if best is None or item[0] > best[0] + 1e-12:
                            best = item
                    if best is not None:
                        put_pcls_decoupled(
                            score_name,
                            rank_pv,
                            rank_pt,
                            "rankavg_sourcefirst_cm_pcls_nli075_"
                            "decision_valselect_small_macro",
                            best[1],
                            best[3],
                            best[4],
                            {
                                "selection": "valselect_small_macro",
                                "candidate_pool": sorted(decision_pool.keys()),
                            },
                        )
                        default_key = "bge_thr"
                        default_yv, default_yt = decision_pool[default_key]
                        default_score, default_metrics = val_utility(
                            yv, rank_pv, default_yv, mode="macro")
                        for margin in (0.005, 0.010):
                            use_best = best[0] >= default_score + margin
                            chosen_key = best[1] if use_best else default_key
                            chosen_yv = best[3] if use_best else default_yv
                            chosen_yt = best[4] if use_best else default_yt
                            put_pcls_decoupled(
                                score_name,
                                rank_pv,
                                rank_pt,
                                "rankavg_sourcefirst_cm_pcls_nli075_"
                                "decision_valselect_bgeprotected_"
                                f"m{int(margin * 1000):03d}",
                                chosen_key,
                                chosen_yv,
                                chosen_yt,
                                {
                                    "selection": "valselect_bgeprotected_macro",
                                    "margin": round(float(margin), 3),
                                    "default": default_key,
                                    "default_utility": round(float(default_score), 4),
                                    "default_select": {
                                        k: round(float(v), 4)
                                        for k, v in default_metrics.items()
                                    },
                                    "candidate_pool": sorted(decision_pool.keys()),
                                },
                            )

        decision_pool_predef = {
            "bge_thr": (bge_v, bge_t),
        }
        decision_pool_all = dict(decision_pool_predef)
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
        predef_fold_candidates = {}
        for proto_name, score_key, decision_key in predef_lowabs_specs:
            pair = score_pair(score_key)
            if (pair is None or decision_key not in decision_heads_t
                    or decision_key not in decision_heads_v):
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
                "protocol": "source_confidence_lowabs_fallback_predef_only",
            }
            predef_fold_candidates[proto_name] = (
                pv_proto,
                pt_proto,
                decision_heads_v[decision_key],
                decision_heads_t[decision_key],
            )
            pool_name = f"predef:{proto_name}"
            decision_pool_predef[pool_name] = (
                decision_heads_v[decision_key],
                decision_heads_t[decision_key],
            )
            decision_pool_all[pool_name] = decision_pool_predef[pool_name]
        switch_self_specs = (
            (
                "sf100src0",
                "rankmix_nli25_hgb_bge_scorefallback_bge100_src0",
                "rankmix_nli25_hgb_bge_scorefallback_bge100_src0_thr",
            ),
            (
                "sf025src0",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_thr",
            ),
            (
                "sf025lowabs",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
                "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs_thr",
            ),
        )
        switch_self_candidates = {}
        for self_name, score_key, decision_key in switch_self_specs:
            pair = score_pair(score_key)
            if (pair is None or decision_key not in decision_heads_t
                    or decision_key not in decision_heads_v):
                continue
            pv_self, pt_self = pair
            switch_self_candidates[self_name] = (
                self_name,
                pv_self,
                pt_self,
                decision_heads_v[decision_key],
                decision_heads_t[decision_key],
            )
        alt_proto = "predef_lowabs_r25_scorefallback_srcconf_bgefallback"
        if alt_proto in predef_fold_candidates and switch_self_candidates:
            alt_pv, alt_pt, alt_yv, alt_yt = predef_fold_candidates[alt_proto]
            alt = ("lowabs_srcconf_bgefallback", alt_pv, alt_pt, alt_yv, alt_yt)
            best_self = None
            for cand in switch_self_candidates.values():
                score, val_metrics = val_utility(yv, cand[1], cand[3], mode="macro")
                item = (score, val_metrics, cand)
                if best_self is None or item[0] > best_self[0] + 1e-12:
                    best_self = item
            switch_bases = dict(switch_self_candidates)
            if best_self is not None:
                switch_bases["sfbest"] = (
                    f"sfbest:{best_self[2][0]}",
                    best_self[2][1],
                    best_self[2][2],
                    best_self[2][3],
                    best_self[2][4],
                )
            for base_name, base in switch_bases.items():
                for fp_slack, override_gain, suffix in (
                    (0.02, None, "fp02"),
                    (0.05, None, "fp05"),
                    (0.02, 0.008, "fp02_gain008"),
                    (0.05, 0.008, "fp05_gain008"),
                ):
                    chosen, switch_meta = choose_binary_switch(
                        yv, val_recs, bge_v, base, alt, fp_slack,
                        override_gain=override_gain)
                    method = (
                        "predef_protocol="
                        f"predef_switch_r25_{base_name}_or_lowabs_srcconf_fp"
                        f"{suffix.replace('fp', '')}"
                    )
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = chosen[2]
                    oof[method]["yhat"][te_idx] = chosen[4]
                    fm[method] = switch_meta
                    decision_pool_all[method] = (chosen[3], chosen[4])
                for fp_hard, macro_margin, recall_slack, rescue, suffix in (
                    (0.08, 0.00, 0.03, 0.008, "fph08_m00_r03_g008"),
                    (0.08, 0.01, 0.05, 0.008, "fph08_m01_r05_g008"),
                ):
                    chosen, switch_meta = choose_protected_default_switch(
                        yv, val_recs, bge_v, base, alt,
                        fp_hard=fp_hard,
                        macro_margin=macro_margin,
                        recall_slack=recall_slack,
                        protected_gain_rescue=rescue,
                    )
                    method = (
                        "predef_protocol="
                        f"predef_switchrev_r25_{base_name}_or_lowabs_srcconf_"
                        f"{suffix}"
                    )
                    oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                            "yhat": np.full(oof["_n"], np.nan)})
                    oof[method]["p"][te_idx] = chosen[2]
                    oof[method]["yhat"][te_idx] = chosen[4]
                    fm[method] = switch_meta
                    decision_pool_all[method] = (chosen[3], chosen[4])
                    if (base_name == "sf100src0"
                            and suffix == "fph08_m00_r03_g008"):
                        groups_v = edit_group_labels(val_recs, "srcbin_conf")
                        groups_t = edit_group_labels(test_recs, "srcbin_conf")
                        for alpha, bge_weight in (
                            (0.15, 0.80),
                            (0.15, 0.90),
                            (0.20, 0.80),
                            (0.20, 0.90),
                        ):
                            score_base_v = rank_mix(p_nli_v, p_bge_v, alpha)
                            score_base_t = rank_mix(p_nli_t, p_bge_t, alpha)
                            score_v, score_t = apply_score_fallback(
                                score_base_v,
                                p_bge_v,
                                groups_v,
                                score_base_t,
                                p_bge_t,
                                groups_t,
                                "src0_src2_3_lowabs",
                                bge_weight,
                            )
                            decoupled = (
                                "predef_protocol="
                                "predef_switchrev_score_"
                                f"a{int(round(alpha * 100)):02d}_"
                                f"bge{int(round(bge_weight * 100)):03d}_"
                                "lowabs_decision_sf100_fph08_m00_r03_g008"
                            )
                            oof.setdefault(
                                decoupled,
                                {"p": np.full(oof["_n"], np.nan),
                                 "yhat": np.full(oof["_n"], np.nan)},
                            )
                            oof[decoupled]["p"][te_idx] = score_t
                            oof[decoupled]["yhat"][te_idx] = chosen[4]
                            fm[decoupled] = {
                                "score_alpha": round(float(alpha), 2),
                                "score_head": "rankmix_custom_hgb_bge",
                                "scorefallback_mode": "src0_src2_3_lowabs",
                                "scorefallback_bge_weight": round(float(bge_weight), 2),
                                "decision_head": method,
                                "decision_selected": switch_meta["selected"],
                                "protocol": "fixed_score_decoupled_reverse_switch",
                            }
        if pcls_combo_scores:
            def source_rich_conf_mask(records, conf_values):
                conf_values = set(conf_values)
                return np.asarray([
                    rec_source_count(r) >= 2
                    and str(r.get("confidence", "")) in conf_values
                    for r in records
                ], bool)

            pcls_switch_decisions = (
                (
                    "switch_sf025lowabs_lowabs_srcconf_fp02_gain008",
                    (
                        "predef_protocol="
                        "predef_switch_r25_sf025lowabs_or_lowabs_srcconf_"
                        "fp02_gain008"
                    ),
                ),
                (
                    "switch_sf025lowabs_lowabs_srcconf_fp05_gain008",
                    (
                        "predef_protocol="
                        "predef_switch_r25_sf025lowabs_or_lowabs_srcconf_"
                        "fp05_gain008"
                    ),
                ),
            )
            for score_name in (
                "rankw_sourcefirst_cm033_nli067",
                "rankw_sourcefirst_cm040_nli060",
            ):
                if score_name not in pcls_combo_scores:
                    continue
                rank_pv, rank_pt = pcls_combo_scores[score_name]
                for suffix, decision_key in pcls_switch_decisions:
                    if decision_key not in decision_pool_all:
                        continue
                    yhat_v, yhat_t = decision_pool_all[decision_key]
                    method = f"{score_name}_decision_{suffix}"
                    oof.setdefault(method, {
                        "p": np.full(oof["_n"], np.nan),
                        "yhat": np.full(oof["_n"], np.nan),
                    })
                    oof[method]["p"][te_idx] = rank_pt
                    oof[method]["yhat"][te_idx] = yhat_t
                    score, val_metrics = val_utility(
                        yv, rank_pv, yhat_v, mode="macro")
                    fm[method] = {
                        "score_head": score_name,
                        "decision_head": decision_key,
                        "utility": round(float(score), 4),
                        "select": {
                            k: round(float(v), 4)
                            for k, v in val_metrics.items()
                        },
                        "protocol": "sourcefirst_cm_nli_rank_switch_decision",
                    }
                    cmbge_key = "rankavg_sourcefirst_cm_pcls_bge_thr"
                    if (decision_key in decision_pool_all
                            and cmbge_key in decision_heads_v
                            and cmbge_key in decision_heads_t):
                        for guard_suffix, conf_values in (
                            ("srcge2_lowmedium_cmbgeprotect",
                             ("low", "medium")),
                            ("srcge2_allconf_cmbgeprotect",
                             ("low", "medium", "high", "absent")),
                        ):
                            guard_v = source_rich_conf_mask(
                                val_recs, conf_values)
                            guard_t = source_rich_conf_mask(
                                test_recs, conf_values)
                            guarded_yv = np.asarray(yhat_v, int).copy()
                            guarded_yt = np.asarray(yhat_t, int).copy()
                            guarded_yv[guard_v] = decision_heads_v[cmbge_key][guard_v]
                            guarded_yt[guard_t] = decision_heads_t[cmbge_key][guard_t]
                            guarded_method = (
                                f"{score_name}_decision_{suffix}_"
                                f"{guard_suffix}"
                            )
                            oof.setdefault(guarded_method, {
                                "p": np.full(oof["_n"], np.nan),
                                "yhat": np.full(oof["_n"], np.nan),
                            })
                            oof[guarded_method]["p"][te_idx] = rank_pt
                            oof[guarded_method]["yhat"][te_idx] = guarded_yt
                            guard_score, guard_metrics = val_utility(
                                yv, rank_pv, guarded_yv, mode="macro")
                            fm[guarded_method] = {
                                "score_head": score_name,
                                "decision_head": decision_key,
                                "guard_decision_head": cmbge_key,
                                "guard": guard_suffix,
                                "guard_confidence": sorted(conf_values),
                                "guard_val_n": int(guard_v.sum()),
                                "guard_test_n": int(guard_t.sum()),
                                "utility": round(float(guard_score), 4),
                                "select": {
                                    k: round(float(v), 4)
                                    for k, v in guard_metrics.items()
                                },
                                "protocol": (
                                    "sourcefirst_cm_nli_rank_switch_"
                                    "source_rich_cmbge_guard"
                                ),
                            }
                            if (
                                    score_name == "rankw_sourcefirst_cm040_nli060"
                                    and suffix == (
                                        "switch_sf025lowabs_lowabs_srcconf_"
                                        "fp05_gain008"
                                    )
                                    and guard_suffix
                                    == "srcge2_lowmedium_cmbgeprotect"):
                                repair_key = (
                                    "rankavg_sourcefirst_cm_pcls_bge_nli075_thr"
                                )
                                if repair_key in decision_heads_v:
                                    score_mask_v = records_mask(
                                        val_recs,
                                        lambda r: (
                                            rec_source_count(r) == 0
                                            or str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    score_mask_t = records_mask(
                                        test_recs,
                                        lambda r: (
                                            rec_source_count(r) == 0
                                            or str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    adaptive_pv = np.asarray(rank_pv, float).copy()
                                    adaptive_pt = np.asarray(rank_pt, float).copy()
                                    adaptive_pv[score_mask_v] = (
                                        0.75 * adaptive_pv[score_mask_v]
                                        + 0.25 * p_cm_v[score_mask_v]
                                    )
                                    adaptive_pt[score_mask_t] = (
                                        0.75 * adaptive_pt[score_mask_t]
                                        + 0.25 * p_cm_t[score_mask_t]
                                    )
                                    repair_v = records_mask(
                                        val_recs,
                                        lambda r: (
                                            rec_source_bin(r) == "src4p"
                                            and str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    repair_t = records_mask(
                                        test_recs,
                                        lambda r: (
                                            rec_source_bin(r) == "src4p"
                                            and str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    adaptive_yv = guarded_yv.copy()
                                    adaptive_yt = guarded_yt.copy()
                                    adaptive_yv[repair_v] = (
                                        decision_heads_v[repair_key][repair_v]
                                    )
                                    adaptive_yt[repair_t] = (
                                        decision_heads_t[repair_key][repair_t]
                                    )
                                    adaptive_method = (
                                        f"{score_name}_score_src0ormedium_"
                                        "cmreinforce025_decision_"
                                        f"{suffix}_{guard_suffix}_"
                                        "src4pmedium_cmbgenli"
                                    )
                                    oof.setdefault(adaptive_method, {
                                        "p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan),
                                    })
                                    oof[adaptive_method]["p"][te_idx] = adaptive_pt
                                    oof[adaptive_method]["yhat"][te_idx] = (
                                        adaptive_yt
                                    )
                                    adaptive_score, adaptive_metrics = val_utility(
                                        yv, adaptive_pv, adaptive_yv, mode="macro")
                                    fm[adaptive_method] = {
                                        "base_score_head": score_name,
                                        "base_decision_head": guarded_method,
                                        "score_reinforce_head": "sourcefirst_cm_pcls",
                                        "score_reinforce_weight": 0.25,
                                        "score_reinforce_rule": (
                                            "source_count == 0 or "
                                            "confidence == medium"
                                        ),
                                        "decision_repair_head": repair_key,
                                        "decision_repair_rule": (
                                            "source_bin == src4p and "
                                            "confidence == medium"
                                        ),
                                        "score_mask_val_n": int(score_mask_v.sum()),
                                        "score_mask_test_n": int(score_mask_t.sum()),
                                        "decision_repair_val_n": int(repair_v.sum()),
                                        "decision_repair_test_n": int(repair_t.sum()),
                                        "utility": round(float(adaptive_score), 4),
                                        "select": {
                                            k: round(float(v), 4)
                                            for k, v in adaptive_metrics.items()
                                        },
                                        "protocol": (
                                            "source_reliability_adaptive_"
                                            "cm_reinforced_score_and_nli_"
                                            "source_rich_decision_repair"
                                        ),
                                    }

                                    ev_score_mask_v = records_mask(
                                        val_recs,
                                        lambda r: (
                                            rec_source_count(r) == 0
                                            or (
                                                rec_evidence_combo(r) == "PO"
                                                and str(r.get("confidence", ""))
                                                == "medium"
                                            )
                                        ),
                                    )
                                    ev_score_mask_t = records_mask(
                                        test_recs,
                                        lambda r: (
                                            rec_source_count(r) == 0
                                            or (
                                                rec_evidence_combo(r) == "PO"
                                                and str(r.get("confidence", ""))
                                                == "medium"
                                            )
                                        ),
                                    )
                                    ev_decision_mask_v = records_mask(
                                        val_recs,
                                        lambda r: (
                                            rec_evidence_combo(r) == "PO"
                                            and str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    ev_decision_mask_t = records_mask(
                                        test_recs,
                                        lambda r: (
                                            rec_evidence_combo(r) == "PO"
                                            and str(r.get("confidence", ""))
                                            == "medium"
                                        ),
                                    )
                                    ev_pv = np.asarray(rank_pv, float).copy()
                                    ev_pt = np.asarray(rank_pt, float).copy()
                                    ev_pv[ev_score_mask_v] = (
                                        0.75 * ev_pv[ev_score_mask_v]
                                        + 0.25 * p_cm_v[ev_score_mask_v]
                                    )
                                    ev_pt[ev_score_mask_t] = (
                                        0.75 * ev_pt[ev_score_mask_t]
                                        + 0.25 * p_cm_t[ev_score_mask_t]
                                    )
                                    ev_yv = guarded_yv.copy()
                                    ev_yt = guarded_yt.copy()
                                    ev_yv[ev_decision_mask_v] = (
                                        adaptive_yv[ev_decision_mask_v]
                                    )
                                    ev_yt[ev_decision_mask_t] = (
                                        adaptive_yt[ev_decision_mask_t]
                                    )
                                    ev_method = (
                                        f"{score_name}_score_src0orpomedium_"
                                        "cmreinforce025_decision_"
                                        f"{suffix}_{guard_suffix}_"
                                        "pomedium_cmbgenli"
                                    )
                                    oof.setdefault(ev_method, {
                                        "p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan),
                                    })
                                    oof[ev_method]["p"][te_idx] = ev_pt
                                    oof[ev_method]["yhat"][te_idx] = ev_yt
                                    ev_score, ev_metrics = val_utility(
                                        yv, ev_pv, ev_yv, mode="macro")
                                    fm[ev_method] = {
                                        "base_score_head": score_name,
                                        "base_decision_head": guarded_method,
                                        "score_reinforce_head": "sourcefirst_cm_pcls",
                                        "score_reinforce_weight": 0.25,
                                        "score_reinforce_rule": (
                                            "source_count == 0 or "
                                            "evidence_combo == PO and "
                                            "confidence == medium"
                                        ),
                                        "decision_repair_head": adaptive_method,
                                        "decision_repair_inner_head": repair_key,
                                        "decision_repair_rule": (
                                            "evidence_combo == PO and "
                                            "confidence == medium"
                                        ),
                                        "score_mask_val_n": int(
                                            ev_score_mask_v.sum()),
                                        "score_mask_test_n": int(
                                            ev_score_mask_t.sum()),
                                        "decision_repair_val_n": int(
                                            ev_decision_mask_v.sum()),
                                        "decision_repair_test_n": int(
                                            ev_decision_mask_t.sum()),
                                        "utility": round(float(ev_score), 4),
                                        "select": {
                                            k: round(float(v), 4)
                                            for k, v in ev_metrics.items()
                                        },
                                        "protocol": (
                                            "evidence_type_adaptive_"
                                            "cm_reinforced_score_and_po_"
                                            "medium_decision_repair"
                                        ),
                                    }
                                    argref_neutral_v = nli_cache_feature(
                                        X[va_idx], "arg_ref", "neutral", "rate35")
                                    argref_neutral_t = nli_cache_feature(
                                        X[te_idx], "arg_ref", "neutral", "rate35")
                                    ev_insuff_pv = rank_weighted(
                                        (ev_pv, 0.95), (argref_neutral_v, 0.05))
                                    ev_insuff_pt = rank_weighted(
                                        (ev_pt, 0.95), (argref_neutral_t, 0.05))
                                    ev_insuff_method = (
                                        f"{score_name}_score_src0orpomedium_"
                                        "cmreinforce025_argrefneutral005_"
                                        "decision_"
                                        f"{suffix}_{guard_suffix}_"
                                        "pomedium_cmbgenli"
                                    )
                                    oof.setdefault(ev_insuff_method, {
                                        "p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan),
                                    })
                                    oof[ev_insuff_method]["p"][te_idx] = (
                                        ev_insuff_pt
                                    )
                                    oof[ev_insuff_method]["yhat"][te_idx] = ev_yt
                                    ev_insuff_score, ev_insuff_metrics = (
                                        val_utility(
                                            yv, ev_insuff_pv, ev_yv,
                                            mode="macro")
                                    )
                                    fm[ev_insuff_method] = {
                                        "base_score_head": ev_method,
                                        "base_decision_head": ev_method,
                                        "score_calibrator": (
                                            "arg_ref neutral posterior rate>=0.35"
                                        ),
                                        "score_calibrator_weight": 0.05,
                                        "decision": "evidence_type_yhat",
                                        "utility": round(
                                            float(ev_insuff_score), 4),
                                        "select": {
                                            k: round(float(v), 4)
                                            for k, v in (
                                                ev_insuff_metrics.items())
                                        },
                                        "protocol": (
                                            "evidence_type_score_with_"
                                            "argument_insufficiency_"
                                            "micro_calibration"
                                        ),
                                    }

                                tax_score_key = "rankw_sourcefirst_cm025_nli075"
                                tax_decision_key = (
                                    "rankw_sourcefirst_cm025_nli075_thr"
                                )
                                if (
                                        tax_score_key in pcls_combo_scores
                                        and tax_decision_key in decision_heads_v):
                                    tax_pv, tax_pt = pcls_combo_scores[tax_score_key]
                                    tax_score_mask_v = records_mask(
                                        val_recs,
                                        lambda r: str(r.get("category", ""))
                                        in ("sports_and_outdoor", "general"),
                                    )
                                    tax_score_mask_t = records_mask(
                                        test_recs,
                                        lambda r: str(r.get("category", ""))
                                        in ("sports_and_outdoor", "general"),
                                    )
                                    tax_decision_mask_v = records_mask(
                                        val_recs,
                                        lambda r: str(r.get("category", ""))
                                        == "sports_and_outdoor",
                                    )
                                    tax_decision_mask_t = records_mask(
                                        test_recs,
                                        lambda r: str(r.get("category", ""))
                                        == "sports_and_outdoor",
                                    )
                                    tax_pv_out = np.asarray(rank_pv, float).copy()
                                    tax_pt_out = np.asarray(rank_pt, float).copy()
                                    tax_pv_out[tax_score_mask_v] = (
                                        tax_pv[tax_score_mask_v]
                                    )
                                    tax_pt_out[tax_score_mask_t] = (
                                        tax_pt[tax_score_mask_t]
                                    )
                                    tax_yv = guarded_yv.copy()
                                    tax_yt = guarded_yt.copy()
                                    tax_yv[tax_decision_mask_v] = (
                                        decision_heads_v[tax_decision_key][
                                            tax_decision_mask_v]
                                    )
                                    tax_yt[tax_decision_mask_t] = (
                                        decision_heads_t[tax_decision_key][
                                            tax_decision_mask_t]
                                    )
                                    tax_method = (
                                        f"{score_name}_score_sportsgeneral_"
                                        "cm025_decision_sports_cm025"
                                    )
                                    oof.setdefault(tax_method, {
                                        "p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan),
                                    })
                                    oof[tax_method]["p"][te_idx] = tax_pt_out
                                    oof[tax_method]["yhat"][te_idx] = tax_yt
                                    tax_score, tax_metrics = val_utility(
                                        yv, tax_pv_out, tax_yv, mode="macro")
                                    fm[tax_method] = {
                                        "base_score_head": score_name,
                                        "base_decision_head": guarded_method,
                                        "taxonomy_score_head": tax_score_key,
                                        "taxonomy_decision_head": tax_decision_key,
                                        "score_taxonomy": [
                                            "general",
                                            "sports_and_outdoor",
                                        ],
                                        "decision_taxonomy": [
                                            "sports_and_outdoor",
                                        ],
                                        "score_mask_val_n": int(
                                            tax_score_mask_v.sum()),
                                        "score_mask_test_n": int(
                                            tax_score_mask_t.sum()),
                                        "decision_mask_val_n": int(
                                            tax_decision_mask_v.sum()),
                                        "decision_mask_test_n": int(
                                            tax_decision_mask_t.sum()),
                                        "utility": round(float(tax_score), 4),
                                        "select": {
                                            k: round(float(v), 4)
                                            for k, v in tax_metrics.items()
                                        },
                                        "protocol": (
                                            "taxonomy_aware_source_reliability_"
                                            "screen_candidate"
                                        ),
                                    }
        for score_tag, score_key in (
            (
                "rankstable075",
                "rankmix_nli25_hgb_bge_scorefallback_bge075_src0_src2_3_lowabs",
            ),
            (
                "rankstable100",
                "rankmix_nli25_hgb_bge_scorefallback_bge100_src0_src2_3_lowabs",
            ),
        ):
            pair = score_pair(score_key)
            if pair is None:
                continue
            fixed_pv, fixed_pt = pair
            for pool_tag, pool in (
                ("predef", decision_pool_predef),
                ("all", decision_pool_all),
            ):
                best = None
                for decision_name, (yhat_v, yhat_t) in pool.items():
                    score, val_metrics = val_utility(
                        yv, fixed_pv, yhat_v, mode="macro")
                    if best is None or score > best[0] + 1e-12:
                        best = (score, decision_name, val_metrics, yhat_t)
                if best is None:
                    continue
                method = (
                    "predef_protocol="
                    f"predef_{score_tag}_decision_valselect_{pool_tag}_macro"
                )
                oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                        "yhat": np.full(oof["_n"], np.nan)})
                oof[method]["p"][te_idx] = fixed_pt
                oof[method]["yhat"][te_idx] = best[3]
                fm[method] = {
                    "score_head": score_key,
                    "selected_decision": best[1],
                    "utility": round(float(best[0]), 4),
                    "select": {k: round(float(v), 4) for k, v in best[2].items()},
                    "candidate_pool": sorted(pool.keys()),
                    "protocol": "fixed_rank_score_valselect_decision",
                }
        for mode in ("macro", "balanced"):
            best = None
            for proto_name, (pv_proto, pt_proto, yhat_v, yhat_t) in (
                    predef_fold_candidates.items()):
                score, val_metrics = val_utility(yv, pv_proto, yhat_v, mode=mode)
                if best is None or score > best[0] + 1e-12:
                    best = (score, proto_name, val_metrics, pt_proto, yhat_t)
            if best is None:
                continue
            method = f"predef_protocol=predef_lowabs_valselect_{mode}"
            oof.setdefault(method, {"p": np.full(oof["_n"], np.nan),
                                    "yhat": np.full(oof["_n"], np.nan)})
            oof[method]["p"][te_idx] = best[3]
            oof[method]["yhat"][te_idx] = best[4]
            fm[method] = {
                "selected": best[1],
                "utility": round(float(best[0]), 4),
                "select": {k: round(float(v), 4) for k, v in best[2].items()},
                "candidate_pool": sorted(predef_fold_candidates.keys()),
                "protocol": "source_confidence_lowabs_valselect_predef_only",
            }
        fold_meta.append(fm)
        print(f"[fold {fi}] done", flush=True)

    rows = {}
    for name in [m for m in oof if m != "_n"]:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = row(y_all[ok], oof[name]["p"][ok],
                         oof[name]["yhat"][ok], c_all[ok])
    ranked = sorted(rows, key=lambda m: (rows[m]["macro_f1"], rows[m]["auprc"]),
                    reverse=True)
    print("=== NLI predef-lowabs candidates ===", flush=True)
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
            print(f"{name:80s} vs bge_lr AP={r['auprc']:.4f} "
                  f"AUROC={r['auroc']:.4f} mF1={r['macro_f1']:.4f} "
                  f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                  f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                  f"dMF1={s['dMacroF1']['mean_delta']:+.4f}"
                  f"(p={s['dMacroF1']['p_a_gt_b']})",
                  flush=True)

    out = {"fold_seed": args.fold_seed, "rows": rows,
           "fold_meta": fold_meta, "significance": sig,
           "protocol": "predef_lowabs_only"}
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
        print(f"[cv_nli_predef_lowabs] oof -> {args.dump_oof}", flush=True)
    print(f"[cv_nli_predef_lowabs] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
