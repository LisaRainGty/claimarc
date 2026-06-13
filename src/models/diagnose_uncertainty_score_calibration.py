"""OOF screen for BGE-uncertainty score calibration.

This diagnostic is intentionally OOF-only. It tests whether a fixed,
evidence-sufficiency rule can improve the ranking score without changing the
already strong source/confidence-aware decision heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


CASES = {
    "fs0_decoupled": {
        "path": "data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_5k.npz",
        "decision": (
            "predef_protocol_predef_switchrev_r25_sf100src0_or_lowabs_srcconf_"
            "fph08_m00_r03_g008__yhat"
        ),
    },
    "fs1_predef": {
        "path": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs.npz",
        "decision": "predef_protocol_predef_lowabs_r25_scorefallback_srcconf_bgefallback__yhat",
    },
    "fs2_bgefallback": {
        "path": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback.npz",
        "decision": (
            "dual_score_rankmix_nli25_hgb_bge__decision_"
            "rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence_"
            "bgefallback_src0_src2_3__yhat"
        ),
    },
}


def rank01(x):
    x = np.asarray(x, float)
    out = np.zeros(len(x), float)
    if len(x) <= 1:
        return out
    order = np.argsort(x, kind="mergesort")
    out[order] = np.arange(len(x), dtype=float)
    return out / float(len(x) - 1)


def macro(y, yhat, w=None):
    return float(f1_score(
        y, yhat, labels=[0, 1], average="macro",
        sample_weight=w, zero_division=0,
    ))


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def paired_bootstrap(y, p_a, yhat_a, p_b, yhat_b, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    dap, dau, dmf = [], [], []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        dap.append(average_precision_score(yy, p_a[idx])
                   - average_precision_score(yy, p_b[idx]))
        dau.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        dmf.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))

    def summ(vals):
        arr = np.asarray(vals, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {"dAP": summ(dap), "dAUROC": summ(dau), "dMacroF1": summ(dmf)}


def source_masks(z):
    src = np.asarray(z["source_bin"]).astype(str)
    conf = np.asarray(z["confidence"]).astype(str)
    sc = np.asarray(z["source_count"], int)
    lowabs = (
        (src == "src0")
        | ((src == "src2_3") & ((conf == "low") | (conf == "absent") | (conf == "")))
    )
    return {
        "all": np.ones(len(src), bool),
        "nonzero": sc > 0,
        "source_ge2": sc >= 2,
        "source_rich_medium": (sc >= 2) & (conf == "medium"),
        "non_lowabs": ~lowabs,
        "non_lowabs_source_ge2": (~lowabs) & (sc >= 2),
        "src1p_medium": (src != "src0") & (conf == "medium"),
    }


def reconstruct_ranks(z):
    folds = np.asarray(z["fold"], int)
    bge_p = np.asarray(z["bge_lr__p"], float)
    r25 = np.asarray(z["rankmix_nli25_hgb_bge__p"], float)
    r50 = np.asarray(z["rankmix_nli50_hgb_bge__p"], float)
    bge_rank = np.full(len(bge_p), np.nan, float)
    nli_rank = np.full(len(bge_p), np.nan, float)
    for fold in sorted(set(folds.tolist())):
        m = folds == fold
        rb = rank01(bge_p[m])
        rn25 = (r25[m] - 0.75 * rb) / 0.25
        rn50 = (r50[m] - 0.50 * rb) / 0.50
        bge_rank[m] = rb
        nli_rank[m] = np.clip((rn25 + rn50) / 2.0, 0.0, 1.0)
    return bge_rank, nli_rank


def build_score(z, alpha, lo, hi, source_mode, lowabs_bge_weight):
    bge_rank, nli_rank = reconstruct_ranks(z)
    score = bge_rank.copy()
    masks = source_masks(z)
    uncertain = (bge_rank >= float(lo)) & (bge_rank <= float(hi))
    target = uncertain & masks[source_mode]
    score[target] = ((1.0 - float(alpha)) * bge_rank[target]
                     + float(alpha) * nli_rank[target])

    if lowabs_bge_weight < 1.0:
        lowabs = ~masks["non_lowabs"]
        base = ((1.0 - float(alpha)) * bge_rank[lowabs]
                + float(alpha) * nli_rank[lowabs])
        w = float(lowabs_bge_weight)
        score[lowabs] = (1.0 - w) * base + w * bge_rank[lowabs]
    return score


def evaluate_case(case_name, cfg, candidates, n_boot):
    z = np.load(cfg["path"], allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    bge_p = np.asarray(z["bge_lr__p"], float)
    bge_yhat = np.asarray(z["bge_lr__yhat"], int)
    yhat = np.asarray(z[cfg["decision"]], int)
    ok_base = np.isfinite(bge_p) & np.isfinite(bge_yhat) & np.isfinite(yhat)
    out = {
        "case": case_name,
        "path": cfg["path"],
        "decision": cfg["decision"],
        "bge": row(y[ok_base], bge_p[ok_base], bge_yhat[ok_base], c[ok_base]),
        "candidates": {},
    }
    for name, score in candidates.items():
        ok = ok_base & np.isfinite(score)
        metrics = row(y[ok], score[ok], yhat[ok], c[ok])
        sig = paired_bootstrap(
            y[ok], score[ok], yhat[ok], bge_p[ok], bge_yhat[ok],
            n_boot=n_boot,
        )
        out["candidates"][name] = {"metrics": metrics, "significance": sig}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/final/cleancl/uncertainty_score_calibration_screen_20260608.json")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()

    # Rank candidates on fs0 first, because fs0 newcache is the current AUROC
    # bottleneck. Then evaluate only the strongest fixed rules on fs1/fs2.
    fs0 = np.load(CASES["fs0_decoupled"]["path"], allow_pickle=True)
    y0 = np.asarray(fs0["y"], int)
    c0 = np.asarray(fs0["c"], float)
    bge0 = np.asarray(fs0["bge_lr__p"], float)
    bgey0 = np.asarray(fs0["bge_lr__yhat"], int)
    dec0 = np.asarray(fs0[CASES["fs0_decoupled"]["decision"]], int)
    grid = []
    score_cache = {}
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        for lo, hi in ((0.20, 0.80), (0.25, 0.75), (0.30, 0.70),
                       (0.35, 0.65), (0.40, 0.60)):
            for mode in ("all", "nonzero", "source_ge2", "source_rich_medium",
                         "non_lowabs", "non_lowabs_source_ge2", "src1p_medium"):
                for lowabs_w in (1.0, 0.9, 0.8):
                    name = (
                        f"uncscore_a{int(round(alpha * 100)):02d}_"
                        f"r{int(lo * 100):02d}_{int(hi * 100):02d}_"
                        f"{mode}_lowabsbge{int(round(lowabs_w * 100)):03d}"
                    )
                    score = build_score(fs0, alpha, lo, hi, mode, lowabs_w)
                    metrics = row(y0, score, dec0, c0)
                    score_cache[name] = score
                    grid.append((
                        -metrics["auroc"], -metrics["auprc"], -metrics["macro_f1"],
                        name, (alpha, lo, hi, mode, lowabs_w), metrics,
                    ))
    grid.sort()
    boot_grid = []
    for _, _, _, name, params, metrics in grid[:80]:
        score = score_cache[name]
        sig = paired_bootstrap(
            y0, score, dec0, bge0, bgey0,
            n_boot=max(300, min(args.n_boot, 1000)),
        )
        max_p = max(sig["dAP"]["p_a_gt_b"],
                    sig["dAUROC"]["p_a_gt_b"],
                    sig["dMacroF1"]["p_a_gt_b"])
        boot_grid.append((max_p, sig["dAUROC"]["p_a_gt_b"],
                          -metrics["auroc"], -metrics["auprc"],
                          name, params, metrics, sig))
    boot_grid.sort()
    selected = boot_grid[:int(args.top_k)]

    candidates_by_case = {}
    for case, cfg in CASES.items():
        z = np.load(cfg["path"], allow_pickle=True)
        candidates = {}
        for _, _, _, _, name, params, _, _ in selected:
            candidates[name] = build_score(z, *params)
        candidates_by_case[case] = evaluate_case(case, cfg, candidates, args.n_boot)

    out = {
        "protocol": "oof_only_bge_uncertainty_score_calibration_screen",
        "selection_note": (
            "candidates ranked on fs0_decoupled OOF only; fs1/fs2 are "
            "cross-case diagnostics, not validation for deployment"
        ),
        "selected_from_fs0": [
            {
                "name": name,
                "params": {
                    "alpha": params[0],
                    "rank_lo": params[1],
                    "rank_hi": params[2],
                    "source_mode": params[3],
                    "lowabs_bge_weight": params[4],
                },
                "fs0_metrics": metrics,
                "fs0_significance": sig,
            }
            for _, _, _, _, name, params, metrics, sig in selected
        ],
        "cases": candidates_by_case,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[diagnose_uncertainty_score_calibration] -> {args.out}")
    for item in out["selected_from_fs0"][:8]:
        sig = item["fs0_significance"]
        m = item["fs0_metrics"]
        print(
            f"{item['name']:70s} AP={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
            f"mF1={m['macro_f1']:.4f} p="
            f"{sig['dAP']['p_a_gt_b']}/"
            f"{sig['dAUROC']['p_a_gt_b']}/"
            f"{sig['dMacroF1']['p_a_gt_b']}"
        )


if __name__ == "__main__":
    main()
