"""OOF diagnostics for source-specific NLI posterior score pooling.

The NLI cache stores fixed set-level posterior features in source blocks:
all / param / ocr / vlm / argument support / argument refutation / evidence gap.
This script aligns those features to the repeated fs0/fs1/fs2 OOF rows and
tests very small score-side calibrations on top of saved evidence-type scores.

It deliberately keeps the decision head from the base method.  Without fold
validation scores for the new blended score, changing thresholds here would be
post-hoc and too optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

try:
    from models.bootstrap_oof_methods import paired_bootstrap
except ModuleNotFoundError:
    from bootstrap_oof_methods import paired_bootstrap


CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
ADAPTIVE = (
    "rankw_sourcefirst_cm040_nli060_score_src0ormedium_cmreinforce025_"
    "decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_"
    "srcge2_lowmedium_cmbgeprotect_src4pmedium_cmbgenli"
)
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"

STATS = ["n", "mean", "max", "min", "std", "top2", "p25", "p75", "rate20", "rate35", "rate50"]
VARS = ["contr", "entail", "neutral", "margin", "maxce", "uncert"]
TYPES = ["all", "param", "ocr", "vlm", "arg_sup", "arg_ref", "arg_gap"]


def metric_row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
        "wF1": round(float(f1_score(
            y, yhat, average="macro", sample_weight=np.clip(c, 0.05, None), zero_division=0
        )), 4),
        "n": int(len(y)),
    }


def rank01(x):
    x = np.asarray(x, float)
    if len(x) <= 1:
        return np.zeros(len(x), dtype=float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=float)
    out[order] = np.arange(len(x), dtype=float)
    return out / float(len(x) - 1)


def feature_idx(typ: str, var: str, stat: str) -> int:
    return TYPES.index(typ) * 66 + VARS.index(var) * 11 + STATS.index(stat)


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def load_records(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def source_feature_table(X_rep: np.ndarray) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    for typ in TYPES:
        for stat in ("mean", "max", "top2", "rate35", "rate50"):
            for var in ("contr", "entail", "neutral", "margin"):
                feats[f"{typ}_{var}_{stat}"] = X_rep[:, feature_idx(typ, var, stat)]

    for stat in ("mean", "max", "top2", "rate35", "rate50"):
        margins = [X_rep[:, feature_idx(t, "margin", stat)] for t in ("param", "ocr", "vlm")]
        contr = [X_rep[:, feature_idx(t, "contr", stat)] for t in ("param", "ocr", "vlm")]
        entail = [X_rep[:, feature_idx(t, "entail", stat)] for t in ("param", "ocr", "vlm")]
        neutral = [X_rep[:, feature_idx(t, "neutral", stat)] for t in ("param", "ocr", "vlm")]
        feats[f"src_max_margin_{stat}"] = np.maximum.reduce(margins)
        feats[f"src_weight_margin_{stat}"] = margins[0] + 0.8 * margins[1] + 0.5 * margins[2]
        feats[f"src_weight_contr_{stat}"] = contr[0] + 0.8 * contr[1] + 0.5 * contr[2]
        feats[f"src_weight_entail_{stat}"] = entail[0] + 0.8 * entail[1] + 0.5 * entail[2]
        feats[f"src_weight_neutral_{stat}"] = neutral[0] + 0.8 * neutral[1] + 0.5 * neutral[2]
    return feats


def grouped_paired_bootstrap(groups, y, p_a, yhat_a, p_b, yhat_b, *, n_boot: int, seed: int):
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups, dtype=object)
    uniq = np.asarray(sorted(set(groups)), dtype=object)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    dap, dau, df1 = [], [], []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        dap.append(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx]))
        dau.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1.append(
            f1_score(yy, yhat_a[idx], average="macro", zero_division=0)
            - f1_score(yy, yhat_b[idx], average="macro", zero_division=0)
        )

    def summarize(values):
        arr = np.asarray(values, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {"dAP": summarize(dap), "dAUROC": summarize(dau), "dMacroF1": summarize(df1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl")
    ap.add_argument("--cache", default="data/final/cleancl/cache_nli_srcargs_a120.npz")
    ap.add_argument("--oof", default="data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz")
    ap.add_argument("--out", default="data/final/cleancl/nli_source_pooling_oof_screen_20260608.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_nli_source_pooling_screen_20260608.npz")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--top_k_bootstrap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260608)
    args = ap.parse_args()

    recs = load_records(args.dataset)
    pair_to_idx = {str(rec.get("pair_id", "")): i for i, rec in enumerate(recs)}
    X = np.load(args.cache)["X"]
    z = np.load(args.oof, allow_pickle=True)
    pair_id = np.asarray(z["pair_id"], dtype=object)
    missing = sorted({str(p) for p in pair_id if str(p) not in pair_to_idx})
    if missing:
        raise KeyError(f"{len(missing)} pair_id values from OOF are missing in dataset; first={missing[:3]}")
    X_rep = X[np.asarray([pair_to_idx[str(p)] for p in pair_id], int)]
    feats = source_feature_table(X_rep)

    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    baselines = {
        "bge_lr": get_method(z, "bge_lr"),
        CMBGE: get_method(z, CMBGE),
        CURRENT: get_method(z, CURRENT),
        ADAPTIVE: get_method(z, ADAPTIVE),
        EVTYPE: get_method(z, EVTYPE),
    }

    feature_specs = {
        # Insufficiency-like argument posterior signals.
        "argref_neutral_rate35": ("arg_ref_neutral_rate35", 1.0),
        "arggap_inv_contr_rate35": ("arg_gap_contr_rate35", -1.0),
        "argref_inv_entail_rate35": ("arg_ref_entail_rate35", -1.0),
        # Source posterior signals retained as a check against source-only pooling.
        "all_inv_contr_rate35": ("all_contr_rate35", -1.0),
        "all_inv_margin_rate35": ("all_margin_rate35", -1.0),
        "param_inv_entail_top2": ("param_entail_top2", -1.0),
        "ocr_entail_rate35": ("ocr_entail_rate35", 1.0),
        "src_weight_entail_rate35": ("src_weight_entail_rate35", 1.0),
        "src_weight_neutral_rate35": ("src_weight_neutral_rate35", 1.0),
    }
    alpha_grid = (0.03, 0.05, 0.08, 0.10, 0.15)
    base_grid = {"evtype": EVTYPE, "current": CURRENT}

    methods = {}
    dump = {
        "y": y,
        "c": c,
        "pair_id": pair_id,
        "fold": np.asarray(z["fold"], int),
        "case": np.asarray(z["case"], dtype=object),
    }
    for base_label, base_method in base_grid.items():
        p_base, yhat_base = baselines[base_method]
        rb = rank01(p_base)
        for feat_label, (feat_key, orient) in feature_specs.items():
            signal = rank01(float(orient) * feats[feat_key])
            for alpha in alpha_grid:
                name = f"{base_label}_score_{feat_label}_a{int(alpha * 100):02d}_decision_{base_label}"
                p = (1.0 - float(alpha)) * rb + float(alpha) * signal
                yhat = yhat_base.copy()
                methods[name] = {
                    "metrics": metric_row(y, p, yhat, c),
                    "base_method": base_method,
                    "feature": feat_key,
                    "orientation": orient,
                    "alpha": alpha,
                    "decision": "base_yhat",
                }
                dump[f"{name}__p"] = p
                dump[f"{name}__yhat"] = yhat

    feature_metrics = {}
    for label, (feat_key, orient) in feature_specs.items():
        s = float(orient) * feats[feat_key]
        feature_metrics[label] = {
            "feature": feat_key,
            "orientation": orient,
            "auprc": round(float(average_precision_score(y, s)), 4),
            "auroc": round(float(roc_auc_score(y, s)), 4),
        }

    baseline_metrics = {name: metric_row(y, p, yhat, c) for name, (p, yhat) in baselines.items()}
    ranked = sorted(methods, key=lambda m: (
        methods[m]["metrics"]["auprc"],
        methods[m]["metrics"]["auroc"],
        methods[m]["metrics"]["macro_f1"],
    ), reverse=True)
    top_methods = ranked[:max(1, int(args.top_k_bootstrap))]

    significance = {"sample_bootstrap": {}, "group_bootstrap": {}}
    if args.n_boot > 0:
        compare_to = ["bge_lr", CMBGE, CURRENT, EVTYPE]
        for name in top_methods:
            p_a = dump[f"{name}__p"]
            yhat_a = dump[f"{name}__yhat"]
            for baseline in compare_to:
                p_b, yhat_b = baselines[baseline]
                key = f"{name}_vs_{baseline}"
                significance["sample_bootstrap"][key] = paired_bootstrap(
                    y, p_a, yhat_a, p_b, yhat_b, n_boot=args.n_boot, seed=args.seed
                )
                significance["group_bootstrap"][key] = grouped_paired_bootstrap(
                    pair_id, y, p_a, yhat_a, p_b, yhat_b,
                    n_boot=args.n_boot, seed=args.seed + 17
                )

    out = {
        "description": (
            "Diagnostic score-side source/argument NLI-posterior pooling over saved OOF predictions. "
            "Decision labels are inherited from the base method because no fold validation scores exist "
            "for these blended scores."
        ),
        "dataset": args.dataset,
        "cache": args.cache,
        "oof": args.oof,
        "n": int(len(y)),
        "n_pair_id": int(len(set(pair_id))),
        "n_boot": int(args.n_boot),
        "top_k_bootstrap": int(args.top_k_bootstrap),
        "feature_specs": feature_specs,
        "alpha_grid": list(alpha_grid),
        "baseline_metrics": baseline_metrics,
        "feature_metrics": feature_metrics,
        "methods": methods,
        "top_methods": top_methods,
        "significance": significance,
        "screening_caution": (
            "Top methods are selected on pooled OOF labels and should be treated as diagnostics; "
            "promising formulas need a fold-level evaluator with validation-side selection."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)

    if args.dump_oof:
        dump_path = Path(args.dump_oof)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **dump)

    print(f"[nli_source_pooling] -> {out_path}", flush=True)
    for name in top_methods[:10]:
        print(name, methods[name]["metrics"], flush=True)


if __name__ == "__main__":
    main()
