"""Enumerate common saved OOF methods across fold_seed screens.

This is a post-hoc audit utility. It reads OOF probabilities and decisions
already emitted by ``cv_nli_predef_lowabs.py`` and ranks every method that is
present in all supplied splits. It does not train, refit, or tune thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from models.bootstrap_oof_methods import paired_bootstrap, row
except ModuleNotFoundError:
    from bootstrap_oof_methods import paired_bootstrap, row


CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
ADAPTIVE = (
    "rankw_sourcefirst_cm040_nli060_score_src0ormedium_cmreinforce025_"
    "decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_"
    "srcge2_lowmedium_cmbgeprotect_src4pmedium_cmbgenli"
)
TAXONOMY = "rankw_sourcefirst_cm040_nli060_score_sportsgeneral_cm025_decision_sports_cm025"

DEFAULT_CASES = {
    "fs0": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_adaptive_quick.npz",
    "fs1": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_adaptive_quick.npz",
    "fs2": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_adaptive_quick.npz",
}

DEFAULT_BASELINES = [
    "bge_lr",
    "rankavg_sourcefirst_cm_pcls_bge",
    CURRENT,
]


def parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.stem, path
    label, path = text.split("=", 1)
    return label, Path(path)


def load_oof(path: Path) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
    methods = {
        key[:-3]
        for key in z.files
        if key.endswith("__p") and f"{key[:-3]}__yhat" in z.files
    }
    return {"path": str(path), "z": z, "y": y, "c": c, "methods": methods}


def get_method(oof: dict[str, object], method: str) -> tuple[np.ndarray, np.ndarray]:
    z = oof["z"]
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def metric_sum(metrics: dict[str, float]) -> float:
    return float(metrics["auprc"] + metrics["auroc"] + metrics["macro_f1"])


def evaluate_method(cases: list[tuple[str, dict[str, object]]], method: str) -> dict[str, object]:
    y_all, c_all, p_all, yhat_all = [], [], [], []
    case_metrics = {}
    for label, oof in cases:
        y = oof["y"]
        c = oof["c"]
        p, yhat = get_method(oof, method)
        ok = ~np.isnan(p)
        case_metrics[label] = row(y[ok], p[ok], yhat[ok], c[ok])
        y_all.append(y[ok])
        c_all.append(c[ok])
        p_all.append(p[ok])
        yhat_all.append(yhat[ok])
    pooled = row(
        np.concatenate(y_all),
        np.concatenate(p_all),
        np.concatenate(yhat_all),
        np.concatenate(c_all),
    )
    return {
        "pooled": pooled,
        "case_metrics": case_metrics,
        "pooled_sum": round(metric_sum(pooled), 4),
        "fs1_sum": round(metric_sum(case_metrics["fs1"]), 4)
        if "fs1" in case_metrics else None,
    }


def rank_entries(
    metrics: dict[str, dict[str, object]],
    metric: str,
    scope: str,
    topn: int,
) -> list[dict[str, object]]:
    def score(item):
        method, info = item
        if scope == "pooled":
            return (float(info["pooled"][metric]), method)
        return (float(info["case_metrics"][scope][metric]), method)

    out = []
    for method, info in sorted(metrics.items(), key=score, reverse=True)[:topn]:
        item_metrics = info["pooled"] if scope == "pooled" else info["case_metrics"][scope]
        out.append({
            "method": method,
            "metrics": item_metrics,
            "pooled": info["pooled"],
            "case_metrics": info["case_metrics"],
        })
    return out


def collect_compare_methods(
    rankings: dict[str, list[dict[str, object]]],
    explicit: list[str] | None,
) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    selected = [CURRENT, ADAPTIVE, TAXONOMY]
    for entries in rankings.values():
        selected.extend(item["method"] for item in entries[:5])
    return list(dict.fromkeys(selected))


def bootstrap_compare(
    cases: list[tuple[str, dict[str, object]]],
    method: str,
    baseline: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    y_all, p_a_all, yhat_a_all, p_b_all, yhat_b_all = [], [], [], [], []
    for _, oof in cases:
        y = np.asarray(oof["y"], int)
        p_a, yhat_a = get_method(oof, method)
        p_b, yhat_b = get_method(oof, baseline)
        ok = (~np.isnan(p_a)) & (~np.isnan(p_b))
        y_all.append(y[ok])
        p_a_all.append(p_a[ok])
        yhat_a_all.append(yhat_a[ok])
        p_b_all.append(p_b[ok])
        yhat_b_all.append(yhat_b[ok])
    return paired_bootstrap(
        np.concatenate(y_all),
        np.concatenate(p_a_all),
        np.concatenate(yhat_a_all),
        np.concatenate(p_b_all),
        np.concatenate(yhat_b_all),
        n_boot=n_boot,
        seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append",
                    help="label=path.npz; defaults to fs0/fs1/fs2 adaptive OOF")
    ap.add_argument("--baseline", action="append", default=[],
                    help="Baseline method for bootstrap comparisons")
    ap.add_argument("--compare_method", action="append", default=[],
                    help="Method to compare against each baseline")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=(
        "data/final/cleancl/common_oof_method_sweep_20260608.json"))
    args = ap.parse_args()

    case_specs = (
        [parse_case(text) for text in args.case]
        if args.case else
        [(label, Path(path)) for label, path in DEFAULT_CASES.items()]
    )
    cases = [(label, load_oof(path)) for label, path in case_specs]
    common = set.intersection(*(oof["methods"] for _, oof in cases))
    metrics = {method: evaluate_method(cases, method) for method in sorted(common)}

    rankings = {
        "pooled_auprc": rank_entries(metrics, "auprc", "pooled", args.topn),
        "pooled_auroc": rank_entries(metrics, "auroc", "pooled", args.topn),
        "pooled_macro_f1": rank_entries(metrics, "macro_f1", "pooled", args.topn),
        "fs1_auprc": rank_entries(metrics, "auprc", "fs1", args.topn),
        "fs1_auroc": rank_entries(metrics, "auroc", "fs1", args.topn),
        "fs1_macro_f1": rank_entries(metrics, "macro_f1", "fs1", args.topn),
    }

    baselines = list(dict.fromkeys(args.baseline or DEFAULT_BASELINES))
    compare_methods = [
        m for m in collect_compare_methods(rankings, args.compare_method) if m in common
    ]
    significance = {}
    for i, method in enumerate(compare_methods):
        for j, baseline in enumerate(baselines):
            if method == baseline or baseline not in common:
                continue
            significance[f"{method}_vs_{baseline}"] = bootstrap_compare(
                cases,
                method,
                baseline,
                n_boot=args.n_boot,
                seed=args.seed + 1000 + i * 37 + j,
            )

    out = {
        "description": (
            "Post-hoc enumeration of every method present in all supplied OOF files. "
            "Metrics use saved probabilities and saved decisions only."
        ),
        "n_common_methods": len(common),
        "cases": {label: {"path": oof["path"], "n": int(len(oof["y"]))}
                  for label, oof in cases},
        "baselines": baselines,
        "compare_methods": compare_methods,
        "n_boot": int(args.n_boot),
        "rankings": rankings,
        "selected_metrics": {method: metrics[method] for method in compare_methods},
        "all_metrics": metrics,
        "significance": significance,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[common_oof_methods] common={len(common)} -> {out_path}", flush=True)
    for key in ("pooled_auprc", "pooled_auroc", "pooled_macro_f1", "fs1_auroc"):
        best = rankings[key][0]
        print(f"[{key}] {best['method']} {best['metrics']}", flush=True)


if __name__ == "__main__":
    main()
