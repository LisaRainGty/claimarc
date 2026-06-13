"""OOF screen for validation-safe adaptive/taxonomy adapters.

This script reads fold-safe OOF predictions plus the fold metadata saved by
``cv_nli_predef_lowabs.py``. It then builds a few small selectors that choose
score and decision heads from validation metrics inside each outer fold.

The goal is to test whether the strong post-hoc taxonomy-aware candidate can
be recovered by a defensible fold-level adapter. It does not train or tune any
model; promising rules still need to be implemented in the evaluator if they
are to become paper claims.
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
    "fs0": {
        "oof": "data/final/cleancl/"
        "oof_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_adaptive_quick.npz",
        "json": "data/final/cleancl/"
        "cv_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_adaptive_quick.json",
    },
    "fs1": {
        "oof": "data/final/cleancl/"
        "oof_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_adaptive_quick.npz",
        "json": "data/final/cleancl/"
        "cv_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_adaptive_quick.json",
    },
    "fs2": {
        "oof": "data/final/cleancl/"
        "oof_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_adaptive_quick.npz",
        "json": "data/final/cleancl/"
        "cv_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_adaptive_quick.json",
    },
}


BASELINES = [
    "bge_lr",
    "rankavg_sourcefirst_cm_pcls_bge",
    CURRENT,
]


def method_key(method: str, suffix: str) -> str:
    return f"{method}__{suffix}"


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[method_key(method, "p")], float),
        np.asarray(z[method_key(method, "yhat")], int),
    )


def val_metric(fold_meta: dict, method: str, mode: str) -> float:
    select = fold_meta[method]["select"]
    if mode == "ap":
        return float(select["ap"])
    if mode == "auroc":
        return float(select["auroc"])
    if mode == "macro":
        return float(select["macro_f1"])
    if mode == "ap_au":
        return 0.5 * float(select["ap"]) + 0.5 * float(select["auroc"])
    if mode == "all":
        return (
            float(select["ap"])
            + float(select["auroc"])
            + float(select["macro_f1"])
        )
    raise KeyError(mode)


def choose_method(
    fold_meta: dict,
    pool: list[str],
    mode: str,
    margin: float,
    default: str = CURRENT,
    min_tax_score_val_n: int = 20,
) -> str:
    best = default
    best_val = val_metric(fold_meta, default, mode)
    for method in pool:
        if method == TAXONOMY:
            if int(fold_meta[method].get("score_mask_val_n", 0)) < min_tax_score_val_n:
                continue
        val = val_metric(fold_meta, method, mode)
        if val > best_val + float(margin):
            best = method
            best_val = val
    return best


SELECTOR_SPECS = {
    "selector_adaptive_score_valauroc0_decision_valall010": {
        "score_pool": [CURRENT, ADAPTIVE],
        "score_mode": "auroc",
        "score_margin": 0.0,
        "decision_pool": [CURRENT, ADAPTIVE],
        "decision_mode": "all",
        "decision_margin": 0.010,
    },
    "selector_adaptive_score_valauroc001_decision_valall010": {
        "score_pool": [CURRENT, ADAPTIVE],
        "score_mode": "auroc",
        "score_margin": 0.001,
        "decision_pool": [CURRENT, ADAPTIVE],
        "decision_mode": "all",
        "decision_margin": 0.010,
    },
    "selector_adaptive_tax_score_valauroc0_decision_valall010": {
        "score_pool": [CURRENT, ADAPTIVE, TAXONOMY],
        "score_mode": "auroc",
        "score_margin": 0.0,
        "decision_pool": [CURRENT, ADAPTIVE, TAXONOMY],
        "decision_mode": "all",
        "decision_margin": 0.010,
    },
    "selector_tax_score_valap0_adapt_decision_valall010": {
        "score_pool": [CURRENT, TAXONOMY],
        "score_mode": "ap",
        "score_margin": 0.0,
        "decision_pool": [CURRENT, ADAPTIVE],
        "decision_mode": "all",
        "decision_margin": 0.010,
    },
}


def load_cases(root: Path) -> list[tuple[str, np.lib.npyio.NpzFile, dict]]:
    cases = []
    for label, spec in DEFAULT_CASES.items():
        oof = np.load(root / spec["oof"], allow_pickle=True)
        meta = json.load(open(root / spec["json"]))
        cases.append((label, oof, meta))
    return cases


def synthesize_case(
    z: np.lib.npyio.NpzFile,
    meta: dict,
    spec: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    fold = np.asarray(z["fold"], int)
    p_out = np.full(len(fold), np.nan, dtype=float)
    yhat_out = np.full(len(fold), np.nan, dtype=float)
    choices = []
    for fi, fold_meta in enumerate(meta["fold_meta"]):
        score_method = choose_method(
            fold_meta,
            spec["score_pool"],
            spec["score_mode"],
            spec["score_margin"],
        )
        decision_method = choose_method(
            fold_meta,
            spec["decision_pool"],
            spec["decision_mode"],
            spec["decision_margin"],
        )
        mask = fold == fi
        p, _ = get_method(z, score_method)
        _, yhat = get_method(z, decision_method)
        p_out[mask] = p[mask]
        yhat_out[mask] = yhat[mask]
        choices.append({
            "fold": int(fi),
            "score_method": score_method,
            "decision_method": decision_method,
            "score_select": fold_meta[score_method]["select"],
            "decision_select": fold_meta[decision_method]["select"],
        })
    return p_out, np.asarray(yhat_out, int), choices


def build_method_arrays(cases, method: str):
    y_all, c_all, p_all, yhat_all = [], [], [], []
    case_metrics = {}
    for label, z, _ in cases:
        y = np.asarray(z["y"], int)
        c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
        p, yhat = get_method(z, method)
        ok = ~np.isnan(p)
        case_metrics[label] = row(y[ok], p[ok], yhat[ok], c[ok])
        y_all.append(y)
        c_all.append(c)
        p_all.append(p)
        yhat_all.append(yhat)
    return (
        case_metrics,
        np.concatenate(y_all),
        np.concatenate(c_all),
        np.concatenate(p_all),
        np.concatenate(yhat_all),
    )


def build_selector_arrays(cases, name: str, spec: dict):
    y_all, c_all, p_all, yhat_all = [], [], [], []
    case_metrics = {}
    choices = {}
    for label, z, meta in cases:
        y = np.asarray(z["y"], int)
        c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
        p, yhat, case_choices = synthesize_case(z, meta, spec)
        ok = ~np.isnan(p)
        case_metrics[label] = row(y[ok], p[ok], yhat[ok], c[ok])
        choices[label] = case_choices
        y_all.append(y)
        c_all.append(c)
        p_all.append(p)
        yhat_all.append(yhat)
    return {
        "name": name,
        "spec": spec,
        "case_metrics": case_metrics,
        "choices": choices,
        "y": np.concatenate(y_all),
        "c": np.concatenate(c_all),
        "p": np.concatenate(p_all),
        "yhat": np.concatenate(yhat_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    cases = load_cases(root)
    out = {
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "baselines": BASELINES,
        "fixed_methods": [ADAPTIVE, TAXONOMY],
        "selectors": {},
        "pooled": {"metrics": {}, "significance": {}},
        "cases": {},
    }

    arrays = {}
    for method in BASELINES + [ADAPTIVE, TAXONOMY]:
        case_metrics, y, c, p, yhat = build_method_arrays(cases, method)
        arrays[method] = {"y": y, "c": c, "p": p, "yhat": yhat}
        out["pooled"]["metrics"][method] = row(y, p, yhat, c)
        for label, metrics in case_metrics.items():
            out["cases"].setdefault(label, {"metrics": {}, "choices": {}})
            out["cases"][label]["metrics"][method] = metrics

    selector_arrays = {}
    for name, spec in SELECTOR_SPECS.items():
        item = build_selector_arrays(cases, name, spec)
        selector_arrays[name] = item
        out["selectors"][name] = {
            "spec": {
                k: ([m for m in v] if isinstance(v, list) else v)
                for k, v in spec.items()
            },
            "choices": item["choices"],
        }
        out["pooled"]["metrics"][name] = row(
            item["y"], item["p"], item["yhat"], item["c"])
        for label, metrics in item["case_metrics"].items():
            out["cases"].setdefault(label, {"metrics": {}, "choices": {}})
            out["cases"][label]["metrics"][name] = metrics
            out["cases"][label]["choices"][name] = item["choices"][label]

    compare_names = [ADAPTIVE, TAXONOMY] + list(SELECTOR_SPECS)
    for mi, method in enumerate(compare_names):
        if method in selector_arrays:
            a = selector_arrays[method]
        else:
            a = arrays[method]
        for bi, baseline in enumerate(BASELINES):
            b = arrays[baseline]
            ok = (~np.isnan(a["p"])) & (~np.isnan(b["p"]))
            out["pooled"]["significance"][f"{method}_vs_{baseline}"] = (
                paired_bootstrap(
                    a["y"][ok],
                    a["p"][ok],
                    a["yhat"][ok],
                    b["p"][ok],
                    b["yhat"][ok],
                    n_boot=args.n_boot,
                    seed=args.seed + 1000 + mi * 19 + bi,
                )
            )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[diagnose_taxonomy_adapter_oof] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
