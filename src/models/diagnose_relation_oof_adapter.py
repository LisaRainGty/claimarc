"""Group-safe OOF relation adapter diagnostics.

This script tests whether observable evidence-relation metadata can learn when
to trust BGE, CM/NLI, or evidence-type scores. It is deliberately post-hoc:
it reads saved OOF predictions and cross-fits a small second-level model by
``pair_id`` groups, so duplicate rows from fs0/fs1/fs2 for the same pair are
always held out together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

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
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"

BASELINES = ["bge_lr", CMBGE, CURRENT, ADAPTIVE, EVTYPE, TAXONOMY]


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def macro(y, pred, w=None):
    return f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)


def best_threshold_macro(y, p):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        val = macro(y, (p >= t).astype(int))
        if val > best:
            best_t, best = float(t), float(val)
    return best_t, best


def one_hot(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    levels = sorted({str(x) for x in values})
    mat = np.column_stack([(values.astype(str) == level).astype(float) for level in levels])
    return mat, levels


def build_features(
    z: np.lib.npyio.NpzFile,
    *,
    include_category: bool = False,
    include_taxonomy_prob: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    p_bge, _ = get_method(z, "bge_lr")
    p_cmbge, _ = get_method(z, CMBGE)
    p_current, _ = get_method(z, CURRENT)
    p_adapt, _ = get_method(z, ADAPTIVE)
    p_evtype, _ = get_method(z, EVTYPE)
    probs = [
        ("p_bge", p_bge),
        ("p_cmbge", p_cmbge),
        ("p_current", p_current),
        ("p_adaptive", p_adapt),
        ("p_evtype", p_evtype),
    ]
    if include_taxonomy_prob:
        p_tax, _ = get_method(z, TAXONOMY)
        probs.append(("p_taxonomy", p_tax))

    source_count = np.asarray(z["source_count"], float)
    numeric = probs + [
        ("d_current_bge", p_current - p_bge),
        ("d_adaptive_current", p_adapt - p_current),
        ("d_evtype_current", p_evtype - p_current),
        ("d_cmbge_bge", p_cmbge - p_bge),
        ("unc_bge", np.abs(p_bge - 0.5)),
        ("unc_current", np.abs(p_current - 0.5)),
        ("unc_adaptive", np.abs(p_adapt - 0.5)),
        ("unc_evtype", np.abs(p_evtype - 0.5)),
        ("source_count_clip10", np.clip(source_count, 0, 10) / 10.0),
        ("source_count_eq0", (source_count == 0).astype(float)),
        ("source_count_ge2", (source_count >= 2).astype(float)),
    ]
    mats = [np.asarray(v, float)[:, None] for _, v in numeric]
    names = [name for name, _ in numeric]
    cat_info = {}
    for field in ["source_bin", "confidence", "evidence_combo"] + (
        ["category"] if include_category else []
    ):
        mat, levels = one_hot(np.asarray(z[field], dtype=object))
        mats.append(mat)
        names.extend([f"{field}={level}" for level in levels])
        cat_info[field] = levels
    return np.concatenate(mats, axis=1), names, cat_info


def inner_train_val(y, groups, train_idx, seed):
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=seed)
    for fit_rel, val_rel in splitter.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]):
        fit_idx = train_idx[fit_rel]
        val_idx = train_idx[val_rel]
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx
    # Fallback: use the first split even if the validation side is imperfect.
    fit_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 999)
        .split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx])
    )
    return train_idx[fit_rel], train_idx[val_rel]


def fit_predict_crossfit(
    X,
    y,
    c,
    groups,
    *,
    model_kind: str,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    unique_groups = np.unique(groups)
    n_splits = min(int(n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    p = np.full(len(y), np.nan, dtype=float)
    yhat = np.full(len(y), -1, dtype=int)
    folds = []
    for fi, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        fit_idx, val_idx = inner_train_val(y, groups, train_idx, seed + fi)
        if model_kind == "lr":
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X[fit_idx])
            X_val = scaler.transform(X[val_idx])
            X_test = scaler.transform(X[test_idx])
            model = LogisticRegression(C=0.25, max_iter=3000, class_weight=None)
            model.fit(X_fit, y[fit_idx], sample_weight=np.clip(c[fit_idx], 0.05, None))
        elif model_kind == "hgb":
            scaler = None
            X_fit, X_val, X_test = X[fit_idx], X[val_idx], X[test_idx]
            model = HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=7,
                l2_regularization=0.1,
                random_state=seed + fi,
            )
            model.fit(X_fit, y[fit_idx], sample_weight=np.clip(c[fit_idx], 0.05, None))
        else:
            raise KeyError(model_kind)

        pv = model.predict_proba(X_val)[:, 1]
        thr, val_macro = best_threshold_macro(y[val_idx], pv)
        pt = model.predict_proba(X_test)[:, 1]
        p[test_idx] = pt
        yhat[test_idx] = (pt >= thr).astype(int)
        folds.append({
            "fold": int(fi),
            "n_fit": int(len(fit_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_test_groups": int(len(np.unique(groups[test_idx]))),
            "threshold": round(float(thr), 4),
            "val_macro": round(float(val_macro), 4),
            "val_ap": round(float(average_precision_score(y[val_idx], pv)), 4)
            if len(np.unique(y[val_idx])) > 1 else None,
            "val_auroc": round(float(roc_auc_score(y[val_idx], pv)), 4)
            if len(np.unique(y[val_idx])) > 1 else None,
        })
    return p, yhat, folds


def grouped_paired_bootstrap(
    groups,
    y,
    p_a,
    yhat_a,
    p_b,
    yhat_b,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
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
        df1.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))

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

    return {
        "dAP": summarize(dap),
        "dAUROC": summarize(dau),
        "dMacroF1": summarize(df1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz")
    ap.add_argument("--out", default="data/final/cleancl/relation_oof_adapter_screen_20260608.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_relation_adapter_screen_20260608.npz")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260608)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    pair_id = np.asarray(z["pair_id"], dtype=object)

    specs = {
        "relation_lr_no_category": {
            "model": "lr",
            "include_category": False,
            "include_taxonomy_prob": False,
        },
        "relation_hgb_no_category": {
            "model": "hgb",
            "include_category": False,
            "include_taxonomy_prob": False,
        },
        "relation_lr_with_category_diag": {
            "model": "lr",
            "include_category": True,
            "include_taxonomy_prob": True,
        },
    }

    built = {}
    for name, spec in specs.items():
        X, feature_names, cat_info = build_features(
            z,
            include_category=bool(spec["include_category"]),
            include_taxonomy_prob=bool(spec["include_taxonomy_prob"]),
        )
        p, yhat, folds = fit_predict_crossfit(
            X,
            y,
            c,
            pair_id,
            model_kind=str(spec["model"]),
            n_splits=args.n_splits,
            seed=args.seed,
        )
        built[name] = {
            "spec": spec,
            "metrics": row(y, p, yhat, c),
            "folds": folds,
            "feature_count": int(X.shape[1]),
            "feature_names": feature_names,
            "categorical_levels": cat_info,
            "p": p,
            "yhat": yhat,
        }

    baseline_metrics = {}
    baseline_arrays = {}
    for method in BASELINES:
        p, yhat = get_method(z, method)
        baseline_metrics[method] = row(y, p, yhat, c)
        baseline_arrays[method] = (p, yhat)

    significance = {"sample_bootstrap": {}, "group_bootstrap": {}}
    for name, info in built.items():
        p_a, yhat_a = info["p"], info["yhat"]
        for baseline, (p_b, yhat_b) in baseline_arrays.items():
            significance["sample_bootstrap"][f"{name}_vs_{baseline}"] = paired_bootstrap(
                y, p_a, yhat_a, p_b, yhat_b, n_boot=args.n_boot, seed=args.seed)
            significance["group_bootstrap"][f"{name}_vs_{baseline}"] = grouped_paired_bootstrap(
                pair_id, y, p_a, yhat_a, p_b, yhat_b,
                n_boot=args.n_boot, seed=args.seed + 17)

    out = {
        "description": (
            "Pair-id grouped cross-fit second-level relation adapter over saved OOF predictions. "
            "This is a diagnostic meta-learner, not yet a fold-level evaluator protocol."
        ),
        "oof": args.oof,
        "n": int(len(y)),
        "n_pair_id": int(len(set(pair_id))),
        "n_splits": int(args.n_splits),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "baseline_metrics": baseline_metrics,
        "methods": {
            name: {k: v for k, v in info.items() if k not in ("p", "yhat")}
            for name, info in built.items()
        },
        "significance": significance,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)

    if args.dump_oof:
        dump = {
            "y": y,
            "c": c,
            "pair_id": pair_id,
            "fold": np.asarray(z["fold"], int),
            "case": np.asarray(z["case"], dtype=object),
        }
        for method, (p, yhat) in baseline_arrays.items():
            dump[f"{method}__p"] = p
            dump[f"{method}__yhat"] = yhat
        for name, info in built.items():
            dump[f"{name}__p"] = info["p"]
            dump[f"{name}__yhat"] = info["yhat"]
        dump_path = Path(args.dump_oof)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **dump)

    print(f"[relation_oof_adapter] -> {out_path}", flush=True)
    for name, info in built.items():
        print(f"{name}: {info['metrics']}", flush=True)


if __name__ == "__main__":
    main()
