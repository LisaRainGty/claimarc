"""Fold-safe selector for evidence-type score/decision adapters.

This is a constrained alternative to a learned meta-head.  For each repeated
CV case and held-out fold, it selects one small, interpretable adapter from
predefined source/evidence masks using only the other folds, then applies the
selected adapter to the held-out fold.
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
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"

DEFAULT_OOF = "data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz"


def method_arrays(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def masks(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    source_count = np.asarray(z["source_count"], int)
    source_bin = np.asarray(z["source_bin"], dtype=object).astype(str)
    confidence = np.asarray(z["confidence"], dtype=object).astype(str)
    combo = np.asarray(z["evidence_combo"], dtype=object).astype(str)
    bge_p, _ = method_arrays(z, BGE)
    no_vlm = np.asarray(["V" not in x for x in combo], bool)
    medium = confidence == "medium"
    lowabs = np.abs(bge_p - 0.5) < 0.25
    po_medium = (combo == "PO") & medium
    out = {
        "none": np.zeros(len(source_count), bool),
        "src0": source_count == 0,
        "po_medium": po_medium,
        "no_vlm_medium": no_vlm & medium,
        "src0_or_po_medium": (source_count == 0) | po_medium,
        "src0_or_no_vlm_medium": (source_count == 0) | (no_vlm & medium),
        "src0_or_medium": (source_count == 0) | medium,
        "source_rich_medium": (source_count >= 2) & medium,
        "src2_3_medium": (source_bin == "src2_3") & medium,
        "po_medium_lowabs": po_medium & lowabs,
        "no_vlm_medium_lowabs": no_vlm & medium & lowabs,
    }
    return out


def synthesize(
    p_current: np.ndarray,
    y_current: np.ndarray,
    p_adapt: np.ndarray,
    y_adapt: np.ndarray,
    score_mask: np.ndarray,
    decision_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = p_current.copy()
    yhat = y_current.copy()
    p[score_mask] = p_adapt[score_mask]
    yhat[decision_mask] = y_adapt[decision_mask]
    return p, yhat


def metric_sum(metric: dict[str, float], objective: str) -> float:
    if objective == "balanced":
        return float(metric["auprc"] + metric["auroc"] + metric["macro_f1"])
    if objective == "macro":
        return float(metric["macro_f1"] + 0.05 * metric["auprc"])
    if objective == "ranking":
        return float(metric["auprc"] + 0.50 * metric["auroc"] + 0.10 * metric["macro_f1"])
    raise ValueError(f"unknown objective: {objective}")


def candidate_specs() -> list[tuple[str, str]]:
    score_rules = [
        "none",
        "src0",
        "src0_or_po_medium",
        "src0_or_no_vlm_medium",
        "src0_or_medium",
        "source_rich_medium",
        "src2_3_medium",
    ]
    decision_rules = [
        "none",
        "po_medium",
        "no_vlm_medium",
        "po_medium_lowabs",
        "no_vlm_medium_lowabs",
        "src2_3_medium",
    ]
    return [(s, d) for s in score_rules for d in decision_rules]


def select_one(
    y: np.ndarray,
    c: np.ndarray,
    p_current: np.ndarray,
    y_current: np.ndarray,
    p_adapt: np.ndarray,
    y_adapt: np.ndarray,
    mask_map: dict[str, np.ndarray],
    train: np.ndarray,
    objective: str,
    min_gain: float,
) -> tuple[str, str, dict[str, object]]:
    base_metric = row(y[train], p_current[train], y_current[train], c[train])
    base_score = metric_sum(base_metric, objective)
    best = {
        "score_rule": "none",
        "decision_rule": "none",
        "metric": base_metric,
        "score": base_score,
        "gain": 0.0,
        "score_mask_n": 0,
        "decision_mask_n": 0,
    }
    for score_rule, decision_rule in candidate_specs():
        p, yhat = synthesize(
            p_current,
            y_current,
            p_adapt,
            y_adapt,
            mask_map[score_rule],
            mask_map[decision_rule],
        )
        metric = row(y[train], p[train], yhat[train], c[train])
        score = metric_sum(metric, objective)
        gain = score - base_score
        score_mask_n = int((mask_map[score_rule] & train).sum())
        decision_mask_n = int((mask_map[decision_rule] & train).sum())
        complexity = (score_rule != "none") + (decision_rule != "none")
        key = (
            score - 0.0005 * complexity,
            metric["macro_f1"],
            metric["auprc"],
            -complexity,
        )
        best_key = (
            best["score"] - 0.0005 * ((best["score_rule"] != "none") + (best["decision_rule"] != "none")),
            best["metric"]["macro_f1"],
            best["metric"]["auprc"],
            -((best["score_rule"] != "none") + (best["decision_rule"] != "none")),
        )
        if gain >= float(min_gain) and key > best_key:
            best = {
                "score_rule": score_rule,
                "decision_rule": decision_rule,
                "metric": metric,
                "score": score,
                "gain": gain,
                "score_mask_n": score_mask_n,
                "decision_mask_n": decision_mask_n,
            }
    return best["score_rule"], best["decision_rule"], best


def crossfit_select(
    z: np.lib.npyio.NpzFile,
    objective: str,
    min_gain: float,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    folds = np.asarray(z["fold"], int)
    cases = np.asarray(z["case"], dtype=object).astype(str)
    p_current, y_current = method_arrays(z, CURRENT)
    p_adapt, y_adapt = method_arrays(z, ADAPTIVE)
    mask_map = masks(z)
    name = f"evtype_cvselect_{objective}_mingain{str(min_gain).replace('.', 'p')}"
    out = {name: {"p": np.full(len(y), np.nan), "yhat": np.full(len(y), np.nan)}}
    fold_meta = []
    for case in sorted(set(cases.tolist())):
        cm = cases == case
        for fold in sorted(set(folds[cm].tolist())):
            te = cm & (folds == fold)
            tr = cm & (folds != fold)
            score_rule, decision_rule, meta = select_one(
                y, c, p_current, y_current, p_adapt, y_adapt,
                mask_map, tr, objective, min_gain)
            p, yhat = synthesize(
                p_current,
                y_current,
                p_adapt,
                y_adapt,
                mask_map[score_rule],
                mask_map[decision_rule],
            )
            out[name]["p"][te] = p[te]
            out[name]["yhat"][te] = yhat[te]
            fold_meta.append({
                "case": case,
                "fold": int(fold),
                "selected_score_rule": score_rule,
                "selected_decision_rule": decision_rule,
                "train_metric": meta["metric"],
                "train_gain": round(float(meta["gain"]), 6),
                "train_score_mask_n": int(meta["score_mask_n"]),
                "train_decision_mask_n": int(meta["decision_mask_n"]),
                "test_score_mask_n": int((mask_map[score_rule] & te).sum()),
                "test_decision_mask_n": int((mask_map[decision_rule] & te).sum()),
            })
            print(
                f"[evtype_select] case={case} fold={fold} "
                f"score={score_rule} decision={decision_rule}",
                flush=True,
            )
    return out, fold_meta


def fixed_adapter(z: np.lib.npyio.NpzFile) -> dict[str, dict[str, np.ndarray]]:
    p_current, y_current = method_arrays(z, CURRENT)
    p_adapt, y_adapt = method_arrays(z, ADAPTIVE)
    mask_map = masks(z)
    fixed = {}
    specs = {
        "evtype_fixed_src0_po_medium__po_medium": (
            "src0_or_po_medium", "po_medium"),
        "evtype_fixed_src0_no_vlm_medium__no_vlm_medium": (
            "src0_or_no_vlm_medium", "no_vlm_medium"),
    }
    for name, (score_rule, decision_rule) in specs.items():
        p, yhat = synthesize(
            p_current,
            y_current,
            p_adapt,
            y_adapt,
            mask_map[score_rule],
            mask_map[decision_rule],
        )
        fixed[name] = {"p": p, "yhat": yhat}
    return fixed


def evaluate(
    z: np.lib.npyio.NpzFile,
    built: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    baselines = [BGE, CMBGE, CURRENT, EVTYPE]
    rows = {}
    for method in baselines:
        p, yhat = method_arrays(z, method)
        rows[method] = row(y, p, yhat, c)
    for method, item in built.items():
        rows[method] = row(y, item["p"], item["yhat"].astype(int), c)
    sig = {}
    if n_boot > 0:
        for mi, method in enumerate(built):
            p_a = built[method]["p"]
            y_a = built[method]["yhat"].astype(int)
            for bi, base in enumerate(baselines):
                p_b, y_b = method_arrays(z, base)
                sig[f"{method}_vs_{base}"] = paired_bootstrap(
                    y, p_a, y_a, p_b, y_b,
                    n_boot=n_boot,
                    seed=seed + mi * 131 + bi * 19,
                )
    return rows, sig


def dump_oof(path: Path, z: np.lib.npyio.NpzFile,
             built: dict[str, dict[str, np.ndarray]]) -> None:
    arrays = {k: np.asarray(z[k]) for k in (
        "y", "c", "fold", "case", "pair_id", "source_count",
        "source_bin", "category", "confidence", "evidence_combo",
    ) if k in z.files}
    for method in (BGE, CMBGE, CURRENT, EVTYPE):
        arrays[f"{method}__p"] = np.asarray(z[f"{method}__p"])
        arrays[f"{method}__yhat"] = np.asarray(z[f"{method}__yhat"])
    for method, item in built.items():
        arrays[f"{method}__p"] = item["p"]
        arrays[f"{method}__yhat"] = item["yhat"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=DEFAULT_OOF)
    ap.add_argument("--objective", choices=["balanced", "macro", "ranking"],
                    default="balanced")
    ap.add_argument("--min_gain", type=float, default=0.0)
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--out", default="data/final/cleancl/evidence_type_selector_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_evidence_type_selector_20260609.npz")
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    selected, fold_meta = crossfit_select(z, args.objective, args.min_gain)
    built = {**selected, **fixed_adapter(z)}
    rows, sig = evaluate(z, built, args.n_boot, args.seed)
    ranked = sorted(
        built,
        key=lambda m: (rows[m]["auprc"], rows[m]["auroc"], rows[m]["macro_f1"]),
        reverse=True,
    )
    out = {
        "description": (
            "Fold-safe constrained evidence-type adapter selector. The selector "
            "chooses only among predefined source/evidence masks on training folds."
        ),
        "oof": args.oof,
        "objective": args.objective,
        "min_gain": float(args.min_gain),
        "fold_meta": fold_meta,
        "ranked_methods": ranked,
        "metrics": rows,
        "n_boot": int(args.n_boot),
        "significance": sig,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    if args.dump_oof:
        dump_oof(Path(args.dump_oof), z, built)
    print(f"[cv_evidence_type_selector] -> {path}", flush=True)
    for method in ranked:
        r = rows[method]
        print(f"{method:64s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
