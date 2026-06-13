"""Small OOF screen for evidence-sufficiency fallback rules.

This script composes a tiny set of interpretable source/evidence masks on top
of the evidence-type adapter.  It is a diagnostic screen only; any promising
rule must be implemented in the fold-level evaluator before being considered
as a paper method.
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
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[f"{method}__p"], float), np.asarray(z[f"{method}__yhat"], int)


def rank01(x):
    x = np.asarray(x, float)
    if len(x) <= 1:
        return np.zeros(len(x), float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    out[order] = np.arange(len(x), dtype=float)
    return out / float(len(x) - 1)


def mask_for(z, name):
    src = np.asarray(z["source_bin"], dtype=object).astype(str)
    conf = np.asarray(z["confidence"], dtype=object).astype(str)
    combo = np.asarray(z["evidence_combo"], dtype=object).astype(str)
    if name == "src2_3_medium":
        return (src == "src2_3") & (conf == "medium")
    if name == "src4p_high":
        return (src == "src4p") & (conf == "high")
    if name == "pov_high":
        return (combo == "POV") & (conf == "high")
    if name == "ov_medium":
        return (combo == "OV") & (conf == "medium")
    if name == "o_low":
        return (combo == "O") & (conf == "low")
    if name == "p_low":
        return (combo == "P") & (conf == "low")
    if name == "none_absent":
        return (combo == "none") & (conf == "absent")
    if name == "src2_3_medium_or_pov_high":
        return mask_for(z, "src2_3_medium") | mask_for(z, "pov_high")
    if name == "src2_3_medium_or_ov_medium":
        return mask_for(z, "src2_3_medium") | mask_for(z, "ov_medium")
    raise KeyError(name)


def synthesize(z, mask_name, score_ref, decision_ref, alpha):
    p_base, y_base = get_method(z, EVTYPE)
    p_ref, y_ref = get_method(z, decision_ref)
    if score_ref == EVTYPE:
        p_score_ref = p_base
    else:
        p_score_ref, _ = get_method(z, score_ref)
    m = mask_for(z, mask_name)
    p = p_base.copy()
    yhat = y_base.copy()
    if alpha >= 1.0:
        p[m] = p_score_ref[m]
    elif alpha > 0:
        # Use raw-score interpolation.  Mask-local ranks are misleading here
        # because they rescale a small stratum against the full OOF population.
        p[m] = (1.0 - alpha) * p_base[m] + alpha * p_score_ref[m]
    yhat[m] = y_ref[m]
    return p, yhat, int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz")
    ap.add_argument("--out", default="data/final/cleancl/evsuff_oof_rule_screen_20260608.json")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260608)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    refs = [BGE, CMBGE, CURRENT, EVTYPE]
    baseline = {name: get_method(z, name) for name in refs}
    baseline_metrics = {name: row(y, p, yhat, c) for name, (p, yhat) in baseline.items()}

    masks = [
        "src2_3_medium",
        "pov_high",
        "src4p_high",
        "ov_medium",
        "o_low",
        "p_low",
        "none_absent",
        "src2_3_medium_or_pov_high",
        "src2_3_medium_or_ov_medium",
    ]
    methods = {}
    for mask_name in masks:
        for ref in (BGE, CMBGE, CURRENT):
            for score_ref, alpha in ((EVTYPE, 0.0), (ref, 1.0), (ref, 0.25)):
                method = (
                    f"evtype_mask_{mask_name}_score_"
                    f"{score_ref.replace('_', '')}_a{int(alpha * 100):03d}_"
                    f"decision_{ref.replace('_', '')}"
                )
                p, yhat, n_mask = synthesize(z, mask_name, score_ref, ref, alpha)
                methods[method] = {
                    "mask": mask_name,
                    "score_ref": score_ref,
                    "score_alpha": alpha,
                    "decision_ref": ref,
                    "mask_n": n_mask,
                    "metrics": row(y, p, yhat, c),
                    "p": p,
                    "yhat": yhat,
                }

    ranked = sorted(
        methods,
        key=lambda m: (
            methods[m]["metrics"]["auprc"],
            methods[m]["metrics"]["auroc"],
            methods[m]["metrics"]["macro_f1"],
        ),
        reverse=True,
    )
    top = ranked[:12]
    sig = {}
    for method in top:
        p_a = methods[method]["p"]
        y_a = methods[method]["yhat"]
        sig[method] = {}
        for base in (BGE, CMBGE, CURRENT, EVTYPE):
            p_b, y_b = baseline[base]
            sig[method][base] = paired_bootstrap(
                y, p_a, y_a, p_b, y_b,
                n_boot=args.n_boot, seed=args.seed)

    out = {
        "description": (
            "Small post-hoc OOF screen for source/evidence fallback masks. "
            "Promising rules require fold-level evaluator verification."
        ),
        "oof": args.oof,
        "n_boot": int(args.n_boot),
        "baseline_metrics": baseline_metrics,
        "top_methods": top,
        "methods": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("p", "yhat")}
            for k, v in methods.items()
        },
        "significance": sig,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"[evsuff_oof_rules] -> {path}", flush=True)
    for method in top[:8]:
        print(method, methods[method]["metrics"], flush=True)


if __name__ == "__main__":
    main()
