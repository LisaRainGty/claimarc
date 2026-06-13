"""Residual diagnostics for the evidence-type adapter OOF bundle.

The goal is to find remaining systematic failure regions after the current
evidence-type adapter, without fitting a new model.  It summarizes false
positive/false negative rates and fixed/broken flips against selected
baselines across source/evidence strata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[f"{method}__p"], float), np.asarray(z[f"{method}__yhat"], int)


def metrics(y, p, yhat):
    out = {
        "n": int(len(y)),
        "pos": int(np.asarray(y, int).sum()),
        "pred_pos": int(np.asarray(yhat, int).sum()),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
    }
    if len(np.unique(y)) > 1:
        out["auprc"] = round(float(average_precision_score(y, p)), 4)
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
    else:
        out["auprc"] = None
        out["auroc"] = None
    neg = np.asarray(y) == 0
    pos = np.asarray(y) == 1
    out["fp_rate"] = round(float((np.asarray(yhat)[neg] == 1).mean()), 4) if neg.any() else None
    out["fn_rate"] = round(float((np.asarray(yhat)[pos] == 0).mean()), 4) if pos.any() else None
    return out


def group_key(z, fields):
    vals = []
    for f in fields:
        vals.append(np.asarray(z[f], dtype=object).astype(str))
    return np.asarray([":".join(parts) for parts in zip(*vals)], dtype=object)


def summarize_groups(z, y, p_ev, y_ev, refs, group_name, keys, *, min_n):
    rows = []
    for key in sorted(set(keys.tolist())):
        m = keys == key
        if int(m.sum()) < int(min_n):
            continue
        row = {
            "group": str(key),
            **metrics(y[m], p_ev[m], y_ev[m]),
        }
        for ref_name, (_, y_ref) in refs.items():
            fixed = int(((y_ev == y) & (y_ref != y) & m).sum())
            broken = int(((y_ev != y) & (y_ref == y) & m).sum())
            row[f"vs_{ref_name}_fixed"] = fixed
            row[f"vs_{ref_name}_broken"] = broken
            row[f"vs_{ref_name}_net"] = fixed - broken
        rows.append(row)
    rows.sort(
        key=lambda r: (
            -(r.get("vs_bge_lr_broken", 0) - r.get("vs_bge_lr_fixed", 0)),
            -r["n"],
        )
    )
    return {
        "grouping": group_name,
        "min_n": int(min_n),
        "rows": rows,
    }


def top_score_error_regions(z, y, p_ev, y_ev, refs, *, min_n):
    groupings = [
        ("case", ["case"]),
        ("source_bin", ["source_bin"]),
        ("confidence", ["confidence"]),
        ("evidence_combo", ["evidence_combo"]),
        ("source_bin_conf", ["source_bin", "confidence"]),
        ("combo_conf", ["evidence_combo", "confidence"]),
        ("case_combo_conf", ["case", "evidence_combo", "confidence"]),
        ("category_combo_conf", ["category", "evidence_combo", "confidence"]),
    ]
    return [
        summarize_groups(z, y, p_ev, y_ev, refs, name, group_key(z, fields), min_n=min_n)
        for name, fields in groupings
    ]


def pair_repeated_consistency(z, y, p_ev, y_ev):
    pair_id = np.asarray(z["pair_id"], dtype=object)
    rows = []
    for pid in sorted(set(pair_id.tolist())):
        idx = np.flatnonzero(pair_id == pid)
        if len(idx) <= 1:
            continue
        pred = y_ev[idx]
        score = p_ev[idx]
        if len(set(pred.tolist())) > 1 or float(score.max() - score.min()) > 0.05:
            rows.append({
                "pair_id": str(pid),
                "y": int(y[idx[0]]),
                "cases": [str(x) for x in np.asarray(z["case"], dtype=object)[idx]],
                "pred": [int(x) for x in pred],
                "score": [round(float(x), 4) for x in score],
                "source_bin": str(np.asarray(z["source_bin"], dtype=object)[idx[0]]),
                "confidence": str(np.asarray(z["confidence"], dtype=object)[idx[0]]),
                "evidence_combo": str(np.asarray(z["evidence_combo"], dtype=object)[idx[0]]),
                "category": str(np.asarray(z["category"], dtype=object)[idx[0]]),
            })
    return {
        "n_inconsistent_or_score_spread": len(rows),
        "top_by_score_spread": sorted(
            rows,
            key=lambda r: max(r["score"]) - min(r["score"]),
            reverse=True,
        )[:100],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz")
    ap.add_argument("--out", default="data/final/cleancl/evtype_residual_diagnosis_20260608.json")
    ap.add_argument("--min_n", type=int, default=25)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    p_ev, y_ev = get_method(z, EVTYPE)
    refs = {
        BGE: get_method(z, BGE),
        CMBGE: get_method(z, CMBGE),
        CURRENT: get_method(z, CURRENT),
    }
    out = {
        "oof": args.oof,
        "method": EVTYPE,
        "overall": metrics(y, p_ev, y_ev),
        "reference_metrics": {
            name: metrics(y, p, yhat)
            for name, (p, yhat) in refs.items()
        },
        "group_diagnostics": top_score_error_regions(
            z, y, p_ev, y_ev, refs, min_n=args.min_n),
        "pair_repeated_consistency": pair_repeated_consistency(z, y, p_ev, y_ev),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"[evtype_residuals] -> {path}", flush=True)
    for diag in out["group_diagnostics"]:
        print(f"== {diag['grouping']} ==")
        for row in diag["rows"][:8]:
            print(row)


if __name__ == "__main__":
    main()
