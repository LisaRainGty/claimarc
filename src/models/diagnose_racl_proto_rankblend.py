"""OOF diagnostic: blend evidence-type score with RACL prototype score.

The prototype verifier can improve ranking but not binary decisions.  This
script keeps the stronger evidence-type decision unchanged and only blends
scores within each case/fold, so the diagnostic remains a score-calibration
probe rather than another decision selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from models.bootstrap_oof_methods import row
except ModuleNotFoundError:
    from bootstrap_oof_methods import row


DEFAULT_OOF = "data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz"
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
PROTO = "rankavg_bge_cm_proto_source_bin"


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def method_arrays(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def rankblend_by_case_fold(
    z: np.lib.npyio.NpzFile,
    base_method: str,
    proto_method: str,
    proto_weight: float,
) -> np.ndarray:
    p_base, _ = method_arrays(z, base_method)
    p_proto, _ = method_arrays(z, proto_method)
    case = np.asarray(z["case"], dtype=object).astype(str)
    fold = np.asarray(z["fold"], int)
    out = np.full(len(p_base), np.nan, float)
    for ca in sorted(set(case.tolist())):
        for fo in sorted(set(fold[case == ca].tolist())):
            m = (case == ca) & (fold == fo)
            out[m] = (
                (1.0 - float(proto_weight)) * rank01(p_base[m])
                + float(proto_weight) * rank01(p_proto[m])
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=DEFAULT_OOF)
    ap.add_argument("--base_method", default=EVTYPE)
    ap.add_argument("--proto_method", default=PROTO)
    ap.add_argument("--decision_method", default=EVTYPE)
    ap.add_argument("--weight", type=float, action="append", default=None,
                    help="Prototype blend weight. May be repeated; defaults to the screening grid.")
    ap.add_argument("--out", default="data/final/cleancl/racl_proto_evtype_rankblend_screen_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_racl_proto_evtype_rankblend_screen_20260609.npz")
    args = ap.parse_args()

    weights = args.weight if args.weight is not None else [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    _, yhat_decision = method_arrays(z, args.decision_method)

    arrays = {name: np.asarray(z[name]) for name in z.files}
    metrics = {}
    names = []
    for weight in weights:
        suffix = str(int(round(float(weight) * 100))).zfill(2)
        name = f"evtype_rankblend_proto{suffix}_decision_evtype"
        p = rankblend_by_case_fold(z, args.base_method, args.proto_method, weight)
        arrays[f"{name}__p"] = p
        arrays[f"{name}__yhat"] = yhat_decision
        metrics[name] = row(y, p, yhat_decision, c)
        names.append(name)

    for method in (
        "bge_lr",
        "rankavg_sourcefirst_cm_pcls_bge",
        "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect",
        args.base_method,
        args.proto_method,
    ):
        if f"{method}__p" in z.files:
            p, yhat = method_arrays(z, method)
            metrics[method] = row(y, p, yhat, c)

    ranked = sorted(
        names,
        key=lambda m: (metrics[m]["auprc"], metrics[m]["auroc"], metrics[m]["macro_f1"]),
        reverse=True,
    )
    out = {
        "description": (
            "Blend evidence-type score and RACL prototype score by case/fold "
            "ranks; retain evidence-type binary decision."
        ),
        "oof": args.oof,
        "base_method": args.base_method,
        "proto_method": args.proto_method,
        "decision_method": args.decision_method,
        "ranked_blends": ranked,
        "metrics": metrics,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    if args.dump_oof:
        dump = Path(args.dump_oof)
        dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump, **arrays)
    print(f"[diagnose_racl_proto_rankblend] -> {path}", flush=True)
    for name in ranked:
        print(name, metrics[name], flush=True)


if __name__ == "__main__":
    main()
