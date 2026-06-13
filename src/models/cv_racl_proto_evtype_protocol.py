"""Fixed RACL-prototype score calibration for the evidence-type adapter.

This is the protocolized counterpart of ``diagnose_racl_proto_rankblend``:
it writes only fixed, named score-calibration methods and keeps the
evidence-type binary decision unchanged.  No validation selector or result
ranking is used.
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
PROTO_CAL = "rankavg_bge_cm_proto_source_bin"
PROTO_RAW = "proto_source_bin"
CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def case_fold_rankblend(
    z: np.lib.npyio.NpzFile,
    base_method: str,
    proto_method: str,
    proto_weight: float,
) -> np.ndarray:
    p_base, _ = get_method(z, base_method)
    p_proto, _ = get_method(z, proto_method)
    case = np.asarray(z["case"], dtype=object).astype(str)
    fold = np.asarray(z["fold"], int)
    out = np.full(len(p_base), np.nan, float)
    for ca in sorted(set(case.tolist())):
        case_mask = case == ca
        for fo in sorted(set(fold[case_mask].tolist())):
            m = case_mask & (fold == fo)
            ok = m & (~np.isnan(p_base)) & (~np.isnan(p_proto))
            if not np.any(ok):
                continue
            out[ok] = (
                (1.0 - float(proto_weight)) * rank01(p_base[ok])
                + float(proto_weight) * rank01(p_proto[ok])
            )
    return out


def metric_row(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> dict:
    ok = (~np.isnan(p)) & (yhat >= 0)
    return row(y[ok], p[ok], yhat[ok].astype(int), c[ok])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=DEFAULT_OOF)
    ap.add_argument("--base_method", default=EVTYPE)
    ap.add_argument("--decision_method", default=EVTYPE)
    ap.add_argument("--decision_label", default="evtype",
                    help="Short label used in emitted method names for the decision head.")
    ap.add_argument("--cal_proto_method", default=PROTO_CAL)
    ap.add_argument("--raw_proto_method", default=PROTO_RAW)
    ap.add_argument("--cal_weight", type=float, default=0.50)
    ap.add_argument("--raw_weight", type=float, default=0.25)
    ap.add_argument("--out", default="data/final/cleancl/racl_proto_evtype_protocol_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz")
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    _, yhat_decision = get_method(z, args.decision_method)

    arrays = {name: np.asarray(z[name]) for name in z.files}
    methods = {
        f"evtype_proto_cal{int(round(args.cal_weight * 100)):02d}_decision_{args.decision_label}": (
            args.cal_proto_method,
            float(args.cal_weight),
        ),
        f"evtype_proto_raw{int(round(args.raw_weight * 100)):02d}_decision_{args.decision_label}": (
            args.raw_proto_method,
            float(args.raw_weight),
        ),
    }

    metrics = {}
    for name, (proto_method, weight) in methods.items():
        p = case_fold_rankblend(z, args.base_method, proto_method, weight)
        arrays[f"{name}__p"] = p
        arrays[f"{name}__yhat"] = yhat_decision
        metrics[name] = metric_row(y, p, yhat_decision, c)

    for method in (BGE, CMBGE, CURRENT, args.base_method, args.cal_proto_method, args.raw_proto_method):
        if f"{method}__p" in z.files:
            p, yhat = get_method(z, method)
            metrics[method] = metric_row(y, p, yhat, c)

    out = {
        "description": (
            "Fixed case/fold rank calibration of the evidence-type score with "
            "RACL prototype relation scores; evidence-type decision is retained."
        ),
        "oof": args.oof,
        "base_method": args.base_method,
        "decision_method": args.decision_method,
        "protocol_methods": {
            name: {"proto_method": proto_method, "proto_weight": weight}
            for name, (proto_method, weight) in methods.items()
        },
        "metrics": metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)

    if args.dump_oof:
        dump_path = Path(args.dump_oof)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **arrays)

    print(f"[cv_racl_proto_evtype_protocol] -> {out_path}", flush=True)
    for name in methods:
        print(name, metrics[name], flush=True)


if __name__ == "__main__":
    main()
