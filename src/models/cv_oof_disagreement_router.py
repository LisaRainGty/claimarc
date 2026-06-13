"""Fold-safe OOF disagreement router diagnostics.

This script uses already cross-fitted OOF predictions and learns only a small
decision router inside each repeated-CV case/fold.  It is meant for late-stage
diagnostics: can a modern embedding baseline fix a narrow slice of errors from
the current guarded/RACL decision without replacing the main mechanism?
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


DEFAULT_BASE = (
    "proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_"
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_"
    "srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)


def macro(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(f1_score(y, pred, average="macro", sample_weight=w, zero_division=0))


def row(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> dict:
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def method_arrays(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return z[f"{method}__p"].astype(float), z[f"{method}__yhat"].astype(int)


@dataclass(frozen=True)
class Spec:
    name: str
    base_uncertain_max: float
    aux_confident_min: float
    source_mode: str
    direction: str = "both"


def source_mask(source_count: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones_like(source_count, dtype=bool)
    if mode == "src0":
        return source_count == 0
    if mode == "src1p":
        return source_count >= 1
    if mode == "src2p":
        return source_count >= 2
    raise ValueError(f"unknown source_mode: {mode}")


def apply_spec(
    base_p: np.ndarray,
    base_yhat: np.ndarray,
    aux_p: np.ndarray,
    aux_yhat: np.ndarray,
    source_count: np.ndarray,
    spec: Spec,
) -> np.ndarray:
    pred = base_yhat.copy()
    if spec.direction == "noop":
        return pred
    base_conf = np.abs(base_p - 0.5)
    aux_conf = np.abs(aux_p - 0.5)
    mask = (
        (base_yhat != aux_yhat)
        & (base_conf <= spec.base_uncertain_max)
        & (aux_conf >= spec.aux_confident_min)
        & source_mask(source_count, spec.source_mode)
    )
    if spec.direction == "promote":
        mask &= (base_yhat == 0) & (aux_yhat == 1)
    elif spec.direction == "veto":
        mask &= (base_yhat == 1) & (aux_yhat == 0)
    elif spec.direction != "both":
        raise ValueError(f"unknown direction: {spec.direction}")
    pred[mask] = aux_yhat[mask]
    return pred


def spec_complexity(spec: Spec) -> float:
    if spec.direction == "noop":
        return 0.0
    source_penalty = {"all": 0.0, "src0": 0.05, "src1p": 0.05, "src2p": 0.08}[spec.source_mode]
    direction_penalty = 0.0 if spec.direction == "both" else 0.05
    return (
        source_penalty
        + direction_penalty
        + max(0.0, 0.30 - spec.base_uncertain_max)
        + max(0.0, spec.aux_confident_min - 0.08)
    )


def build_specs() -> list[Spec]:
    specs = []
    for bmax in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.00]:
        for qmin in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for source in ["all", "src0", "src1p", "src2p"]:
                for direction in ["both", "promote", "veto"]:
                    specs.append(
                        Spec(
                            name=(
                                f"b{bmax:.2f}_q{qmin:.2f}_{source}_{direction}"
                                .replace(".", "")
                            ),
                            base_uncertain_max=bmax,
                            aux_confident_min=qmin,
                            source_mode=source,
                            direction=direction,
                        )
                    )
    return specs


def select_spec(
    y: np.ndarray,
    c: np.ndarray,
    base_p: np.ndarray,
    base_yhat: np.ndarray,
    aux_p: np.ndarray,
    aux_yhat: np.ndarray,
    source_count: np.ndarray,
    specs: list[Spec],
    objective: str,
    min_val_delta: float = 0.0,
    max_val_flip_rate: float = 1.0,
) -> tuple[Spec, dict]:
    base_metric = row(y, base_p, base_yhat, c)
    best_key = None
    best_spec = Spec("noop", 0.0, 1.0, "all", "noop")
    best_metric: dict | None = {**base_metric, "flips": 0, "val_delta": 0.0, "flip_rate": 0.0}
    for spec in specs:
        pred = apply_spec(base_p, base_yhat, aux_p, aux_yhat, source_count, spec)
        m = row(y, base_p, pred, c)
        flips = int(np.sum(pred != base_yhat))
        val_delta = float(m["macro_f1"] - base_metric["macro_f1"])
        flip_rate = float(flips / max(1, len(y)))
        if val_delta < min_val_delta or flip_rate > max_val_flip_rate:
            continue
        if objective == "macro":
            score = m["macro_f1"]
        elif objective == "macro_wf1":
            score = m["macro_f1"] + 0.25 * m["wF1"]
        else:
            raise ValueError(f"unknown objective: {objective}")
        key = (score - 0.0005 * spec_complexity(spec), m["macro_f1"], m["wF1"], -flips)
        if best_key is None or key > best_key:
            best_key = key
            best_spec = spec
            best_metric = {**m, "flips": flips, "val_delta": round(val_delta, 6), "flip_rate": round(flip_rate, 6)}
    assert best_metric is not None
    return best_spec, best_metric


def method_suffix(args: argparse.Namespace) -> str:
    bits = []
    if args.min_val_delta > 0:
        bits.append(f"mind{args.min_val_delta:.4f}".replace(".", ""))
    if args.non_veto_min_val_delta > 0:
        bits.append(f"nvmind{args.non_veto_min_val_delta:.4f}".replace(".", ""))
    if args.max_val_flip_rate < 1.0:
        bits.append(f"maxfr{args.max_val_flip_rate:.3f}".replace(".", ""))
    return ("_" + "_".join(bits)) if bits else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--base_method", default=DEFAULT_BASE)
    ap.add_argument("--aux_method", default="qwen3emb06b_lr")
    ap.add_argument("--objective", choices=["macro", "macro_wf1"], default="macro")
    ap.add_argument("--min_val_delta", type=float, default=0.0,
                    help="Require selected rule to beat the base decision on training folds by this Macro-F1 margin.")
    ap.add_argument("--non_veto_min_val_delta", type=float, default=0.0,
                    help="If the best rule is promote/both and its validation gain is below this, reselect among veto rules.")
    ap.add_argument("--max_val_flip_rate", type=float, default=1.0,
                    help="Maximum allowed training-fold flip rate for selected rules.")
    ap.add_argument("--fixed_base_uncertain_max", type=float, default=0.30)
    ap.add_argument("--fixed_aux_confident_min", type=float, default=0.15)
    ap.add_argument("--fixed_source_mode", default="all", choices=["all", "src0", "src1p", "src2p"])
    ap.add_argument("--fixed_direction", default="both", choices=["both", "promote", "veto"])
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--out", default="data/final/cleancl/oof_disagreement_router_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_disagreement_router_20260609.npz")
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = z["y"].astype(int)
    c = z["c"].astype(float) if "c" in z.files else np.ones_like(y, dtype=float)
    case = z["case"]
    fold = z["fold"].astype(int)
    source_count = z["source_count"].astype(int)

    base_p, base_yhat = method_arrays(z, args.base_method)
    aux_p, aux_yhat = method_arrays(z, args.aux_method)

    specs = build_specs()
    select_name = f"router_select_{args.objective}{method_suffix(args)}_{args.aux_method}_on_{args.base_method}"
    fixed_spec = Spec(
        name=(
            f"b{args.fixed_base_uncertain_max:.2f}_q{args.fixed_aux_confident_min:.2f}_"
            f"{args.fixed_source_mode}_{args.fixed_direction}"
        ).replace(".", ""),
        base_uncertain_max=args.fixed_base_uncertain_max,
        aux_confident_min=args.fixed_aux_confident_min,
        source_mode=args.fixed_source_mode,
        direction=args.fixed_direction,
    )
    fixed_name = f"router_fixed_{fixed_spec.name}_{args.aux_method}_on_{args.base_method}"

    pred_select = base_yhat.copy()
    chosen = []
    for ca in np.unique(case):
        for fo in np.unique(fold[case == ca]):
            tr = (case == ca) & (fold != fo)
            te = (case == ca) & (fold == fo)
            spec, metric = select_spec(
                y[tr],
                c[tr],
                base_p[tr],
                base_yhat[tr],
                aux_p[tr],
                aux_yhat[tr],
                source_count[tr],
                specs,
                args.objective,
                min_val_delta=args.min_val_delta,
                max_val_flip_rate=args.max_val_flip_rate,
            )
            if spec.direction not in {"veto", "noop"} and metric["val_delta"] < args.non_veto_min_val_delta:
                spec, metric = select_spec(
                    y[tr],
                    c[tr],
                    base_p[tr],
                    base_yhat[tr],
                    aux_p[tr],
                    aux_yhat[tr],
                    source_count[tr],
                    [s for s in specs if s.direction == "veto"],
                    args.objective,
                    min_val_delta=args.min_val_delta,
                    max_val_flip_rate=args.max_val_flip_rate,
                )
            pred_select[te] = apply_spec(
                base_p[te], base_yhat[te], aux_p[te], aux_yhat[te], source_count[te], spec
            )
            chosen.append({"case": str(ca), "fold": int(fo), "spec": spec.name, "val_metric": metric})

    pred_fixed = apply_spec(base_p, base_yhat, aux_p, aux_yhat, source_count, fixed_spec)

    metrics = {
        args.base_method: row(y, base_p, base_yhat, c),
        args.aux_method: row(y, aux_p, aux_yhat, c),
        select_name: row(y, base_p, pred_select, c),
        fixed_name: row(y, base_p, pred_fixed, c),
    }
    for baseline in args.baseline:
        if baseline in metrics:
            continue
        bp, by = method_arrays(z, baseline)
        metrics[baseline] = row(y, bp, by, c)

    arrays = {k: z[k] for k in z.files}
    arrays[f"{select_name}__p"] = base_p
    arrays[f"{select_name}__yhat"] = pred_select
    arrays[f"{fixed_name}__p"] = base_p
    arrays[f"{fixed_name}__yhat"] = pred_fixed
    Path(args.dump_oof).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dump_oof, **arrays)

    payload = {
        "oof": args.oof,
        "base_method": args.base_method,
        "aux_method": args.aux_method,
        "objective": args.objective,
        "min_val_delta": args.min_val_delta,
        "non_veto_min_val_delta": args.non_veto_min_val_delta,
        "max_val_flip_rate": args.max_val_flip_rate,
        "fixed_spec": fixed_spec.__dict__,
        "metrics": metrics,
        "chosen": chosen,
        "select_method": select_name,
        "fixed_method": fixed_name,
        "dump_oof": args.dump_oof,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[cv_oof_disagreement_router] -> {args.out}", flush=True)
    print(f"[cv_oof_disagreement_router] -> {args.dump_oof}", flush=True)


if __name__ == "__main__":
    main()
