"""Fold-safe RACL-prototype decision features for evidence-type predictions.

The prototype score improves ranking; this script tests whether it can also
repair the binary decision boundary.  It keeps the score head fixed and applies
small, interpretable decision edits selected within each repeated-CV case using
only the other outer folds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from models.bootstrap_oof_methods import paired_bootstrap, row
except ModuleNotFoundError:
    from bootstrap_oof_methods import paired_bootstrap, row


DEFAULT_OOF = "data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz"
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
SCORE_RAW25 = "evtype_proto_raw25_decision_evtype"
SCORE_CAL50 = "evtype_proto_cal50_decision_evtype"
CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"


@dataclass(frozen=True)
class Spec:
    name: str
    feature: str
    mask: str
    veto_lt: float | None = None
    promote_gt: float | None = None


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def rank_by_case_fold(z: np.lib.npyio.NpzFile, p: np.ndarray) -> np.ndarray:
    case = np.asarray(z["case"], dtype=object).astype(str)
    fold = np.asarray(z["fold"], int)
    out = np.full(len(p), np.nan, float)
    for ca in sorted(set(case.tolist())):
        cm = case == ca
        for fo in sorted(set(fold[cm].tolist())):
            m = cm & (fold == fo) & (~np.isnan(p))
            out[m] = rank01(p[m])
    return out


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def build_context(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    p_raw = np.asarray(z["proto_source_bin__p"], float)
    p_cal = np.asarray(z["rankavg_bge_cm_proto_source_bin__p"], float)
    p_bge, _ = get_method(z, BGE)
    source_count = np.asarray(z["source_count"], int)
    source_bin = np.asarray(z["source_bin"], dtype=object).astype(str)
    confidence = np.asarray(z["confidence"], dtype=object).astype(str)
    combo = np.asarray(z["evidence_combo"], dtype=object).astype(str)
    return {
        "raw": rank_by_case_fold(z, p_raw),
        "cal": rank_by_case_fold(z, p_cal),
        "source_count": source_count,
        "src0": source_count == 0,
        "lowabs": np.abs(p_bge - 0.5) < 0.25,
        "po_medium": (combo == "PO") & (confidence == "medium"),
        "src2_3_medium": (source_bin == "src2_3") & (confidence == "medium"),
        "source_rich_medium": (source_count >= 2) & (confidence == "medium"),
    }


def mask_for(ctx: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "all":
        return np.ones_like(ctx["src0"], bool)
    if name == "src0":
        return ctx["src0"]
    if name == "lowabs":
        return ctx["lowabs"]
    if name == "src0_or_lowabs":
        return ctx["src0"] | ctx["lowabs"]
    if name == "po_medium":
        return ctx["po_medium"]
    if name == "source_rich_medium":
        return ctx["source_rich_medium"]
    if name == "src2_3_medium":
        return ctx["src2_3_medium"]
    raise KeyError(name)


def apply_spec(yhat_base: np.ndarray, ctx: dict[str, np.ndarray], spec: Spec) -> np.ndarray:
    yhat = yhat_base.copy()
    if spec.name == "noop":
        return yhat
    feature = ctx[spec.feature]
    mask = mask_for(ctx, spec.mask)
    if spec.veto_lt is not None:
        veto = (yhat == 1) & mask & (feature < float(spec.veto_lt))
        yhat[veto] = 0
    if spec.promote_gt is not None:
        promote = (yhat == 0) & mask & (feature > float(spec.promote_gt))
        yhat[promote] = 1
    return yhat


def candidate_specs() -> list[Spec]:
    specs = [Spec("noop", "raw", "all")]
    masks = ["src0", "lowabs", "src0_or_lowabs", "all"]
    for feature in ("raw", "cal"):
        for mask in masks:
            for lo in (0.15, 0.20, 0.25):
                specs.append(Spec(f"veto_{feature}_lt{lo:.2f}_{mask}", feature, mask, veto_lt=lo))
            for hi in (0.75, 0.80, 0.85):
                specs.append(Spec(f"promote_{feature}_gt{hi:.2f}_{mask}", feature, mask, promote_gt=hi))
            for lo in (0.15, 0.20, 0.25):
                for hi in (0.75, 0.80, 0.85):
                    specs.append(
                        Spec(
                            f"veto{lo:.2f}_promote{hi:.2f}_{feature}_{mask}",
                            feature,
                            mask,
                            veto_lt=lo,
                            promote_gt=hi,
                        )
                    )
    # A few mechanism-specific masks from earlier residual analyses.
    for feature in ("raw", "cal"):
        for mask in ("po_medium", "source_rich_medium", "src2_3_medium"):
            specs.append(Spec(f"veto_{feature}_lt0.20_{mask}", feature, mask, veto_lt=0.20))
            specs.append(
                Spec(
                    f"veto0.20_promote0.80_{feature}_{mask}",
                    feature,
                    mask,
                    veto_lt=0.20,
                    promote_gt=0.80,
                )
            )
    return specs


def source0_threshold_specs() -> list[Spec]:
    specs = [Spec("noop", "raw", "src0")]
    for lo in (0.15, 0.20, 0.25, 0.30):
        specs.append(Spec(f"src0nested_veto_raw_lt{lo:.2f}", "raw", "src0", veto_lt=lo))
    for hi in (0.70, 0.75, 0.80, 0.85):
        specs.append(Spec(f"src0nested_promote_raw_gt{hi:.2f}", "raw", "src0", promote_gt=hi))
    for lo in (0.15, 0.20, 0.25, 0.30):
        for hi in (0.70, 0.75, 0.80, 0.85):
            specs.append(
                Spec(
                    f"src0nested_veto{lo:.2f}_promote{hi:.2f}_raw",
                    "raw",
                    "src0",
                    veto_lt=lo,
                    promote_gt=hi,
                )
            )
    return specs


def complexity(spec: Spec) -> int:
    if spec.name == "noop":
        return 0
    return int(spec.veto_lt is not None) + int(spec.promote_gt is not None) + int(spec.mask != "all")


def objective(metric: dict[str, float], kind: str) -> float:
    if kind == "macro":
        return float(metric["macro_f1"] + 0.25 * metric["wF1"])
    if kind == "balanced":
        return float(metric["macro_f1"] + metric["wF1"] + 0.10 * metric["auprc"])
    raise ValueError(kind)


def select_spec(
    y: np.ndarray,
    c: np.ndarray,
    p_score: np.ndarray,
    yhat_base: np.ndarray,
    ctx: dict[str, np.ndarray],
    train: np.ndarray,
    specs: list[Spec],
    kind: str,
    min_gain: float,
) -> tuple[Spec, dict[str, object]]:
    base_metric = row(y[train], p_score[train], yhat_base[train], c[train])
    base_obj = objective(base_metric, kind)
    best = specs[0]
    best_metric = base_metric
    best_obj = base_obj
    for spec in specs:
        yhat = apply_spec(yhat_base, ctx, spec)
        metric = row(y[train], p_score[train], yhat[train], c[train])
        obj = objective(metric, kind)
        key = (obj - 0.0005 * complexity(spec), metric["macro_f1"], metric["wF1"], -complexity(spec))
        best_key = (
            best_obj - 0.0005 * complexity(best),
            best_metric["macro_f1"],
            best_metric["wF1"],
            -complexity(best),
        )
        if obj - base_obj >= min_gain and key > best_key:
            best = spec
            best_metric = metric
            best_obj = obj
    return best, {
        "train_base_metric": base_metric,
        "train_selected_metric": best_metric,
        "train_gain": round(float(best_obj - base_obj), 6),
    }


def build_crossfit(
    z: np.lib.npyio.NpzFile,
    score_method: str,
    decision_method: str,
    kind: str,
    min_gain: float,
    specs: list[Spec] | None = None,
    name_suffix: str | None = None,
) -> tuple[str, dict[str, np.ndarray], list[dict[str, object]]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    case = np.asarray(z["case"], dtype=object).astype(str)
    fold = np.asarray(z["fold"], int)
    p_score, _ = get_method(z, score_method)
    _, yhat_base = get_method(z, decision_method)
    ctx = build_context(z)
    specs = specs if specs is not None else candidate_specs()
    suffix = name_suffix or f"{kind}_{score_method.replace('evtype_proto_', '').replace('_decision_evtype', '')}"
    name = f"proto_decision_cvselect_{suffix}"
    out = {"p": np.full(len(y), np.nan, float), "yhat": np.full(len(y), -1, int)}
    fold_meta = []
    for ca in sorted(set(case.tolist())):
        cm = case == ca
        for fo in sorted(set(fold[cm].tolist())):
            te = cm & (fold == fo)
            tr = cm & (fold != fo)
            spec, meta = select_spec(
                y, c, p_score, yhat_base, ctx, tr, specs, kind, min_gain)
            yhat = apply_spec(yhat_base, ctx, spec)
            out["p"][te] = p_score[te]
            out["yhat"][te] = yhat[te]
            fold_meta.append({
                "case": ca,
                "fold": int(fo),
                "selected_spec": spec.name,
                **meta,
                "test_flips": int((yhat[te] != yhat_base[te]).sum()),
            })
            print(f"[proto_decision] case={ca} fold={fo} score={score_method} spec={spec.name}", flush=True)
    return name, out, fold_meta


def fixed_methods(
    z: np.lib.npyio.NpzFile,
    decision_method: str,
    score_methods: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    _, yhat_base = get_method(z, decision_method)
    ctx = build_context(z)
    methods = {}
    fixed = [
        Spec("veto0.20_promote0.75_raw_src0", "raw", "src0", veto_lt=0.20, promote_gt=0.75),
        Spec("veto_raw_lt0.20_src0", "raw", "src0", veto_lt=0.20),
        Spec("veto0.20_promote0.80_cal_src0_or_lowabs", "cal", "src0_or_lowabs", veto_lt=0.20, promote_gt=0.80),
    ]
    fixed_score_methods = list(dict.fromkeys(score_methods + [decision_method]))
    for score_method in fixed_score_methods:
        if f"{score_method}__p" not in z.files:
            continue
        p, _ = get_method(z, score_method)
        for spec in fixed:
            name = f"proto_decision_fixed_{spec.name}_score_{score_method.replace('evtype_proto_', '').replace('_decision_evtype', '')}"
            methods[name] = {"p": p.copy(), "yhat": apply_spec(yhat_base, ctx, spec)}
    return methods


def evaluate(
    z: np.lib.npyio.NpzFile,
    methods: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
    baselines: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    rows = {}
    for method in baselines:
        if f"{method}__p" in z.files:
            p, yhat = get_method(z, method)
            rows[method] = row(y, p, yhat, c)
    for method, item in methods.items():
        ok = (~np.isnan(item["p"])) & (item["yhat"] >= 0)
        rows[method] = row(y[ok], item["p"][ok], item["yhat"][ok], c[ok])
    sig = {}
    if n_boot > 0:
        for mi, (method, item) in enumerate(methods.items()):
            for bi, base in enumerate(baselines):
                if f"{base}__p" not in z.files:
                    continue
                p_b, y_b = get_method(z, base)
                ok = (~np.isnan(item["p"])) & (item["yhat"] >= 0) & (~np.isnan(p_b))
                sig[f"{method}_vs_{base}"] = paired_bootstrap(
                    y[ok],
                    item["p"][ok],
                    item["yhat"][ok],
                    p_b[ok],
                    y_b[ok],
                    n_boot=n_boot,
                    seed=seed + mi * 97 + bi * 13,
                )
    return rows, sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=DEFAULT_OOF)
    ap.add_argument("--decision_method", default=EVTYPE,
                    help="Method whose yhat is edited by prototype decision rules.")
    ap.add_argument("--score_method", action="append",
                    help="Score method(s) used as p in emitted decision-feature methods.")
    ap.add_argument("--source0_score_method", default="",
                    help="Score method for the source0 nested selector; defaults to first --score_method.")
    ap.add_argument("--baseline", action="append",
                    help="Baseline/method rows to report and compare against.")
    ap.add_argument("--objective", choices=["macro", "balanced"], default="macro")
    ap.add_argument("--min_gain", type=float, default=0.001)
    ap.add_argument("--n_boot", type=int, default=0,
                    help="Default 0 for screening; use bootstrap_oof_methods for targeted 5k checks.")
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--out", default="data/final/cleancl/racl_proto_decision_feature_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_racl_proto_decision_feature_20260609.npz")
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    score_methods = args.score_method or [SCORE_RAW25, SCORE_CAL50]
    baselines = args.baseline or [BGE, CMBGE, CURRENT, EVTYPE, SCORE_RAW25, SCORE_CAL50]
    methods = fixed_methods(z, args.decision_method, score_methods)
    fold_meta = []
    for score_method in score_methods:
        if f"{score_method}__p" not in z.files:
            print(f"[proto_decision] skip missing score_method={score_method}", flush=True)
            continue
        name, built, meta = build_crossfit(
            z, score_method, args.decision_method, args.objective, args.min_gain)
        methods[name] = built
        fold_meta.extend(meta)
    source0_score = args.source0_score_method or (score_methods[0] if score_methods else SCORE_RAW25)
    name, built, meta = build_crossfit(
        z,
        source0_score,
        args.decision_method,
        args.objective,
        args.min_gain,
        specs=source0_threshold_specs(),
        name_suffix=(
            f"src0nested_{args.objective}_"
            f"{source0_score.replace('evtype_proto_', '').replace('_decision_evtype', '')}"
        ),
    )
    methods[name] = built
    fold_meta.extend(meta)

    metrics, sig = evaluate(z, methods, args.n_boot, args.seed, baselines)
    ranked = sorted(
        methods,
        key=lambda m: (metrics[m]["macro_f1"], metrics[m]["wF1"], metrics[m]["auprc"]),
        reverse=True,
    )
    out = {
        "description": (
            "Prototype-rank decision edits for evidence-type predictions. "
            "Cross-fit methods select edits using other folds within each repeated-CV case."
        ),
        "oof": args.oof,
        "decision_method": args.decision_method,
        "score_methods": score_methods,
        "baselines": baselines,
        "objective": args.objective,
        "min_gain": float(args.min_gain),
        "ranked_methods": ranked,
        "metrics": metrics,
        "fold_meta": fold_meta,
        "n_boot": int(args.n_boot),
        "significance": sig,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)

    if args.dump_oof:
        arrays = {name: np.asarray(z[name]) for name in z.files}
        for method, item in methods.items():
            arrays[f"{method}__p"] = item["p"]
            arrays[f"{method}__yhat"] = item["yhat"].astype(int)
        dump_path = Path(args.dump_oof)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **arrays)

    print(f"[cv_racl_proto_decision_feature] -> {out_path}", flush=True)
    for method in ranked[:12]:
        print(method, metrics[method], flush=True)


if __name__ == "__main__":
    main()
