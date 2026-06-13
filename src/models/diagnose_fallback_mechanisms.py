"""Diagnose fallback mechanisms from saved OOF predictions.

This script compares a selected CLAIMARC/RACL-NLI candidate against the fair
BGE+LR baseline using only saved OOF ``.npz`` files. It reports where the
candidate fixes BGE errors, where it breaks BGE-correct cases, and which
source/confidence slices explain the net gain.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CASE_SPECS = {
    "fs0_old_lowabs": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick5k.npz",
        "method": "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
        "description": "Old-cache fs0 strict score-side fallback on source0 and source2/3 low/absent confidence.",
    },
    "fs0_newcache_src0": {
        "oof": "data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_5k.npz",
        "method": "rankmix_nli25_hgb_bge_scorefallback_bge100_src0",
        "description": "New-cache fs0 predef-only strict score-side full BGE fallback on source0.",
    },
    "fs1_predef_lowabs": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs.npz",
        "method": "predef_protocol_predef_lowabs_r25_scorefallback_srcconf_bgefallback",
        "description": "fs1 fixed predef lowabs protocol: NLI+BGE score with source/confidence BGE fallback.",
    },
    "fs2_decision_bgefallback": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback.npz",
        "method": "dual_score_rankmix_nli25_hgb_bge__decision_rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
        "description": "fs2 strict dual-head candidate: NLI+BGE score with source0/source2-3 BGE decision fallback.",
    },
}


def as_int(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(int)


def as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(float)


def as_str(a: np.ndarray) -> np.ndarray:
    return np.asarray([str(x) for x in np.asarray(a, dtype=object)], dtype=object)


def source_bin_from_count(n: int) -> str:
    if n <= 0:
        return "src0"
    if n == 1:
        return "src1"
    if n <= 3:
        return "src2_3"
    return "src4p"


def bge_conf_bin(p: float) -> str:
    d = abs(float(p) - 0.5)
    if d < 0.08:
        return "bge_c00_08"
    if d < 0.16:
        return "bge_c08_16"
    if d < 0.28:
        return "bge_c16_28"
    return "bge_c28p"


def lowabs(conf: str) -> bool:
    return str(conf).lower() in {"low", "absent", "", "none", "nan"}


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = as_int(y)
    score = as_float(score)
    pos = int(y.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / pos)


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = as_float(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = as_int(y)
    score = as_float(score)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(score)
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def f1_for_label(y: np.ndarray, yhat: np.ndarray, label: int, w: np.ndarray | None = None) -> float:
    y = as_int(y)
    yhat = as_int(yhat)
    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = as_float(w)
    tp = float(w[(y == label) & (yhat == label)].sum())
    fp = float(w[(y != label) & (yhat == label)].sum())
    fn = float(w[(y == label) & (yhat != label)].sum())
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom <= 0 else 2.0 * tp / denom


def macro_f1(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None = None) -> float:
    return 0.5 * (f1_for_label(y, yhat, 0, w) + f1_for_label(y, yhat, 1, w))


def confusion(y: np.ndarray, yhat: np.ndarray) -> dict[str, int]:
    y = as_int(y)
    yhat = as_int(yhat)
    return {
        "tp": int(((y == 1) & (yhat == 1)).sum()),
        "fp": int(((y == 0) & (yhat == 1)).sum()),
        "tn": int(((y == 0) & (yhat == 0)).sum()),
        "fn": int(((y == 1) & (yhat == 0)).sum()),
    }


def metrics(y: np.ndarray, score: np.ndarray, yhat: np.ndarray, c: np.ndarray | None = None) -> dict[str, float | int | dict[str, int]]:
    out = {
        "n": int(len(y)),
        "pos": int(as_int(y).sum()),
        "ap": round(average_precision(y, score), 4),
        "auroc": round(auroc(y, score), 4),
        "macro_f1": round(macro_f1(y, yhat), 4),
        "accuracy": round(float((as_int(y) == as_int(yhat)).mean()), 4),
        "pred_pos_rate": round(float(as_int(yhat).mean()), 4),
        "confusion": confusion(y, yhat),
    }
    if c is not None:
        out["weighted_macro_f1"] = round(macro_f1(y, yhat, np.clip(as_float(c), 0.05, None)), 4)
    return out


def round_float(x: float) -> float:
    return round(float(x), 4)


def summarize_slice(
    name: str,
    group_type: str,
    mask: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    bge_p: np.ndarray,
    bge_yhat: np.ndarray,
    cand_p: np.ndarray,
    cand_yhat: np.ndarray,
    source_count: np.ndarray,
) -> dict[str, object] | None:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return None
    yy = y[mask]
    bp = bge_p[mask]
    by = bge_yhat[mask]
    cp = cand_p[mask]
    cy = cand_yhat[mask]
    cc = c[mask]
    bge_ok = by == yy
    cand_ok = cy == yy
    fixed = (~bge_ok) & cand_ok
    broken = bge_ok & (~cand_ok)
    bge_fp = (yy == 0) & (by == 1)
    bge_fn = (yy == 1) & (by == 0)
    cand_fp = (yy == 0) & (cy == 1)
    cand_fn = (yy == 1) & (cy == 0)
    fixed_fp_to_tn = fixed & bge_fp
    fixed_fn_to_tp = fixed & bge_fn
    broken_tn_to_fp = broken & (yy == 0) & (by == 0) & (cy == 1)
    broken_tp_to_fn = broken & (yy == 1) & (by == 1) & (cy == 0)
    out = {
        "group": name,
        "group_type": group_type,
        "n": int(mask.sum()),
        "pos": int(yy.sum()),
        "pos_rate": round_float(float(yy.mean())),
        "bge_acc": round_float(float(bge_ok.mean())),
        "method_acc": round_float(float(cand_ok.mean())),
        "net_acc_gain": round_float(float(cand_ok.mean() - bge_ok.mean())),
        "bge_errors": int((~bge_ok).sum()),
        "method_errors": int((~cand_ok).sum()),
        "fixed_total": int(fixed.sum()),
        "broken_total": int(broken.sum()),
        "net_flip_gain": int(fixed.sum() - broken.sum()),
        "fixed_bge_fp_to_tn": int(fixed_fp_to_tn.sum()),
        "fixed_bge_fn_to_tp": int(fixed_fn_to_tp.sum()),
        "broken_tn_to_fp": int(broken_tn_to_fp.sum()),
        "broken_tp_to_fn": int(broken_tp_to_fn.sum()),
        "bge_fp": int(bge_fp.sum()),
        "bge_fn": int(bge_fn.sum()),
        "method_fp": int(cand_fp.sum()),
        "method_fn": int(cand_fn.sum()),
        "bge_pred_pos_rate": round_float(float(by.mean())),
        "method_pred_pos_rate": round_float(float(cy.mean())),
        "mean_bge_p": round_float(float(bp.mean())),
        "mean_method_p": round_float(float(cp.mean())),
        "mean_score_delta": round_float(float((cp - bp).mean())),
        "mean_source_count": round_float(float(source_count[mask].mean())),
        "mean_weight": round_float(float(cc.mean())),
    }
    return out


def add_named_slices(
    groups: list[tuple[str, str, np.ndarray]],
    source_count: np.ndarray,
    source_bin: np.ndarray,
    confidence: np.ndarray,
    bge_p: np.ndarray,
) -> None:
    lowabs_mask = np.asarray([lowabs(x) for x in confidence], dtype=bool)
    src0 = source_bin == "src0"
    src2_3 = source_bin == "src2_3"
    groups.extend(
        [
            ("mechanism:src0", "mechanism", src0),
            ("mechanism:src2_3", "mechanism", src2_3),
            ("mechanism:src2_3_lowabs", "mechanism", src2_3 & lowabs_mask),
            ("mechanism:src0_or_src2_3", "mechanism", src0 | src2_3),
            ("mechanism:src0_or_src2_3_lowabs", "mechanism", src0 | (src2_3 & lowabs_mask)),
            ("mechanism:lowabs", "mechanism", lowabs_mask),
            ("mechanism:src4p", "mechanism", source_bin == "src4p"),
            ("mechanism:bge_uncertain_008", "mechanism", np.abs(bge_p - 0.5) < 0.08),
            ("mechanism:bge_confident_028p", "mechanism", np.abs(bge_p - 0.5) >= 0.28),
            ("mechanism:source_count_ge2", "mechanism", source_count >= 2),
        ]
    )


def build_group_masks(
    fold: np.ndarray,
    source_count: np.ndarray,
    source_bin: np.ndarray,
    confidence: np.ndarray,
    category: np.ndarray,
    bge_p: np.ndarray,
) -> list[tuple[str, str, np.ndarray]]:
    groups: list[tuple[str, str, np.ndarray]] = []
    add_named_slices(groups, source_count, source_bin, confidence, bge_p)
    bge_bins = np.asarray([bge_conf_bin(x) for x in bge_p], dtype=object)
    sources = sorted(set(str(x) for x in source_bin))
    confidences = sorted(set(str(x) for x in confidence))
    categories = sorted(set(str(x) for x in category))
    folds = sorted(set(int(x) for x in fold))
    bge_conf_bins = sorted(set(str(x) for x in bge_bins))
    for value in folds:
        groups.append((f"fold={value}", "fold", fold == value))
    for value in sources:
        groups.append((f"source_bin={value}", "source_bin", source_bin == value))
    for value in confidences:
        groups.append((f"confidence={value}", "confidence", confidence == value))
    for value in bge_conf_bins:
        groups.append((f"bge_conf_bin={value}", "bge_conf_bin", bge_bins == value))
    for src in sources:
        for conf in confidences:
            groups.append((f"source_bin={src}|confidence={conf}", "source_confidence", (source_bin == src) & (confidence == conf)))
    for src in sources:
        for bin_name in bge_conf_bins:
            groups.append((f"source_bin={src}|bge_conf_bin={bin_name}", "source_bge_conf", (source_bin == src) & (bge_bins == bin_name)))
    for cat in categories:
        groups.append((f"category={cat}", "category", category == cat))
    return groups


def load_case(case_name: str, spec: dict[str, str], root: Path, min_group_n: int) -> dict[str, object]:
    path = root / spec["oof"]
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    method = spec["method"]
    required = ["y", "bge_lr__p", "bge_lr__yhat", f"{method}__p", f"{method}__yhat"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{case_name} missing OOF keys: {missing}")

    y = as_int(z["y"])
    c = as_float(z["c"]) if "c" in z.files else np.ones(len(y), dtype=float)
    bge_p = as_float(z["bge_lr__p"])
    bge_yhat = as_int(z["bge_lr__yhat"])
    cand_p = as_float(z[f"{method}__p"])
    cand_yhat = as_int(z[f"{method}__yhat"])
    fold = as_int(z["fold"]) if "fold" in z.files else np.zeros(len(y), dtype=int)
    source_count = as_int(z["source_count"]) if "source_count" in z.files else np.zeros(len(y), dtype=int)
    if "source_bin" in z.files:
        source_bin = as_str(z["source_bin"])
    else:
        source_bin = np.asarray([source_bin_from_count(int(n)) for n in source_count], dtype=object)
    confidence = as_str(z["confidence"]) if "confidence" in z.files else np.asarray(["unknown"] * len(y), dtype=object)
    category = as_str(z["category"]) if "category" in z.files else np.asarray(["unknown"] * len(y), dtype=object)

    valid = np.isfinite(bge_p) & np.isfinite(cand_p)
    y = y[valid]
    c = c[valid]
    bge_p = bge_p[valid]
    bge_yhat = bge_yhat[valid]
    cand_p = cand_p[valid]
    cand_yhat = cand_yhat[valid]
    fold = fold[valid]
    source_count = source_count[valid]
    source_bin = source_bin[valid]
    confidence = confidence[valid]
    category = category[valid]

    bge_ok = bge_yhat == y
    cand_ok = cand_yhat == y
    fixed = (~bge_ok) & cand_ok
    broken = bge_ok & (~cand_ok)
    group_summaries: list[dict[str, object]] = []
    for name, group_type, mask in build_group_masks(fold, source_count, source_bin, confidence, category, bge_p):
        if int(mask.sum()) < min_group_n:
            continue
        summary = summarize_slice(name, group_type, mask, y, c, bge_p, bge_yhat, cand_p, cand_yhat, source_count)
        if summary is not None:
            group_summaries.append(summary)

    top_gain = sorted(group_summaries, key=lambda r: (int(r["net_flip_gain"]), float(r["net_acc_gain"]), int(r["n"])), reverse=True)[:20]
    top_loss = sorted(group_summaries, key=lambda r: (int(r["net_flip_gain"]), float(r["net_acc_gain"]), -int(r["n"])))[:20]
    top_fp_fix = sorted(group_summaries, key=lambda r: (int(r["fixed_bge_fp_to_tn"]), int(r["net_flip_gain"])), reverse=True)[:20]
    top_fn_fix = sorted(group_summaries, key=lambda r: (int(r["fixed_bge_fn_to_tp"]), int(r["net_flip_gain"])), reverse=True)[:20]

    overall_bge = metrics(y, bge_p, bge_yhat, c)
    overall_method = metrics(y, cand_p, cand_yhat, c)
    overall_delta = {
        "ap": round_float(float(overall_method["ap"]) - float(overall_bge["ap"])),
        "auroc": round_float(float(overall_method["auroc"]) - float(overall_bge["auroc"])),
        "macro_f1": round_float(float(overall_method["macro_f1"]) - float(overall_bge["macro_f1"])),
        "weighted_macro_f1": round_float(float(overall_method["weighted_macro_f1"]) - float(overall_bge["weighted_macro_f1"])),
        "accuracy": round_float(float(overall_method["accuracy"]) - float(overall_bge["accuracy"])),
    }

    return {
        "case": case_name,
        "description": spec["description"],
        "oof": str(path),
        "method": method,
        "baseline": "bge_lr",
        "valid_n": int(len(y)),
        "overall": {
            "bge_lr": overall_bge,
            "method": overall_method,
            "delta_method_minus_bge": overall_delta,
            "flips": {
                "fixed_total": int(fixed.sum()),
                "broken_total": int(broken.sum()),
                "net_flip_gain": int(fixed.sum() - broken.sum()),
                "both_ok": int((bge_ok & cand_ok).sum()),
                "both_wrong": int((~bge_ok & ~cand_ok).sum()),
                "fixed_bge_fp_to_tn": int((fixed & (y == 0) & (bge_yhat == 1)).sum()),
                "fixed_bge_fn_to_tp": int((fixed & (y == 1) & (bge_yhat == 0)).sum()),
                "broken_tn_to_fp": int((broken & (y == 0) & (cand_yhat == 1)).sum()),
                "broken_tp_to_fn": int((broken & (y == 1) & (cand_yhat == 0)).sum()),
            },
        },
        "top_gain_slices": top_gain,
        "top_loss_slices": top_loss,
        "top_bge_fp_fixed_slices": top_fp_fix,
        "top_bge_fn_fixed_slices": top_fn_fix,
        "all_slices_min_n": group_summaries,
    }


def parse_extra_case(raw: str) -> tuple[str, dict[str, str]]:
    try:
        name, rest = raw.split("=", 1)
        oof, method = rest.split("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("extra case must be NAME=OOF_PATH::METHOD_BASE") from exc
    return name, {"oof": oof, "method": method, "description": "User supplied case."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/final/cleancl/fallback_mechanism_diagnosis_20260608.json")
    ap.add_argument("--case", action="append", choices=sorted(CASE_SPECS), help="Default case to include; repeatable.")
    ap.add_argument("--extra_case", action="append", default=[], help="Add NAME=OOF_PATH::METHOD_BASE.")
    ap.add_argument("--min_group_n", type=int, default=25)
    args = ap.parse_args()

    root = Path.cwd()
    specs = dict(CASE_SPECS)
    for raw in args.extra_case:
        name, spec = parse_extra_case(raw)
        specs[name] = spec
    selected = args.case or list(CASE_SPECS)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": "bge_lr",
        "min_group_n": int(args.min_group_n),
        "cases": {},
    }
    for name in selected:
        result["cases"][name] = load_case(name, specs[name], root, args.min_group_n)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, case in result["cases"].items():
        delta = case["overall"]["delta_method_minus_bge"]
        flips = case["overall"]["flips"]
        best = case["top_gain_slices"][0] if case["top_gain_slices"] else {"group": "none", "net_flip_gain": 0}
        print(
            f"{name}: dAP={delta['ap']:+.4f} dAUROC={delta['auroc']:+.4f} "
            f"dMacroF1={delta['macro_f1']:+.4f} net_flips={flips['net_flip_gain']:+d} "
            f"top_gain={best['group']}({best['net_flip_gain']:+d})"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
