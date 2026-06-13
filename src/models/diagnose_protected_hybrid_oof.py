"""OOF-level screen for protected BGE + RACL/NLI hybrid rules.

This is an exploratory, no-training screen. It combines fold-safe OOF scores
and decisions that were already produced by the CV evaluators, then evaluates
small predefined routing rules against BGE+LR. Promising rules still need to be
implemented in a fold-level evaluator before becoming a paper claim.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from models.diagnose_fallback_mechanisms import (
        auroc,
        average_precision,
        lowabs,
        macro_f1,
        metrics,
    )
except ModuleNotFoundError:
    from diagnose_fallback_mechanisms import (
        auroc,
        average_precision,
        lowabs,
        macro_f1,
        metrics,
    )


CASE_SPECS = {
    "fs0_old": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick5k.npz",
        "reference": "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
    },
    "fs0_newcache": {
        "oof": "data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_5k.npz",
        "reference": "rankmix_nli25_hgb_bge_scorefallback_bge100_src0",
    },
    "fs1_predef": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs.npz",
        "reference": "predef_protocol_predef_lowabs_r25_scorefallback_srcconf_bgefallback",
    },
    "fs2_bgefallback": {
        "oof": "data/final/cleancl/oof_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback.npz",
        "reference": "dual_score_rankmix_nli25_hgb_bge__decision_rankmix_nli25_hgb_bge_scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3",
    },
}


def as_int(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(int)


def as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(float)


def as_str(x: np.ndarray) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(x, dtype=object)], dtype=object)


def rank01(x: np.ndarray) -> np.ndarray:
    x = as_float(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def source_bin_from_count(sc: np.ndarray) -> np.ndarray:
    out = []
    for n in as_int(sc):
        if n <= 0:
            out.append("src0")
        elif n == 1:
            out.append("src1")
        elif n <= 3:
            out.append("src2_3")
        else:
            out.append("src4p")
    return np.asarray(out, dtype=object)


def blend_score(base: np.ndarray, bge: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    out = as_float(base).copy()
    bge_rank = rank01(bge)
    m = np.asarray(mask, dtype=bool)
    out[m] = (1.0 - float(weight)) * out[m] + float(weight) * bge_rank[m]
    return out


def protect_decision(base: np.ndarray, bge: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = as_int(base).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = as_int(bge)[m]
    return out


def scoreguard_keys() -> list[str]:
    return [
        "dual_score_rankmix_nli25_hgb_bge__decision_scoreguard_clip_drop20_min30_confidence",
        "dual_score_rankmix_nli25_hgb_bge__decision_scoreguard_clip_drop20_min30_srcbin_conf",
    ]


def score_key(z: np.lib.npyio.NpzFile, base: str) -> tuple[np.ndarray, np.ndarray] | None:
    p_key = f"{base}__p"
    y_key = f"{base}__yhat"
    if p_key in z.files and y_key in z.files:
        return as_float(z[p_key]), as_int(z[y_key])
    return None


def paired_bootstrap(y, p_a, yhat_a, p_b, yhat_b, n_boot: int, seed: int = 0) -> dict[str, dict[str, object]]:
    rng = np.random.RandomState(seed)
    y = as_int(y)
    p_a = as_float(p_a)
    p_b = as_float(p_b)
    yhat_a = as_int(yhat_a)
    yhat_b = as_int(yhat_b)
    dap_l, dau_l, df1_l = [], [], []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, len(y), len(y))
        yy = y[idx]
        if int(yy.sum()) == 0 or int(yy.sum()) == len(yy):
            continue
        dap_l.append(average_precision(yy, p_a[idx]) - average_precision(yy, p_b[idx]))
        dau_l.append(auroc(yy, p_a[idx]) - auroc(yy, p_b[idx]))
        df1_l.append(macro_f1(yy, yhat_a[idx]) - macro_f1(yy, yhat_b[idx]))

    def summ(vals: list[float]) -> dict[str, object]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {"dAP": summ(dap_l), "dAUROC": summ(dau_l), "dMacroF1": summ(df1_l)}


def build_candidates(z: np.lib.npyio.NpzFile) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    y = as_int(z["y"])
    bge_p = as_float(z["bge_lr__p"])
    bge_yhat = as_int(z["bge_lr__yhat"])
    source_count = as_int(z["source_count"]) if "source_count" in z.files else np.zeros(len(y), dtype=int)
    source_bin = as_str(z["source_bin"]) if "source_bin" in z.files else source_bin_from_count(source_count)
    confidence = as_str(z["confidence"]) if "confidence" in z.files else np.asarray(["unknown"] * len(y), dtype=object)
    lowabs_mask = np.asarray([lowabs(x) for x in confidence], dtype=bool)
    src0 = source_bin == "src0"
    src2_3 = source_bin == "src2_3"
    src_rich = source_count >= 2
    protected_lowabs = src0 | (src2_3 & lowabs_mask)
    protected_src0_src23 = src0 | src2_3
    bge_uncertain = np.abs(bge_p - 0.5) < 0.08
    bge_uncertain_wide = np.abs(bge_p - 0.5) < 0.16

    rank25 = score_key(z, "rankmix_nli25_hgb_bge")
    if rank25 is None:
        return {}
    rank25_p, rank25_y = rank25

    score_heads: dict[str, np.ndarray] = {
        "rank25": rank25_p,
        "rank25_bge025_lowabs": blend_score(rank25_p, bge_p, protected_lowabs, 0.25),
        "rank25_bge050_lowabs": blend_score(rank25_p, bge_p, protected_lowabs, 0.50),
        "rank25_bge025_unc08": blend_score(rank25_p, bge_p, bge_uncertain, 0.25),
        "rank25_bge025_unc16": blend_score(rank25_p, bge_p, bge_uncertain_wide, 0.25),
        "rank25_bge025_src0_unc08": blend_score(rank25_p, bge_p, src0 | bge_uncertain, 0.25),
    }
    for base in (
        "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs",
        "rankmix_nli25_hgb_bge_scorefallback_bge100_src0",
        "rankmix_nli25_hgb_bge_scorefallback_bge025_src0",
        "rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3",
    ):
        got = score_key(z, base)
        if got is not None:
            score_heads[base.replace("rankmix_nli25_hgb_bge_", "")] = got[0]

    decision_heads: dict[str, np.ndarray] = {
        "rank25_thr": rank25_y,
        "protect_src0_rank25": protect_decision(rank25_y, bge_yhat, src0),
        "protect_lowabs_rank25": protect_decision(rank25_y, bge_yhat, protected_lowabs),
        "protect_src0_src23_rank25": protect_decision(rank25_y, bge_yhat, protected_src0_src23),
        "protect_lowabs_or_unc08_rank25": protect_decision(rank25_y, bge_yhat, protected_lowabs | bge_uncertain),
        "protect_src0_else_rank25_on_srcge2": protect_decision(rank25_y, bge_yhat, (~src_rich) | src0),
    }
    for base in scoreguard_keys():
        got = score_key(z, base)
        if got is None:
            continue
        key = base.replace("dual_score_rankmix_nli25_hgb_bge__decision_", "")
        decision_heads[key] = got[1]
        decision_heads[f"protect_lowabs_{key}"] = protect_decision(got[1], bge_yhat, protected_lowabs)
        decision_heads[f"protect_src0_src23_{key}"] = protect_decision(got[1], bge_yhat, protected_src0_src23)

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for score_name, p in score_heads.items():
        for dec_name, yhat in decision_heads.items():
            candidates[f"score={score_name}__decision={dec_name}"] = (p, yhat)
    return candidates


def summarize_case(name: str, spec: dict[str, str], root: Path, n_boot: int, top_k: int) -> dict[str, object]:
    path = root / spec["oof"]
    z = np.load(path, allow_pickle=True)
    y = as_int(z["y"])
    c = as_float(z["c"]) if "c" in z.files else np.ones(len(y), dtype=float)
    bge_p = as_float(z["bge_lr__p"])
    bge_yhat = as_int(z["bge_lr__yhat"])
    out: dict[str, object] = {
        "oof": str(path),
        "baseline": metrics(y, bge_p, bge_yhat, c),
        "reference": {},
        "candidates": {},
    }
    ref = spec.get("reference", "")
    ref_got = score_key(z, ref) if ref else None
    if ref_got is not None:
        rp, ry = ref_got
        out["reference"] = {
            "method": ref,
            "metrics": metrics(y, rp, ry, c),
            "vs_bge": paired_bootstrap(y, rp, ry, bge_p, bge_yhat, n_boot=n_boot),
        }

    candidates = build_candidates(z)
    rows = []
    for cand_name, (p, yhat) in candidates.items():
        m = metrics(y, p, yhat, c)
        delta = {
            "ap": round(float(m["ap"]) - float(out["baseline"]["ap"]), 4),
            "auroc": round(float(m["auroc"]) - float(out["baseline"]["auroc"]), 4),
            "macro_f1": round(float(m["macro_f1"]) - float(out["baseline"]["macro_f1"]), 4),
            "weighted_macro_f1": round(float(m["weighted_macro_f1"]) - float(out["baseline"]["weighted_macro_f1"]), 4),
        }
        rows.append((cand_name, m, delta, p, yhat))
    rows.sort(key=lambda r: (r[2]["macro_f1"], r[2]["ap"], r[2]["auroc"]), reverse=True)
    for cand_name, m, delta, p, yhat in rows[:top_k]:
        out["candidates"][cand_name] = {
            "metrics": m,
            "delta_vs_bge": delta,
            "vs_bge": paired_bootstrap(y, p, yhat, bge_p, bge_yhat, n_boot=n_boot),
        }
    return out


def cross_case_summary(cases: dict[str, object]) -> list[dict[str, object]]:
    by_name: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for case_name, case in cases.items():
        for cand_name, row in case["candidates"].items():
            by_name.setdefault(cand_name, []).append((case_name, row))
    summary = []
    for cand_name, vals in by_name.items():
        pos_all = 0
        strict_all = 0
        for _, row in vals:
            d = row["delta_vs_bge"]
            sig = row["vs_bge"]
            if d["ap"] > 0 and d["auroc"] > 0 and d["macro_f1"] > 0:
                pos_all += 1
            if (
                sig["dAP"]["p_a_gt_b"] <= 0.05
                and sig["dAUROC"]["p_a_gt_b"] <= 0.05
                and sig["dMacroF1"]["p_a_gt_b"] <= 0.05
            ):
                strict_all += 1
        summary.append({
            "candidate": cand_name,
            "n_cases_ranked": len(vals),
            "n_cases_positive_all3": pos_all,
            "n_cases_strict_all3": strict_all,
            "cases": {case_name: row["delta_vs_bge"] for case_name, row in vals},
        })
    summary.sort(
        key=lambda r: (
            int(r["n_cases_strict_all3"]),
            int(r["n_cases_positive_all3"]),
            int(r["n_cases_ranked"]),
        ),
        reverse=True,
    )
    return summary[:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/final/cleancl/protected_hybrid_oof_screen_20260608.json")
    ap.add_argument("--case", action="append", choices=sorted(CASE_SPECS))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    root = Path.cwd()
    selected = args.case or list(CASE_SPECS)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "OOF-level exploratory protected BGE + RACL/NLI hybrid screen",
        "n_boot": int(args.n_boot),
        "top_k_per_case": int(args.top_k),
        "cases": {},
    }
    for name in selected:
        result["cases"][name] = summarize_case(name, CASE_SPECS[name], root, args.n_boot, args.top_k)
    result["cross_case_summary"] = cross_case_summary(result["cases"])

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for case_name, case in result["cases"].items():
        print(f"\n## {case_name}")
        ref = case.get("reference") or {}
        if ref:
            print(f"reference {ref['method']} {ref['metrics']} {ref['vs_bge']}")
        for cand_name, row in list(case["candidates"].items())[:5]:
            sig = row["vs_bge"]
            d = row["delta_vs_bge"]
            print(
                f"{cand_name} d={d} "
                f"p=({sig['dAP']['p_a_gt_b']},{sig['dAUROC']['p_a_gt_b']},{sig['dMacroF1']['p_a_gt_b']})"
            )
    print("\n## cross-case")
    for row in result["cross_case_summary"][:10]:
        print(row)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
