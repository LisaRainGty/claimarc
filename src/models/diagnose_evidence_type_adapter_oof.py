"""OOF screen for evidence-type adaptive adapters.

This script tests whether adaptive repairs can be restricted by observable
evidence-source structure instead of product taxonomy. It reads saved OOF
predictions and joins dataset evidence fields by ``pair_id``; no model is
trained or refit here.
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
TAXONOMY = "rankw_sourcefirst_cm040_nli060_score_sportsgeneral_cm025_decision_sports_cm025"


DEFAULT_CASES = {
    "fs0": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_adaptive_quick.npz",
    "fs1": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_adaptive_quick.npz",
    "fs2": "data/final/cleancl/"
    "oof_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_adaptive_quick.npz",
}


BASELINES = [
    "bge_lr",
    "rankavg_sourcefirst_cm_pcls_bge",
    CURRENT,
]


FIXED_METHODS = [
    ADAPTIVE,
    TAXONOMY,
]


CANDIDATES = {
    "evtype_adapt_score_src0_po_medium_decision_po_medium": {
        "score_rule": "src0_or_po_medium",
        "decision_rule": "po_medium",
        "description": (
            "Use adaptive score for source0 or params+OCR medium-confidence "
            "cases; use adaptive decision only for params+OCR medium-confidence cases."
        ),
        "primary": True,
    },
    "evtype_adapt_score_src0_no_vlm_medium_decision_no_vlm_medium": {
        "score_rule": "src0_or_no_vlm_medium",
        "decision_rule": "no_vlm_medium",
        "description": (
            "Evidence-type version without naming the params+OCR combo: apply "
            "adaptive repairs to medium-confidence cases without VLM evidence."
        ),
        "primary": True,
    },
    "evtype_adapt_score_src0_po_medium_decision_po_medium_bgeunc028": {
        "score_rule": "src0_or_po_medium_bgeunc028",
        "decision_rule": "po_medium_bgeunc028",
        "description": "Same as PO-medium adapter but only when BGE is not very confident.",
        "primary": False,
    },
    "evtype_diag_score_src0_po_medium_not_food_decision_po_medium_not_food": {
        "score_rule": "src0_or_po_medium_not_food",
        "decision_rule": "po_medium_not_food",
        "description": (
            "Diagnostic only: excludes food category after residual inspection; "
            "not suitable as a main method without prior justification."
        ),
        "primary": False,
    },
}


def method_key(method: str, suffix: str) -> str:
    return f"{method}__{suffix}"


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[method_key(method, "p")], float),
        np.asarray(z[method_key(method, "yhat")], int),
    )


def nonempty(value) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def evidence_combo(rec: dict) -> str:
    parts = []
    if nonempty(rec.get("evidence_params")):
        parts.append("P")
    if nonempty(rec.get("evidence_ocr")):
        parts.append("O")
    if nonempty(rec.get("evidence_vlm")):
        parts.append("V")
    return "".join(parts) if parts else "none"


def load_evidence_meta(path: Path) -> dict[str, dict[str, object]]:
    meta = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            pid = str(rec["pair_id"])
            combo = evidence_combo(rec)
            meta[pid] = {
                "evidence_combo": combo,
                "has_params": "P" in combo,
                "has_ocr": "O" in combo,
                "has_vlm": "V" in combo,
            }
    return meta


def context(z: np.lib.npyio.NpzFile, evidence_meta: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    combos, has_vlm, has_params, has_ocr = [], [], [], []
    for pid in np.asarray(z["pair_id"], dtype=object):
        item = evidence_meta[str(pid)]
        combos.append(item["evidence_combo"])
        has_vlm.append(item["has_vlm"])
        has_params.append(item["has_params"])
        has_ocr.append(item["has_ocr"])
    return {
        "evidence_combo": np.asarray(combos, dtype=object),
        "has_vlm": np.asarray(has_vlm, dtype=bool),
        "has_params": np.asarray(has_params, dtype=bool),
        "has_ocr": np.asarray(has_ocr, dtype=bool),
        "source_count": np.asarray(z["source_count"], int),
        "source_bin": np.asarray([str(x) for x in z["source_bin"]], dtype=object),
        "confidence": np.asarray([str(x) for x in z["confidence"]], dtype=object),
        "category": np.asarray([str(x) for x in z["category"]], dtype=object),
        "bge_p": np.asarray(z["bge_lr__p"], float),
    }


def mask_for(ctx: dict[str, np.ndarray], rule: str) -> np.ndarray:
    src0 = ctx["source_count"] == 0
    po_medium = (ctx["evidence_combo"] == "PO") & (ctx["confidence"] == "medium")
    no_vlm_medium = (~ctx["has_vlm"]) & (ctx["confidence"] == "medium")
    po_medium_bgeunc028 = po_medium & (np.abs(ctx["bge_p"] - 0.5) < 0.28)
    po_medium_not_food = po_medium & (ctx["category"] != "food_and_beverages")

    if rule == "src0":
        return src0
    if rule == "po_medium":
        return po_medium
    if rule == "no_vlm_medium":
        return no_vlm_medium
    if rule == "po_medium_bgeunc028":
        return po_medium_bgeunc028
    if rule == "po_medium_not_food":
        return po_medium_not_food
    if rule == "src0_or_po_medium":
        return src0 | po_medium
    if rule == "src0_or_no_vlm_medium":
        return src0 | no_vlm_medium
    if rule == "src0_or_po_medium_bgeunc028":
        return src0 | po_medium_bgeunc028
    if rule == "src0_or_po_medium_not_food":
        return src0 | po_medium_not_food
    raise KeyError(rule)


def synthesize_case(
    z: np.lib.npyio.NpzFile,
    evidence_meta: dict[str, dict[str, object]],
    spec: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ctx = context(z, evidence_meta)
    p_out, yhat_out = get_method(z, CURRENT)
    p_out = p_out.copy()
    yhat_out = yhat_out.copy()
    p_adapt, yhat_adapt = get_method(z, ADAPTIVE)
    score_mask = mask_for(ctx, spec["score_rule"])
    decision_mask = mask_for(ctx, spec["decision_rule"])
    p_out[score_mask] = p_adapt[score_mask]
    yhat_out[decision_mask] = yhat_adapt[decision_mask]
    return p_out, yhat_out, {
        "score_mask_n": int(score_mask.sum()),
        "decision_mask_n": int(decision_mask.sum()),
        "po_medium_n": int(mask_for(ctx, "po_medium").sum()),
        "no_vlm_medium_n": int(mask_for(ctx, "no_vlm_medium").sum()),
    }


def load_cases(root: Path) -> list[tuple[str, np.lib.npyio.NpzFile]]:
    return [(label, np.load(root / path, allow_pickle=True)) for label, path in DEFAULT_CASES.items()]


def build_fixed(cases, method: str):
    y_all, c_all, p_all, yhat_all = [], [], [], []
    case_metrics = {}
    for label, z in cases:
        y = np.asarray(z["y"], int)
        c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
        p, yhat = get_method(z, method)
        case_metrics[label] = row(y, p, yhat, c)
        y_all.append(y)
        c_all.append(c)
        p_all.append(p)
        yhat_all.append(yhat)
    y = np.concatenate(y_all)
    c = np.concatenate(c_all)
    p = np.concatenate(p_all)
    yhat = np.concatenate(yhat_all)
    return {
        "metrics": row(y, p, yhat, c),
        "case_metrics": case_metrics,
        "y": y,
        "c": c,
        "p": p,
        "yhat": yhat,
    }


def build_candidate(cases, evidence_meta, name: str, spec: dict):
    y_all, c_all, p_all, yhat_all = [], [], [], []
    case_metrics, counts = {}, {}
    for label, z in cases:
        y = np.asarray(z["y"], int)
        c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
        p, yhat, count = synthesize_case(z, evidence_meta, spec)
        case_metrics[label] = row(y, p, yhat, c)
        counts[label] = count
        y_all.append(y)
        c_all.append(c)
        p_all.append(p)
        yhat_all.append(yhat)
    y = np.concatenate(y_all)
    c = np.concatenate(c_all)
    p = np.concatenate(p_all)
    yhat = np.concatenate(yhat_all)
    return {
        "description": spec["description"],
        "primary": bool(spec.get("primary", False)),
        "score_rule": spec["score_rule"],
        "decision_rule": spec["decision_rule"],
        "metrics": row(y, p, yhat, c),
        "case_metrics": case_metrics,
        "counts": counts,
        "y": y,
        "c": c,
        "p": p,
        "yhat": yhat,
    }


def dump_oof(path: Path, cases, evidence_meta, built: dict[str, dict]) -> None:
    y_all, c_all, fold_all, case_all = [], [], [], []
    pair_all, source_count_all, source_bin_all, category_all, confidence_all, combo_all = [], [], [], [], [], []
    for label, z in cases:
        y_all.append(np.asarray(z["y"], int))
        c_all.append(np.asarray(z["c"], float))
        fold_all.append(np.asarray(z["fold"], int))
        case_all.append(np.asarray([label] * len(z["y"]), dtype=object))
        pair_ids = np.asarray(z["pair_id"], dtype=object)
        pair_all.append(pair_ids)
        source_count_all.append(np.asarray(z["source_count"], int))
        source_bin_all.append(np.asarray(z["source_bin"], dtype=object))
        category_all.append(np.asarray(z["category"], dtype=object))
        confidence_all.append(np.asarray(z["confidence"], dtype=object))
        combo_all.append(np.asarray([evidence_meta[str(pid)]["evidence_combo"] for pid in pair_ids], dtype=object))

    arrays = {
        "y": np.concatenate(y_all),
        "c": np.concatenate(c_all),
        "fold": np.concatenate(fold_all),
        "case": np.concatenate(case_all),
        "pair_id": np.concatenate(pair_all),
        "source_count": np.concatenate(source_count_all),
        "source_bin": np.concatenate(source_bin_all),
        "category": np.concatenate(category_all),
        "confidence": np.concatenate(confidence_all),
        "evidence_combo": np.concatenate(combo_all),
    }
    for name, item in built.items():
        arrays[f"{name}__p"] = item["p"]
        arrays[f"{name}__yhat"] = item["yhat"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--dataset",
        default="data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl",
    )
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_oof", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    evidence_meta = load_evidence_meta(root / args.dataset)
    cases = load_cases(root)
    methods = BASELINES + FIXED_METHODS
    built = {method: build_fixed(cases, method) for method in methods}
    for name, spec in CANDIDATES.items():
        built[name] = build_candidate(cases, evidence_meta, name, spec)

    out = {
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "dataset": args.dataset,
        "baselines": BASELINES,
        "fixed_methods": FIXED_METHODS,
        "candidates": {
            name: {
                k: v
                for k, v in CANDIDATES[name].items()
                if k in {"score_rule", "decision_rule", "description", "primary"}
            }
            for name in CANDIDATES
        },
        "cases": {},
        "pooled": {"metrics": {}, "significance": {}},
    }

    for label, _ in cases:
        out["cases"][label] = {
            "metrics": {
                name: item["case_metrics"][label]
                for name, item in built.items()
                if label in item.get("case_metrics", {})
            },
            "counts": {
                name: item["counts"][label]
                for name, item in built.items()
                if "counts" in item and label in item["counts"]
            },
        }

    for name, item in built.items():
        out["pooled"]["metrics"][name] = item["metrics"]

    for name, item in built.items():
        if name in BASELINES:
            continue
        for base in BASELINES:
            out["pooled"]["significance"][f"{name}_vs_{base}"] = paired_bootstrap(
                item["y"],
                item["p"],
                item["yhat"],
                built[base]["p"],
                built[base]["yhat"],
                n_boot=args.n_boot,
                seed=args.seed,
            )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    if args.dump_oof:
        dump_oof(root / args.dump_oof, cases, evidence_meta, built)
    print(json.dumps({
        "out": str(out_path),
        "dump_oof": args.dump_oof,
        "primary": {
            name: out["pooled"]["metrics"][name]
            for name, spec in CANDIDATES.items()
            if spec.get("primary", False)
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
