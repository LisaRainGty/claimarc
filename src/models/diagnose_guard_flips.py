"""Diagnose binary-decision flips between two saved OOF methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from models.bootstrap_oof_methods import row
except ModuleNotFoundError:
    from bootstrap_oof_methods import row


def get_method(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def macro(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(f1_score(y, yhat, average="macro", zero_division=0))


def summarize_group(
    y: np.ndarray,
    y_base: np.ndarray,
    y_new: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "flips": 0,
            "veto": 0,
            "promote": 0,
            "net_correct": 0,
            "fixed_fp": 0,
            "introduced_fn": 0,
            "fixed_fn": 0,
            "introduced_fp": 0,
        }
    mb = mask
    flips = mb & (y_base != y_new)
    veto = flips & (y_base == 1) & (y_new == 0)
    promote = flips & (y_base == 0) & (y_new == 1)
    before_correct = (y_base[flips] == y[flips]).sum()
    after_correct = (y_new[flips] == y[flips]).sum()
    return {
        "n": int(mb.sum()),
        "flips": int(flips.sum()),
        "veto": int(veto.sum()),
        "promote": int(promote.sum()),
        "net_correct": int(after_correct - before_correct),
        "fixed_fp": int((veto & (y == 0)).sum()),
        "introduced_fn": int((veto & (y == 1)).sum()),
        "fixed_fn": int((promote & (y == 1)).sum()),
        "introduced_fp": int((promote & (y == 0)).sum()),
        "base_macro": round(macro(y[mb], y_base[mb]), 4),
        "new_macro": round(macro(y[mb], y_new[mb]), 4),
    }


def table_by(
    z: np.lib.npyio.NpzFile,
    y: np.ndarray,
    y_base: np.ndarray,
    y_new: np.ndarray,
    key: str,
) -> dict[str, dict[str, float | int]]:
    if key not in z.files:
        return {}
    values = np.asarray(z[key], dtype=object).astype(str)
    out = {}
    for value in sorted(set(values.tolist())):
        out[value] = summarize_group(y, y_base, y_new, values == value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float) if "c" in z.files else np.ones_like(y, float)
    p_base, y_base = get_method(z, args.base)
    p_new, y_new = get_method(z, args.method)
    if not np.allclose(p_base, p_new, equal_nan=True):
        score_delta = {
            "max_abs": float(np.nanmax(np.abs(p_new - p_base))),
            "mean_abs": float(np.nanmean(np.abs(p_new - p_base))),
        }
    else:
        score_delta = {"max_abs": 0.0, "mean_abs": 0.0}

    groups = {}
    for key in ("case", "source_count", "source_bin", "confidence", "evidence_combo", "category"):
        groups[key] = table_by(z, y, y_base, y_new, key)

    source0 = np.asarray(z["source_count"], int) == 0 if "source_count" in z.files else np.zeros_like(y, bool)
    flips = y_base != y_new
    out = {
        "oof": args.oof,
        "base": args.base,
        "method": args.method,
        "score_delta": score_delta,
        "metrics": {
            args.base: row(y, p_base, y_base, c),
            args.method: row(y, p_new, y_new, c),
        },
        "overall": summarize_group(y, y_base, y_new, np.ones_like(y, bool)),
        "source0": summarize_group(y, y_base, y_new, source0),
        "non_source0": summarize_group(y, y_base, y_new, ~source0),
        "groups": groups,
        "flip_rate": round(float(flips.mean()), 6),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[diagnose_guard_flips] -> {out_path}")
    print(json.dumps({
        "metrics": out["metrics"],
        "overall": out["overall"],
        "source0": out["source0"],
        "flip_rate": out["flip_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
