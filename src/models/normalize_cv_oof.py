"""Normalize cv_eval OOF dumps for prototype/evidence post-processing.

``cv_eval`` writes method arrays as ``p__method``/``yhat__method`` and stores
fold ids in ``fold_id``.  Later diagnostic scripts use ``method__p``/
``method__yhat`` plus explicit ``case``/``fold`` fields.  This utility bridges
those formats and can concatenate multiple repeated-CV cases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_labeled_path(text: str) -> tuple[str | None, Path]:
    if "=" not in text:
        return None, Path(text)
    label, path = text.split("=", 1)
    return label, Path(path)


def parse_rename(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --rename value: {item!r}; expected old=new")
        old, new = item.split("=", 1)
        out[old] = new
    return out


def source_bin(source_count: np.ndarray) -> np.ndarray:
    sc = np.asarray(source_count, int)
    out = np.full(len(sc), "src4p", dtype=object)
    out[sc <= 0] = "src0"
    out[sc == 1] = "src1"
    out[(sc >= 2) & (sc <= 3)] = "src2_3"
    return out


def method_names(z: np.lib.npyio.NpzFile, renames: dict[str, str]) -> list[str]:
    names = set()
    for key in z.files:
        if key.startswith("p__"):
            raw = key[3:]
            if f"yhat__{raw}" in z.files:
                names.add(renames.get(raw, raw))
        elif key.endswith("__p"):
            raw = key[:-3]
            if f"{raw}__yhat" in z.files:
                names.add(renames.get(raw, raw))
    return sorted(names)


def read_method(
    z: np.lib.npyio.NpzFile,
    name: str,
    inverse_renames: dict[str, str],
) -> tuple[np.ndarray, np.ndarray] | None:
    raw = inverse_renames.get(name, name)
    if f"p__{raw}" in z.files and f"yhat__{raw}" in z.files:
        p = np.asarray(z[f"p__{raw}"], float)
        yhat = np.asarray(z[f"yhat__{raw}"], float)
        return p, np.where(np.isnan(yhat), -1, yhat).astype(int)
    if f"{raw}__p" in z.files and f"{raw}__yhat" in z.files:
        p = np.asarray(z[f"{raw}__p"], float)
        yhat = np.asarray(z[f"{raw}__yhat"], float)
        return p, np.where(np.isnan(yhat), -1, yhat).astype(int)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="OOF npz path or label=OOF npz path. Repeat to concatenate cases.")
    ap.add_argument("--rename", action="append", default=[],
                    help="Rename method while normalizing, e.g. CLAIMARC_pcls=sourcefirst_cm_pcls_saved.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    renames = parse_rename(args.rename)
    inverse_renames = {new: old for old, new in renames.items()}
    inputs = [parse_labeled_path(x) for x in args.input]

    loaded = []
    all_methods = set()
    for label, path in inputs:
        z = np.load(path, allow_pickle=True)
        loaded.append((label, path, z))
        all_methods.update(method_names(z, renames))
    methods = sorted(all_methods)

    fields: dict[str, list[np.ndarray]] = {
        "y": [],
        "c": [],
        "case": [],
        "pair_id": [],
        "room_id": [],
        "attribute_id": [],
        "category": [],
        "source_count": [],
        "source_bin": [],
        "confidence": [],
        "evidence_combo": [],
        "fold": [],
    }
    method_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        m: {"p": [], "yhat": []} for m in methods
    }

    for label, path, z in loaded:
        n = len(z["y"])
        case_value = label
        if case_value is None and "case" in z.files:
            case_arr = np.asarray(z["case"], dtype=object)
        else:
            case_value = case_value or path.stem
            case_arr = np.full(n, case_value, dtype=object)

        sc = np.asarray(z["source_count"], float).astype(int)
        fields["y"].append(np.asarray(z["y"], float).astype(int))
        fields["c"].append(np.asarray(z["c"], float))
        fields["case"].append(case_arr.astype(object))
        for name in ("pair_id", "room_id", "attribute_id", "category", "confidence",
                     "evidence_combo"):
            fields[name].append(np.asarray(z[name], dtype=object))
        fields["source_count"].append(sc)
        if "source_bin" in z.files:
            fields["source_bin"].append(np.asarray(z["source_bin"], dtype=object))
        else:
            fields["source_bin"].append(source_bin(sc))
        if "fold" in z.files:
            fields["fold"].append(np.asarray(z["fold"], int))
        else:
            fields["fold"].append(np.asarray(z["fold_id"], int))

        for method in methods:
            item = read_method(z, method, inverse_renames)
            if item is None:
                method_arrays[method]["p"].append(np.full(n, np.nan, float))
                method_arrays[method]["yhat"].append(np.full(n, -1, int))
            else:
                p, yhat = item
                method_arrays[method]["p"].append(p)
                method_arrays[method]["yhat"].append(yhat)

    arrays = {name: np.concatenate(parts) for name, parts in fields.items()}
    for method in methods:
        arrays[f"{method}__p"] = np.concatenate(method_arrays[method]["p"])
        arrays[f"{method}__yhat"] = np.concatenate(method_arrays[method]["yhat"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[normalize_cv_oof] -> {out} cases={len(inputs)} n={len(arrays['y'])} methods={len(methods)}")
    for method in methods:
        ok = np.isfinite(arrays[f"{method}__p"]) & (arrays[f"{method}__yhat"] >= 0)
        print(f"  {method}: n={int(ok.sum())}")


if __name__ == "__main__":
    main()
