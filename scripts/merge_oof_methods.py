#!/usr/bin/env python
"""Merge method probability/yhat arrays from one OOF npz into another."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def key_rows(d: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    if "case" not in d or "pair_id" not in d:
        raise KeyError("both OOF files must contain case and pair_id arrays")
    case = np.asarray(d["case"], dtype=object).astype(str)
    pair = np.asarray(d["pair_id"], dtype=object).astype(str)
    return list(zip(case.tolist(), pair.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load(args.base)
    src = load(args.src)
    base_keys = key_rows(base)
    src_keys = key_rows(src)
    index = {k: i for i, k in enumerate(src_keys)}
    if len(index) != len(src_keys):
        raise ValueError("source OOF has duplicate (case, pair_id) keys")
    order = np.asarray([index[k] for k in base_keys], int)
    out = {k: np.asarray(v) for k, v in base.items()}
    if "y" in src:
        y_src = np.asarray(src["y"])[order]
        if "y" in out and not np.array_equal(np.asarray(out["y"]), y_src):
            raise ValueError("y mismatch after aligning source OOF")
    for method in args.method:
        for suffix in ("__p", "__yhat"):
            key = method + suffix
            if key not in src:
                raise KeyError(f"{key} not found in source OOF")
            out[key] = np.asarray(src[key])[order]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path} with {len(out)} arrays")


if __name__ == "__main__":
    main()
