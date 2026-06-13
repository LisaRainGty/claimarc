#!/usr/bin/env python
"""Subset an OOF npz by values in its ``case`` array."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    if "case" not in z.files:
        raise KeyError(f"{args.oof} has no case array")
    cases = np.asarray(z["case"], dtype=object).astype(str)
    keep_values = set(str(x) for x in args.case)
    keep = np.asarray([x in keep_values for x in cases], dtype=bool)
    if not np.any(keep):
        raise ValueError(f"no rows matched case values {sorted(keep_values)}")

    n = len(cases)
    out = {}
    for key in z.files:
        value = np.asarray(z[key])
        if value.shape[:1] == (n,):
            out[key] = value[keep]
        else:
            out[key] = value
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path} with n={int(keep.sum())}")


if __name__ == "__main__":
    main()
