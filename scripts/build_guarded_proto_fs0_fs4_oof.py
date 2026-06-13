#!/usr/bin/env python
"""Build a five-split OOF file with guarded CM/NLI decisions and RACL prototypes."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path("data/final/cleancl")

FS012 = ROOT / "oof_racl_proto_evtype_protocol_room_20260609.npz"
NLI = {
    "fs3": ROOT / "oof_nli_predef_lowabs_srcargs_drop_fs3_s0_nondropbge_cmpcls_evtype_fixed_room_alias_20260609.npz",
    "fs4": ROOT / "oof_nli_predef_lowabs_srcargs_drop_fs4_s0_nondropbge_cmpcls_evtype_fixed_room_alias_20260609.npz",
}
PROTO = {
    "fs3": ROOT / "oof_racl_proto_decision_feature_fs3_bge_room_20260609.npz",
    "fs4": ROOT / "oof_racl_proto_decision_feature_fs4_bge_20260609.npz",
}
OUT = ROOT / "oof_guarded_proto_fs0_fs4_room_20260609.npz"

PROTO_KEYS = [
    "bge_lr_saved__p",
    "bge_lr_saved__yhat",
    "proto_source_bin__p",
    "proto_source_bin__yhat",
    "rankavg_cm_proto_source_bin__p",
    "rankavg_cm_proto_source_bin__yhat",
    "rankavg_bge_cm_proto_source_bin__p",
    "rankavg_bge_cm_proto_source_bin__yhat",
]
PROTO_META_KEYS = ["evidence_combo", "attribute_id"]


def load(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def reorder_by_pair(src: dict[str, np.ndarray], pair_ids: np.ndarray, key: str) -> np.ndarray:
    src_pair = np.asarray(src["pair_id"], dtype=object).astype(str)
    index = {pid: i for i, pid in enumerate(src_pair)}
    return np.asarray(src[key])[[index[str(pid)] for pid in pair_ids]]


def build_case(case: str) -> dict[str, np.ndarray]:
    nli = load(NLI[case])
    proto = load(PROTO[case])
    pair_ids = np.asarray(nli["pair_id"], dtype=object).astype(str)
    out = {k: np.asarray(v) for k, v in nli.items()}
    out["case"] = np.asarray([case] * len(pair_ids), dtype=object)
    for key in PROTO_KEYS:
        out[key] = reorder_by_pair(proto, pair_ids, key)
    for key in PROTO_META_KEYS:
        if key in proto and key not in out:
            out[key] = reorder_by_pair(proto, pair_ids, key)
    for key in ("proto_source_bin", "rankavg_cm_proto_source_bin", "rankavg_bge_cm_proto_source_bin"):
        p_key = f"{key}__p"
        y_key = f"{key}__yhat"
        if p_key in out and y_key in out:
            continue
        raise KeyError(f"{case}: missing merged prototype method {key}")
    return out


def concat_blocks(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    common = set(blocks[0])
    for block in blocks[1:]:
        common &= set(block)
    out = {}
    for key in sorted(common):
        out[key] = np.concatenate([np.asarray(block[key]) for block in blocks])
    return out


def main() -> None:
    blocks = [load(FS012), build_case("fs3"), build_case("fs4")]
    out = concat_blocks(blocks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} with n={len(out['y'])} and {len(out)} arrays")


if __name__ == "__main__":
    main()
