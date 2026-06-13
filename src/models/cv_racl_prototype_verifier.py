"""RACL embedding prototype verifier over saved grouped-CV bundles.

This script probes whether the CLAIMARC retrieval embedding ``g`` contains a
usable relation/sufficiency geometry.  For every outer fold it builds positive
and negative prototypes from that fold's training embeddings only, scores the
validation/test rows by prototype similarity gaps, chooses thresholds on the
validation split, and emits OOF predictions for the held-out fold.

No model is retrained here.  The experiment is intended as a leakage-safe
diagnostic before investing GPU time into a trainable verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

try:
    from models.bootstrap_oof_methods import paired_bootstrap, row
except ModuleNotFoundError:
    from bootstrap_oof_methods import paired_bootstrap, row


BASE_OOF = "data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz"
DATASET = "data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl"
CASES = {
    "fs0": "data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs0_s0",
    "fs1": "data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs1_s0",
    "fs2": "data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs2_s0",
}

BGE = "bge_lr"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
CURRENT = (
    "rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_"
    "lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect"
)
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"


def macro(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(f1_score(y, yhat, average="macro", zero_division=0))


def best_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best = 0.5, -1.0
    for t in np.linspace(float(np.min(p)), float(np.max(p)), 49):
        score = macro(y, (p >= t).astype(int))
        if score > best:
            best = score
            best_t = float(t)
    return best_t


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


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


def source_count(rec: dict) -> int:
    ev = rec.get("evidence_count", {}) or {}
    return sum(int(ev.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))


def source_bin(sc: int) -> str:
    if sc <= 0:
        return "src0"
    if sc == 1:
        return "src1"
    if sc <= 3:
        return "src2_3"
    return "src4p"


def load_meta(path: str | Path) -> dict[str, dict[str, object]]:
    meta = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec["pair_id"])
            sc = source_count(rec)
            combo = evidence_combo(rec)
            conf = str(rec.get("confidence", ""))
            meta[pid] = {
                "combo": combo,
                "source_count": sc,
                "source_bin": source_bin(sc),
                "confidence": conf,
                "source_conf": f"{source_bin(sc)}:{conf}",
                "combo_conf": f"{combo}:{conf}",
                "category": str(rec.get("category", "")),
            }
    return meta


def bundle_to_np(bundle: dict, split: str) -> dict[str, np.ndarray | list[str]]:
    item = bundle[split]
    out = {}
    for key in ("g", "p", "y", "c"):
        value = item.get(key)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        out[key] = np.asarray(value)
    out["attr"] = [str(x) for x in item["attr"]]
    out["pair_id"] = [str(x) for x in item["pair_id"]]
    return out


def load_torch(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def weighted_proto(g: np.ndarray, w: np.ndarray) -> np.ndarray:
    proto = (g * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)
    norm = np.linalg.norm(proto)
    return proto / max(float(norm), 1e-8)


def build_proto_map(
    g: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    keys: Iterable[str],
    min_class: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    g = normalize(g)
    y = np.asarray(y, int)
    c = np.clip(np.asarray(c, float), 0.05, None)
    keys = np.asarray([str(k) for k in keys], dtype=object)
    out = {}
    for key in sorted(set(keys.tolist())):
        m = keys == key
        pos = m & (y == 1)
        neg = m & (y == 0)
        if int(pos.sum()) < int(min_class) or int(neg.sum()) < int(min_class):
            continue
        out[str(key)] = (weighted_proto(g[pos], c[pos]), weighted_proto(g[neg], c[neg]))
    return out


def score_from_proto(
    g: np.ndarray,
    keys: Iterable[str],
    protos: dict[str, tuple[np.ndarray, np.ndarray]],
    global_proto: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    g = normalize(g)
    keys = [str(k) for k in keys]
    out = np.zeros(len(keys), float)
    for i, key in enumerate(keys):
        pos, neg = protos.get(key, global_proto)
        out[i] = float(g[i] @ pos - g[i] @ neg)
    return out


def meta_keys(pair_ids: list[str], meta: dict[str, dict[str, object]], field: str) -> list[str]:
    return [str(meta[pid][field]) for pid in pair_ids]


def fold_proto_scores(
    train: dict,
    query: dict,
    meta: dict[str, dict[str, object]],
    min_class: int,
) -> dict[str, np.ndarray]:
    gtr = np.asarray(train["g"], float)
    ytr = np.asarray(train["y"], int)
    ctr = np.asarray(train["c"], float)
    gq = np.asarray(query["g"], float)

    global_proto = (
        weighted_proto(normalize(gtr)[ytr == 1], np.clip(ctr[ytr == 1], 0.05, None)),
        weighted_proto(normalize(gtr)[ytr == 0], np.clip(ctr[ytr == 0], 0.05, None)),
    )
    key_defs = {
        "global": (["all"] * len(train["pair_id"]), ["all"] * len(query["pair_id"])),
        "attr": (train["attr"], query["attr"]),
        "combo": (
            meta_keys(train["pair_id"], meta, "combo"),
            meta_keys(query["pair_id"], meta, "combo"),
        ),
        "source_conf": (
            meta_keys(train["pair_id"], meta, "source_conf"),
            meta_keys(query["pair_id"], meta, "source_conf"),
        ),
        "combo_conf": (
            meta_keys(train["pair_id"], meta, "combo_conf"),
            meta_keys(query["pair_id"], meta, "combo_conf"),
        ),
        "source_bin": (
            meta_keys(train["pair_id"], meta, "source_bin"),
            meta_keys(query["pair_id"], meta, "source_bin"),
        ),
        "confidence": (
            meta_keys(train["pair_id"], meta, "confidence"),
            meta_keys(query["pair_id"], meta, "confidence"),
        ),
    }
    scores = {}
    for name, (tr_keys, q_keys) in key_defs.items():
        protos = build_proto_map(gtr, ytr, ctr, tr_keys, min_class)
        scores[f"proto_{name}"] = score_from_proto(gq, q_keys, protos, global_proto)
    scores["proto_mean_struct"] = np.mean(
        np.vstack([
            scores["proto_attr"],
            scores["proto_combo_conf"],
            scores["proto_source_conf"],
            scores["proto_global"],
        ]),
        axis=0,
    )
    scores["proto_mean_all"] = np.mean(np.vstack(list(scores.values())), axis=0)
    return scores


def put_method(
    out: dict[str, dict[str, np.ndarray]],
    name: str,
    idx: np.ndarray,
    val_y: np.ndarray,
    val_p: np.ndarray,
    test_p: np.ndarray,
) -> float:
    thr = best_thr(val_y, val_p)
    out.setdefault(name, {
        "p": np.full(out["_n"], np.nan, float),
        "yhat": np.full(out["_n"], np.nan, float),
    })
    out[name]["p"][idx] = test_p
    out[name]["yhat"][idx] = (test_p >= thr).astype(int)
    return float(thr)


def method_metric(y: np.ndarray, p: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    return (
        float(average_precision_score(y, p)),
        float(roc_auc_score(y, p)),
        float(macro(y, yhat)),
    )


def choose_proto(val_y: np.ndarray, val_scores: dict[str, np.ndarray]) -> str:
    best = None
    for name, score in val_scores.items():
        thr = best_thr(val_y, score)
        yhat = (score >= thr).astype(int)
        ap, au, mf = method_metric(val_y, score, yhat)
        obj = ap + 0.5 * au + 0.2 * mf
        if best is None or obj > best[0]:
            best = (obj, name)
    return str(best[1])


def parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.stem, path
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--base_oof", default=BASE_OOF)
    ap.add_argument("--case", action="append",
                    help="label=tmpdir; defaults to fs0/fs1/fs2 drop-src0args bundles")
    ap.add_argument("--cm_seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_class", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=0,
                    help="Default 0 for fast screening; use bootstrap_oof_methods for targeted comparisons.")
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--out", default="data/final/cleancl/racl_prototype_verifier_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_racl_prototype_verifier_20260609.npz")
    args = ap.parse_args()

    base = np.load(args.base_oof, allow_pickle=True)
    y = np.asarray(base["y"], int)
    c = np.asarray(base["c"], float)
    cases = np.asarray(base["case"], dtype=object).astype(str)
    pair_ids = np.asarray(base["pair_id"], dtype=object).astype(str)
    index = {(str(case), str(pid)): i for i, (case, pid) in enumerate(zip(cases, pair_ids))}
    meta = load_meta(args.dataset)
    case_specs = (
        [parse_case(x) for x in args.case]
        if args.case else [(k, Path(v)) for k, v in CASES.items()]
    )

    out: dict[str, object] = {"_n": len(y)}
    fold_meta = []
    proto_names = [
        "proto_global", "proto_attr", "proto_combo", "proto_source_conf",
        "proto_combo_conf", "proto_source_bin", "proto_confidence",
        "proto_mean_struct", "proto_mean_all",
    ]

    for case, tmpdir in case_specs:
        for fold in range(int(args.folds)):
            cm = load_torch(tmpdir / f"cv_cm_f{fold}_s{args.cm_seed}.pt")
            bge = load_torch(tmpdir / f"cv_bge_lr_f{fold}.pt")
            tr = bundle_to_np(cm, "train")
            va = bundle_to_np(cm, "val")
            te = bundle_to_np(cm, "test")
            idx = np.asarray([index[(case, pid)] for pid in te["pair_id"]], int)
            yv = np.asarray(va["y"], int)
            bge_v = np.asarray(bge["val"]["p"], float)
            bge_t = np.asarray(bge["test"]["p"], float)
            cm_v = np.asarray(va["p"], float)
            cm_t = np.asarray(te["p"], float)
            proto_v = fold_proto_scores(tr, va, meta, args.min_class)
            proto_t = fold_proto_scores(tr, te, meta, args.min_class)

            put_method(out, "sourcefirst_cm_pcls_saved", idx, yv, cm_v, cm_t)
            put_method(out, "bge_lr_saved", idx, yv, bge_v, bge_t)
            score_v = 0.5 * rank01(cm_v) + 0.5 * rank01(bge_v)
            score_t = 0.5 * rank01(cm_t) + 0.5 * rank01(bge_t)
            put_method(out, "rankavg_saved_cm_bge", idx, yv, score_v, score_t)

            selected_proto = choose_proto(yv, {k: proto_v[k] for k in proto_names})
            for name in proto_names:
                put_method(out, name, idx, yv, proto_v[name], proto_t[name])
                rv = 0.5 * rank01(cm_v) + 0.5 * rank01(proto_v[name])
                rt = 0.5 * rank01(cm_t) + 0.5 * rank01(proto_t[name])
                put_method(out, f"rankavg_cm_{name}", idx, yv, rv, rt)
                rv3 = (rank01(bge_v) + rank01(cm_v) + rank01(proto_v[name])) / 3.0
                rt3 = (rank01(bge_t) + rank01(cm_t) + rank01(proto_t[name])) / 3.0
                put_method(out, f"rankavg_bge_cm_{name}", idx, yv, rv3, rt3)
            put_method(
                out,
                "proto_valselect",
                idx,
                yv,
                proto_v[selected_proto],
                proto_t[selected_proto],
            )
            put_method(
                out,
                "rankavg_bge_cm_proto_valselect",
                idx,
                yv,
                (rank01(bge_v) + rank01(cm_v) + rank01(proto_v[selected_proto])) / 3.0,
                (rank01(bge_t) + rank01(cm_t) + rank01(proto_t[selected_proto])) / 3.0,
            )
            fold_meta.append({
                "case": case,
                "fold": fold,
                "n_train": int(len(tr["y"])),
                "n_val": int(len(va["y"])),
                "n_test": int(len(te["y"])),
                "selected_proto": selected_proto,
            })
            print(f"[racl_proto] case={case} fold={fold} proto={selected_proto}", flush=True)

    methods = [k for k in out if k != "_n"]
    rows = {}
    for method in methods:
        p = out[method]["p"]
        yhat = out[method]["yhat"]
        ok = (~np.isnan(p)) & (~np.isnan(yhat))
        rows[method] = row(y[ok], p[ok], yhat[ok].astype(int), c[ok])

    baselines = [BGE, CMBGE, CURRENT, EVTYPE]
    for method in baselines:
        if f"{method}__p" in base.files:
            rows[method] = row(
                y,
                np.asarray(base[f"{method}__p"], float),
                np.asarray(base[f"{method}__yhat"], int),
                c,
            )

    ranked = sorted(
        methods,
        key=lambda m: (rows[m]["auprc"], rows[m]["auroc"], rows[m]["macro_f1"]),
        reverse=True,
    )

    sig = {}
    if args.n_boot > 0:
        compare = ranked[:8]
        for mi, method in enumerate(compare):
            p_a = out[method]["p"]
            y_a = out[method]["yhat"]
            for bi, base_name in enumerate(baselines):
                if f"{base_name}__p" not in base.files:
                    continue
                p_b = np.asarray(base[f"{base_name}__p"], float)
                y_b = np.asarray(base[f"{base_name}__yhat"], int)
                ok = (~np.isnan(p_a)) & (~np.isnan(y_a)) & (~np.isnan(p_b))
                sig[f"{method}_vs_{base_name}"] = paired_bootstrap(
                    y[ok], p_a[ok], y_a[ok].astype(int), p_b[ok], y_b[ok],
                    n_boot=args.n_boot,
                    seed=args.seed + mi * 101 + bi * 17,
                )

    result = {
        "description": (
            "RACL embedding positive/negative prototype verifier. Prototypes are "
            "built from each outer fold's train embeddings only; validation is "
            "used only for thresholds and the diagnostic prototype choice."
        ),
        "dataset": args.dataset,
        "base_oof": args.base_oof,
        "cm_seed": int(args.cm_seed),
        "min_class": int(args.min_class),
        "fold_meta": fold_meta,
        "ranked_methods": ranked,
        "metrics": rows,
        "n_boot": int(args.n_boot),
        "significance": sig,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=2)

    if args.dump_oof:
        arrays = {
            "y": y,
            "c": c,
            "case": cases,
            "pair_id": pair_ids,
            "fold": np.asarray(base["fold"], int),
            "source_count": np.asarray(base["source_count"], int),
            "source_bin": np.asarray(base["source_bin"], dtype=object),
            "category": np.asarray(base["category"], dtype=object),
            "confidence": np.asarray(base["confidence"], dtype=object),
            "evidence_combo": np.asarray(base["evidence_combo"], dtype=object),
        }
        for meta_key in ("room_id", "attribute_id"):
            if meta_key in base.files:
                arrays[meta_key] = np.asarray(base[meta_key], dtype=object)
        for method in baselines:
            if f"{method}__p" in base.files:
                arrays[f"{method}__p"] = np.asarray(base[f"{method}__p"], float)
                arrays[f"{method}__yhat"] = np.asarray(base[f"{method}__yhat"], int)
        for method in methods:
            arrays[f"{method}__p"] = out[method]["p"]
            yhat = out[method]["yhat"]
            arrays[f"{method}__yhat"] = np.where(np.isnan(yhat), -1, yhat).astype(int)
        dump_path = Path(args.dump_oof)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **arrays)

    print(f"[cv_racl_prototype_verifier] -> {out_path}", flush=True)
    for method in ranked[:12]:
        r = rows[method]
        print(f"{method:72s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
