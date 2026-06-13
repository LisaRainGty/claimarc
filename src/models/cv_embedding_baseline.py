"""Grouped-CV frozen embedding + LR baselines for arbitrary SentenceTransformer encoders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

try:
    from models.baselines import claim_text, evidence_text
    from models.bootstrap_oof_methods import paired_bootstrap, row
except ModuleNotFoundError:
    from baselines import claim_text, evidence_text
    from bootstrap_oof_methods import paired_bootstrap, row


def load_recs(path: str | Path) -> list[dict]:
    recs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def make_folds(recs: list[dict], n_folds: int, seed: int):
    y = np.asarray([int(r["y"]) for r in recs], int)
    g = np.asarray([r.get("room_id", r.get("product_id", i)) for i, r in enumerate(recs)], dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(recs)), y, g)), y, g


def val_carve(train_idx, g, frac=0.15, seed=0):
    rng = np.random.RandomState(seed)
    rooms = sorted({g[i] for i in train_idx}, key=lambda x: str(x))
    rng.shuffle(rooms)
    nval = max(1, int(len(rooms) * frac))
    val_rooms = set(rooms[:nval])
    val = np.asarray([i for i in train_idx if g[i] in val_rooms], int)
    tr = np.asarray([i for i in train_idx if g[i] not in val_rooms], int)
    return tr, val


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def best_thr(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    lo, hi = float(np.nanmin(p)), float(np.nanmax(p))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.5
    grid = np.linspace(lo, hi, 101)
    return float(max(grid, key=lambda t: f1_score(y, p >= t, average="macro", zero_division=0)))


def method_name(model_name: str) -> str:
    name = model_name.rstrip("/").split("/")[-1].lower()
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() else "_")
    return "emb_" + "".join(keep).strip("_") + "_lr"


def encode_texts(model, texts: list[str], batch_size: int, prompt_name: str = "", prompt: str = "") -> np.ndarray:
    kwargs = {
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "show_progress_bar": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    if prompt:
        kwargs["prompt"] = prompt
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def load_or_encode(args, recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(args.cache) if args.cache else None
    pair_ids = np.asarray([str(r.get("pair_id", "")) for r in recs], dtype=object)
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(np.asarray(z["pair_id"], dtype=object).astype(str), pair_ids.astype(str)):
            print(f"[cache] loaded {cache}", flush=True)
            return np.asarray(z["claim"], np.float32), np.asarray(z["evidence"], np.float32)
        raise ValueError(f"cache pair_id mismatch: {cache}")

    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(args.model_name, device=args.device, trust_remote_code=args.trust_remote_code)
    except TypeError:
        # Older sentence-transformers releases do not expose trust_remote_code.
        model = SentenceTransformer(args.model_name, device=args.device)
    claims = [claim_text(r) for r in recs]
    evidence = [evidence_text(r) for r in recs]
    print(f"[encode] model={args.model_name} n={len(recs)} device={args.device}", flush=True)
    Xc = encode_texts(model, claims, args.batch_size, args.claim_prompt_name, args.claim_prompt)
    Xe = encode_texts(model, evidence, args.batch_size, args.evidence_prompt_name, args.evidence_prompt)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, pair_id=pair_ids, claim=Xc, evidence=Xe,
                            model_name=args.model_name)
        print(f"[cache] saved {cache}", flush=True)
    return Xc, Xe


def build_features(Xc: np.ndarray, Xe: np.ndarray, idx: np.ndarray) -> np.ndarray:
    c = Xc[idx]
    e = Xe[idx]
    return np.concatenate([c, e, c - e, c * e], axis=1)


def run(args) -> tuple[dict, dict[str, np.ndarray]]:
    recs = load_recs(args.dataset)
    Xc, Xe = load_or_encode(args, recs)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    mname = args.method_name or method_name(args.model_name)

    pieces = []
    meta = []
    for fold_seed in args.fold_seed:
        folds, y, g = make_folds(recs, args.folds, seed=fold_seed)
        p_oof = np.full(len(recs), np.nan, float)
        yhat_oof = np.full(len(recs), -1, int)
        fold_oof = np.full(len(recs), -1, int)
        for fi, (trainval_idx, test_idx) in enumerate(folds):
            train_idx, val_idx = val_carve(trainval_idx, g, seed=fold_seed * 1000 + fi)
            Xtr = build_features(Xc, Xe, train_idx)
            Xv = build_features(Xc, Xe, val_idx)
            Xt = build_features(Xc, Xe, test_idx)
            clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                                     class_weight=("balanced" if args.class_weight_balanced else None))
            clf.fit(Xtr, y[train_idx], sample_weight=np.clip(c_all[train_idx], 0.05, None))
            pv = clf.predict_proba(Xv)[:, 1]
            pt = clf.predict_proba(Xt)[:, 1]
            thr = best_thr(y[val_idx], pv)
            p_oof[test_idx] = pt
            yhat_oof[test_idx] = (pt >= thr).astype(int)
            fold_oof[test_idx] = fi
            meta.append({
                "case": f"fs{fold_seed}",
                "fold": int(fi),
                "thr": round(float(thr), 6),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
            })
            print(f"[fold] fs{fold_seed} f{fi} thr={thr:.4f}", flush=True)
        pieces.append({
            "case": np.asarray([f"fs{fold_seed}"] * len(recs), dtype=object),
            "fold": fold_oof,
            f"{mname}__p": p_oof,
            f"{mname}__yhat": yhat_oof,
        })

    arrays = {
        "y": np.tile(y_all, len(pieces)),
        "c": np.tile(c_all, len(pieces)),
        "pair_id": np.tile(np.asarray([str(r.get("pair_id", "")) for r in recs], dtype=object), len(pieces)),
        "room_id": np.tile(np.asarray([str(r.get("room_id", "")) for r in recs], dtype=object), len(pieces)),
        "attribute_id": np.tile(np.asarray([str(r.get("attribute_id", "")) for r in recs], dtype=object), len(pieces)),
        "category": np.tile(np.asarray([str(r.get("category", "")) for r in recs], dtype=object), len(pieces)),
        "source_count": np.tile(np.asarray([sum(int((r.get("evidence_count", {}) or {}).get(k, 0) or 0) for k in ("params", "ocr", "vlm")) for r in recs], int), len(pieces)),
        "case": np.concatenate([p["case"] for p in pieces]),
        "fold": np.concatenate([p["fold"] for p in pieces]),
        f"{mname}__p": np.concatenate([p[f"{mname}__p"] for p in pieces]),
        f"{mname}__yhat": np.concatenate([p[f"{mname}__yhat"] for p in pieces]),
    }
    y = arrays["y"]
    c = arrays["c"]
    p = arrays[f"{mname}__p"]
    yhat = arrays[f"{mname}__yhat"]
    ok = (~np.isnan(p)) & (yhat >= 0)
    out = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "method": mname,
        "fold_seed": args.fold_seed,
        "folds": int(args.folds),
        "C": float(args.C),
        "class_weight_balanced": bool(args.class_weight_balanced),
        "metrics": {mname: row(y[ok], p[ok], yhat[ok], c[ok])},
        "fold_meta": meta,
    }
    return out, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--method_name", default="")
    ap.add_argument("--fold_seed", type=int, action="append", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--claim_prompt_name", default="")
    ap.add_argument("--evidence_prompt_name", default="")
    ap.add_argument("--claim_prompt", default="")
    ap.add_argument("--evidence_prompt", default="")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=3000)
    ap.add_argument("--class_weight_balanced", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_oof", required=True)
    args = ap.parse_args()

    out, arrays = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    Path(args.dump_oof).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dump_oof, **arrays)
    print(f"[cv_embedding_baseline] -> {args.out}", flush=True)
    print(json.dumps(out["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"[dump_oof] -> {args.dump_oof}", flush=True)


if __name__ == "__main__":
    main()
