"""Lightweight label learnability diagnostic.

This is not a paper baseline. It checks whether claim/evidence text contains a
learnable signal under the current labels using a deliberately simple
character TF-IDF + logistic regression model. The model never reads
adjudication rationales or adjudication labels; those fields are only used by
dataset construction scripts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    segs = claim.get("segments") or []
    text = " ".join(str(s.get("text", "") or "").strip() for s in segs if s.get("text"))
    return text or str(claim.get("passage", "") or "")


def evidence_text(rec: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for item in rec.get(key) or []:
            text = str(item.get(field, "") or "").strip()
            if text:
                parts.append(text)
    args = rec.get("arguments") or {}
    for key in ("supporting_argument", "refuting_argument", "evidence_gap"):
        text = str(args.get(key, "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def model_text(rec: dict[str, Any]) -> str:
    return (
        f"类目:{rec.get('category', '')} 属性:{rec.get('attribute_name', '')} "
        f"主播:{claim_text(rec)} 证据:{evidence_text(rec)}"
    )


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        score = f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0)
        if score > best_score:
            best_thr, best_score = float(thr), float(score)
    return best_thr


def val_carve(train_idx: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rooms = np.array(sorted({groups[i] for i in train_idx}, key=str), dtype=object)
    rng.shuffle(rooms)
    n_val = max(1, int(round(len(rooms) * 0.15)))
    val_rooms = set(rooms[:n_val])
    val = np.array([i for i in train_idx if groups[i] in val_rooms], dtype=int)
    train = np.array([i for i in train_idx if groups[i] not in val_rooms], dtype=int)
    return train, val


def metric_row(y: np.ndarray, p: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
    }


def evaluate(path: str, folds: int, seed: int) -> dict[str, Any]:
    recs = read_jsonl(path)
    y = np.array([int(r.get("y", 0)) for r in recs])
    groups = np.array([str(r.get("room_id", r.get("product_id", i))) for i, r in enumerate(recs)])
    text = np.array([model_text(r) for r in recs], dtype=object)
    c = np.array([max(float(r.get("c", 0.05) or 0.05), 0.05) for r in recs])

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    p_oof = np.full(len(recs), np.nan)
    yhat_oof = np.full(len(recs), np.nan)
    fold_rows = []
    for fold, (train_full, test_idx) in enumerate(splitter.split(np.zeros(len(recs)), y, groups)):
        train_idx, val_idx = val_carve(train_full, groups, seed * 100 + fold)
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        )
        x_train = vec.fit_transform(text[train_idx])
        x_val = vec.transform(text[val_idx])
        x_test = vec.transform(text[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        clf.fit(x_train, y[train_idx], sample_weight=c[train_idx])
        p_val = clf.predict_proba(x_val)[:, 1]
        p_test = clf.predict_proba(x_test)[:, 1]
        thr = best_threshold(y[val_idx], p_val)
        yhat_test = (p_test >= thr).astype(int)
        p_oof[test_idx] = p_test
        yhat_oof[test_idx] = yhat_test
        fold_rows.append({
            "fold": fold,
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "test_pos": int(y[test_idx].sum()),
            "thr": round(thr, 3),
            **metric_row(y[test_idx], p_test, yhat_test),
        })

    ok = ~np.isnan(p_oof)
    return {
        "dataset": path,
        "n": len(recs),
        "labels": dict(Counter(int(v) for v in y)),
        "rooms": int(len(set(groups))),
        "folds": folds,
        "seed": seed,
        "pooled_oof": metric_row(y[ok], p_oof[ok], yhat_oof[ok]),
        "fold_rows": fold_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/final/label_learnability_20260612.json")
    args = ap.parse_args()

    report = {"results": [evaluate(path, args.folds, args.seed) for path in args.dataset]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in report["results"]:
        print(row["dataset"], row["pooled_oof"])
    print(f"[eval_label_learnability] report={out}")


if __name__ == "__main__":
    main()
