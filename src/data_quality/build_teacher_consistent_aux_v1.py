"""Build a teacher-consistent auxiliary dataset.

The script keeps a trusted base auxiliary set intact, then selectively admits
new rows from a larger candidate set. Candidate rows must pass out-of-fold
teacher agreement and simple evidence-quality gates; a per-attribute/label cap
prevents expansion from being dominated by a few frequent attributes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from data_quality.eval_label_learnability import claim_text, evidence_text, model_text


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_id(rec: dict[str, Any]) -> str:
    if rec.get("atomic_id"):
        return str(rec["atomic_id"])
    h = hashlib.sha1()
    h.update(str(rec.get("pair_id", "")).encode("utf-8"))
    h.update(b"\0")
    h.update(claim_text(rec).encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(evidence_text(rec).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def teacher_oof(rows: list[dict[str, Any]], folds: int, seed: int) -> np.ndarray:
    y = np.asarray([int(r.get("y", 0)) for r in rows], dtype=int)
    groups = np.asarray(
        [str(r.get("room_id", r.get("product_id", i))) for i, r in enumerate(rows)],
        dtype=object,
    )
    texts = np.asarray([model_text(r) for r in rows], dtype=object)
    c = np.asarray([max(float(r.get("c", 0.05) or 0.05), 0.05) for r in rows])

    min_class = int(np.bincount(y, minlength=2).min())
    n_groups = len(set(groups.tolist()))
    n_splits = min(int(folds), min_class, n_groups)
    if n_splits >= 2:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(rows)), y, groups)
    else:
        n_splits = min(int(folds), min_class)
        if n_splits < 2:
            raise ValueError("Need at least two examples per class for OOF teacher.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(rows)), y)

    prob = np.full(len(rows), np.nan, dtype=float)
    for tr, te in split_iter:
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        )
        x_tr = vec.fit_transform(texts[tr])
        x_te = vec.transform(texts[te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        clf.fit(x_tr, y[tr], sample_weight=c[tr])
        prob[te] = clf.predict_proba(x_te)[:, 1]

    if np.isnan(prob).any():
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        )
        x = vec.fit_transform(texts)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        clf.fit(x, y, sample_weight=c)
        miss = np.isnan(prob)
        prob[miss] = clf.predict_proba(x[miss])[:, 1]
    return np.clip(prob, 1e-4, 1 - 1e-4)


def source_count(rec: dict[str, Any]) -> int:
    cnt = rec.get("evidence_count") or {}
    return sum(int(cnt.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))


def accept_candidate(rec: dict[str, Any], p_teacher: float, args) -> tuple[bool, str]:
    y = int(rec.get("y", 0))
    cov = int(rec.get("coverage", 0) or 0)
    conf = str(rec.get("confidence", "") or "")
    c = float(rec.get("c", 0.05) or 0.05)

    if source_count(rec) <= 0:
        return False, "no_source"
    if y == 1:
        if p_teacher < args.pos_min_teacher:
            return False, "pos_teacher_disagree"
        if c < args.pos_min_c and conf not in {"medium", "high"}:
            return False, "pos_low_confidence"
        if cov < args.pos_min_coverage:
            return False, "pos_low_coverage"
    else:
        if p_teacher > args.neg_max_teacher:
            return False, "neg_teacher_disagree"
        if cov < args.neg_min_coverage:
            return False, "neg_low_coverage"
    return True, "accepted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos_min_teacher", type=float, default=0.56)
    ap.add_argument("--neg_max_teacher", type=float, default=0.44)
    ap.add_argument("--pos_min_c", type=float, default=0.12)
    ap.add_argument("--pos_min_coverage", type=int, default=1)
    ap.add_argument("--neg_min_coverage", type=int, default=2)
    ap.add_argument("--max_add_per_attr_label", type=int, default=24)
    args = ap.parse_args()

    base_rows = read_jsonl(args.base)
    cand_rows = read_jsonl(args.candidate)
    base_ids = {row_id(r) for r in base_rows}
    added_pool = [r for r in cand_rows if row_id(r) not in base_ids]

    teacher_rows = base_rows + added_pool
    probs = teacher_oof(teacher_rows, args.folds, args.seed)
    added_probs = probs[len(base_rows):]

    accepted: list[tuple[dict[str, Any], float]] = []
    rejected = Counter()
    for rec, p_teacher in zip(added_pool, added_probs):
        ok, reason = accept_candidate(rec, float(p_teacher), args)
        if ok:
            accepted.append((rec, float(p_teacher)))
        else:
            rejected[reason] += 1

    # Highest-margin rows are admitted first within each attribute/label block.
    buckets: dict[tuple[str, int], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for rec, p_teacher in accepted:
        buckets[(str(rec.get("attribute_id", "")), int(rec.get("y", 0)))].append((rec, p_teacher))

    capped: list[dict[str, Any]] = []
    cap_rejected = 0
    for (_attr, y), items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        items.sort(
            key=lambda rp: (
                -abs(rp[1] - 0.5),
                -float(rp[0].get("c", 0.05) or 0.05),
                -int(rp[0].get("coverage", 0) or 0),
                str(row_id(rp[0])),
            )
        )
        for rec, p_teacher in items[: args.max_add_per_attr_label]:
            out = dict(rec)
            out["_teacher_consistent_aux_v1"] = {
                "teacher_p_oof": round(float(p_teacher), 6),
                "source": Path(args.candidate).name,
            }
            capped.append(out)
        cap_rejected += max(0, len(items) - args.max_add_per_attr_label)

    out_rows = base_rows + capped
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "base": args.base,
        "candidate": args.candidate,
        "out": args.out,
        "n_base": len(base_rows),
        "n_candidate_total": len(cand_rows),
        "n_candidate_new": len(added_pool),
        "n_new_accepted_before_cap": len(accepted),
        "n_new_accepted_after_cap": len(capped),
        "n_out": len(out_rows),
        "labels_out": dict(Counter(int(r.get("y", 0)) for r in out_rows)),
        "labels_added": dict(Counter(int(r.get("y", 0)) for r in capped)),
        "rejected": dict(rejected),
        "cap_rejected": cap_rejected,
        "added_by_confidence": dict(Counter(str(r.get("confidence", "")) for r in capped)),
        "added_by_coverage": dict(Counter(str(r.get("coverage", "")) for r in capped)),
        "added_by_source_family": dict(Counter(str(r.get("source_family", "")) for r in capped)),
        "params": vars(args),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
