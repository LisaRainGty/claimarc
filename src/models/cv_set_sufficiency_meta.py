"""Fold-safe set-level sufficiency meta heads over saved OOF screens.

The evidence-type adapter currently uses hand-specified source/evidence masks.
This diagnostic asks a tighter question: can a tiny, interpretable set-level
head learn the same kind of score/decision repair from observable evidence
structure without seeing the held-out fold?

For each repeated-CV case (fs0/fs1/fs2) and each outer fold, the script trains
only on the other folds. Hyper-parameters and the decision threshold are chosen
from inner OOF predictions on those training folds, then a final head is fit on
all outer-train rows and applied once to the held-out fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
EVTYPE = "evtype_adapt_score_src0_po_medium_decision_po_medium"
CMBGE = "rankavg_sourcefirst_cm_pcls_bge"
BGE = "bge_lr"

DEFAULT_OOF = "data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz"


def macro(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(f1_score(y, yhat, average="macro", zero_division=0))


def best_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        score = macro(y, (p >= t).astype(int))
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=float)
    out[order] = np.arange(len(x), dtype=float)
    return (out + 0.5) / max(1, len(x))


def method_arrays(z: np.lib.npyio.NpzFile, method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z[f"{method}__p"], float),
        np.asarray(z[f"{method}__yhat"], int),
    )


def one_hot(values: np.ndarray, vocab: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=object).astype(str)
    return np.asarray([[float(v == item) for item in vocab] for v in values], float)


def build_features(
    z: np.lib.npyio.NpzFile,
    include_category: bool = False,
) -> tuple[np.ndarray, list[str]]:
    score_names = [BGE, CMBGE, CURRENT]
    if f"{ADAPTIVE}__p" in z.files:
        score_names.append(ADAPTIVE)

    cols: list[np.ndarray] = []
    names: list[str] = []
    scores: dict[str, np.ndarray] = {}
    for name in score_names:
        p, _ = method_arrays(z, name)
        scores[name] = p
        short = {
            BGE: "bge",
            CMBGE: "cmbge",
            CURRENT: "current",
            ADAPTIVE: "adaptive",
        }.get(name, name[:12])
        for suffix, arr in (
            ("p", p),
            ("logit", logit(p)),
            ("unc", np.abs(p - 0.5)),
        ):
            cols.append(arr)
            names.append(f"{short}_{suffix}")

    pairs = [(CURRENT, BGE), (CMBGE, BGE)]
    if ADAPTIVE in scores:
        pairs.extend([(ADAPTIVE, CURRENT), (ADAPTIVE, BGE)])
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        aa, bb = scores[a], scores[b]
        short_a = "adaptive" if a == ADAPTIVE else ("current" if a == CURRENT else "cmbge")
        short_b = "bge" if b == BGE else "current"
        cols.extend([aa - bb, np.abs(aa - bb)])
        names.extend([f"{short_a}_minus_{short_b}", f"{short_a}_absdiff_{short_b}"])

    score_stack = np.vstack([scores[n] for n in score_names])
    cols.extend([score_stack.mean(axis=0), score_stack.std(axis=0),
                 score_stack.max(axis=0) - score_stack.min(axis=0)])
    names.extend(["score_mean", "score_std", "score_range"])

    source_count = np.asarray(z["source_count"], float)
    cols.extend([
        source_count,
        np.log1p(source_count),
        (source_count == 0).astype(float),
        (source_count == 1).astype(float),
        ((source_count >= 2) & (source_count <= 3)).astype(float),
        (source_count >= 4).astype(float),
    ])
    names.extend([
        "source_count", "log_source_count", "src0", "src1", "src2_3", "src4p",
    ])

    combo = np.asarray(z["evidence_combo"], dtype=object).astype(str)
    for token in ("P", "O", "V"):
        cols.append(np.asarray([float(token in x) for x in combo], float))
        names.append(f"has_{token.lower()}")
    cols.append(np.asarray([float("V" not in x) for x in combo], float))
    names.append("no_vlm")

    combo_vocab = sorted(set(combo.tolist()))
    conf_vocab = sorted(set(np.asarray(z["confidence"], dtype=object).astype(str).tolist()))
    src_vocab = sorted(set(np.asarray(z["source_bin"], dtype=object).astype(str).tolist()))
    for prefix, mat, vocab in (
        ("combo", one_hot(combo, combo_vocab), combo_vocab),
        ("confidence", one_hot(z["confidence"], conf_vocab), conf_vocab),
        ("source_bin", one_hot(z["source_bin"], src_vocab), src_vocab),
    ):
        for j, item in enumerate(vocab):
            cols.append(mat[:, j])
            names.append(f"{prefix}={item}")

    if include_category:
        cat_vocab = sorted(set(np.asarray(z["category"], dtype=object).astype(str).tolist()))
        cat_mat = one_hot(z["category"], cat_vocab)
        for j, item in enumerate(cat_vocab):
            cols.append(cat_mat[:, j])
            names.append(f"category={item}")

    return np.column_stack(cols).astype(float), names


def inner_oof_lr(
    X: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    folds: np.ndarray,
    c_value: float,
    class_weight,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, float)
    for f in sorted(set(folds.tolist())):
        tr = folds != f
        te = folds == f
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            C=float(c_value),
            max_iter=3000,
            solver="liblinear",
            class_weight=class_weight,
        )
        clf.fit(Xtr, y[tr], sample_weight=np.clip(c[tr], 0.05, None))
        pred[te] = clf.predict_proba(Xte)[:, 1]
    if np.isnan(pred).any():
        raise RuntimeError("inner LR OOF left missing predictions")
    return pred


def fit_lr_crossfit(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    ctr: np.ndarray,
    fold_tr: np.ndarray,
    Xte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    best = None
    for c_value in (0.01, 0.03, 0.1, 0.3, 1.0):
        for class_weight in (None, "balanced"):
            pv = inner_oof_lr(Xtr, ytr, ctr, fold_tr, c_value, class_weight)
            thr = best_thr(ytr, pv)
            yhat = (pv >= thr).astype(int)
            score = (
                float(macro(ytr, yhat))
                + 0.05 * float(average_precision_score(ytr, pv))
                + 0.02 * float(roc_auc_score(ytr, pv))
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "thr": float(thr),
                    "c_value": float(c_value),
                    "class_weight": class_weight,
                    "inner_ap": float(average_precision_score(ytr, pv)),
                    "inner_auroc": float(roc_auc_score(ytr, pv)),
                    "inner_macro": float(macro(ytr, yhat)),
                }

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xtr)
    Xt = scaler.transform(Xte)
    clf = LogisticRegression(
        C=best["c_value"],
        max_iter=3000,
        solver="liblinear",
        class_weight=best["class_weight"],
    )
    clf.fit(Xs, ytr, sample_weight=np.clip(ctr, 0.05, None))
    pt = clf.predict_proba(Xt)[:, 1]
    yhat_t = (pt >= best["thr"]).astype(int)
    meta = {k: v for k, v in best.items() if k != "score"}
    meta["class_weight"] = best["class_weight"] or "none"
    return pt, yhat_t, meta


def inner_oof_hgb(
    X: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    folds: np.ndarray,
    learning_rate: float,
    l2: float,
    leaves: int,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, float)
    for f in sorted(set(folds.tolist())):
        tr = folds != f
        te = folds == f
        clf = HistGradientBoostingClassifier(
            learning_rate=float(learning_rate),
            l2_regularization=float(l2),
            max_leaf_nodes=int(leaves),
            max_iter=120,
            random_state=0,
        )
        clf.fit(X[tr], y[tr], sample_weight=np.clip(c[tr], 0.05, None))
        pred[te] = clf.predict_proba(X[te])[:, 1]
    if np.isnan(pred).any():
        raise RuntimeError("inner HGB OOF left missing predictions")
    return pred


def fit_hgb_crossfit(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    ctr: np.ndarray,
    fold_tr: np.ndarray,
    Xte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    best = None
    for learning_rate in (0.03, 0.06):
        for l2 in (0.05, 0.2, 1.0):
            for leaves in (7, 15):
                pv = inner_oof_hgb(Xtr, ytr, ctr, fold_tr, learning_rate, l2, leaves)
                thr = best_thr(ytr, pv)
                yhat = (pv >= thr).astype(int)
                score = (
                    float(macro(ytr, yhat))
                    + 0.05 * float(average_precision_score(ytr, pv))
                    + 0.02 * float(roc_auc_score(ytr, pv))
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "thr": float(thr),
                        "learning_rate": float(learning_rate),
                        "l2": float(l2),
                        "leaves": int(leaves),
                        "inner_ap": float(average_precision_score(ytr, pv)),
                        "inner_auroc": float(roc_auc_score(ytr, pv)),
                        "inner_macro": float(macro(ytr, yhat)),
                    }

    clf = HistGradientBoostingClassifier(
        learning_rate=best["learning_rate"],
        l2_regularization=best["l2"],
        max_leaf_nodes=best["leaves"],
        max_iter=120,
        random_state=0,
    )
    clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
    pt = clf.predict_proba(Xte)[:, 1]
    yhat_t = (pt >= best["thr"]).astype(int)
    meta = {k: v for k, v in best.items() if k != "score"}
    return pt, yhat_t, meta


def crossfit_heads(
    z: np.lib.npyio.NpzFile,
    X: np.ndarray,
    run_hgb: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    folds = np.asarray(z["fold"], int)
    cases = np.asarray(z["case"], dtype=object).astype(str) if "case" in z.files else np.asarray(["case"] * len(y), object)
    out = {
        "set_suff_lr_no_cat": {"p": np.full(len(y), np.nan), "yhat": np.full(len(y), np.nan)},
    }
    if run_hgb:
        out["set_suff_hgb_no_cat"] = {
            "p": np.full(len(y), np.nan),
            "yhat": np.full(len(y), np.nan),
        }
    fold_meta: list[dict[str, object]] = []

    for case in sorted(set(cases.tolist())):
        cm = cases == case
        case_folds = sorted(set(folds[cm].tolist()))
        for f in case_folds:
            te = cm & (folds == f)
            tr = cm & (folds != f)
            if int(te.sum()) == 0 or int(tr.sum()) == 0:
                continue
            p_lr, y_lr, meta_lr = fit_lr_crossfit(
                X[tr], y[tr], c[tr], folds[tr], X[te])
            out["set_suff_lr_no_cat"]["p"][te] = p_lr
            out["set_suff_lr_no_cat"]["yhat"][te] = y_lr
            item = {
                "case": case,
                "fold": int(f),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "lr": meta_lr,
            }
            if run_hgb:
                p_hgb, y_hgb, meta_hgb = fit_hgb_crossfit(
                    X[tr], y[tr], c[tr], folds[tr], X[te])
                out["set_suff_hgb_no_cat"]["p"][te] = p_hgb
                out["set_suff_hgb_no_cat"]["yhat"][te] = y_hgb
                item["hgb"] = meta_hgb
            fold_meta.append(item)
            print(f"[set_suff] case={case} fold={f} done", flush=True)
    return out, fold_meta


def add_decoupled_scores(
    z: np.lib.npyio.NpzFile,
    built: dict[str, dict[str, np.ndarray]],
) -> None:
    cases = np.asarray(z["case"], dtype=object).astype(str) if "case" in z.files else np.asarray(["case"] * len(z["y"]), object)
    ev_p = np.asarray(z[f"{EVTYPE}__p"], float) if f"{EVTYPE}__p" in z.files else None
    cur_p = np.asarray(z[f"{CURRENT}__p"], float) if f"{CURRENT}__p" in z.files else None
    for base_name in ("set_suff_lr_no_cat", "set_suff_hgb_no_cat"):
        if base_name not in built:
            continue
        p_meta = built[base_name]["p"]
        y_meta = built[base_name]["yhat"]
        p_rank = np.full_like(p_meta, np.nan, dtype=float)
        for case in sorted(set(cases.tolist())):
            m = cases == case
            p_rank[m] = rank01(p_meta[m])
        built[f"{base_name}_rankscore"] = {"p": p_rank, "yhat": y_meta.copy()}
        if ev_p is not None:
            built[f"evtype_score__decision_{base_name}"] = {
                "p": ev_p.copy(),
                "yhat": y_meta.copy(),
            }
        if cur_p is not None:
            built[f"current_score__decision_{base_name}"] = {
                "p": cur_p.copy(),
                "yhat": y_meta.copy(),
            }


def metrics_and_sig(
    z: np.lib.npyio.NpzFile,
    built: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    y = np.asarray(z["y"], int)
    c = np.asarray(z["c"], float)
    baselines = [m for m in (BGE, CMBGE, CURRENT, EVTYPE) if f"{m}__p" in z.files]
    rows: dict[str, dict[str, float]] = {}
    for name in baselines:
        p, yhat = method_arrays(z, name)
        rows[name] = row(y, p, yhat, c)
    for name, item in built.items():
        ok = ~np.isnan(item["p"])
        rows[name] = row(y[ok], item["p"][ok], item["yhat"][ok].astype(int), c[ok])

    sig = {}
    if n_boot > 0:
        for mi, name in enumerate(built):
            p_a = built[name]["p"]
            y_a = built[name]["yhat"].astype(int)
            for bi, base in enumerate(baselines):
                p_b, y_b = method_arrays(z, base)
                ok = (~np.isnan(p_a)) & (~np.isnan(p_b))
                sig[f"{name}_vs_{base}"] = paired_bootstrap(
                    y[ok],
                    p_a[ok],
                    y_a[ok],
                    p_b[ok],
                    y_b[ok],
                    n_boot=n_boot,
                    seed=seed + mi * 101 + bi * 17,
                )
    return rows, sig


def dump_oof(
    path: Path,
    z: np.lib.npyio.NpzFile,
    built: dict[str, dict[str, np.ndarray]],
) -> None:
    arrays = {k: np.asarray(z[k]) for k in (
        "y", "c", "fold", "case", "pair_id", "source_count",
        "source_bin", "category", "confidence", "evidence_combo",
    ) if k in z.files}
    for method in (BGE, CMBGE, CURRENT, EVTYPE):
        if f"{method}__p" in z.files:
            arrays[f"{method}__p"] = np.asarray(z[f"{method}__p"])
            arrays[f"{method}__yhat"] = np.asarray(z[f"{method}__yhat"])
    for name, item in built.items():
        arrays[f"{name}__p"] = item["p"]
        arrays[f"{name}__yhat"] = item["yhat"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=DEFAULT_OOF)
    ap.add_argument("--out", default="data/final/cleancl/set_sufficiency_meta_20260609.json")
    ap.add_argument("--dump_oof", default="data/final/cleancl/oof_set_sufficiency_meta_20260609.npz")
    ap.add_argument("--include_category", action="store_true",
                    help="Diagnostic only; primary method keeps category out.")
    ap.add_argument("--run_hgb", action="store_true",
                    help="Also run the slower tree meta-head diagnostic.")
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()

    z = np.load(args.oof, allow_pickle=True)
    X, feature_names = build_features(z, include_category=args.include_category)
    built, fold_meta = crossfit_heads(z, X, run_hgb=args.run_hgb)
    add_decoupled_scores(z, built)
    rows, sig = metrics_and_sig(z, built, args.n_boot, args.seed)

    ranked = sorted(
        built,
        key=lambda m: (rows[m]["auprc"], rows[m]["auroc"], rows[m]["macro_f1"]),
        reverse=True,
    )
    out = {
        "description": (
            "Fold-safe set-level sufficiency meta-head over saved strict OOF. "
            "Each repeated-CV case is cross-fitted independently; category is "
            "excluded unless --include_category is set."
        ),
        "oof": args.oof,
        "include_category": bool(args.include_category),
        "run_hgb": bool(args.run_hgb),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "fold_meta": fold_meta,
        "ranked_methods": ranked,
        "metrics": rows,
        "n_boot": int(args.n_boot),
        "significance": sig,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    if args.dump_oof:
        dump_oof(Path(args.dump_oof), z, built)
    print(f"[cv_set_sufficiency_meta] -> {path}", flush=True)
    for name in ranked:
        r = rows[name]
        print(f"{name:64s} AP={r['auprc']:.4f} AUROC={r['auroc']:.4f} "
              f"mF1={r['macro_f1']:.4f} wF1={r['wF1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
