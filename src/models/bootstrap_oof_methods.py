"""Bootstrap saved OOF method pairs without rerunning fold evaluators.

This is for post-hoc, leakage-safe significance checks on methods that were
already produced by CV scripts with ``--dump_oof``. It reads only OOF
probabilities and OOF decisions; it does not fit or retune any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def macro(y, pred, w=None):
    return f1_score(
        y, pred, average="macro", sample_weight=w, zero_division=0)


def row(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def paired_bootstrap(y, p_a, yhat_a, p_b, yhat_b, c=None, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    if c is None:
        c = np.ones_like(y, dtype=float)
    c = np.asarray(c, float)
    dap, dau, df1, dwf1 = [], [], [], []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        dap.append(
            average_precision_score(yy, p_a[idx])
            - average_precision_score(yy, p_b[idx])
        )
        dau.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))
        ww = np.clip(c[idx], 0.05, None)
        dwf1.append(macro(yy, yhat_a[idx], ww) - macro(yy, yhat_b[idx], ww))

    def summarize(values):
        arr = np.asarray(values, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {
        "dAP": summarize(dap),
        "dAUROC": summarize(dau),
        "dMacroF1": summarize(df1),
        "dWF1": summarize(dwf1),
    }


def grouped_paired_bootstrap(groups, y, p_a, yhat_a, p_b, yhat_b, c=None,
                             n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    if c is None:
        c = np.ones_like(y, dtype=float)
    c = np.asarray(c, float)
    groups = np.asarray(groups, dtype=object)
    uniq = np.asarray(sorted(set(groups.tolist())), dtype=object)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    dap, dau, df1, dwf1 = [], [], [], []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        dap.append(
            average_precision_score(yy, p_a[idx])
            - average_precision_score(yy, p_b[idx])
        )
        dau.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1.append(macro(yy, yhat_a[idx]) - macro(yy, yhat_b[idx]))
        ww = np.clip(c[idx], 0.05, None)
        dwf1.append(macro(yy, yhat_a[idx], ww) - macro(yy, yhat_b[idx], ww))

    def summarize(values):
        arr = np.asarray(values, float)
        return {
            "mean_delta": round(float(arr.mean()), 4),
            "ci": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
            "p_a_gt_b": round(float((arr <= 0).mean()), 4),
        }

    return {
        "dAP": summarize(dap),
        "dAUROC": summarize(dau),
        "dMacroF1": summarize(df1),
        "dWF1": summarize(dwf1),
    }


def load_oof(path):
    z = np.load(path, allow_pickle=True)
    out = {
        "y": np.asarray(z["y"], int),
        "c": np.asarray(z["c"], float) if "c" in z.files else None,
    }
    if out["c"] is None:
        out["c"] = np.ones_like(out["y"], dtype=float)
    out["keys"] = set(z.files)
    out["z"] = z
    return out


def get_method(oof, method):
    p_key = f"{method}__p"
    yhat_key = f"{method}__yhat"
    if p_key not in oof["keys"] and f"p__{method}" in oof["keys"]:
        p_key = f"p__{method}"
    if yhat_key not in oof["keys"] and f"yhat__{method}" in oof["keys"]:
        yhat_key = f"yhat__{method}"
    missing = [k for k in (p_key, yhat_key) if k not in oof["keys"]]
    if missing:
        raise KeyError(f"{method}: missing keys {missing}")
    return (
        np.asarray(oof["z"][p_key], float),
        np.asarray(oof["z"][yhat_key], int),
    )


def parse_case(text):
    if "=" not in text:
        path = Path(text)
        return path.stem, path
    label, path = text.split("=", 1)
    return label, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True,
                    help="label=path.npz; may be supplied multiple times")
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--baseline", action="append", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group_key", default=None,
                    help="Optional OOF array key, e.g. pair_id, for grouped bootstrap.")
    ap.add_argument("--skip_case", action="store_true",
                    help="Skip per-case significance/metric blocks and only write pooled results.")
    ap.add_argument("--only_group", action="store_true",
                    help="With --group_key, compute only grouped significance, not sample bootstrap.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = [(label, load_oof(path)) for label, path in map(parse_case, args.case)]
    out = {
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "methods": args.method,
        "baselines": args.baseline,
        "cases": {},
        "pooled": {},
    }

    if args.only_group and not args.group_key:
        raise ValueError("--only_group requires --group_key")

    if not args.skip_case:
        for ci, (label, oof) in enumerate(cases):
            case_out = {"metrics": {}, "significance": {}}
            if args.group_key:
                case_out["group_significance"] = {}
            y, c = oof["y"], oof["c"]
            groups = None
            if args.group_key:
                if args.group_key not in oof["keys"]:
                    raise KeyError(f"group_key {args.group_key!r} not found in {label}")
                groups = np.asarray(oof["z"][args.group_key], dtype=object)
            for method in list(dict.fromkeys(args.baseline + args.method)):
                p, yhat = get_method(oof, method)
                ok = (~np.isnan(p)) & (yhat >= 0)
                case_out["metrics"][method] = row(y[ok], p[ok], yhat[ok], c[ok])
            for method in args.method:
                p_a, yhat_a = get_method(oof, method)
                for baseline in args.baseline:
                    p_b, yhat_b = get_method(oof, baseline)
                    ok = (~np.isnan(p_a)) & (yhat_a >= 0) & (~np.isnan(p_b)) & (yhat_b >= 0)
                    if not args.only_group:
                        case_out["significance"][f"{method}_vs_{baseline}"] = (
                            paired_bootstrap(
                                y[ok], p_a[ok], yhat_a[ok], p_b[ok], yhat_b[ok],
                                c=c[ok],
                                n_boot=args.n_boot, seed=args.seed + ci)
                        )
                    if groups is not None:
                        case_out["group_significance"][f"{method}_vs_{baseline}"] = (
                            grouped_paired_bootstrap(
                                groups[ok], y[ok], p_a[ok], yhat_a[ok],
                                p_b[ok], yhat_b[ok],
                                c=c[ok],
                                n_boot=args.n_boot, seed=args.seed + 503 + ci)
                        )
            out["cases"][label] = case_out

    pooled_y = np.concatenate([oof["y"] for _, oof in cases])
    pooled_c = np.concatenate([oof["c"] for _, oof in cases])
    pooled_groups = None
    if args.group_key:
        pooled_groups = np.concatenate([
            np.asarray(oof["z"][args.group_key], dtype=object)
            for _, oof in cases
        ])
    pooled = {"metrics": {}, "significance": {}}
    if args.group_key:
        pooled["group_significance"] = {}
    for method in list(dict.fromkeys(args.baseline + args.method)):
        p_all, yhat_all = [], []
        for _, oof in cases:
            p, yhat = get_method(oof, method)
            p_all.append(p)
            yhat_all.append(yhat)
        p = np.concatenate(p_all)
        yhat = np.concatenate(yhat_all)
        ok = (~np.isnan(p)) & (yhat >= 0)
        pooled["metrics"][method] = row(
            pooled_y[ok], p[ok], yhat[ok], pooled_c[ok])

    for mi, method in enumerate(args.method):
        p_a = np.concatenate([get_method(oof, method)[0] for _, oof in cases])
        yhat_a = np.concatenate([get_method(oof, method)[1] for _, oof in cases])
        for bi, baseline in enumerate(args.baseline):
            p_b = np.concatenate(
                [get_method(oof, baseline)[0] for _, oof in cases])
            yhat_b = np.concatenate(
                [get_method(oof, baseline)[1] for _, oof in cases])
            ok = (~np.isnan(p_a)) & (yhat_a >= 0) & (~np.isnan(p_b)) & (yhat_b >= 0)
            if not args.only_group:
                pooled["significance"][f"{method}_vs_{baseline}"] = (
                    paired_bootstrap(
                        pooled_y[ok], p_a[ok], yhat_a[ok], p_b[ok], yhat_b[ok],
                        c=pooled_c[ok],
                        n_boot=args.n_boot, seed=args.seed + 1000 + mi * 17 + bi)
                )
            if pooled_groups is not None:
                pooled["group_significance"][f"{method}_vs_{baseline}"] = (
                    grouped_paired_bootstrap(
                        pooled_groups[ok], pooled_y[ok], p_a[ok], yhat_a[ok],
                        p_b[ok], yhat_b[ok],
                        c=pooled_c[ok],
                        n_boot=args.n_boot, seed=args.seed + 1703 + mi * 17 + bi)
                )
    out["pooled"] = pooled

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[bootstrap_oof_methods] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
