"""Diagnose source-first rank-average failures under grouped CV.

The script reconstructs OOF predictions from saved cv_eval bundles and reports
where a candidate such as rankavg(sourcefirst_args_pcls, BGE) helps or hurts the
strong BGE+LR baseline. It is intentionally diagnostic: no training, no new
claims, just fold-safe thresholds and subgroup summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import best_thr, build_split_features, load_bundles, macro
from models.cv_reliability_gate import rank01


def source_count(rec):
    ev = rec.get("evidence_count", {}) or {}
    return int(ev.get("params", 0) or 0) + int(ev.get("ocr", 0) or 0) + int(ev.get("vlm", 0) or 0)


def source_len(rec):
    total = 0
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for item in rec.get(key, []) or []:
            total += len(str(item.get(field, "") or ""))
    return total


def arg_len(rec):
    args = rec.get("arguments", {}) or {}
    return sum(len(str(args.get(k, "") or "")) for k in
               ("supporting_argument", "refuting_argument", "evidence_gap"))


def bin_source_count(n):
    if n == 0:
        return "src0"
    if n == 1:
        return "src1"
    if n <= 3:
        return "src2_3"
    return "src4p"


def bin_conf(p):
    c = abs(float(p) - 0.5)
    if c < 0.08:
        return "bge_c00_08"
    if c < 0.16:
        return "bge_c08_16"
    if c < 0.28:
        return "bge_c16_28"
    return "bge_c28p"


def metric(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
    }


def add_group(groups, key, row):
    groups[key].append(row)


def summarize_group(rows, min_n):
    if len(rows) < min_n:
        return None
    y = np.asarray([r["y"] for r in rows], int)
    out = {
        "n": len(rows),
        "pos_rate": round(float(y.mean()), 4),
        "bge_acc": round(float(np.mean([r["bge_yhat"] == r["y"] for r in rows])), 4),
        "rankavg_acc": round(float(np.mean([r["rankavg_yhat"] == r["y"] for r in rows])), 4),
        "bge_correct_rankavg_wrong": int(sum((r["bge_yhat"] == r["y"]) and (r["rankavg_yhat"] != r["y"]) for r in rows)),
        "rankavg_correct_bge_wrong": int(sum((r["rankavg_yhat"] == r["y"]) and (r["bge_yhat"] != r["y"]) for r in rows)),
        "mean_bge_p": round(float(np.mean([r["bge_p"] for r in rows])), 4),
        "mean_args_p": round(float(np.mean([r["args_p"] for r in rows])), 4),
        "mean_rankavg_p": round(float(np.mean([r["rankavg_p"] for r in rows])), 4),
        "mean_source_count": round(float(np.mean([r["source_count"] for r in rows])), 2),
        "mean_arg_len": round(float(np.mean([r["arg_len"] for r in rows])), 1),
    }
    out["net_acc_gain"] = round(out["rankavg_acc"] - out["bge_acc"], 4)
    out["net_flip_gain"] = out["rankavg_correct_bge_wrong"] - out["bge_correct_rankavg_wrong"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl")
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--min_group_n", type=int, default=25)
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    n = len(recs)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    oof = {
        "bge": {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)},
        "args": {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)},
        "rankavg": {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)},
    }
    fold_rows = []
    rows = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"fold {fi} missing {missing}")
        ar_b = load_bundles(ar_paths)
        _, p_args_v, yv, cv, _ = build_split_features(ar_b, "val")
        _, p_args_t, yt, ct, _ = build_split_features(ar_b, "test")
        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        p_rank_v = 0.5 * rank01(p_args_v) + 0.5 * rank01(p_bge_v)
        p_rank_t = 0.5 * rank01(p_args_t) + 0.5 * rank01(p_bge_t)
        fold_data = {
            "bge": (p_bge_v, p_bge_t),
            "args": (p_args_v, p_args_t),
            "rankavg": (p_rank_v, p_rank_t),
        }
        for name, (pv, pt) in fold_data.items():
            thr = best_thr(yv, pv)
            yhat = (pt >= thr).astype(int)
            oof[name]["p"][te_idx] = pt
            oof[name]["yhat"][te_idx] = yhat
        fold_metric = {"fold": fi, "n_test": int(len(te_idx)), "pos_test": int(y_all[te_idx].sum())}
        for name in ("bge", "args", "rankavg"):
            fold_metric[name] = metric(y_all[te_idx], oof[name]["p"][te_idx],
                                       oof[name]["yhat"][te_idx], c_all[te_idx])
        fold_rows.append(fold_metric)

        for local_i, idx in enumerate(te_idx):
            rec = recs[idx]
            row = {
                "idx": int(idx),
                "fold": fi,
                "pair_id": rec.get("pair_id", ""),
                "category": str(rec.get("category", "")),
                "attribute_id": str(rec.get("attribute_id", "")),
                "confidence": str(rec.get("confidence", "")),
                "coverage": float(rec.get("coverage", 0.0) or 0.0),
                "source_count": source_count(rec),
                "source_len": source_len(rec),
                "arg_len": arg_len(rec),
                "y": int(y_all[idx]),
                "c": float(c_all[idx]),
                "bge_p": float(oof["bge"]["p"][idx]),
                "args_p": float(oof["args"]["p"][idx]),
                "rankavg_p": float(oof["rankavg"]["p"][idx]),
                "bge_yhat": int(oof["bge"]["yhat"][idx]),
                "args_yhat": int(oof["args"]["yhat"][idx]),
                "rankavg_yhat": int(oof["rankavg"]["yhat"][idx]),
            }
            row["flip"] = (
                "bge_ok_rankavg_bad" if row["bge_yhat"] == row["y"] and row["rankavg_yhat"] != row["y"]
                else "rankavg_ok_bge_bad" if row["rankavg_yhat"] == row["y"] and row["bge_yhat"] != row["y"]
                else "both_ok" if row["bge_yhat"] == row["y"] and row["rankavg_yhat"] == row["y"]
                else "both_bad"
            )
            row["source_bin"] = bin_source_count(row["source_count"])
            row["bge_conf_bin"] = bin_conf(row["bge_p"])
            rows.append(row)

    ok = ~np.isnan(oof["bge"]["p"])
    overall = {}
    for name in ("bge", "args", "rankavg"):
        overall[name] = metric(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])

    groups = defaultdict(list)
    for row in rows:
        for key in (
            f"fold={row['fold']}",
            f"category={row['category']}",
            f"confidence={row['confidence']}",
            f"source_bin={row['source_bin']}",
            f"bge_conf_bin={row['bge_conf_bin']}",
            f"source_bin={row['source_bin']}|bge_conf_bin={row['bge_conf_bin']}",
            f"confidence={row['confidence']}|source_bin={row['source_bin']}",
        ):
            add_group(groups, key, row)
    group_summary = {}
    for key, vals in groups.items():
        s = summarize_group(vals, args.min_group_n)
        if s:
            group_summary[key] = s

    worst = sorted(group_summary.items(), key=lambda kv: (kv[1]["net_flip_gain"], kv[1]["net_acc_gain"]))[:30]
    best = sorted(group_summary.items(), key=lambda kv: (kv[1]["net_flip_gain"], kv[1]["net_acc_gain"]), reverse=True)[:30]
    output = {
        "fold_seed": args.fold_seed,
        "overall": overall,
        "folds": fold_rows,
        "worst_groups": [{"group": k, **v} for k, v in worst],
        "best_groups": [{"group": k, **v} for k, v in best],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "out_json": args.out_json,
        "out_csv": args.out_csv,
        "overall": overall,
        "worst_groups": output["worst_groups"][:8],
        "best_groups": output["best_groups"][:8],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
