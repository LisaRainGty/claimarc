"""Diagnose dual-head router failures under grouped CV.

This is a leakage-safe diagnostic companion to ``cv_dual_head_router``. It
reconstructs OOF scores/predictions for BGE, rank-average, source masks, and a
few representative dual-head candidates, then reports subgroup flip gains.
No model is trained here.
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from models.cv_dual_head_router import (
    apply_group_thresholds,
    greedy_group_thresholds,
    group_labels,
    source_count,
    source_len,
)
from models.cv_eval import make_folds, val_carve
from models.cv_reliability_gate import (
    fit_conf_advantage_switch,
    fit_uncertain_switch,
    rank01,
)
from models.cv_source_condition_gate import apply_mask, fixed_masks
from models.data import load_split
from models.fusion_eval import best_thr, build_split_features, load_bundles, macro


def arg_len(rec):
    args = rec.get("arguments", {}) or {}
    return sum(len(str(args.get(k, "") or "")) for k in
               ("supporting_argument", "refuting_argument", "evidence_gap"))


def source_bin(n):
    if n <= 0:
        return "src0"
    if n == 1:
        return "src1"
    if n <= 3:
        return "src2_3"
    return "src4p"


def source_len_bin(n):
    if n <= 0:
        return "slen0"
    if n < 20:
        return "slen1_19"
    if n < 80:
        return "slen20_79"
    return "slen80p"


def conf_bin(p):
    d = abs(float(p) - 0.5)
    if d < 0.08:
        return "bge_c00_08"
    if d < 0.16:
        return "bge_c08_16"
    if d < 0.28:
        return "bge_c16_28"
    return "bge_c28p"


def metric(y, p, yhat, c):
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(float(macro(y, yhat)), 4),
        "pos_f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "wF1": round(float(macro(y, yhat, w=np.clip(c, 0.05, None))), 4),
        "n": int(len(y)),
        "pos": int(np.asarray(y, int).sum()),
    }


def put(oof, name, idx, yv, pv, pt):
    thr = best_thr(yv, pv)
    oof[name]["p"][idx] = pt
    oof[name]["yhat"][idx] = (pt >= thr).astype(int)
    return float(thr)


def summarize(rows, candidate, min_n):
    if len(rows) < min_n:
        return None
    y = np.asarray([r["y"] for r in rows], int)
    bge_ok = np.asarray([r["bge_yhat"] == r["y"] for r in rows], bool)
    cand_ok = np.asarray([r[f"{candidate}_yhat"] == r["y"] for r in rows], bool)
    bge_pos = np.asarray([r["bge_yhat"] == 1 for r in rows], bool)
    cand_pos = np.asarray([r[f"{candidate}_yhat"] == 1 for r in rows], bool)
    out = {
        "n": len(rows),
        "pos_rate": round(float(y.mean()), 4),
        "bge_acc": round(float(bge_ok.mean()), 4),
        "cand_acc": round(float(cand_ok.mean()), 4),
        "net_acc_gain": round(float(cand_ok.mean() - bge_ok.mean()), 4),
        "bge_correct_cand_wrong": int((bge_ok & ~cand_ok).sum()),
        "cand_correct_bge_wrong": int((cand_ok & ~bge_ok).sum()),
        "both_wrong": int((~bge_ok & ~cand_ok).sum()),
        "both_ok": int((bge_ok & cand_ok).sum()),
        "mean_bge_p": round(float(np.mean([r["bge_p"] for r in rows])), 4),
        "mean_cand_p": round(float(np.mean([r[f"{candidate}_p"] for r in rows])), 4),
        "bge_pred_pos_rate": round(float(bge_pos.mean()), 4),
        "cand_pred_pos_rate": round(float(cand_pos.mean()), 4),
        "mean_source_count": round(float(np.mean([r["source_count"] for r in rows])), 2),
        "mean_source_len": round(float(np.mean([r["source_len"] for r in rows])), 1),
        "mean_arg_len": round(float(np.mean([r["arg_len"] for r in rows])), 1),
    }
    out["net_flip_gain"] = out["cand_correct_bge_wrong"] - out["bge_correct_cand_wrong"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--noargs_tmp", required=True)
    ap.add_argument("--args_tmp", required=True)
    ap.add_argument("--bge_tmp", required=True)
    ap.add_argument("--fold_seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--candidate", default="dual_rank_mask_ge4")
    ap.add_argument("--min_group_n", type=int, default=25)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    recs_by_split = load_split(args.dataset)
    recs = recs_by_split["train"] + recs_by_split["val"] + recs_by_split["test"]
    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    n = len(recs)
    y_all = np.asarray([int(r["y"]) for r in recs], int)
    c_all = np.asarray([float(r.get("c", 0.05)) for r in recs], float)
    methods = [
        "bge",
        "args",
        "noargs",
        "rankavg_args_bge",
        "rankavg_no_bge",
        "mask_src_ge2",
        "mask_src_ge4",
        "mask_src_len20",
        "switch_confadv_macro_rankavg_no_bge",
        "switch_confadv_rank_rankavg_args_bge",
        "dual_rank_mask_ge4",
        "dual_rank_mask_slen20",
        "groupthr_srcbin_rankavg_args_bge",
        "groupthr_srcbin_switch_confadv_macro_rankavg_no_bge",
    ]
    oof = {m: {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)} for m in methods}
    fold_meta = []
    fold_by_idx = {}
    for fi, (_, te_idx) in enumerate(folds):
        for idx in te_idx:
            fold_by_idx[int(idx)] = int(fi)

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        no_paths = [f"{args.noargs_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        ar_paths = [f"{args.args_tmp}/cv_cm_f{fi}_s{s}.pt" for s in args.cm_seeds]
        bge_path = f"{args.bge_tmp}/cv_bge_lr_f{fi}.pt"
        missing = [p for p in no_paths + ar_paths + [bge_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"fold {fi} missing {missing}")

        no_b = load_bundles(no_paths)
        ar_b = load_bundles(ar_paths)
        _, p_no_v, yv, _, _ = build_split_features(no_b, "val")
        _, p_no_t, yt, _, _ = build_split_features(no_b, "test")
        _, p_args_v, yv2, _, _ = build_split_features(ar_b, "val")
        _, p_args_t, yt2, _, _ = build_split_features(ar_b, "test")
        if not (np.all(yv == yv2) and np.all(yt == yt2)):
            raise ValueError(f"fold {fi} noargs/args labels differ")
        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        p_bge_v = np.asarray(bge["val"]["p"], float)
        p_bge_t = np.asarray(bge["test"]["p"], float)
        p_rank_v = 0.5 * rank01(p_args_v) + 0.5 * rank01(p_bge_v)
        p_rank_t = 0.5 * rank01(p_args_t) + 0.5 * rank01(p_bge_t)
        p_rank_no_v = 0.5 * rank01(p_no_v) + 0.5 * rank01(p_bge_v)
        p_rank_no_t = 0.5 * rank01(p_no_t) + 0.5 * rank01(p_bge_t)

        val_recs = [recs[i] for i in va_idx]
        test_recs = [recs[i] for i in te_idx]
        meta = {"fold": fi, "n_val": len(va_idx), "n_test": len(te_idx)}
        meta["bge_thr"] = put(oof, "bge", te_idx, yv, p_bge_v, p_bge_t)
        meta["args_thr"] = put(oof, "args", te_idx, yv, p_args_v, p_args_t)
        meta["noargs_thr"] = put(oof, "noargs", te_idx, yv, p_no_v, p_no_t)
        meta["rankavg_thr"] = put(oof, "rankavg_args_bge", te_idx, yv, p_rank_v, p_rank_t)
        meta["rankavg_no_bge_thr"] = put(oof, "rankavg_no_bge", te_idx, yv,
                                         p_rank_no_v, p_rank_no_t)

        mv = fixed_masks(val_recs, p_bge_v)
        mt = fixed_masks(test_recs, p_bge_t)
        mask_specs = {
            "mask_src_ge2": "src_ge2",
            "mask_src_ge4": "src_ge4",
            "mask_src_len20": "src_len_ge20",
        }
        for name, mask_name in mask_specs.items():
            pv = apply_mask(p_bge_v, p_rank_v, mv[mask_name])
            pt = apply_mask(p_bge_t, p_rank_t, mt[mask_name])
            meta[f"{name}_thr"] = put(oof, name, te_idx, yv, pv, pt)
            meta[f"{name}_test_rate"] = round(float(mt[mask_name].mean()), 4)

        _, pv_sw, pt_sw, rv, rt = fit_conf_advantage_switch(
            yv, p_bge_v, p_bge_t, p_rank_v, p_rank_t, "rank")
        name = "switch_confadv_rank_rankavg_args_bge"
        meta[f"{name}_thr"] = put(oof, name, te_idx, yv, pv_sw, pt_sw)
        meta[f"{name}_val_rate"] = round(float(rv), 4)
        meta[f"{name}_test_rate"] = round(float(rt), 4)

        _, pv_sw_macro_no, pt_sw_macro_no, rv, rt = fit_conf_advantage_switch(
            yv, p_bge_v, p_bge_t, p_rank_no_v, p_rank_no_t, "macro")
        name = "switch_confadv_macro_rankavg_no_bge"
        meta[f"{name}_thr"] = put(oof, name, te_idx, yv, pv_sw_macro_no, pt_sw_macro_no)
        meta[f"{name}_val_rate"] = round(float(rv), 4)
        meta[f"{name}_test_rate"] = round(float(rt), 4)

        for dual_name, decision_name in (
            ("dual_rank_mask_ge4", "mask_src_ge4"),
            ("dual_rank_mask_slen20", "mask_src_len20"),
        ):
            oof[dual_name]["p"][te_idx] = pt_sw
            oof[dual_name]["yhat"][te_idx] = oof[decision_name]["yhat"][te_idx]

        gv = group_labels(val_recs, "srcbin")
        gt = group_labels(test_recs, "srcbin")
        global_thr, by_group, val_macro = greedy_group_thresholds(yv, p_rank_v, gv)
        name = "groupthr_srcbin_rankavg_args_bge"
        oof[name]["p"][te_idx] = p_rank_t
        oof[name]["yhat"][te_idx] = apply_group_thresholds(p_rank_t, gt, global_thr, by_group)
        meta[f"{name}_global_thr"] = round(float(global_thr), 3)
        meta[f"{name}_group_thr"] = {k: round(float(v), 3) for k, v in by_group.items()}
        meta[f"{name}_val_macro"] = round(float(val_macro), 4)

        global_thr, by_group, val_macro = greedy_group_thresholds(yv, pv_sw_macro_no, gv)
        name = "groupthr_srcbin_switch_confadv_macro_rankavg_no_bge"
        oof[name]["p"][te_idx] = pt_sw_macro_no
        oof[name]["yhat"][te_idx] = apply_group_thresholds(
            pt_sw_macro_no, gt, global_thr, by_group)
        meta[f"{name}_global_thr"] = round(float(global_thr), 3)
        meta[f"{name}_group_thr"] = {k: round(float(v), 3) for k, v in by_group.items()}
        meta[f"{name}_val_macro"] = round(float(val_macro), 4)
        fold_meta.append(meta)

    ok = ~np.isnan(oof["bge"]["p"])
    overall = {}
    for m in methods:
        overall[m] = metric(y_all[ok], oof[m]["p"][ok], oof[m]["yhat"][ok], c_all[ok])

    rows = []
    for idx, rec in enumerate(recs):
        row = {
            "idx": int(idx),
            "pair_id": rec.get("pair_id", ""),
            "fold": fold_by_idx[int(idx)],
            "category": str(rec.get("category", "")),
            "attribute_id": str(rec.get("attribute_id", "")),
            "confidence": str(rec.get("confidence", "")),
            "coverage": float(rec.get("coverage", 0.0) or 0.0),
            "y": int(y_all[idx]),
            "c": float(c_all[idx]),
            "source_count": source_count(rec),
            "source_len": source_len(rec),
            "arg_len": arg_len(rec),
        }
        row["source_bin"] = source_bin(row["source_count"])
        row["source_len_bin"] = source_len_bin(row["source_len"])
        row["bge_conf_bin"] = conf_bin(oof["bge"]["p"][idx])
        row["label"] = "pos" if row["y"] else "neg"
        for m in methods:
            row[f"{m}_p"] = float(oof[m]["p"][idx])
            row[f"{m}_yhat"] = int(oof[m]["yhat"][idx])
        rows.append(row)

    candidate = args.candidate
    if candidate not in methods:
        raise ValueError(f"candidate must be one of {methods}")
    groups = defaultdict(list)
    for row in rows:
        group_keys = [
            f"fold={row['fold']}",
            f"category={row['category']}",
            f"attribute_id={row['attribute_id']}",
            f"confidence={row['confidence']}",
            f"source_bin={row['source_bin']}",
            f"source_len_bin={row['source_len_bin']}",
            f"bge_conf_bin={row['bge_conf_bin']}",
            f"label={row['label']}",
            f"source_bin={row['source_bin']}|bge_conf_bin={row['bge_conf_bin']}",
            f"confidence={row['confidence']}|source_bin={row['source_bin']}",
            f"fold={row['fold']}|source_bin={row['source_bin']}",
            f"fold={row['fold']}|confidence={row['confidence']}",
            f"category={row['category']}|source_bin={row['source_bin']}",
        ]
        for key in group_keys:
            groups[key].append(row)
    group_summary = {}
    for key, vals in groups.items():
        s = summarize(vals, candidate, args.min_group_n)
        if s:
            group_summary[key] = s
    worst = sorted(group_summary.items(),
                   key=lambda kv: (kv[1]["net_flip_gain"], kv[1]["net_acc_gain"]))[:40]
    best = sorted(group_summary.items(),
                  key=lambda kv: (kv[1]["net_flip_gain"], kv[1]["net_acc_gain"]),
                  reverse=True)[:40]

    output = {
        "fold_seed": args.fold_seed,
        "candidate": candidate,
        "overall": overall,
        "fold_meta": fold_meta,
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
        "candidate": candidate,
        "overall": overall,
        "worst_groups": output["worst_groups"][:10],
        "best_groups": output["best_groups"][:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
