"""分组 K 折交叉验证主对比（提升统计功效）。

固定划分 n_test=337 域内噪声受限，强方法统计打平。本脚本按 room_id 分组做
StratifiedGroupKFold(5)，每个样本恰被预测一次 → 汇集 OOF 预测 n≈1694，
CI 收窄 ~2.2×，更可能让 CLAIMARC 在主指标(Macro-F1/wF1/AUPRC)上达到显著。

每折：test=该折；从其余折按 room 切 val（阈值/标定/模型选择）；其余为 train。
- CLAIMARC：n_seeds 个 lora48 canonical → blend2(2特征检索融合)+Platt（与主方法一致）。
- 基线：roberta_cls / bert_cls / bge_lr（最强三个对手），同 val 选阈值。
汇集 OOF → 指标 + 配对 bootstrap。

用法：python -m models.cv_eval --dataset ../data/final/dataset_verify_faithful.jsonl \
        --folds 5 --cm_seeds 0 1 2 --out ../data/final/v2/cv.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from models.data import load_split, source_count, evidence_combo, confidence_bin
from models.train import train
from models import baselines_ft, baselines_extra
from models.fusion_eval import (load_bundles, build_split_features, fit_blend2,
                                apply_blend2, fit_platt, apply_platt, best_thr,
                                metric_row, paired_bootstrap, macro,
                                fit_selective_rkc, apply_selective_rkc)

CANON = dict(no_cl=False, no_fusion=False, n_fusion=2, fusion_dropout=0.2, no_lora=False,
             no_weight=False, lora_rank=48, heads=8, tau=0.07, lambda_cl=0.5, Kp=3, Kn=5,
             global_neg=False, backbone="bge", xattn_dir="both", indep_proj=False,
             ffn="swiglu", swa=False, loss="asl", gamma_neg=4.0, gamma_pos=0.0,
             enc_train="lora", unfreeze_top=0, no_ret_disc=False,
             encoder_name="BAAI/bge-large-zh-v1.5",
             cl_c_min=0.0, cl_neg_c_min=0.0,
             cl_teacher_mode="off", cl_teacher_conf_min=0.0,
             cl_neg_filter="none", cl_neg_bonus=0.0,
             cl_neg_bonus_filter="none",
             source0_ce_scale=1.0, source0_cl_scale=1.0,
             source_rich_ce_scale=1.0, source_rich_cl_scale=1.0,
             distill_bge_weight=0.0, distill_bge_folds=5,
             distill_teacher_seed=0, distill_temp=1.0, distill_conf_min=0.0,
             distill_c_min=0.0, distill_mode="all",
             evidence_policy="", evidence_policy_mix="",
             view_consistency_mix="", view_ce_weight=0.0,
             view_logit_weight=0.0, view_embed_weight=0.0,
             view_consistency_in_warmup=False,
             source_aux_combo_weight=0.0, source_aux_conf_weight=0.0,
             source_aux_count_weight=0.0, source_aux_in_warmup=False,
             proto_aux_weight=0.0, proto_aux_group="source_bin",
             proto_aux_mode="ce", proto_aux_margin=0.15,
             proto_aux_tau=0.10, proto_aux_min_class=3,
             proto_aux_c_min=0.10, proto_aux_in_warmup=False,
             bs=16, accum=2, lr=2e-5, lr_head=1e-4, warmup=3, cl_epochs=6, pos_weight=-1.0)


def make_folds(recs, n_folds, seed=0):
    y = np.array([int(r["y"]) for r in recs])
    g = np.array([r.get("room_id", r.get("product_id", i)) for i, r in enumerate(recs)])
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(recs)), y, g)), y, g


def val_carve(train_idx, recs, g, frac=0.15, seed=0):
    """从 train 折按 room 切出 val（无泄漏）。"""
    rng = np.random.RandomState(seed)
    rooms = sorted({g[i] for i in train_idx}, key=lambda x: str(x))
    rng.shuffle(rooms)
    nval = max(1, int(len(rooms) * frac))
    val_rooms = set(rooms[:nval])
    val = [i for i in train_idx if g[i] in val_rooms]
    tr = [i for i in train_idx if g[i] not in val_rooms]
    return tr, val


def choose_threshold(yv, pv, args, train_prior=None):
    if getattr(args, "threshold_policy", "val_macro") == "fixed":
        return float(getattr(args, "fixed_thr", 0.5))
    if getattr(args, "threshold_policy", "val_macro") == "prior_stable":
        prior = float(train_prior if train_prior is not None else np.mean(yv))
        penalty = float(getattr(args, "prior_penalty", 0.5))
        best_t, best_s = 0.5, -1e9
        for t in np.linspace(0.02, 0.98, 49):
            pred = (pv >= t).astype(int)
            pred_rate = float(pred.mean())
            s = macro(yv, pred) - penalty * abs(pred_rate - prior)
            if s > best_s:
                best_t, best_s = float(t), float(s)
        return best_t
    return best_thr(yv, pv)


def apply_evidence_policy(records, policy):
    if not policy or policy == "record":
        return
    for r in records:
        r["_evidence_policy"] = policy


def load_aux_records(path: str) -> list[dict]:
    if not path:
        return []
    split = load_split(path)
    recs = split["train"] + split["val"] + split["test"]
    for r in recs:
        r["_aux_train_source"] = path
    return recs


def select_aux_train(aux_records, eval_records, scale=1.0, max_per_fold=0):
    if not aux_records:
        return [], {"available": 0, "blocked": 0, "added": 0}
    blocked_rooms = {str(r.get("room_id", "")) for r in eval_records if r.get("room_id")}
    blocked_products = {str(r.get("product_id", "")) for r in eval_records if r.get("product_id")}
    blocked_pairs = {str(r.get("pair_id", "")) for r in eval_records if r.get("pair_id")}
    out = []
    blocked = 0
    for r in aux_records:
        rid = str(r.get("room_id", ""))
        pid = str(r.get("product_id", ""))
        pair = str(r.get("pair_id", ""))
        if (rid and rid in blocked_rooms) or (pid and pid in blocked_products) or (pair and pair in blocked_pairs):
            blocked += 1
            continue
        nr = copy.deepcopy(r)
        nr["split"] = "train"
        nr["_aux_train"] = True
        nr["c"] = max(0.05, min(1.0, float(nr.get("c", 0.05)) * float(scale)))
        out.append(nr)
    if max_per_fold and len(out) > max_per_fold:
        # Deterministic, confidence-first cap to keep auxiliary supervision small.
        out.sort(key=lambda r: (-float(r.get("c", 0.0)), str(r.get("pair_id", ""))))
        out = out[:max_per_fold]
    return out, {"available": len(aux_records), "blocked": blocked, "added": len(out)}


def claimarc_fold(splits, seeds, tmpdir, fold):
    """训练 n 种子 CLAIMARC，返回多个推理头的 val/test 概率。

    CLAIMARC_pcls 保留训练期 RACL 表征学习，但不做推理时 kNN 融合；
    CLAIMARC_selectiveRKC 仅在验证集学到可信门控时让检索小步修正；
    CLAIMARC_v2 保留旧的 blend2+Platt，作为融合头对照。
    """
    import gc
    paths = []
    for s in seeds:
        out = f"{tmpdir}/cv_cm_f{fold}_s{s}.pt"
        if not os.path.exists(out):
            args = SimpleNamespace(seed=s, tag=f"cv_f{fold}", dataset="", save_emb=out, **CANON)
            train(args, splits=splits, return_model=False)
            gc.collect(); torch.cuda.empty_cache()
        paths.append(out)
    bundles = load_bundles(paths)
    Xv, pcls_v, yv, cv, _ = build_split_features(bundles, "val")
    Xt, pcls_t, yt, ct, _ = build_split_features(bundles, "test")
    pknn_v, pknn_t = Xv[:, 1], Xt[:, 1]
    out = {
        "CLAIMARC_pcls": (pcls_v, pcls_t),
    }
    sel_rule = fit_selective_rkc(Xv, yv, cv)
    sel_v, _ = apply_selective_rkc(sel_rule, Xv)
    sel_t, _ = apply_selective_rkc(sel_rule, Xt)
    out["CLAIMARC_selectiveRKC"] = (sel_v, sel_t)
    b2 = fit_blend2(pcls_v, pknn_v, yv, cv)
    bv, bt = apply_blend2(b2, pcls_v, pknn_v), apply_blend2(b2, pcls_t, pknn_t)
    platt = fit_platt(bv, yv, cv)
    out["CLAIMARC_v2"] = (apply_platt(platt, bv), apply_platt(platt, bt))
    return out, yv, yt, ct


def baseline_fold(kind, splits, seed, tmpdir, fold):
    out = f"{tmpdir}/cv_{kind}_f{fold}.pt"
    if not os.path.exists(out):
        if kind in ("bge_lr",):
            a = SimpleNamespace(dataset="", kind=kind, seed=seed, save_pred=out)
            # baselines_extra.main reads argv; call its core via a thin wrapper
            _run_extra(a, splits, out)
        else:
            a = SimpleNamespace(dataset="", kind=kind, seed=seed, save_pred=out,
                                bs=16, lr=2e-5, epochs=4, loss="asl", gamma_neg=4.0)
            baselines_ft.run(a, splits=splits)
    try:
        d = torch.load(out, weights_only=False)
    except TypeError:
        d = torch.load(out)
    return d["val"]["p"], d["val"]["y"], d["test"]["p"], d["test"]["y"], d["test"]["c"]


def _run_extra(args, splits, out):
    """bge_lr：复用 baselines_extra 的编码+LR，但喂自定义 splits。"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = {s: baselines_extra.encode(splits[s], device) for s in ("train", "val", "test")}
    Y = {s: np.array([int(r.get("y", 0)) for r in splits[s]]) for s in splits}
    C = {s: np.array([float(r.get("c", 0.05)) for r in splits[s]]) for s in splits}
    ct, et = feats["train"]
    Xtr = np.concatenate([ct, et, ct - et, ct * et], 1)
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(Xtr, Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
    pr = {}
    for s in ("val", "test"):
        cs, es = feats[s]
        pr[s] = clf.predict_proba(np.concatenate([cs, es, cs - es, cs * es], 1))[:, 1]
    torch.save({"val": {"p": pr["val"], "y": Y["val"]},
                "test": {"p": pr["test"], "y": Y["test"], "c": C["test"]}}, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0,
                    help="Optional early-stop screen: 0 runs all folds; N runs the first N folds.")
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--cm_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--baselines", nargs="*", default=["roberta_cls", "bert_cls", "bge_lr"])
    ap.add_argument("--tmpdir", default="../data/final/v2/cv_tmp")
    ap.add_argument("--out", default="../data/final/v2/cv.json")
    ap.add_argument("--dump_oof", default="",
                    help="可选：保存 pooled OOF 概率/yhat/元数据为 npz，供后续 evidence pooling 诊断")
    ap.add_argument("--aux_train_dataset", default="",
                    help="Optional train-only auxiliary JSONL. Fold-specific room/product/pair guards prevent eval leakage.")
    ap.add_argument("--aux_train_weight_scale", type=float, default=0.35,
                    help="Multiplier for auxiliary sample confidence weights.")
    ap.add_argument("--aux_train_max_per_fold", type=int, default=0,
                    help="Optional deterministic cap for auxiliary rows added to each fold; 0 means no cap.")
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="Paired bootstrap repetitions. Set 0 to skip during fast/backup-first runs.")
    ap.add_argument("--threshold_policy", default="val_macro",
                    choices=["val_macro", "prior_stable", "fixed"])
    ap.add_argument("--fixed_thr", type=float, default=0.5)
    ap.add_argument("--prior_penalty", type=float, default=0.5)
    ap.add_argument("--evidence_policy", default=CANON["evidence_policy"],
                    choices=["", "record", "args_first", "source_first", "no_args",
                             "source_only", "sources_only", "args_only", "params_only",
                             "ocr_only", "vlm_only", "params_args", "ocr_args", "vlm_args"],
                    help="覆盖记录内 _evidence_policy；用于训练 params/OCR/VLM/argument 分源专家")
    ap.add_argument("--evidence_policy_mix", default=CANON["evidence_policy_mix"],
                    help="逗号或空格分隔的 train-only evidence views；用于 evidence-view dropout/consistency")
    ap.add_argument("--view_consistency_mix", default=CANON["view_consistency_mix"],
                    help="逗号或空格分隔的辅助 evidence views；用于同样本多视图一致性")
    ap.add_argument("--view_ce_weight", type=float, default=CANON["view_ce_weight"])
    ap.add_argument("--view_logit_weight", type=float, default=CANON["view_logit_weight"])
    ap.add_argument("--view_embed_weight", type=float, default=CANON["view_embed_weight"])
    ap.add_argument("--view_consistency_in_warmup", action="store_true",
                    default=CANON["view_consistency_in_warmup"])
    ap.add_argument("--source_aux_combo_weight", type=float,
                    default=CANON["source_aux_combo_weight"])
    ap.add_argument("--source_aux_conf_weight", type=float,
                    default=CANON["source_aux_conf_weight"])
    ap.add_argument("--source_aux_count_weight", type=float,
                    default=CANON["source_aux_count_weight"])
    ap.add_argument("--source_aux_in_warmup", action="store_true",
                    default=CANON["source_aux_in_warmup"])
    ap.add_argument("--proto_aux_weight", type=float,
                    default=CANON["proto_aux_weight"])
    ap.add_argument("--proto_aux_group", default=CANON["proto_aux_group"],
                    choices=["global", "attr", "source_bin", "evidence_combo",
                             "confidence", "combo_conf", "source_conf"])
    ap.add_argument("--proto_aux_mode", default=CANON["proto_aux_mode"],
                    choices=["ce", "margin"])
    ap.add_argument("--proto_aux_margin", type=float,
                    default=CANON["proto_aux_margin"])
    ap.add_argument("--proto_aux_tau", type=float,
                    default=CANON["proto_aux_tau"])
    ap.add_argument("--proto_aux_min_class", type=int,
                    default=CANON["proto_aux_min_class"])
    ap.add_argument("--proto_aux_c_min", type=float,
                    default=CANON["proto_aux_c_min"])
    ap.add_argument("--proto_aux_in_warmup", action="store_true",
                    default=CANON["proto_aux_in_warmup"])
    ap.add_argument("--encoder_name", default=CANON["encoder_name"])
    ap.add_argument("--n_fusion", type=int, default=CANON["n_fusion"])
    ap.add_argument("--lora_rank", type=int, default=CANON["lora_rank"])
    ap.add_argument("--warmup", type=int, default=CANON["warmup"])
    ap.add_argument("--cl_epochs", type=int, default=CANON["cl_epochs"])
    ap.add_argument("--lambda_cl", type=float, default=CANON["lambda_cl"])
    ap.add_argument("--tau", type=float, default=CANON["tau"])
    ap.add_argument("--Kp", type=int, default=CANON["Kp"])
    ap.add_argument("--Kn", type=int, default=CANON["Kn"])
    ap.add_argument("--global_neg", action="store_true", default=CANON["global_neg"])
    ap.add_argument("--bs", type=int, default=CANON["bs"])
    ap.add_argument("--accum", type=int, default=CANON["accum"])
    ap.add_argument("--cl_c_min", type=float, default=CANON["cl_c_min"])
    ap.add_argument("--cl_neg_c_min", type=float, default=CANON["cl_neg_c_min"])
    ap.add_argument("--cl_teacher_mode", default=CANON["cl_teacher_mode"],
                    choices=["off", "agree", "agree_pos"])
    ap.add_argument("--cl_teacher_conf_min", type=float, default=CANON["cl_teacher_conf_min"])
    ap.add_argument("--cl_neg_filter", default=CANON["cl_neg_filter"],
                    choices=["none", "same_evtype", "same_evtype_conf", "medium_evtype_conf"])
    ap.add_argument("--cl_neg_bonus", type=float, default=CANON["cl_neg_bonus"])
    ap.add_argument("--cl_neg_bonus_filter", default=CANON["cl_neg_bonus_filter"],
                    choices=["none", "same_evtype", "same_evtype_conf", "medium_evtype_conf"])
    ap.add_argument("--source0_ce_scale", type=float, default=CANON["source0_ce_scale"])
    ap.add_argument("--source0_cl_scale", type=float, default=CANON["source0_cl_scale"])
    ap.add_argument("--source_rich_ce_scale", type=float, default=CANON["source_rich_ce_scale"])
    ap.add_argument("--source_rich_cl_scale", type=float, default=CANON["source_rich_cl_scale"])
    ap.add_argument("--distill_bge_weight", type=float, default=CANON["distill_bge_weight"])
    ap.add_argument("--distill_bge_folds", type=int, default=CANON["distill_bge_folds"])
    ap.add_argument("--distill_teacher_seed", type=int, default=CANON["distill_teacher_seed"])
    ap.add_argument("--distill_temp", type=float, default=CANON["distill_temp"])
    ap.add_argument("--distill_conf_min", type=float, default=CANON["distill_conf_min"])
    ap.add_argument("--distill_c_min", type=float, default=CANON["distill_c_min"])
    ap.add_argument("--distill_mode", default=CANON["distill_mode"], choices=["all", "disagree"])
    args = ap.parse_args()
    os.makedirs(args.tmpdir, exist_ok=True)
    for key in ("encoder_name", "n_fusion", "lora_rank", "warmup", "cl_epochs",
                "lambda_cl", "tau", "Kp", "Kn", "global_neg",
                "bs", "accum", "cl_c_min", "cl_neg_c_min",
                "cl_teacher_mode", "cl_teacher_conf_min", "cl_neg_filter",
                "cl_neg_bonus", "cl_neg_bonus_filter",
                "source0_ce_scale", "source0_cl_scale",
                "source_rich_ce_scale", "source_rich_cl_scale",
                "distill_bge_weight", "distill_bge_folds", "distill_teacher_seed",
                "distill_temp", "distill_conf_min", "distill_c_min", "distill_mode",
                "evidence_policy", "evidence_policy_mix",
                "view_consistency_mix", "view_ce_weight", "view_logit_weight",
                "view_embed_weight", "view_consistency_in_warmup",
                "source_aux_combo_weight", "source_aux_conf_weight",
                "source_aux_count_weight", "source_aux_in_warmup",
                "proto_aux_weight", "proto_aux_group", "proto_aux_mode",
                "proto_aux_margin", "proto_aux_tau",
                "proto_aux_min_class", "proto_aux_c_min", "proto_aux_in_warmup"):
        CANON[key] = getattr(args, key)

    full = load_split(args.dataset)
    recs = full["train"] + full["val"] + full["test"]
    aux_recs = load_aux_records(args.aux_train_dataset)
    if aux_recs:
        print(f"[aux_train_dataset] loaded={len(aux_recs)} path={args.aux_train_dataset} "
              f"weight_scale={args.aux_train_weight_scale} "
              f"max_per_fold={args.aux_train_max_per_fold}", flush=True)
    apply_evidence_policy(recs, args.evidence_policy)
    folds, y_all, g_all = make_folds(recs, args.folds, seed=args.fold_seed)

    cm_methods = ["CLAIMARC_pcls", "CLAIMARC_selectiveRKC", "CLAIMARC_v2"]
    methods = cm_methods + args.baselines
    oof = {m: {"p": np.full(len(recs), np.nan), "yhat": np.full(len(recs), np.nan)} for m in methods}
    y_oof = np.array([int(r["y"]) for r in recs], float)
    c_oof = np.array([float(r.get("c", 0.05)) for r in recs], float)
    fold_oof = np.full(len(recs), -1, dtype=int)

    for fi, (tr_full, te_idx) in enumerate(folds):
        if args.max_folds > 0 and fi >= args.max_folds:
            break
        tr_idx, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        splits = {"train": [recs[i] for i in tr_idx],
                  "val": [recs[i] for i in va_idx],
                  "test": [recs[i] for i in te_idx]}
        train_prior = float(np.mean([int(r["y"]) for r in splits["train"]]))
        if aux_recs:
            aux_fold, aux_info = select_aux_train(
                aux_recs,
                splits["val"] + splits["test"],
                scale=args.aux_train_weight_scale,
                max_per_fold=args.aux_train_max_per_fold,
            )
            splits["train"] = splits["train"] + aux_fold
            print(f"[aux_train] fold={fi} added={aux_info['added']} "
                  f"blocked={aux_info['blocked']} available={aux_info['available']}",
                  flush=True)
        print(f"\n==== FOLD {fi}: train={len(splits['train'])} main_train={len(tr_idx)} "
              f"val={len(va_idx)} test={len(te_idx)} "
              f"test_pos={int(y_all[te_idx].sum())} ====", flush=True)

        cm_probs, yv, yt, ct = claimarc_fold(splits, args.cm_seeds, args.tmpdir, fi)
        for name, (pv, pt) in cm_probs.items():
            thr = choose_threshold(yv, pv, args, train_prior=train_prior)
            oof[name]["p"][te_idx] = pt
            oof[name]["yhat"][te_idx] = (pt >= thr).astype(int)
            fold_oof[te_idx] = fi

        import gc
        for kind in args.baselines:
            bpv, byv, bpt, byt, bct = baseline_fold(kind, splits, 0, args.tmpdir, fi)
            bthr = choose_threshold(byv, bpv, args, train_prior=train_prior)
            oof[kind]["p"][te_idx] = bpt
            oof[kind]["yhat"][te_idx] = (bpt >= bthr).astype(int)
            gc.collect(); torch.cuda.empty_cache()
        # 每折落盘进度
        json.dump({"fold_done": fi}, open(args.out + ".progress", "w"))

    # === 汇集 OOF 指标 + 配对 bootstrap ===
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = {}
    for m in methods:
        p = oof[m]["p"]; yhat = oof[m]["yhat"]
        ok = ~np.isnan(p)
        rows[m] = {
            "auprc": round(float(average_precision_score(y_oof[ok], p[ok])), 4),
            "auroc": round(float(roc_auc_score(y_oof[ok], p[ok])), 4),
            "macro_f1": round(macro(y_oof[ok], yhat[ok]), 4),
            "wF1": round(macro(y_oof[ok], yhat[ok], w=np.clip(c_oof[ok], 0.05, None)), 4),
            "n": int(ok.sum()),
        }
        print(f"  {m:16s} AP={rows[m]['auprc']} AUROC={rows[m]['auroc']} "
              f"mF1={rows[m]['macro_f1']} wF1={rows[m]['wF1']} n={rows[m]['n']}", flush=True)

    if args.dump_oof:
        dump = {
            "y": y_oof,
            "c": c_oof,
            "fold_id": fold_oof,
            "pair_id": np.array([r.get("pair_id", "") for r in recs], dtype=object),
            "room_id": np.array([r.get("room_id", "") for r in recs], dtype=object),
            "attribute_id": np.array([r.get("attribute_id", "") for r in recs], dtype=object),
            "category": np.array([r.get("category", "") for r in recs], dtype=object),
            "source_count": np.array([source_count(r) for r in recs], dtype=float),
            "evidence_combo": np.array([evidence_combo(r) for r in recs], dtype=object),
            "confidence": np.array([confidence_bin(r) for r in recs], dtype=object),
        }
        for m in methods:
            safe = m.replace("/", "_").replace(" ", "_")
            dump[f"p__{safe}"] = oof[m]["p"]
            dump[f"yhat__{safe}"] = oof[m]["yhat"]
        np.savez_compressed(args.dump_oof, **dump)
        print(f"[dump_oof] -> {args.dump_oof}", flush=True)

    sig = {}
    if args.n_boot > 0:
        print("=== Paired bootstrap on pooled OOF (CLAIMARC method vs baseline) ===", flush=True)
        for cm_name in cm_methods:
            pa = oof[cm_name]["p"]
            for m in args.baselines:
                pb = oof[m]["p"]
                ok = (~np.isnan(pa)) & (~np.isnan(pb))
                s = paired_bootstrap(y_oof[ok], pa[ok], pb[ok], c_oof[ok],
                                     n_boot=args.n_boot)
                key = f"{cm_name}_vs_{m}"
                sig[key] = s
                print(f"  {cm_name:22s} vs {m:14s} "
                      f"dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                      f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                      f"dMacroF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                      flush=True)
    else:
        print("=== Paired bootstrap skipped (--n_boot=0) ===", flush=True)

    json.dump({"folds": args.folds, "max_folds": args.max_folds,
               "fold_seed": args.fold_seed,
               "cm_seeds": args.cm_seeds,
               "evidence_policy": args.evidence_policy,
               "cl_teacher": {
                   "mode": args.cl_teacher_mode,
                   "conf_min": args.cl_teacher_conf_min,
                   "neg_filter": args.cl_neg_filter,
                   "neg_bonus": args.cl_neg_bonus,
                   "neg_bonus_filter": args.cl_neg_bonus_filter,
               },
               "source_domain_weighting": {
                   "source0_ce_scale": args.source0_ce_scale,
                   "source0_cl_scale": args.source0_cl_scale,
                   "source_rich_ce_scale": args.source_rich_ce_scale,
                   "source_rich_cl_scale": args.source_rich_cl_scale,
               },
               "distill": {
                   "bge_weight": args.distill_bge_weight,
                   "bge_folds": args.distill_bge_folds,
                   "teacher_seed": args.distill_teacher_seed,
                   "temp": args.distill_temp,
                   "conf_min": args.distill_conf_min,
                   "c_min": args.distill_c_min,
                   "mode": args.distill_mode,
               },
               "proto_aux": {
                   "weight": args.proto_aux_weight,
                   "group": args.proto_aux_group,
                   "mode": args.proto_aux_mode,
                   "margin": args.proto_aux_margin,
                   "tau": args.proto_aux_tau,
                   "min_class": args.proto_aux_min_class,
                   "c_min": args.proto_aux_c_min,
                   "in_warmup": args.proto_aux_in_warmup,
               },
               "n_boot": int(args.n_boot),
               "threshold_policy": {
                   "name": args.threshold_policy,
                   "fixed_thr": args.fixed_thr,
                   "prior_penalty": args.prior_penalty,
               },
               "aux_train": {
                   "dataset": args.aux_train_dataset,
                   "weight_scale": args.aux_train_weight_scale,
                   "max_per_fold": args.aux_train_max_per_fold,
               },
               "rows": rows, "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[cv] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
