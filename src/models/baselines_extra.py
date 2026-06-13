"""补充基线（§4.4.5 消融 1 的双流退化对照 + 通用嵌入对照）。

- bge_lr     ：冻结 BGE 分别编码 claim/evidence → 拼接 → 逻辑回归（通用嵌入分类上限）。
- dual_cos   ：冻结 BGE 双编码器 + 余弦相似度（无交叉注意力的退化）。
- bge_knn    ：冻结 BGE 拼接嵌入 + 可信度加权 kNN 投票（非对比检索对照）。
- setfit_lr  ：BGE 句向量 + 监督对比微调思想的近似（这里用冻结向量 + 平衡 LR，作轻量对照）。

均与 CLAIMARC 同划分/同监督权重 c，val 上按 Macro-F1 选阈值，保存预测供 fusion_eval。

用法：python -m models.baselines_extra --dataset .../dataset_verify_faithful.jsonl \
        --kind bge_lr --seed 0 --save_pred out.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from models.data import load_split, resolve_bge_path
from models.baselines import claim_text, evidence_text
from models.train import macro_f1, best_threshold_macroF1, ece


def encode(recs, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(resolve_bge_path(), device=device)
    c = m.encode([claim_text(r) for r in recs], normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False)
    e = m.encode([evidence_text(r) for r in recs], normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False)
    return np.asarray(c), np.asarray(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--kind", required=True, choices=["bge_lr", "dual_cos", "bge_knn", "setfit_lr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_pred", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    sp = load_split(args.dataset)

    feats = {s: encode(sp[s], device) for s in ("train", "val", "test")}
    Y = {s: np.array([int(r.get("y", 0)) for r in sp[s]]) for s in sp}
    C = {s: np.array([float(r.get("c", 0.05)) for r in sp[s]]) for s in sp}

    def probs(kind):
        ct, et = feats["train"]
        if kind in ("bge_lr", "setfit_lr"):
            Xtr = np.concatenate([ct, et, ct - et, ct * et], 1)
            clf = LogisticRegression(C=1.0, max_iter=3000,
                                     class_weight=("balanced" if kind == "setfit_lr" else None))
            clf.fit(Xtr, Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
            out = {}
            for s in ("val", "test"):
                cs, es = feats[s]
                Xs = np.concatenate([cs, es, cs - es, cs * es], 1)
                out[s] = clf.predict_proba(Xs)[:, 1]
            return out
        if kind == "dual_cos":
            out = {}
            for s in ("val", "test"):
                cs, es = feats[s]
                out[s] = ((cs * es).sum(1) + 1.0) / 2.0  # cosine→[0,1]
            return out
        if kind == "bge_knn":
            Xtr = np.concatenate([ct, et], 1)
            Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
            out = {}
            for s in ("val", "test"):
                cs, es = feats[s]
                Xs = np.concatenate([cs, es], 1)
                Xs /= (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
                sims = Xs @ Xtr.T
                k = 15
                pr = np.zeros(len(Xs))
                ytr = Y["train"].astype(float); ctr = np.clip(C["train"], 0.05, None)
                for i in range(len(Xs)):
                    top = np.argpartition(-sims[i], k - 1)[:k]
                    w = ctr[top] * np.clip(sims[i, top], 0, None)
                    pr[i] = (w * ytr[top]).sum() / (w.sum() + 1e-8)
                out[s] = pr
            return out

    pr = probs(args.kind)
    pv, p = pr["val"], pr["test"]
    yv, y, c = Y["val"], Y["test"], C["test"]
    thr = best_threshold_macroF1(yv, pv)
    pred = (p >= thr).astype(int)
    res = {"tag": args.kind, "seed": args.seed, "thr": round(float(thr), 3),
           "macro_f1": round(macro_f1(y, pred), 4),
           "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
           "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
           "auprc": round(average_precision_score(y, p), 4),
           "auroc": round(roc_auc_score(y, p), 4), "ece": round(ece(y, p), 4),
           "n_test": int(len(y)), "pos_test": int(y.sum())}
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
    if args.save_pred:
        torch.save({"thr": res["thr"], "val": {"p": pv, "y": yv},
                    "test": {"p": p, "y": y, "c": c,
                             "attr": [r.get("attribute_id", "") for r in sp["test"]]}},
                   args.save_pred)
        print(f"[save_pred] -> {args.save_pred}", flush=True)


if __name__ == "__main__":
    main()
