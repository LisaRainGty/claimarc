"""用已训练模型在"可核验子集"(n_aligned>0)上重算指标，验证数据子集假设。

不重训：加载 emb_clarc_s*.pt / pred_*_s*.pt（同一 test 顺序），按 pair_id 的
n_aligned>0 掩码过滤，对比 CLAIMARC 集成 vs 基线集成的 AP/AUROC。
"""
import glob
import json
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

DS = "../data/final/dataset_claim.jsonl"
EMB = "../data/final"


def load_align():
    m = {}
    for l in open(DS):
        r = json.loads(l)
        m[r["pair_id"]] = r["label_audit"].get("n_aligned", 0)
    return m


def ens(files, key="p"):
    ps, y = [], None
    for f in sorted(glob.glob(files)):
        d = torch.load(f, map_location="cpu", weights_only=False)
        ps.append(np.asarray(d["test"][key], dtype=float))
        y = np.asarray(d["test"]["y"], dtype=float)
    return np.mean(ps, axis=0), y


def best_f1(p, y, mask):
    best = 0
    for t in np.unique(p[mask]):
        pred = (p[mask] >= t).astype(int)
        best = max(best, f1_score(y[mask], pred, average="macro"))
    return best


def main():
    align = load_align()
    # CLAIMARC：取 pair_id 顺序，构造 n_aligned 掩码
    d0 = torch.load(f"{EMB}/emb_clarc_s0.pt", map_location="cpu", weights_only=False)
    pid = d0["test"]["pair_id"]
    na = np.array([align.get(p, 0) for p in pid])
    mask = na > 0
    print(f"test total={len(pid)}  verifiable(n_aligned>0)={mask.sum()}  pos_rate_sub="
          f"{np.asarray(d0['test']['y'])[mask].mean():.3f}")

    specs = {
        "CLAIMARC": (f"{EMB}/emb_clarc_s*.pt", "p"),
        "CLAIMARC_RKC": (f"{EMB}/emb_clarc_s*.pt", "p_rkc"),
        "BERT": (f"{EMB}/pred_bert_s*.pt", "p"),
        "BERT_BCE": (f"{EMB}/pred_bertbce_s*.pt", "p"),
        "RoBERTa": (f"{EMB}/pred_roberta_s*.pt", "p"),
        "RoBERTa_BCE": (f"{EMB}/pred_robertabce_s*.pt", "p"),
        "ESIM": (f"{EMB}/pred_esim_s*.pt", "p"),
    }
    yref = np.asarray(d0["test"]["y"], dtype=float)
    print(f"\n{'method':<14}{'AP_full':>9}{'AP_sub':>9}{'AUROC_full':>11}{'AUROC_sub':>11}{'F1_sub*':>9}")
    rows = []
    for name, (pat, key) in specs.items():
        files = sorted(glob.glob(pat))
        if not files:
            continue
        p, y = ens(pat, key)
        # 对齐性检查：y 与参考一致
        if not np.array_equal(y, yref):
            tag = " [Y-MISMATCH]"
        else:
            tag = ""
        apf = average_precision_score(y, p)
        aps = average_precision_score(y[mask], p[mask])
        auf = roc_auc_score(y, p)
        aus = roc_auc_score(y[mask], p[mask])
        f1s = best_f1(p, y, mask)
        rows.append((name, aps, aus))
        print(f"{name:<14}{apf:>9.4f}{aps:>9.4f}{auf:>11.4f}{aus:>11.4f}{f1s:>9.4f}{tag}")
    rows.sort(key=lambda x: -x[1])
    print("\n按子集 AP 排序:", [(n, round(a, 4)) for n, a, _ in rows])


if __name__ == "__main__":
    main()
