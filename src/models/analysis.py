"""§4.4.3 边界样本判别 + §4.4.4 表征几何分析（基于已训练模型保存的嵌入）。

边界集构造（§4.4.3）：test 中与某 anchor 同标准化属性、商品事实(evidence) BGE cosine≥0.85、
但消费者感知标签相反的样本对；anchor∪confounder 构成诊断子集。
报告各方法在该子集上的 Boundary Macro-F1 / AUPRC，以及同属性反标签 separation。

几何（§4.4.4）：同属性同标签紧度、同属性反标签分离度、属性间正交度、alignment、uniformity。

用法：python -m models.analysis --dataset ../data/final/dataset.jsonl \
        --emb full=emb_full.pt no_cl=emb_nocl.pt global_neg=emb_gneg.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

from models.data import load_split, resolve_bge_path
from models.baselines import evidence_text


def macro(y, pred, w=None):
    return f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)


def evidence_bge(recs, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(resolve_bge_path(), device=device)
    txt = [evidence_text(r) for r in recs]
    return torch.tensor(m.encode(txt, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False))


def build_boundary(fe, attr, y, sim_thr=0.85):
    """返回布尔 mask：参与边界集的 test 样本（自身 + 其 confounder）。"""
    attr = np.array(attr); y = np.array(y)
    sims = (fe @ fe.T).numpy()
    n = len(y)
    in_set = np.zeros(n, dtype=bool)
    pairs = 0
    for i in range(n):
        cand = (attr == attr[i]) & (y != y[i]) & (sims[i] >= sim_thr)
        cand[i] = False
        if cand.any():
            in_set[i] = True
            in_set[cand] = True
            pairs += int(cand.sum())
    return in_set, pairs


def geometry(g, attr, y):
    g = F.normalize(g, dim=-1)
    attr = np.array(attr); y = np.array(y)
    sims = (g @ g.T).numpy()
    n = len(y)
    iu = np.triu_indices(n, 1)
    same_a = (attr[:, None] == attr[None, :])[iu]
    same_y = (y[:, None] == y[None, :])[iu]
    s = sims[iu]
    def mean(mask):
        return float(s[mask].mean()) if mask.any() else None
    tight = mean(same_a & same_y)          # 同属性同标签紧度（高好）
    sep = mean(same_a & ~same_y)           # 同属性反标签分离度（低好）
    cross = mean(~same_a)                  # 属性间正交度
    # alignment / uniformity (Wang & Isola 2020)
    pos = same_a & same_y
    align = float(((2 - 2 * s[pos])).mean()) if pos.any() else None
    unif = float(np.log(np.exp(-2 * (1 - s)).mean()))
    return {"attr_same_label_tight": round(tight, 4) if tight else None,
            "attr_opp_label_sep": round(sep, 4) if sep else None,
            "cross_attr": round(cross, 4) if cross else None,
            "alignment": round(align, 4) if align else None,
            "uniformity": round(unif, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset.jsonl")
    ap.add_argument("--emb", nargs="+", required=True, help="tag=path.pt ...")
    ap.add_argument("--out", default="../data/final/analysis.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = load_split(args.dataset)
    test = sp["test"]
    fe = evidence_bge(test, device)
    bundles = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.emb}
    # 用任一 bundle 的 attr/y 构造边界集（test 顺序一致）
    any_b = torch.load(list(bundles.values())[0], map_location="cpu", weights_only=False)
    attr = any_b["test"]["attr"]; y = any_b["test"]["y"]
    in_set, npairs = build_boundary(fe, attr, y)
    print(f"[boundary] |set|={int(in_set.sum())} pairs={npairs} of {len(y)} test", flush=True)

    rows = []
    for tag, path in bundles.items():
        b = torch.load(path, map_location="cpu", weights_only=False)
        t = b["test"]; thr = b.get("thr", 0.5)
        p = np.array(t["p"]); yy = np.array(t["y"]); cc = np.array(t["c"])
        pred = (p >= thr).astype(int)
        row = {"tag": tag,
               "overall_macro_f1": round(macro(yy, pred), 4),
               "overall_auprc": round(average_precision_score(yy, p), 4),
               "boundary_macro_f1": round(macro(yy[in_set], pred[in_set]), 4),
               "boundary_auprc": round(average_precision_score(yy[in_set], p[in_set]), 4)
               if len(set(yy[in_set])) > 1 else None,
               "boundary_wF1": round(macro(yy[in_set], pred[in_set],
                                            w=np.clip(cc[in_set], 0.05, None)), 4)}
        if "g" in t:  # 几何指标仅对有检索表征的 CLAIMARC 变体计算
            row.update({f"geom_{k}": v for k, v in geometry(t["g"], attr, y).items()})
        rows.append(row)
        print("ANALYSIS", json.dumps(row, ensure_ascii=False), flush=True)
    json.dump({"boundary_size": int(in_set.sum()), "rows": rows},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[analysis] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
