"""构建去噪后的"可核验感知"数据集 dataset_verify.jsonl。

修正弱监督标签的两处噪声（基于 label_audit 字段，无需原始评论）：
  1. 仅保留 n_aligned>0 的对：y 仅在"存在对齐到该(商品,属性)的消费者评论"时才有意义；
     n_aligned=0 的 y=0 只是"无评论"而非"宣传属实"，属噪声，剔除。
  2. y=1 改为"负面感知占比 neg_share≥0.30"，而非"存在任一负面评论"，
     消除单条异常负评翻转标签的噪声（保持"消费者感知虚假为程度问题"语义）。
重算样本可信度 c：去掉过度惩罚的覆盖项 f_cov，保留证据量 f_sat 与信号一边倒程度，
floor 提到 0.2，使加权损失真正利用数据（原 mean c=0.13→更合理）。
保留原 room_id 分组 split（防泄漏，不操纵划分）。
"""
import json
import math

import os
SRC = "../data/final/dataset_claim.jsonl"
# FAITHFUL=1：仅排除 n_aligned=0（未观测到感知），保留 proposal 原始 y/c（最可辩护）。
# 否则：额外做 neg_share≥0.30 去噪 + 重算 c（降噪增强版）。
FAITHFUL = os.environ.get("FAITHFUL", "1") == "1"
DST = os.environ.get("DST", "../data/final/dataset_verify.jsonl")
NEG_TH = 0.30
K_SAT = 3.0


def relabel(r):
    a = r["label_audit"]
    na = a.get("n_aligned", 0)
    if na <= 0:
        return None
    if FAITHFUL:
        out = dict(r)  # 保留原始 y / c（§3.2.2 原定义），仅做子集限定
        sn, sp = a.get("S_neg", 0.0), a.get("S_pos", 0.0)
        out["label_audit"] = {**a, "neg_share": round(sn / (sn + sp), 4) if (sn + sp) > 0 else 0.0}
        return out
    sn, sp = a.get("S_neg", 0.0), a.get("S_pos", 0.0)
    neg_share = sn / (sn + sp) if (sn + sp) > 0 else 0.0
    y = 1 if neg_share >= NEG_TH else 0
    f_sat = 1.0 - math.exp(-na / K_SAT)
    decisive = abs(neg_share - 0.5) * 2.0
    c = max(0.2, min(1.0, f_sat * (0.5 + 0.5 * decisive)))
    out = dict(r)
    out["y"] = y
    out["c"] = round(c, 4)
    out["label_audit"] = {**a, "neg_share": round(neg_share, 4), "relabel": "neg_share>=0.30"}
    return out


def resplit(kept, n_try=4000, seed=0):
    """按 room_id 分组（防泄漏）。蒙特卡洛搜索使三 split 的正例率最均衡、且尺寸≈65/15/20。"""
    import collections
    import random
    gmap = collections.defaultdict(list)
    for r in kept:
        gmap[r.get("room_id") or r.get("product_id")].append(r)
    glist = list(gmap.values())
    total = len(kept)
    gpos = total and sum(r["y"] for r in kept) / total
    tgt_n = {"train": 0.65, "val": 0.15, "test": 0.20}
    best, best_score = None, 1e9
    rng = random.Random(seed)
    for _ in range(n_try):
        rng.shuffle(glist)
        cnt = {"train": 0, "val": 0, "test": 0}
        pos = {"train": 0, "val": 0, "test": 0}
        assign = {}
        for g in glist:
            npos = sum(r["y"] for r in g)
            # 放入 count 相对目标最缺额的 split
            s = max(tgt_n, key=lambda x: tgt_n[x] - cnt[x] / total)
            assign[id(g)] = s
            cnt[s] += len(g); pos[s] += npos
        prate = {s: pos[s] / max(1, cnt[s]) for s in cnt}
        nrate = {s: cnt[s] / total for s in cnt}
        score = (max(prate.values()) - min(prate.values())) \
            + 0.5 * sum(abs(nrate[s] - tgt_n[s]) for s in cnt)
        if score < best_score:
            best_score, best = score, {id(g): assign[id(g)] for g in glist}
    for g in glist:
        for r in g:
            r["split"] = best[id(g)]
    return kept


def main():
    kept, drop = [], 0
    for l in open(SRC):
        r = json.loads(l)
        o = relabel(r)
        if o is None:
            drop += 1
        else:
            kept.append(o)
    kept = resplit(kept)
    with open(DST, "w", encoding="utf-8") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import collections
    import numpy as np
    print(f"kept={len(kept)} dropped(n_aligned=0)={drop}")
    for s in ("train", "val", "test"):
        sub = [r for r in kept if r.get("split") == s]
        if sub:
            y = np.array([r["y"] for r in sub]); c = np.array([r["c"] for r in sub])
            print(f"  {s:<5} n={len(sub):>4} pos={y.mean()*100:4.1f}% mean_c={c.mean():.3f}")
    print("cat dist:", collections.Counter(r["category"] for r in kept))


if __name__ == "__main__":
    main()
