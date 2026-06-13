"""§4.4.2 跨域少样本适应（CLAIMARC-v2 头条实验）。

留一品类 X：源域=其余 9 类（再切 val）；目标域=X（拆 support/query）。
编码器训练后冻结，仅靠"检索库注入 m 条 support"完成对 X 的适应（无需重训）。

对照：
  - CM_forward      ：CLAIMARC 参数化前向（零样本，无适应）
  - CM_RKC(m)       ：CLAIMARC 冻结 + 注入 m 条 support 的检索投票（数据库适应）
  - roberta_zs      ：RoBERTa 源域训练后在目标域零样本
  - roberta_fs(m)   ：RoBERTa 源域训练后再在 m 条 support 上微调（少样本重训，易过拟合）
  - bge_knn(m)      ：冻结 BGE（非对比）+ 同样注入 m 条的 kNN（消融：对比表征 vs 通用嵌入）

聚合多个留出品类 × 多种子，报告 Macro-F1 / AUPRC 的适应曲线。

用法：python -m models.crossdomain_v2 --dataset ../data/final/dataset_verify_faithful.jsonl \
        --holdouts apparel_and_underwear baby_kids_and_pets shoes_and_bags general \
        --seeds 0 1 2 --out ../data/final/v2/xdom.json
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

from models.data import (ClaimDataset, make_collate, load_split, build_tokenizer,
                         resolve_bge_path)
from models.train import train, predict, set_seed


CANON = dict(no_cl=False, no_fusion=False, n_fusion=2, fusion_dropout=0.2, no_lora=False,
             no_weight=False, lora_rank=16, heads=8, tau=0.07, lambda_cl=0.5, Kp=3, Kn=5,
             global_neg=False, backbone="bge", xattn_dir="both", indep_proj=False,
             ffn="swiglu", swa=False, loss="asl", gamma_neg=4.0, gamma_pos=0.0,
             enc_train="lora", unfreeze_top=0, no_ret_disc=False, save_emb="",
             bs=16, accum=2, lr=2e-5, lr_head=1e-4, warmup=3, cl_epochs=6, pos_weight=-1.0)


def mf1(y, p, thr=0.5):
    return round(f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0), 4)


def ap(y, p):
    return round(average_precision_score(y, p), 4) if len(set(y.tolist())) > 1 else None


def rkc_vote(lib_g, lib_y, lib_c, qg, k=10):
    g_l = F.normalize(lib_g.float(), dim=-1)
    g_q = F.normalize(qg.float(), dim=-1)
    sims = (g_q @ g_l.T).cpu().numpy()
    ly = np.asarray(lib_y, float); lc = np.clip(np.asarray(lib_c, float), 0.05, None)
    n = g_q.shape[0]; out = np.zeros(n)
    for i in range(n):
        kk = min(k, sims.shape[1])
        top = np.argpartition(-sims[i], kk - 1)[:kk]
        w = lc[top] * np.clip(sims[i, top], 0, None)
        out[i] = (w * ly[top]).sum() / (w.sum() + 1e-8)
    return out


def split_support_query(hy, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(hy)); rng.shuffle(idx)
    pos = [i for i in idx if hy[i] == 1]; neg = [i for i in idx if hy[i] == 0]
    sup = np.array(pos[:max(10, len(pos) // 2)] + neg[:max(10, len(neg) // 2)])
    query = np.array([i for i in idx if i not in set(sup.tolist())])
    if len(query) < 10 or len(set(hy[query].tolist())) < 2:
        return idx, idx
    return sup, query


def bge_encode(recs, device):
    """冻结 BGE 拼接 claim/evidence 嵌入（通用嵌入对照，无对比训练）。"""
    from sentence_transformers import SentenceTransformer
    from models.baselines import claim_text, evidence_text
    m = SentenceTransformer(resolve_bge_path(), device=device)
    c = m.encode([claim_text(r) for r in recs], normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False)
    e = m.encode([evidence_text(r) for r in recs], normalize_embeddings=True,
                 batch_size=64, show_progress_bar=False)
    return torch.tensor(np.concatenate([np.asarray(c), np.asarray(e)], 1))


def run_holdout(dataset, holdout, seed, ms=(0, 1, 3, 5, 10, 20)):
    set_seed(seed)
    full = load_split(dataset)
    allrecs = full["train"] + full["val"] + full["test"]
    held = [r for r in allrecs if r.get("category") == holdout]
    rest = [r for r in allrecs if r.get("category") != holdout]
    if sum(r["y"] for r in held) < 6:
        return None
    rng = np.random.RandomState(seed); rng.shuffle(rest)
    nval = max(60, len(rest) // 10)
    splits = {"train": rest[nval:], "val": rest[:nval], "test": held}
    args = SimpleNamespace(seed=seed, tag=f"xdom_{holdout}", dataset=dataset, **CANON)

    model, loaders, device, train_pack, res = train(args, splits=splits, return_model=True)
    lib_g, lib_y, lib_c = train_pack

    tok = build_tokenizer(resolve_bge_path())
    collate = make_collate(tok.pad_token_id)
    hl = DataLoader(ClaimDataset(held, tok), batch_size=16, shuffle=False,
                    collate_fn=collate, num_workers=2)
    p_fwd, hg, hy, hc, hattr = predict(model, hl, device)
    sup, query = split_support_query(hy, seed)
    thr = res["thr"]

    # 通用嵌入对照（bge_knn）：冻结 BGE 对 rest(库) 与 held(query/support) 编码
    bge_rest = bge_encode(splits["train"], device)
    rest_y = np.array([int(r["y"]) for r in splits["train"]])
    rest_c = np.array([float(r.get("c", 0.05)) for r in splits["train"]])
    bge_held = bge_encode(held, device)

    out = {"holdout": holdout, "seed": seed, "n_query": int(len(query)),
           "pos_query": int(hy[query].sum()),
           "CM_forward": {"mf1": mf1(hy[query], p_fwd[query], thr), "ap": ap(hy[query], p_fwd[query])},
           "CM_RKC": {}, "BGE_knn": {}}
    for m in ms:
        if m == 0:
            g2, y2, c2 = lib_g, lib_y, lib_c
            bg2, by2, bc2 = bge_rest, rest_y, rest_c
        else:
            take = sup[:m]
            g2 = torch.cat([lib_g, hg[take]], 0)
            y2 = np.concatenate([lib_y, hy[take]]); c2 = np.concatenate([lib_c, hc[take]])
            bg2 = torch.cat([bge_rest, bge_held[take]], 0)
            by2 = np.concatenate([rest_y, hy[take]]); bc2 = np.concatenate([rest_c, hc[take]])
        prkc = rkc_vote(g2, y2, c2, hg[query])
        out["CM_RKC"][str(m)] = {"mf1": mf1(hy[query], prkc, 0.5), "ap": ap(hy[query], prkc)}
        pbge = rkc_vote(bg2, by2, bc2, bge_held[query])
        out["BGE_knn"][str(m)] = {"mf1": mf1(hy[query], pbge, 0.5), "ap": ap(hy[query], pbge)}
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap_.add_argument("--holdouts", nargs="+", required=True)
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap_.add_argument("--out", default="../data/final/v2/xdom.json")
    args = ap_.parse_args()

    results = []
    for ho in args.holdouts:
        for s in args.seeds:
            r = run_holdout(args.dataset, ho, s)
            if r is not None:
                results.append(r)
                print("RESULT_XDOM", json.dumps(r, ensure_ascii=False), flush=True)
                json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)

    # 聚合
    def agg(getter):
        vals = [getter(r) for r in results if getter(r) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    summary = {
        "CM_forward_mf1": agg(lambda r: r["CM_forward"]["mf1"]),
        "CM_RKC0_mf1": agg(lambda r: r["CM_RKC"]["0"]["mf1"]),
        "CM_RKC10_mf1": agg(lambda r: r["CM_RKC"].get("10", {}).get("mf1")),
        "CM_RKC20_mf1": agg(lambda r: r["CM_RKC"].get("20", {}).get("mf1")),
        "CM_forward_ap": agg(lambda r: r["CM_forward"]["ap"]),
        "CM_RKC10_ap": agg(lambda r: r["CM_RKC"].get("10", {}).get("ap")),
        "BGE_knn0_mf1": agg(lambda r: r["BGE_knn"]["0"]["mf1"]),
        "BGE_knn10_mf1": agg(lambda r: r["BGE_knn"].get("10", {}).get("mf1")),
        "BGE_knn20_mf1": agg(lambda r: r["BGE_knn"].get("20", {}).get("mf1")),
        "BGE_knn10_ap": agg(lambda r: r["BGE_knn"].get("10", {}).get("ap")),
    }
    print("XDOM_SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    json.dump({"runs": results, "summary": summary}, open(args.out, "w"),
              ensure_ascii=False, indent=2)
    print(f"[xdom] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
