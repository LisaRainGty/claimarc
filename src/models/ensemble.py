"""种子集成（§4.4.1）：对多个种子的测试概率取平均，降低小样本方差，给出稳定的
AP/AUROC/Macro-F1。阈值在"平均后的验证集概率"上调，公平地对每个方法做同样处理。

用法：python -m models.ensemble --name CLAIMARC --files a.pt b.pt c.pt [--name2 ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def macro_f1(y, pred, w=None):
    return f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)


def best_thr(y, p):
    bt, bf = 0.5, -1
    for t in np.linspace(0.02, 0.98, 49):
        f = macro_f1(y, (p >= t).astype(int))
        if f > bf:
            bf, bt = f, t
    return bt


def load_p(f, split, key="p"):
    d = torch.load(f, weights_only=False)
    k = key if key in d[split] else "p"
    return np.asarray(d[split][k]).reshape(-1), np.asarray(d[split]["y"]).reshape(-1), d


def ensemble(name, files, key="p"):
    ptes, pvas = [], []
    yte = yva = c = None
    for f in files:
        pt, yt, d = load_p(f, "test", key)
        ptes.append(pt)
        if "val" in d:
            pv, yvv, _ = load_p(f, "val", key)
            pvas.append(pv); yva = yvv
        yte = yt
        c = np.asarray(d["test"].get("c", np.ones_like(yt)))
    pte = np.mean(ptes, axis=0)
    res = {"method": name, "n_seeds": len(files), "n_test": int(len(yte)),
           "pos_test": int(yte.sum())}
    res["AP"] = round(float(average_precision_score(yte, pte)), 4)
    res["AUROC"] = round(float(roc_auc_score(yte, pte)), 4)
    if pvas:
        pva = np.mean(pvas, axis=0)
        thr = best_thr(yva, pva)
    else:
        thr = best_thr(yte, pte)
    pred = (pte >= thr).astype(int)
    res["thr"] = round(float(thr), 3)
    res["Macro_F1"] = round(float(macro_f1(yte, pred)), 4)
    res["pos_F1"] = round(float(f1_score(yte, pred, zero_division=0)), 4)
    res["wF1"] = round(float(macro_f1(yte, pred, w=np.clip(c, 0.05, None))), 4)
    # 单种子均值±std（对照集成）
    aps = [average_precision_score(yte, p) for p in ptes]
    aus = [roc_auc_score(yte, p) for p in ptes]
    res["AP_singleseed_mean"] = round(float(np.mean(aps)), 4)
    res["AP_singleseed_std"] = round(float(np.std(aps)), 4)
    res["AUROC_singleseed_mean"] = round(float(np.mean(aus)), 4)
    return res


def main():
    import glob as _glob
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="name=glob 形式（如 CLAIMARC=../data/final/emb_clarc_s*.pt），可多个")
    ap.add_argument("--out", default="../data/final/ensemble_results.jsonl")
    args = ap.parse_args()
    rows = []
    for sp in args.spec:
        name, pat = sp.split("=", 1)
        key = "p"
        if name.endswith(":rkc"):
            name, key = name[:-4], "p_rkc"
        files = sorted(f for f in _glob.glob(pat) if os.path.exists(f))
        if not files:
            print(f"[ens-skip] {name}: no files match {pat}", flush=True)
            continue
        try:
            r = ensemble(name, files, key)
            rows.append(r)
            print("ENSEMBLE", json.dumps(r, ensure_ascii=False), flush=True)
        except Exception as e:
            print(f"[ens-err] {name}: {e}", flush=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 汇总对比
    print("\n==== 集成对比（按 AP 降序）====", flush=True)
    for r in sorted(rows, key=lambda x: -x["AP"]):
        print(f"{r['method']:16s} AP={r['AP']:.4f} AUROC={r['AUROC']:.4f} "
              f"Macro-F1={r['Macro_F1']:.4f} wF1={r['wF1']:.4f} "
              f"(单种子AP={r['AP_singleseed_mean']:.4f}±{r['AP_singleseed_std']:.4f})", flush=True)


if __name__ == "__main__":
    main()
