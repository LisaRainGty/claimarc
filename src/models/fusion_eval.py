"""CLAIMARC-v2 离线推理融合：Adaptive Retrieval Fusion (ARF) + 多种子集成 + 标定。

保留 RACL 核心（属性分块对比表征 g + 检索 kNN），但把原先"全局标量 α"的朴素
插值（P=α·P_cls+(1−α)·P_knn）升级为"逐样本置信度自适应门控"（Zhang et al. 2023,
Retrieval-Augmented Classification with Decoupled Representation 的学习式插值思想）：

  λ_i = σ( w·[p_cls, p_knn, top1_sim, mean_sim_k, agreement, weight_mass, attr_support] )
  P_i = ARF 门控直接输出（c 加权逻辑回归，在 val 上拟合，避免 train 的 p_cls 过拟合泄漏）。

多种子集成：逐种子在各自表征空间内算 kNN 特征 → 跨种子平均特征 → 单一门控。
基线用同样的"多种子概率平均 + val 选阈值"协议，保证公平。

用法：
  python -m models.fusion_eval --dataset ../data/final/dataset_verify_faithful.jsonl \
      --cm cap_c_lora16_s0.pt cap_c_lora16_s1.pt cap_c_lora16_s2.pt \
      --baseline "roberta=rob_s0.pt,rob_s1.pt;bert=bert_s0.pt,bert_s1.pt" \
      --boundary --out ../data/final/v2_compare.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


# ----------------------------- 基础指标 -----------------------------
def macro(y, pred, w=None):
    return f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)


def _macro_fast(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = ((y == 1) & (pred == 1)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    tn = ((y == 0) & (pred == 0)).sum()
    f1_pos = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    f1_neg = 0.0 if (2 * tn + fp + fn) == 0 else (2 * tn) / (2 * tn + fp + fn)
    return float((f1_pos + f1_neg) / 2)


def best_thr_score(y, p):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f = _macro_fast(y, p >= t)
        if f > best:
            best, best_t = f, t
    return best_t, best


def best_thr(y, p):
    return best_thr_score(y, p)[0]


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        hi = p <= bins[i + 1] if i == n_bins - 1 else p < bins[i + 1]
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - (y[m] == (p[m] >= 0.5)).mean())
    return float(e)


def metric_row(y, p, c, thr, mask=None):
    if mask is not None:
        y, p, c = y[mask], p[mask], c[mask]
    pred = (p >= thr).astype(int)
    two = len(set(y.tolist())) > 1
    return {
        "macro_f1": round(macro(y, pred), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(macro(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(average_precision_score(y, p), 4) if two else None,
        "auroc": round(roc_auc_score(y, p), 4) if two else None,
        "ece": round(ece(y, p), 4),
        "n": int(len(y)), "pos": int(y.sum()),
    }


# ----------------------------- 属性分块 kNN 特征 -----------------------------
def knn_features(store_g, store_y, store_c, store_attr, q_g, q_attr, k=15, self_idx=None):
    """同属性近邻（不足 3 回退全局）的可信度加权投票 + 检索置信特征。
    self_idx: 当 query==store（train OOF）时，逐样本自身下标，需从近邻中剔除。"""
    g_s = F.normalize(store_g.float(), dim=-1)
    g_q = F.normalize(q_g.float(), dim=-1)
    sims = (g_q @ g_s.T).cpu().numpy()
    sa = np.asarray(store_attr); qa = np.asarray(q_attr)
    sy = np.asarray(store_y, dtype=float)
    sc = np.clip(np.asarray(store_c, dtype=float), 0.05, None)
    n = g_q.shape[0]
    feat = {key: np.zeros(n) for key in
            ("p_knn", "top1", "meansim", "agree", "wmass", "support")}
    all_idx = np.arange(len(sa))
    for i in range(n):
        idx = np.where(sa == qa[i])[0]
        if self_idx is not None:
            idx = idx[idx != self_idx[i]]
        support = 1.0 if len(idx) >= 3 else 0.0
        if len(idx) < 3:
            idx = all_idx if self_idx is None else all_idx[all_idx != self_idx[i]]
        s = sims[i, idx]
        kk = min(k, len(idx))
        top = np.argpartition(-s, kk - 1)[:kk]
        j = idx[top]
        sj = np.clip(s[top], 0.0, None)
        w = sc[j] * sj
        wsum = w.sum() + 1e-8
        pk = float((w * sy[j]).sum() / wsum)
        feat["p_knn"][i] = pk
        feat["top1"][i] = float(s[top].max())
        feat["meansim"][i] = float(sj.mean())
        feat["agree"][i] = abs(2 * pk - 1)
        feat["wmass"][i] = float(min(wsum, 5.0) / 5.0)
        feat["support"][i] = support
    return feat


FEAT_KEYS = ("p_cls", "p_knn", "top1", "meansim", "agree", "wmass", "support")


def stack_feats(p_cls, kf):
    return np.column_stack([p_cls] + [kf[key] for key in FEAT_KEYS[1:]])


# ----------------------------- 多种子特征聚合 -----------------------------
def build_split_features(bundles, split, k=15):
    """对一个 split 聚合多种子的 kNN 特征 + p_cls（均在各自种子空间内算后跨种子平均）。"""
    feats = []
    p_cls_list = []
    y = c = attr = None
    for b in bundles:
        tr, q = b["train"], b[split]
        self_idx = np.arange(len(q["y"])) if split == "train" else None
        kf = knn_features(tr["g"], tr["y"], tr["c"], tr["attr"],
                          q["g"], q["attr"], k=k, self_idx=self_idx)
        feats.append(stack_feats(np.asarray(q["p"], float), kf))
        p_cls_list.append(np.asarray(q["p"], float))
        y = np.asarray(q["y"]); c = np.asarray(q["c"], float); attr = q["attr"]
    X = np.mean(feats, axis=0)            # 跨种子平均特征
    p_cls = np.mean(p_cls_list, axis=0)   # 集成 p_cls
    return X, p_cls, y, c, attr


def fit_arf(Xv, yv, cv):
    """在 val 上拟合 ARF 门控（c 加权逻辑回归 + 标准化）。诊断用：易在小 val 过拟合。"""
    mu = Xv.mean(0); sd = Xv.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    clf.fit((Xv - mu) / sd, yv, sample_weight=np.clip(cv, 0.05, None))
    return (clf, mu, sd)


def apply_arf(model, X):
    clf, mu, sd = model
    return clf.predict_proba((X - mu) / sd)[:, 1]


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(p_v, y_v, c_v):
    """单调 Platt 标定：σ(a·logit(p)+b)，a>0 时保序 → 不改变 AUROC/AP，仅修 ECE 与阈值迁移。"""
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(_logit(p_v).reshape(-1, 1), y_v, sample_weight=np.clip(c_v, 0.05, None))
    return clf


def apply_platt(clf, p):
    return clf.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def scalar_alpha(p_cls_v, p_knn_v, y_v):
    """val 上按 AP 选检索融合标量 α（保留原 RKC 协议；域内常≈1，跨域>0）。"""
    best_a, best_s = 1.0, -1.0
    for a in np.linspace(0, 1, 21):
        s = average_precision_score(y_v, a * p_cls_v + (1 - a) * p_knn_v)
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def fit_blend2(p_cls_v, p_knn_v, y_v, c_v):
    """2 特征检索融合门控（[logit(p_cls), logit(p_knn)] 上的 c 加权逻辑回归）。
    比标量 α 更灵活、比 7 特征 ARF 更抗过拟合（仅 2 参 + 截距），在小 val 上稳健。"""
    X = np.column_stack([_logit(p_cls_v), _logit(p_knn_v)])
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y_v, sample_weight=np.clip(c_v, 0.05, None))
    return clf


def apply_blend2(clf, p_cls, p_knn):
    return clf.predict_proba(np.column_stack([_logit(p_cls), _logit(p_knn)]))[:, 1]


def fit_selective_rkc(Xv, yv, cv):
    """保守检索门控：只在主分类器不确定且近邻证据一致时，让 RKC 小步修正。

    这是对小验证集更稳健的非参数融合：若 val 上相对 p_cls 没有清晰增益，自动退化为
    p_cls。这样保留检索增强机制，但避免低质量邻居把模型推向错误标签。
    """
    pcls = Xv[:, 0]
    pknn = Xv[:, 1]
    base_thr, base_mf = best_thr_score(yv, pcls)
    base_ap = average_precision_score(yv, pcls) if len(set(yv.tolist())) > 1 else 0.0
    base = {
        "noop": True, "thr": round(float(base_thr), 3),
        "macro_f1": float(base_mf), "auprc": float(base_ap),
        "uncert": 0.0, "agree": 1.0, "top1": 1.0, "support": 1.0,
        "wmass": 1.0, "beta": 0.0, "gate_rate": 0.0,
    }
    best = dict(base)
    best_score = base_mf + 0.15 * base_ap
    grid = {
        "uncert": (0.12, 0.20, 0.30),
        "agree": (0.25, 0.45, 0.60),
        "top1": (0.0, 0.50, 0.65),
        "support": (0.0, 1.0),
        "wmass": (0.0, 0.10),
        "beta": (0.35, 0.55, 0.75),
    }
    for uncert in grid["uncert"]:
        uncertain = np.abs(pcls - 0.5) <= uncert
        if uncertain.sum() < 3:
            continue
        for agree in grid["agree"]:
            agreed = Xv[:, 4] >= agree
            for top1 in grid["top1"]:
                similar = Xv[:, 2] >= top1
                for support in grid["support"]:
                    supported = Xv[:, 6] >= support
                    for wmass in grid["wmass"]:
                        trusted = uncertain & agreed & similar & supported & (Xv[:, 5] >= wmass)
                        if trusted.sum() < 3:
                            continue
                        for beta in grid["beta"]:
                            q = pcls.copy()
                            q[trusted] = pcls[trusted] + beta * (pknn[trusted] - pcls[trusted])
                            q = np.clip(q, 0.0, 1.0)
                            thr, mf = best_thr_score(yv, q)
                            ap = average_precision_score(yv, q) if len(set(yv.tolist())) > 1 else 0.0
                            score = mf + 0.15 * ap - 0.0005 * trusted.sum()
                            if score > best_score:
                                best_score = score
                                best = {
                                    "noop": False, "thr": round(float(thr), 3),
                                    "macro_f1": float(mf), "auprc": float(ap),
                                    "uncert": float(uncert), "agree": float(agree),
                                    "top1": float(top1), "support": float(support),
                                    "wmass": float(wmass), "beta": float(beta),
                                    "gate_rate": float(trusted.mean()),
                                }
    if best["macro_f1"] < base_mf + 0.005 and best["auprc"] < base_ap + 0.005:
        return base
    return best


def apply_selective_rkc(rule, X):
    pcls = X[:, 0].copy()
    if rule.get("noop", True):
        return pcls, np.zeros(len(pcls), dtype=bool)
    trusted = (
        (np.abs(X[:, 0] - 0.5) <= rule["uncert"]) &
        (X[:, 4] >= rule["agree"]) &
        (X[:, 2] >= rule["top1"]) &
        (X[:, 6] >= rule["support"]) &
        (X[:, 5] >= rule["wmass"])
    )
    out = pcls.copy()
    out[trusted] = X[:, 0][trusted] + rule["beta"] * (X[:, 1][trusted] - X[:, 0][trusted])
    return np.clip(out, 0.0, 1.0), trusted


# ----------------------------- 主流程 -----------------------------
def torch_load_compat(path, **kwargs):
    """torch.load compatible with both torch>=2.6 and older CUDA builds."""
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_bundles(paths):
    return [torch_load_compat(p, map_location="cpu") for p in paths]


def ensemble_pcls(bundles, split):
    return np.mean([np.asarray(b[split]["p"], float) for b in bundles], axis=0)


def paired_bootstrap(y, p_a, p_b, c, n_boot=2000, seed=0):
    """配对 bootstrap：返回 (ΔAP, ΔAUROC, ΔmF1) 的 p 值（a 优于 b 的单侧）+ 均值差。
    mF1 用各自在原样本上的最优阈值固定后再 bootstrap（近似）。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    ta = best_thr(y, p_a); tb = best_thr(y, p_b)
    d_ap = d_au = d_f1 = 0
    s_ap = s_au = s_f1 = []
    dap_l, dau_l, df1_l = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        dap_l.append(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx]))
        dau_l.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
        df1_l.append(macro(yy, (p_a[idx] >= ta).astype(int)) - macro(yy, (p_b[idx] >= tb).astype(int)))
    def summ(arr):
        arr = np.array(arr)
        return {"mean_delta": round(float(arr.mean()), 4),
                "ci": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
                "p_a_gt_b": round(float((arr <= 0).mean()), 4)}
    return {"dAP": summ(dap_l), "dAUROC": summ(dau_l), "dMacroF1": summ(df1_l)}


def boundary_mask(dataset, test_attr, test_y, sim_thr=0.85):
    from models.data import load_split, resolve_bge_path
    from models.baselines import evidence_text
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_split(dataset)["test"]
    m = SentenceTransformer(resolve_bge_path(), device=device)
    fe = torch.tensor(m.encode([evidence_text(r) for r in test],
                               normalize_embeddings=True, batch_size=64, show_progress_bar=False))
    sims = (fe @ fe.T).numpy()
    attr = np.asarray(test_attr); y = np.asarray(test_y)
    n = len(y); in_set = np.zeros(n, dtype=bool)
    for i in range(n):
        cand = (attr == attr[i]) & (y != y[i]) & (sims[i] >= sim_thr)
        cand[i] = False
        if cand.any():
            in_set[i] = True; in_set[cand] = True
    return in_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--cm", nargs="+", required=True, help="CLAIMARC seed .pt paths")
    ap.add_argument("--baseline", default="", help="name=p1,p2;name2=...")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--boundary", action="store_true")
    ap.add_argument("--out", default="../data/final/v2_compare.json")
    args = ap.parse_args()

    bundles = load_bundles(args.cm)
    print(f"[fusion] CLAIMARC seeds={len(bundles)}", flush=True)

    # --- val/test 特征 ---
    Xv, pcls_v, yv, cv, attr_v = build_split_features(bundles, "val", k=args.k)
    Xt, pcls_t, yt, ct, attr_t = build_split_features(bundles, "test", k=args.k)
    pknn_v, pknn_t = Xv[:, 1], Xt[:, 1]

    # boundary 子集
    bmask = boundary_mask(args.dataset, attr_t, yt) if args.boundary else None
    if bmask is not None:
        print(f"[boundary] |set|={int(bmask.sum())}/{len(yt)}", flush=True)

    rows = {}
    test_probs = {}

    def record(name, p_v, p_t, meta=None):
        thr = best_thr(yv, p_v)
        r = {"thr": round(float(thr), 3), "overall": metric_row(yt, p_t, ct, thr)}
        if meta is not None:
            r["meta"] = meta
        if bmask is not None and bmask.any():
            r["boundary"] = metric_row(yt, p_t, ct, thr, mask=bmask)
        rows[name] = r
        test_probs[name] = p_t
        ov = r["overall"]
        print(f"  {name:22s} AP={ov['auprc']} AUROC={ov['auroc']} mF1={ov['macro_f1']} "
              f"wF1={ov['wF1']} ECE={ov['ece']}", flush=True)

    print("=== CLAIMARC variants ===", flush=True)
    record("CM_pcls_ens", pcls_v, pcls_t)                       # 纯参数化集成
    record("CM_knn_only", pknn_v, pknn_t)                       # 纯检索
    sel_rule = fit_selective_rkc(Xv, yv, cv)
    sel_v, sel_mask_v = apply_selective_rkc(sel_rule, Xv)
    sel_t, sel_mask_t = apply_selective_rkc(sel_rule, Xt)
    sel_meta = {"rule": sel_rule, "gate_val": round(float(sel_mask_v.mean()), 4),
                "gate_test": round(float(sel_mask_t.mean()), 4)}
    record("CM_selectiveRKC", sel_v, sel_t, sel_meta)
    pl_sel = fit_platt(sel_v, yv, cv)
    record("CLAIMARC_v3", apply_platt(pl_sel, sel_v), apply_platt(pl_sel, sel_t), sel_meta)
    # 检索融合候选 1：标量 α（val 上按 AP 选）
    a = scalar_alpha(pcls_v, pknn_v, yv)
    blendA_v = a * pcls_v + (1 - a) * pknn_v
    blendA_t = a * pcls_t + (1 - a) * pknn_t
    record(f"CM_blendScalar_a{a:.2f}", blendA_v, blendA_t)
    # 检索融合候选 2：2 特征逻辑门控（稳健）
    b2 = fit_blend2(pcls_v, pknn_v, yv, cv)
    blendB_v = apply_blend2(b2, pcls_v, pknn_v)
    blendB_t = apply_blend2(b2, pcls_t, pknn_t)
    record("CM_blend2", blendB_v, blendB_t)
    # === CLAIMARC-v2 主方法：2 特征学习型检索融合(blend2) + 单调 Platt 标定 ===
    # blend2 在各配置上对主指标(Macro-F1/wF1)稳健占优，是 §3.2.9 检索-参数协同判定的落地形式。
    platt = fit_platt(blendB_v, yv, cv)
    record("CLAIMARC_v2", apply_platt(platt, blendB_v), apply_platt(platt, blendB_t))
    # ARF 7 特征自适应门控（诊断：小 val 上易过拟合，非主方法）
    arf = fit_arf(Xv, yv, cv)
    record("CM_ARF_diag", apply_arf(arf, Xv), apply_arf(arf, Xt))

    # --- 基线 ---
    if args.baseline:
        print("=== Baselines (seed-ensembled) ===", flush=True)
        for spec in args.baseline.split(";"):
            if not spec.strip():
                continue
            name, paths = spec.split("=", 1)
            bb = load_bundles(paths.split(","))
            pv = ensemble_pcls(bb, "val")
            pt = ensemble_pcls(bb, "test")
            # 基线 test 顺序与 CM 一致（同划分）；y 取基线自身
            yv_b = np.asarray(bb[0]["val"]["y"]); yt_b = np.asarray(bb[0]["test"]["y"])
            cv_b = np.asarray(bb[0]["val"].get("c", np.ones_like(yv_b)), float)
            ct_b = np.asarray(bb[0]["test"]["c"], float)
            # 同 CLAIMARC-v2 协议：Platt 标定后再选阈值，保证公平
            pl = fit_platt(pv, yv_b, cv_b)
            pv, pt = apply_platt(pl, pv), apply_platt(pl, pt)
            thr = best_thr(yv_b, pv)
            r = {"thr": round(float(thr), 3), "overall": metric_row(yt_b, pt, ct_b, thr)}
            if bmask is not None and bmask.any():
                r["boundary"] = metric_row(yt_b, pt, ct_b, thr, mask=bmask)
            rows["base_" + name] = r
            test_probs["base_" + name] = pt
            ov = r["overall"]
            print(f"  base_{name:17s} AP={ov['auprc']} AUROC={ov['auroc']} "
                  f"mF1={ov['macro_f1']} wF1={ov['wF1']} ECE={ov['ece']}", flush=True)

    # === 配对 bootstrap 显著性：主 CLAIMARC 版本 vs 各基线 ===
    sig = {}
    main_name = "CLAIMARC_v3" if "CLAIMARC_v3" in test_probs else "CLAIMARC_v2"
    if main_name in test_probs:
        pa = test_probs[main_name]
        print(f"=== Paired bootstrap ({main_name} vs baseline; p = P(baseline>=method)) ===", flush=True)
        for name, pb in test_probs.items():
            if not name.startswith("base_"):
                continue
            s = paired_bootstrap(yt, pa, pb, ct)
            sig[name] = s
            print(f"  vs {name:18s} dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                  f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                  f"dMacroF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})", flush=True)

    json.dump({"k": args.k, "seeds": len(bundles), "rows": rows, "significance": sig},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[fusion] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
