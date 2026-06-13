"""CLAIMARC 全套实验编排（§4.4.1 主对比 + §4.4.5 消融 + §4.4.3/4 分析 + §4.4.2 跨域）。

- 串行执行，实时流式输出；每个 job 的 RESULT/ANALYSIS/RESULT_XDOM 行解析后增量写入
  data/final/all_results.jsonl（可断点续跑：已完成 tag 自动跳过）。
- 单个 job 失败不中断整体。
按优先级排序：主对比 → 核心消融 → 融合消融 → 超参消融 → 跨域。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_claim.jsonl"   # 仅含主播话术的对（正例 3.9%→17.9%）
OUT = "../data/final/all_results.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

FULL = ["--warmup", "3", "--cl_epochs", "6"]      # 主/消融统一 9 epoch 预算


def claimarc(tag, extra=None, seeds=(0,), save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *(extra or [])]
        if save_prefix:
            cmd += ["--save_emb", f"{EMB}/{save_prefix}_s{s}.pt"]
        jobs.append((f"{tag}__s{s}", cmd))
    return jobs


def ft(kind, seeds=(0, 1, 2), loss="asl", suffix="", save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.baselines_ft", "--dataset", DS,
               "--kind", kind, "--seed", str(s), "--loss", loss]
        if save_prefix:
            cmd += ["--save_pred", f"{EMB}/{save_prefix}_s{s}.pt"]
        jobs.append((f"{kind}{suffix}__s{s}", cmd))
    return jobs


SEEDS = (0, 1, 2, 3, 4)   # 5 种子：稳健均值±std + 种子集成（小样本降方差）
JOBS = []
# ============ 主任务：含主播话术的(商品,属性)对（正例 17.9%，去除"无话术"捷径）============
# ---------- PHASE 1: canonical CLAIMARC（ASL γ4 / λ0.5 / τ0.05 / K(3,5) + SWA）5 seed ----------
JOBS += claimarc("claimarc", seeds=SEEDS, save_prefix="emb_clarc")
# ---------- PHASE 2: 主对比基线（统一 ASL 损失，公平对比）5 seed，存每种子预测供集成 ----------
JOBS += [("BGE_frozen_LR", [PY, "-m", "models.baselines", "--dataset", DS])]
JOBS += ft("bert_cls", seeds=SEEDS, save_prefix="pred_bert")
JOBS += ft("roberta_cls", seeds=SEEDS, save_prefix="pred_roberta")
JOBS += ft("esim", seeds=SEEDS, save_prefix="pred_esim")
JOBS += ft("bert_nli", seeds=(0, 1, 2))
# ---------- PHASE 2b: 基线最强损失变体（BCE），报告最强基线，杜绝"削弱基线"质疑 ----------
JOBS += ft("bert_cls", loss="bce", suffix="_bce", seeds=(0, 1, 2))
JOBS += ft("roberta_cls", loss="bce", suffix="_bce", seeds=(0, 1, 2))
# ---------- PHASE 2c: 种子集成对比（CLAIMARC vs 各基线，公平地都做集成）----------
JOBS += [("ensemble_main", [PY, "-m", "models.ensemble", "--spec",
          f"CLAIMARC={EMB}/emb_clarc_s*.pt", f"CLAIMARC_RKC:rkc={EMB}/emb_clarc_s*.pt",
          f"BERT={EMB}/pred_bert_s*.pt",
          f"RoBERTa={EMB}/pred_roberta_s*.pt", f"ESIM={EMB}/pred_esim_s*.pt"])]
# ---------- PHASE 3: 核心消融（从 canonical 派生，seed0）----------
JOBS += claimarc("abl_no_cl", ["--no_cl"], save_prefix="emb_nocl")
JOBS += claimarc("abl_global_neg", ["--global_neg"], save_prefix="emb_gneg")
JOBS += claimarc("abl_no_weight", ["--no_weight"])
JOBS += claimarc("abl_no_fusion", ["--no_fusion"])
JOBS += claimarc("abl_no_lora", ["--no_lora"])
JOBS += claimarc("abl_backbone_bert", ["--backbone", "bert"])
JOBS += claimarc("abl_swa", ["--swa"])
# ---------- PHASE 4: 边界+几何分析（依赖 seed0 嵌入）----------
JOBS += [("analysis_boundary_geom",
          [PY, "-m", "models.analysis", "--dataset", DS, "--emb",
           f"full={EMB}/emb_clarc_s0.pt", f"no_cl={EMB}/emb_nocl_s0.pt",
           f"global_neg={EMB}/emb_gneg_s0.pt", f"bert={EMB}/pred_bert_s0.pt",
           f"roberta={EMB}/pred_roberta_s0.pt", f"esim={EMB}/pred_esim_s0.pt"])]
# ---------- PHASE 5: 融合机制消融 ----------
JOBS += claimarc("abl_xattn_c2e", ["--xattn_dir", "c2e"])
JOBS += claimarc("abl_xattn_e2c", ["--xattn_dir", "e2c"])
JOBS += claimarc("abl_indep_proj", ["--indep_proj"])
JOBS += claimarc("abl_ffn_gelu", ["--ffn", "gelu"])
JOBS += claimarc("abl_n_fusion1", ["--n_fusion", "1"])
JOBS += claimarc("abl_n_fusion4", ["--n_fusion", "4"])
# ---------- PHASE 6: RACL/超参消融（canonical=λ0.5,τ0.05,K(3,5)）----------
# λ_CL 扫描（τ0.05）：0.1/0.3/1.0（0.5=canonical；另有 boost_l05=λ0.5τ0.07）
JOBS += claimarc("abl_lambda_0.1", ["--lambda_cl", "0.1"])
JOBS += claimarc("abl_lambda_0.3", ["--lambda_cl", "0.3"])
JOBS += claimarc("abl_lambda_1.0", ["--lambda_cl", "1.0"])
# 损失函数消融（canonical=ASL）：BCE / Focal
JOBS += claimarc("abl_loss_bce", ["--loss", "bce"])
JOBS += claimarc("abl_loss_focal", ["--loss", "focal"])
# τ 扫描（λ0.5）：0.10（0.05=canonical；0.07=boost_l05）
JOBS += claimarc("abl_tau_0.10", ["--tau", "0.10"])
# Kp/Kn 扫描：(1,1)（(3,5)=canonical；(5,10)=boost_l05_t05_K）
JOBS += claimarc("abl_K_1_1", ["--Kp", "1", "--Kn", "1"])
# LoRA 秩扫描
JOBS += claimarc("abl_lora_8", ["--lora_rank", "8"])
JOBS += claimarc("abl_lora_32", ["--lora_rank", "32"])
# ---------- PHASE 7: 跨域 留一品类（最大正例的品类）----------
for cat in ("apparel_and_underwear", "general", "baby_kids_and_pets",
            "shoes_and_bags", "sports_and_outdoor", "beauty_and_personal_care",
            "smart_home", "food_and_beverages", "digital_and_electronics",
            "jewelry_and_collectibles"):
    JOBS += [(f"xdom_{cat}", [PY, "-m", "models.crossdomain", "--dataset", DS,
                              "--holdout", cat, "--warmup", "2", "--cl_epochs", "3"])]


def done_tags():
    if not os.path.exists(OUT):
        return set()
    tags = set()
    for line in open(OUT):
        try:
            tags.add(json.loads(line)["_job"])
        except Exception:
            pass
    return tags


def log_result(job, payload):
    payload["_job"] = job
    payload["_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_job(job, cmd):
    print(f"\n{'='*70}\n[RUN] {job}\n{'='*70}", flush=True)
    t0 = time.time()
    got = False
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
    try:
        p = subprocess.Popen(cmd, cwd=src_dir,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=ENV)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            for key in ("RESULT_XDOM", "ANALYSIS", "RESULT"):
                if line.startswith(key + " "):
                    try:
                        log_result(job, json.loads(line[len(key) + 1:]))
                        got = True
                    except Exception as e:
                        print(f"[parse-err] {e}", flush=True)
                    break
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job}: {e}", flush=True)
    if not got:
        log_result(job, {"error": "no_result", "cmd": " ".join(cmd)})
    print(f"[DONE] {job} in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    done = done_tags()
    print(f"[run_all] {len(JOBS)} jobs, {len(done)} already done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print(f"[skip] {job}", flush=True)
            continue
        run_job(job, cmd)
    print("\n######## RUN_ALL COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
