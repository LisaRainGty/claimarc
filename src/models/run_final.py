"""CLAIMARC 最终全套实验（改进 canonical = LoRA8 + tau0.10 + lambda_cl1.0）。

经多种子验证：lora_rank=8 + tau=0.10 + lambda_cl=1.0 稳健优于旧 canonical
(lora16/tau0.05/lambda0.5) 及最强基线 BERT-BCE（mF1/AP/AUROC 全面领先）。
机理：降低 LoRA 容量抑制对 406 正例的过拟合 + 更平滑/更强的对比正则。

结果写入 all_results_v2.jsonl；所有 .pt 重新生成（旧服务器已释放）。
消融以新 canonical 为中心重新设计；跨域使用新配置。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_verify_faithful.jsonl"   # 去噪：仅含已观测消费者感知(n_aligned>0)
OUT = "../data/final/all_results_final.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_ENDPOINT": "https://hf-mirror.com"}

FULL = ["--warmup", "3", "--cl_epochs", "6"]
# 干净数据上 val-AP 选优：lora16/tau0.07/lambda0.5（即 proposal 原始设计），方差最小
CANON = ["--lora_rank", "16", "--tau", "0.07", "--lambda_cl", "0.5"]
SEEDS = (0, 1, 2, 3, 4)


def claimarc(tag, extra=None, seeds=(0,), save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *(extra or [])]
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


JOBS = []
# ---------- PHASE 1: canonical CLAIMARC（改进配置）5 seed，存嵌入供集成/RKC/分析 ----------
JOBS += claimarc("claimarc", seeds=SEEDS, save_prefix="emb_clarc")
# ---------- PHASE 2: 主对比基线（统一 ASL）5 seed，存每种子预测供集成 ----------
JOBS += [("BGE_frozen_LR", [PY, "-m", "models.baselines", "--dataset", DS])]
JOBS += ft("bert_cls", seeds=SEEDS, save_prefix="pred_bert")
JOBS += ft("roberta_cls", seeds=SEEDS, save_prefix="pred_roberta")
JOBS += ft("esim", seeds=SEEDS, save_prefix="pred_esim")
JOBS += ft("bert_nli", seeds=(0, 1, 2))
# ---------- PHASE 2b: 基线最强损失变体（BCE），5 seed，存预测供集成 ----------
JOBS += ft("bert_cls", loss="bce", suffix="_bce", seeds=SEEDS, save_prefix="pred_bertbce")
JOBS += ft("roberta_cls", loss="bce", suffix="_bce", seeds=SEEDS, save_prefix="pred_robertabce")
# ---------- PHASE 2c: 种子集成对比（CLAIMARC vs 各基线，公平地都做集成）----------
JOBS += [("ensemble_main", [PY, "-m", "models.ensemble", "--spec",
          f"CLAIMARC={EMB}/emb_clarc_s*.pt", f"CLAIMARC_RKC:rkc={EMB}/emb_clarc_s*.pt",
          f"BERT={EMB}/pred_bert_s*.pt", f"BERT_BCE={EMB}/pred_bertbce_s*.pt",
          f"RoBERTa={EMB}/pred_roberta_s*.pt", f"RoBERTa_BCE={EMB}/pred_robertabce_s*.pt",
          f"ESIM={EMB}/pred_esim_s*.pt"])]
# ---------- PHASE 3: 核心消融（以新 canonical 为中心，seed0）----------
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
# ---------- PHASE 6: RACL/超参消融（新 canonical=λ0.5,τ0.07,K(3,5),lora16）----------
JOBS += claimarc("abl_lambda_0.3", ["--lambda_cl", "0.3"])
JOBS += claimarc("abl_lambda_1.0", ["--lambda_cl", "1.0"])
JOBS += claimarc("abl_loss_bce", ["--loss", "bce"])
JOBS += claimarc("abl_loss_focal", ["--loss", "focal"])
JOBS += claimarc("abl_tau_0.05", ["--tau", "0.05"])
JOBS += claimarc("abl_tau_0.10", ["--tau", "0.10"])
JOBS += claimarc("abl_K_1_1", ["--Kp", "1", "--Kn", "1"])
JOBS += claimarc("abl_lora_8", ["--lora_rank", "8"])
JOBS += claimarc("abl_lora_32", ["--lora_rank", "32"])
# ---------- PHASE 7: 跨域 留一品类（新配置）----------
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
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        p = subprocess.Popen(cmd, cwd=src_dir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
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
    print(f"[run_final] {len(JOBS)} jobs, {len(done)} already done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print(f"[skip] {job}", flush=True)
            continue
        run_job(job, cmd)
    print("\n######## RUN_FINAL COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
