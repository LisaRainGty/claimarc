"""改进配置多种子验证（防止 seed0 噪声误导）。

消融(seed0)提示 canonical(lora16/tau0.05/lambda0.5) 过拟合 406 正例：
lora_8 / tau_0.10 / lambda_1.0 / xattn_e2c 的 seed0 AP/AUROC 明显更高。
本脚本在 seeds 0,1,2 上验证若干候选配置，找出稳健优于 BERT-BCE 的配置。
结果写入 all_results_verify.jsonl（独立文件，避免污染主结果）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_claim.jsonl"
OUT = "../data/final/all_results_verify.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_ENDPOINT": "https://hf-mirror.com"}
FULL = ["--warmup", "3", "--cl_epochs", "6"]
SEEDS = (0, 1, 2)


def claimarc(tag, extra=None, seeds=SEEDS, save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *(extra or [])]
        if save_prefix:
            cmd += ["--save_emb", f"{EMB}/{save_prefix}_s{s}.pt"]
        jobs.append((f"{tag}__s{s}", cmd))
    return jobs


def ft(kind, seeds, loss, suffix="", save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.baselines_ft", "--dataset", DS,
               "--kind", kind, "--seed", str(s), "--loss", loss]
        if save_prefix:
            cmd += ["--save_pred", f"{EMB}/{save_prefix}_s{s}.pt"]
        jobs.append((f"{kind}{suffix}__s{s}", cmd))
    return jobs


JOBS = []
# 候选改进配置（降容量 / 强正则）
JOBS += claimarc("imp_l8", ["--lora_rank", "8"])
JOBS += claimarc("imp_l8_t10", ["--lora_rank", "8", "--tau", "0.10"])
JOBS += claimarc("imp_l8_t10_lam1", ["--lora_rank", "8", "--tau", "0.10", "--lambda_cl", "1.0"])
JOBS += claimarc("imp_t10_lam1", ["--tau", "0.10", "--lambda_cl", "1.0"])
JOBS += claimarc("imp_l8_e2c", ["--lora_rank", "8", "--xattn_dir", "e2c"])
# BERT-BCE 补到 5 种子（已有 0,1,2 在主结果；此处补 3,4 以便公平 5 种子对比）
JOBS += ft("bert_cls", seeds=(3, 4), loss="bce", suffix="_bce", save_prefix="pred_bertbce")


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
            if line.startswith("RESULT "):
                try:
                    log_result(job, json.loads(line[len("RESULT "):]))
                    got = True
                except Exception as e:
                    print(f"[parse-err] {e}", flush=True)
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job}: {e}", flush=True)
    if not got:
        log_result(job, {"error": "no_result", "cmd": " ".join(cmd)})
    print(f"[DONE] {job} in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    done = done_tags()
    print(f"[verify] {len(JOBS)} jobs, {len(done)} done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print(f"[skip] {job}", flush=True); continue
        run_job(job, cmd)
    print("\n######## VERIFY COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
