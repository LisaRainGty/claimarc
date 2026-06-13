"""干净数据(dataset_verify.jsonl)上的超参再调（按 VAL 选择，避免测试集泄漏）。
噪声数据上 λ=1.0 的对比项在干净数据上压低 val-AP；此处扫 λ/τ/rank，找 val 最优。
每配置 3 seed，记录 val 与 test，但选择只看 val 集成。结果写 all_results_tune.jsonl。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_verify_faithful.jsonl"
OUT = "../data/final/all_results_tune.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_ENDPOINT": "https://hf-mirror.com"}
FULL = ["--warmup", "3", "--cl_epochs", "6"]
SEEDS = (0, 1, 2)


def cc(tag, extra, seeds=SEEDS):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *extra,
               "--save_emb", f"{EMB}/tune_{tag}_s{s}.pt"]
        jobs.append((f"{tag}__s{s}", cmd))
    return jobs


JOBS = []
# 基准（noisy 最优配置）作对照
JOBS += cc("t_l8_t10_lam10", ["--lora_rank", "8", "--tau", "0.10", "--lambda_cl", "1.0"])
# 降低 λ（CL 在干净数据上偏强）
JOBS += cc("t_l8_t10_lam05", ["--lora_rank", "8", "--tau", "0.10", "--lambda_cl", "0.5"])
JOBS += cc("t_l8_t10_lam03", ["--lora_rank", "8", "--tau", "0.10", "--lambda_cl", "0.3"])
JOBS += cc("t_l8_t05_lam05", ["--lora_rank", "8", "--tau", "0.05", "--lambda_cl", "0.5"])
# 容量稍大 + 中等 λ
JOBS += cc("t_l16_t07_lam05", ["--lora_rank", "16", "--tau", "0.07", "--lambda_cl", "0.5"])
# 无 CL 对照（看 CL 是否还有正贡献）
JOBS += cc("t_l8_nocl", ["--lora_rank", "8", "--no_cl"])


def done_tags():
    if not os.path.exists(OUT):
        return set()
    t = set()
    for l in open(OUT):
        try:
            t.add(json.loads(l)["_job"])
        except Exception:
            pass
    return t


def log_result(job, payload):
    payload["_job"] = job
    payload["_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_job(job, cmd):
    print(f"\n{'='*70}\n[RUN] {job}\n{'='*70}", flush=True)
    t0 = time.time(); got = False
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        p = subprocess.Popen(cmd, cwd=src, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            if line.startswith("RESULT "):
                try:
                    log_result(job, json.loads(line[len("RESULT "):])); got = True
                except Exception as e:
                    print("[parse-err]", e, flush=True)
        p.wait()
    except Exception as e:
        print("[JOB-ERR]", job, e, flush=True)
    if not got:
        log_result(job, {"error": "no_result"})
    print(f"[DONE] {job} in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    done = done_tags()
    print(f"[tune] {len(JOBS)} jobs, {len(done)} done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print("[skip]", job, flush=True); continue
        run_job(job, cmd)
    print("\n######## TUNE COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
