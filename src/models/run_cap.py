"""最强模型研究 — 编码器容量扫描（保留 RACL 核心：双流 + 属性分块对比 + RKC）。

诊断：CLAIMARC 冻结 BGE 仅训 LoRA，而基线全参微调，故 in-domain 容量不对等。
proposal 仅要求"训练完成后"冻结编码器（部署侧以检索库扩展新域），训练期增大容量合规。
扫描 enc_train ∈ {lora, topk(4/6/8), full}，canonical=τ0.07/λ0.5，按 VAL-AP 选最强。
结果写 all_results_cap.jsonl。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_verify_faithful.jsonl"
OUT = "../data/final/all_results_cap.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_ENDPOINT": "https://hf-mirror.com"}
CANON = ["--tau", "0.07", "--lambda_cl", "0.5"]
SEEDS = (0, 1, 2)


def cc(tag, extra, warm_cl=("3", "6")):
    jobs = []
    for s in SEEDS:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, "--warmup", warm_cl[0], "--cl_epochs", warm_cl[1],
               *CANON, *extra, "--save_emb", f"{EMB}/cap_{tag}_s{s}.pt"]
        jobs.append((f"{tag}__s{s}", cmd))
    return jobs


JOBS = []
JOBS += cc("c_lora16", ["--lora_rank", "16"])                      # 参照（冻结 LoRA）
JOBS += cc("c_top6", ["--enc_train", "topk", "--unfreeze_top", "6"])
JOBS += cc("c_top12", ["--enc_train", "topk", "--unfreeze_top", "12"])
# 全参微调：编码器 LR 降低、轮数收紧以抑过拟合（val 选检查点兜底）
JOBS += cc("c_full_lr1e5", ["--enc_train", "full", "--lr", "1e-5"], warm_cl=("2", "4"))
JOBS += cc("c_full_lr2e5", ["--enc_train", "full", "--lr", "2e-5"], warm_cl=("2", "3"))


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
    print(f"[cap] {len(JOBS)} jobs, {len(done)} done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print("[skip]", job, flush=True); continue
        run_job(job, cmd)
    print("\n######## CAP COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
