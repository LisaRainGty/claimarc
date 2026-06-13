"""去噪可核验数据集(dataset_verify.jsonl)上的决定性对比：
CLAIMARC(改进配置) vs 最强基线，3 seed，判断 (1) 任务是否可学习(AUROC/AP 上升)
(2) CLAIMARC 是否明显领先。结果写 all_results_verifytest.jsonl。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = "/root/miniconda3/envs/myconda/bin/python"
DS = "../data/final/dataset_verify.jsonl"
OUT = "../data/final/all_results_verifytest.jsonl"
EMB = "../data/final"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_ENDPOINT": "https://hf-mirror.com"}
FULL = ["--warmup", "3", "--cl_epochs", "6"]
CANON = ["--lora_rank", "8", "--tau", "0.10", "--lambda_cl", "1.0"]
SEEDS = (0, 1, 2)


def claimarc(tag, extra=None, seeds=SEEDS, save_prefix=None):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *(extra or [])]
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
JOBS += claimarc("vt_claimarc", seeds=SEEDS, save_prefix="vemb_clarc")
JOBS += [("vt_BGE_LR", [PY, "-m", "models.baselines", "--dataset", DS])]
JOBS += ft("bert_cls", seeds=SEEDS, loss="bce", suffix="_bce", save_prefix="vpred_bertbce")
JOBS += ft("roberta_cls", seeds=SEEDS, loss="bce", suffix="_bce", save_prefix="vpred_robertabce")
JOBS += ft("bert_cls", seeds=SEEDS, loss="asl", suffix="_asl", save_prefix="vpred_bert")
JOBS += ft("esim", seeds=SEEDS, loss="asl", save_prefix="vpred_esim")
JOBS += [("vt_ensemble", [PY, "-m", "models.ensemble", "--spec",
          f"CLAIMARC={EMB}/vemb_clarc_s*.pt", f"CLAIMARC_RKC:rkc={EMB}/vemb_clarc_s*.pt",
          f"BERT_BCE={EMB}/vpred_bertbce_s*.pt", f"RoBERTa_BCE={EMB}/vpred_robertabce_s*.pt",
          f"BERT={EMB}/vpred_bert_s*.pt", f"ESIM={EMB}/vpred_esim_s*.pt"])]


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
    print(f"[verifytest] {len(JOBS)} jobs, {len(done)} done", flush=True)
    for job, cmd in JOBS:
        if job in done:
            print("[skip]", job, flush=True); continue
        run_job(job, cmd)
    print("\n######## VERIFYTEST COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
