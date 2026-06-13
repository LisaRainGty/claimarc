"""CLAIMARC-v2 实验编排：训练 CLAIMARC(canonical) 与基线多种子 → 保存嵌入/预测 →
离线 ARF 融合对比。可断点续跑（已存在的 .pt 跳过）。

用法：
  python -m models.run_v2 --seeds 0 1 2 3 4 --dataset ../data/final/dataset_verify_faithful.jsonl
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PY = sys.executable


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--baselines", nargs="+",
                    default=["roberta_cls", "bert_cls", "bert_nli", "esim"])
    ap.add_argument("--outdir", default="../data/final/v2")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    DS = args.dataset

    # ---- CLAIMARC canonical：lora16 / τ0.07 / λ0.5 / ASL γ4 / n_fusion2 ----
    cm_paths = []
    for s in args.seeds:
        out = f"{args.outdir}/cm_s{s}.pt"
        cm_paths.append(out)
        if os.path.exists(out):
            print(f"[skip] {out}", flush=True); continue
        sh(f"{PY} -m models.train --dataset {DS} --seed {s} --tag cm "
           f"--warmup 3 --cl_epochs 6 --lora_rank 16 --tau 0.07 --lambda_cl 0.5 "
           f"--loss asl --gamma_neg 4 --n_fusion 2 --fusion_dropout 0.2 "
           f"--enc_train lora --save_emb {out}")

    # ---- 基线 ----
    base_specs = []
    for kind in args.baselines:
        paths = []
        for s in args.seeds:
            out = f"{args.outdir}/{kind}_s{s}.pt"
            paths.append(out)
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True); continue
            sh(f"{PY} -m models.baselines_ft --dataset {DS} --kind {kind} --seed {s} "
               f"--epochs 4 --loss asl --gamma_neg 4 --save_pred {out}")
        base_specs.append(f"{kind}=" + ",".join(paths))

    # ---- 离线 ARF 融合对比 ----
    cm_arg = " ".join(cm_paths)
    base_arg = ";".join(base_specs)
    sh(f"{PY} -m models.fusion_eval --dataset {DS} --cm {cm_arg} "
       f"--baseline \"{base_arg}\" --boundary --k {args.k} "
       f"--out {args.outdir}/v2_compare.json")
    print("\n######## RUN_V2 DONE ########", flush=True)


if __name__ == "__main__":
    main()
