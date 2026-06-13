"""CLAIMARC-v2 域内架构/容量扫描：在保留 RACL 核心（双流+对比+检索）前提下，
逐一对比不同编码器容量/融合配置，按"3 种子标定集成"的 AP/AUROC/mF1 与基线对照，
目标是让 CLAIMARC 明显最优。可断点续跑。

每个 config 训练 N 种子 → 保存嵌入 → fusion_eval 与既有基线对照 → 汇总。

用法：python -m models.run_arch --seeds 0 1 2 --dataset ../data/final/dataset_verify_faithful.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PY = sys.executable

# 各配置：保留 canonical RACL 核心，仅调编码器容量/融合（命令行增量参数）
CONFIGS = {
    "lora16":   "--lora_rank 16 --enc_train lora",
    "lora32":   "--lora_rank 32 --enc_train lora",
    "lora48":   "--lora_rank 48 --enc_train lora",
    "lora64":   "--lora_rank 64 --enc_train lora",
    "lora32cl1":"--lora_rank 32 --enc_train lora --lambda_cl 1.0",
    "lora32f3": "--lora_rank 32 --enc_train lora --n_fusion 3",
    "full1e5":  "--enc_train full --lr 1e-5",
    "top10":    "--enc_train topk --unfreeze_top 10 --lr 1e-5",
    "fus3h16":  "--lora_rank 16 --enc_train lora --n_fusion 3 --heads 16",
    # 骨干对比：在 CLAIMARC 双流+RACL 框架内换骨干并全量微调（回应"哪个基座最优"）
    "robFull":  "--backbone roberta --enc_train full --lr 2e-5",
    "robLora":  "--backbone roberta --lora_rank 32 --enc_train lora --lr 2e-5",
    "bgeFull":  "--backbone bge --enc_train full --lr 1e-5",
    # 检索表征质量提升：更强对比（更高 λ、更尖 τ、更多负例）→ 抬升 kNN/blend 的 AUPRC
    "rich":     "--lora_rank 48 --enc_train lora --lambda_cl 1.0 --tau 0.05 --Kp 5 --Kn 10",
    "rich2":    "--lora_rank 48 --enc_train lora --lambda_cl 1.5 --tau 0.05 --Kp 5 --Kn 10 --cl_epochs 8",
    "rich32":   "--lora_rank 32 --enc_train lora --lambda_cl 1.0 --tau 0.05 --Kp 5 --Kn 10",
    # §4.4.5 消融（基于锁定的 lora48 canonical，逐一移除单项）
    "abl_nocl":   "--lora_rank 48 --enc_train lora --no_cl",
    "abl_gneg":   "--lora_rank 48 --enc_train lora --global_neg",
    "abl_nofus":  "--lora_rank 48 --enc_train lora --no_fusion",
    "abl_bertbb": "--lora_rank 48 --enc_train lora --backbone bert",
}


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--outdir", default="../data/final/v2")
    args = ap.parse_args()
    DS = args.dataset
    def seeds_glob(kind):
        ps = [f"{args.outdir}/{kind}_s{s}.pt" for s in args.seeds
              if os.path.exists(f"{args.outdir}/{kind}_s{s}.pt")]
        return ",".join(ps)
    base = ";".join(f"{k}={seeds_glob(k)}" for k in
                    ("roberta_cls", "bert_cls", "bert_nli", "esim") if seeds_glob(k))

    summary = {}
    for cfg in args.configs:
        extra = CONFIGS[cfg]
        paths = []
        for s in args.seeds:
            out = f"{args.outdir}/arch_{cfg}_s{s}.pt"
            paths.append(out)
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True); continue
            sh(f"{PY} -m models.train --dataset {DS} --seed {s} --tag arch_{cfg} "
               f"--warmup 3 --cl_epochs 6 --tau 0.07 --lambda_cl 0.5 --loss asl "
               f"--gamma_neg 4 --fusion_dropout 0.2 {extra} --save_emb {out}")
        cmp_out = f"{args.outdir}/cmp_{cfg}.json"
        sh(f"{PY} -m models.fusion_eval --dataset {DS} --cm {' '.join(paths)} "
           f"--baseline \"{base}\" --out {cmp_out}")
        try:
            summary[cfg] = json.load(open(cmp_out))["rows"].get("CLAIMARC_v2")
        except Exception as e:
            summary[cfg] = {"err": str(e)}
    print("\n######## ARCH SWEEP SUMMARY (CLAIMARC_v2 overall) ########", flush=True)
    for cfg, r in summary.items():
        ov = (r or {}).get("overall", {}) if isinstance(r, dict) else {}
        print(f"  {cfg:10s} AP={ov.get('auprc')} AUROC={ov.get('auroc')} "
              f"mF1={ov.get('macro_f1')} wF1={ov.get('wF1')}", flush=True)
    json.dump(summary, open(f"{args.outdir}/arch_summary.json", "w"),
              ensure_ascii=False, indent=2)
    print("######## ARCH SWEEP DONE ########", flush=True)


if __name__ == "__main__":
    main()
