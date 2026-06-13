"""跑全套实验并汇总结果表：基线 + CLAIMARC 全量 + 消融。

每个配置作为独立子进程跑（GPU 显存隔离），解析其 stdout 里的 `RESULT {json}`。
产出 data/final/results.json 与 results_table.md。

用法：python -m models.run_experiments --dataset ../data/final/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RESULT_RE = re.compile(r"RESULT\s+(\{.*\})")


def run_cmd(cmd: list[str]) -> list[dict]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    out = []
    for line in proc.stdout:  # 实时流式，避免缓冲丢失
        line = line.rstrip("\n")
        print(line, flush=True)
        m = RESULT_RE.search(line)
        if m:
            try:
                out.append(json.loads(m.group(1)))
            except Exception:
                pass
    proc.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/final/dataset.jsonl")
    ap.add_argument("--quick", action="store_true", help="缩短 epoch 做冒烟")
    args = ap.parse_args()
    py = sys.executable
    we = 1 if args.quick else 2
    cl = 1 if args.quick else 3
    bs = "16"

    def t(tag, *extra):
        return [py, "-m", "models.train", "--dataset", args.dataset, "--tag", tag,
                "--warmup", str(we), "--cl_epochs", str(cl), "--bs", bs, *extra]

    configs = [
        ("baselines", [py, "-m", "models.baselines", "--dataset", args.dataset]),
        ("claimarc_full", t("claimarc_full")),
        ("abl_no_cl", t("abl_no_cl", "--no_cl")),
        ("abl_no_fusion", t("abl_no_fusion", "--no_fusion")),
        ("abl_no_lora", t("abl_no_lora", "--no_lora")),
    ]
    all_res = []
    for _, cmd in configs:
        all_res += run_cmd(cmd)

    out_json = Path(args.dataset).parent / "results.json"
    out_md = Path(args.dataset).parent / "results_table.md"
    out_json.write_text(json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["tag", "f1", "precision", "recall", "auc", "ap", "f1_rkc", "f1_selective", "abstain_rate", "thr", "pos_test", "n_test"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in all_res:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n-> {out_json}\n-> {out_md}")


if __name__ == "__main__":
    main()
