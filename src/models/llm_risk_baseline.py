"""Direct LLM risk baseline for CLAIMARC.

This script scores each product-attribute pair with an OpenAI-compatible LLM
without using weak labels, consumer comments, or label_audit fields. It then
evaluates the cached scores under the same grouped-CV threshold protocol used
by CLAIMARC.

Typical usage:
  source env.sh
  export MATPOOL_API_KEY=...
  PYTHONPATH=src python -m models.llm_risk_baseline \
      --dataset data/final/dataset_verify_faithful_args.jsonl \
      --pred_out data/final/cleancl/llm_qwen_flash_args.jsonl \
      --eval_out data/final/cleancl/llm_qwen_flash_args_fs1.json \
      --fold_seed 1 \
      --bge_tmp data/final/cleancl/cv_tmp_args_small_e3_c10_det_fairbase_fs1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common.llm import chat_json
from models.cv_eval import make_folds, val_carve
from models.data import load_split
from models.fusion_eval import best_thr, macro, paired_bootstrap


def trim(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "..."


def claim_text(r: dict) -> str:
    claim = r.get("claim", {}) or {}
    passage = (claim.get("passage") or "").strip()
    if passage:
        return passage
    segs = claim.get("segments", []) or []
    return "\n".join((s.get("text", "") or "").strip() for s in segs if s.get("text"))


def evidence_text(r: dict, *, include_arguments: bool = True) -> str:
    parts: list[str] = []
    for label, key, field in (
        ("PARAM", "evidence_params", "raw_text"),
        ("OCR", "evidence_ocr", "raw_text"),
        ("VLM", "evidence_vlm", "raw_quote"),
    ):
        for it in r.get(key, []) or []:
            txt = trim(str(it.get(field, "") or ""), 360)
            conf = it.get("confidence")
            if txt:
                suffix = f" confidence={conf}" if conf is not None else ""
                parts.append(f"[{label}{suffix}] {txt}")
    if include_arguments:
        args = r.get("arguments") or {}
        for label, key in (
            ("ARG_SUPPORT", "supporting_argument"),
            ("ARG_REFUTE", "refuting_argument"),
            ("ARG_GAP", "evidence_gap"),
        ):
            txt = trim(str(args.get(key, "") or ""), 420)
            if txt:
                parts.append(f"[{label}] {txt}")
    return "\n".join(parts)


def make_prompt(r: dict, *, include_arguments: bool = True, policy: str = "broad") -> str:
    claim = trim(claim_text(r), 1200) or "(无明确主播话术)"
    evidence = trim(evidence_text(r, include_arguments=include_arguments), 2400)
    if not evidence:
        evidence = "(无商品参数、OCR、VLM 或论证证据)"
    if policy == "conservative":
        scoring = """评分规则必须保守：
- 0.00-0.20：证据明确支持宣传，或宣传只是无法核验的主观感受且无明显客观承诺。
- 0.25-0.45：只有一般性证据不足，没有明确矛盾，也没有具体可量化承诺。
- 0.50-0.65：关键客观/量化宣传缺少覆盖证据，可能误导但尚无明确反证。
- 0.70-1.00：存在明确矛盾、规格不一致、关键承诺被证据削弱，或高价值客观承诺完全无证据覆盖。
不要因为“证据不足”四个字就一律给高分。"""
    else:
        scoring = """评分规则：
- 分数越高，表示虚假/误导风险越高。
- 明确矛盾应给高分；关键证据缺失也可以给中高分；证据支持则给低分。"""
    return f"""你是直播电商宣传风险核验员。请只根据给出的主播话术和商品事实证据，判断该商品-属性 pair 是否存在“主播宣传与可核验证据不一致、证据无法覆盖关键宣传、或消费者可能据此感到被误导”的风险。

禁止使用消费者评论、弱标签、外部搜索或常识补全；证据缺失时可以提高风险，但要区分“明确矛盾”和“证据不足”。

{scoring}

商品类目：{r.get("category", "")}
属性：{r.get("attribute_name", "")}

主播话术：
{claim}

商品事实证据/结构化论证：
{evidence}

请输出严格 JSON，不要 Markdown。schema:
{{
  "risk_score": 0到1之间的小数，越高表示虚假/误导风险越高,
  "decision": 0或1，1表示有风险，0表示暂未发现风险,
  "confidence": 0到1之间的小数,
  "evidence_state": "contradiction|insufficient|supported|unclear",
  "rationale": "一句话说明主要依据"
}}"""


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid = r.get("pair_id")
            if pid:
                done[pid] = r
    return done


def clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return float(min(1.0, max(0.0, v)))


def score_records(args: argparse.Namespace) -> None:
    if not os.environ.get("MATPOOL_API_KEY"):
        raise RuntimeError("MATPOOL_API_KEY is not set in the process environment.")
    recs = load_jsonl(args.dataset)
    if args.stride > 1:
        if args.offset < 0 or args.offset >= args.stride:
            raise ValueError("--offset must be in [0, stride)")
        recs = [r for i, r in enumerate(recs) if i % args.stride == args.offset]
    if args.limit > 0:
        recs = recs[:args.limit]
    out_path = Path(args.pred_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    include_arguments = not args.no_arguments

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(recs, 1):
            pid = r.get("pair_id", "")
            if pid in done:
                continue
            row = {
                "pair_id": pid,
                "model": args.model,
                "include_arguments": include_arguments,
                "policy": args.policy,
            }
            try:
                obj = chat_json(
                    make_prompt(r, include_arguments=include_arguments, policy=args.policy),
                    system="你是严谨的中文事实核验助手，只输出 JSON。",
                    model=args.model,
                    temperature=args.temperature,
                    namespace=args.namespace,
                    max_tokens=args.max_tokens,
                )
                row.update({
                    "risk_score": clamp01(obj.get("risk_score")),
                    "decision": int(obj.get("decision", 1 if clamp01(obj.get("risk_score")) >= 0.5 else 0)),
                    "confidence": clamp01(obj.get("confidence"), default=0.5),
                    "evidence_state": str(obj.get("evidence_state", ""))[:40],
                    "rationale": str(obj.get("rationale", ""))[:500],
                })
            except Exception as e:  # noqa: BLE001
                row.update({"risk_score": None, "decision": None, "__error__": repr(e)[:400]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % args.progress_every == 0:
                print(f"[llm_risk] written={n_new} seen={i}/{len(recs)} out={out_path}", flush=True)
    print(f"[llm_risk] scoring done new={n_new} out={out_path}", flush=True)


def load_scores(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for r in load_jsonl(path):
        pid = r.get("pair_id")
        score = r.get("risk_score")
        if pid and score is not None:
            scores[pid] = clamp01(score)
    return scores


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / max(1, len(x))


def metric_row(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> dict:
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "macro_f1": round(macro(y, yhat), 4),
        "wF1": round(macro(y, yhat, w=np.clip(c, 0.05, None)), 4),
        "n": int(len(y)),
    }


def load_bge_fold(path: str | Path) -> dict:
    import torch  # Imported lazily so scoring/eval without BGE does not require torch.

    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(args: argparse.Namespace) -> dict:
    scores = load_scores(args.pred_out)
    recs_by = load_split(args.dataset)
    recs = recs_by["train"] + recs_by["val"] + recs_by["test"]
    missing = [r.get("pair_id", "") for r in recs if r.get("pair_id", "") not in scores]
    if missing and args.require_complete:
        raise RuntimeError(f"Missing {len(missing)} LLM scores; first missing={missing[:3]}")

    folds, _, g_all = make_folds(recs, args.folds, seed=args.fold_seed)
    n = len(recs)
    y_all = np.array([int(r["y"]) for r in recs], float)
    c_all = np.array([float(r.get("c", 0.05)) for r in recs], float)
    p_llm_all = np.array([scores.get(r.get("pair_id", ""), np.nan) for r in recs], float)
    methods = ["llm_fixed05", "llm_valthr"]
    if args.bge_tmp:
        methods += ["bge_lr", "rankavg_llm+bge_lr"]
    oof = {m: {"p": np.full(n, np.nan), "yhat": np.full(n, np.nan)} for m in methods}
    fold_meta: list[dict] = []

    for fi, (tr_full, te_idx) in enumerate(folds):
        _, va_idx = val_carve(tr_full, recs, g_all, seed=args.fold_seed * 100 + fi)
        pv = p_llm_all[va_idx]
        pt = p_llm_all[te_idx]
        okv = ~np.isnan(pv)
        okt = ~np.isnan(pt)
        if okv.sum() == 0 or okt.sum() == 0:
            continue
        yv = y_all[va_idx][okv]
        pv_ok = pv[okv]
        thr = best_thr(yv, pv_ok)
        oof["llm_fixed05"]["p"][te_idx[okt]] = pt[okt]
        oof["llm_fixed05"]["yhat"][te_idx[okt]] = (pt[okt] >= 0.5).astype(int)
        oof["llm_valthr"]["p"][te_idx[okt]] = pt[okt]
        oof["llm_valthr"]["yhat"][te_idx[okt]] = (pt[okt] >= thr).astype(int)
        meta = {"fold": fi, "n_val_scored": int(okv.sum()), "n_test_scored": int(okt.sum()),
                "llm_thr": round(float(thr), 3),
                "val_pos": round(float(yv.mean()), 4),
                "test_pos": round(float(y_all[te_idx][okt].mean()), 4)}

        if args.bge_tmp:
            bge = load_bge_fold(Path(args.bge_tmp) / f"cv_bge_lr_f{fi}.pt")
            bge_v = np.asarray(bge["val"]["p"], float)
            bge_t = np.asarray(bge["test"]["p"], float)
            bthr = best_thr(np.asarray(bge["val"]["y"], float), bge_v)
            oof["bge_lr"]["p"][te_idx] = bge_t
            oof["bge_lr"]["yhat"][te_idx] = (bge_t >= bthr).astype(int)
            rv = 0.5 * rank01(pv) + 0.5 * rank01(bge_v)
            rt = 0.5 * rank01(pt) + 0.5 * rank01(bge_t)
            rthr = best_thr(np.asarray(bge["val"]["y"], float), rv)
            oof["rankavg_llm+bge_lr"]["p"][te_idx] = rt
            oof["rankavg_llm+bge_lr"]["yhat"][te_idx] = (rt >= rthr).astype(int)
            meta["bge_thr"] = round(float(bthr), 3)
            meta["rank_llm_bge_thr"] = round(float(rthr), 3)
        fold_meta.append(meta)

    rows = {}
    for name in methods:
        ok = ~np.isnan(oof[name]["p"])
        rows[name] = metric_row(y_all[ok], oof[name]["p"][ok], oof[name]["yhat"][ok], c_all[ok])
        print(f"{name:20s} AP={rows[name]['auprc']} AUROC={rows[name]['auroc']} "
              f"mF1={rows[name]['macro_f1']} wF1={rows[name]['wF1']} n={rows[name]['n']}",
              flush=True)

    sig = {}
    if args.bge_tmp:
        for name in ("llm_valthr", "rankavg_llm+bge_lr"):
            ok = ~np.isnan(oof[name]["p"]) & ~np.isnan(oof["bge_lr"]["p"])
            sig[f"{name}_vs_bge_lr"] = paired_bootstrap(
                y_all[ok], oof[name]["p"][ok], oof["bge_lr"]["p"][ok], c_all[ok],
                n_boot=args.n_boot,
            )
            s = sig[f"{name}_vs_bge_lr"]
            print(f"{name} vs bge_lr: dAP={s['dAP']['mean_delta']:+.4f}(p={s['dAP']['p_a_gt_b']}) "
                  f"dAUROC={s['dAUROC']['mean_delta']:+.4f}(p={s['dAUROC']['p_a_gt_b']}) "
                  f"dMF1={s['dMacroF1']['mean_delta']:+.4f}(p={s['dMacroF1']['p_a_gt_b']})",
                  flush=True)

    out = {
        "dataset": args.dataset,
        "pred_out": args.pred_out,
        "model": args.model,
        "fold_seed": args.fold_seed,
        "rows": rows,
        "fold_meta": fold_meta,
        "significance": sig,
    }
    if args.eval_out:
        Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.eval_out, "w"), ensure_ascii=False, indent=2)
        print(f"[llm_risk] eval -> {args.eval_out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--pred_out", default="data/final/cleancl/llm_risk_preds.jsonl")
    ap.add_argument("--eval_out", default="")
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--namespace", default="llm_risk_baseline")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--policy", choices=["broad", "conservative"], default="broad")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--no_arguments", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=1)
    ap.add_argument("--bge_tmp", default="")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--require_complete", action="store_true")
    args = ap.parse_args()

    if not args.eval_only:
        score_records(args)
    if not args.skip_eval:
        evaluate(args)


if __name__ == "__main__":
    main()
