"""LLM adjudication of claim-evidence states without using consumer labels.

The adjudicator is a data-quality instrument. It never receives y, c,
label_audit, reviews, or split. It only sees the attribute, grounded livestream
claim text, and product evidence. The output can later be cross-tabulated with
consumer-derived weak labels to select clean/silver samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.llm import chat_json, run_many


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done[str(obj.get("pair_id"))] = obj
            except Exception:
                continue
    return done


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    segs = claim.get("segments") or []
    txt = "\n".join(str(s.get("text", "") or "").strip() for s in segs if s.get("text"))
    return txt or str(claim.get("passage", "") or "").strip()


def evidence_text(rec: dict[str, Any], cap: int = 2200) -> str:
    parts: list[str] = []
    for label, key, field in (
        ("PARAM", "evidence_params", "raw_text"),
        ("OCR", "evidence_ocr", "raw_text"),
        ("VLM", "evidence_vlm", "raw_quote"),
    ):
        for it in rec.get(key, []) or []:
            txt = str(it.get(field, "") or "").strip()
            if txt:
                parts.append(f"[{label}] {txt}")
    text = "\n".join(parts)
    return text[:cap]


def trim(text: str, cap: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap else text[:cap] + "..."


def make_prompt(rec: dict[str, Any]) -> str:
    claim = trim(claim_text(rec), 1400) or "【无明确主播属性话术】"
    evidence = trim(evidence_text(rec), 2200) or "【无商品参数/OCR/VLM证据】"
    return f"""你是直播电商 claim-evidence 数据质量裁决员。请只根据给定的属性、主播话术和商品证据判断证据状态。

严禁使用消费者评论、弱标签、销量、外部搜索或常识扩写。若证据不足，请说不足，不要猜测。

商品类目：{rec.get("category", "")}
属性：{rec.get("attribute_name", "")}

主播话术：
{claim}

商品证据：
{evidence}

请输出严格 JSON：
{{
  "claim_quality": "clear|mixed|garbled|no_claim",
  "evidence_state": "supported|contradicted|insufficient|not_verifiable",
  "misleading_risk": "none|low|medium|high",
  "key_claim": "最关键的主播宣称短语，原文摘录，不超过40字",
  "key_evidence": "最关键的商品证据摘录；无则写空字符串",
  "rationale": "一句话说明为什么是该 evidence_state，不超过80字",
  "flags": ["可选：garbled_srt|mixed_product|missing_source|comparative_claim|medical_or_safety_claim|numeric_claim|subjective_claim"]
}}"""


def adjudicate_one(rec: dict[str, Any], model: str, max_tokens: int) -> dict[str, Any]:
    obj = chat_json(
        make_prompt(rec),
        system="你是严谨的数据质量裁决员，只输出 JSON。",
        model=model,
        temperature=0.0,
        namespace="claim_evidence_adjudication",
        max_tokens=max_tokens,
    )
    out = {
        "pair_id": rec.get("pair_id"),
        "product_id": rec.get("product_id"),
        "attribute_id": rec.get("attribute_id"),
        "category": rec.get("category"),
        "attribute_name": rec.get("attribute_name"),
        "claim_quality": str(obj.get("claim_quality", ""))[:32],
        "evidence_state": str(obj.get("evidence_state", ""))[:32],
        "misleading_risk": str(obj.get("misleading_risk", ""))[:32],
        "key_claim": str(obj.get("key_claim", ""))[:80],
        "key_evidence": str(obj.get("key_evidence", ""))[:120],
        "rationale": str(obj.get("rationale", ""))[:180],
        "flags": [str(x)[:50] for x in (obj.get("flags", []) or [])[:8]],
        "model": model,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_hq_silver_v1.jsonl")
    ap.add_argument("--out", default="data/final/claim_evidence_adjudication_hq_silver_v1.jsonl")
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=350)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if str(r.get("pair_id")) not in done]

    def fn(rec: dict[str, Any]) -> dict[str, Any]:
        try:
            return adjudicate_one(rec, args.model, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            return {"pair_id": rec.get("pair_id"), "__error__": repr(exc)[:300], "model": args.model}

    results = run_many(todo, fn, concurrency=args.concurrency, desc="llm_adjudicate")
    with open(out_path, "a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[llm_claim_evidence_adjudicate] new={len(results)} done={len(done)+len(results)} out={out_path}")


if __name__ == "__main__":
    main()
