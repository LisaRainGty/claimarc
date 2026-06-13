"""LLM adjudication for regeneration-manifest pairs.

This is a targeted data-repair tool. It reads the regeneration manifest,
product-scope training dataset, and repaired pair records, then asks an LLM to
separate three signals without seeing y/c/split:

1. whether the livestream claim is usable;
2. whether product evidence supports or contradicts the claim;
3. whether consumer comments are actually responding to that claim.

The output is meant to guide regeneration and relabeling. It should not be used
as a hidden oracle unless reported as an LLM-assisted data curation step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.llm import chat_json, run_many


VALID_ACTIONS = {
    "keep_clean",
    "keep_silver",
    "drop_bad_claim",
    "drop_bad_attribute",
    "rerun_claim_extraction",
    "rerun_product_evidence",
    "schema_repair_review",
    "needs_human_review",
}

VALID_LABEL_RECOMMENDATIONS = {"positive_risk", "negative_clean", "drop_or_regenerate"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[str(obj.get("pair_id"))] = obj
            except Exception:
                continue
    return out


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    passage = str(claim.get("passage", "") or "").strip()
    if passage:
        return passage
    segs = claim.get("segments") or []
    return "\n".join(str(s.get("text", "") or "").strip() for s in segs if s.get("text"))


def evidence_text(rec: dict[str, Any], cap: int = 2400) -> str:
    parts: list[str] = []
    for label, key, field in (
        ("PARAM", "evidence_params", "raw_text"),
        ("OCR", "evidence_ocr", "raw_text"),
        ("VLM", "evidence_vlm", "raw_quote"),
    ):
        for item in rec.get(key, []) or []:
            txt = str(item.get(field, "") or "").strip()
            if txt:
                parts.append(f"[{label}] {txt}")
    return "\n".join(parts)[:cap]


def review_text(pair_rec: dict[str, Any], cap: int = 1800) -> str:
    rows = []
    reviews = pair_rec.get("reviews") or []
    chosen = [
        c for c in reviews
        if int(c.get("y_supportability", 0) or 0) == 1
        or c.get("explicit_fact_hit")
        or c.get("mention_strength") == "strong"
    ]
    if not chosen:
        chosen = reviews[:5]
    for idx, c in enumerate(chosen[:10], 1):
        pol = str(c.get("polarity", "neu"))
        strength = str(c.get("mention_strength", "weak"))
        explicit = "explicit" if c.get("explicit_fact_hit") else "implicit"
        text = str(c.get("text") or c.get("evidence_span") or "").strip()
        if text:
            rows.append(f"{idx}. [{pol}; {strength}; {explicit}] {text[:180]}")
    return "\n".join(rows)[:cap]


def trim(text: str, cap: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap else text[:cap] + "..."


def make_prompt(rec: dict[str, Any], pair_rec: dict[str, Any], manifest: dict[str, Any]) -> str:
    claim = trim(claim_text(rec), 1400) or "【无明确主播属性话术】"
    evidence = trim(evidence_text(rec), 2400) or "【无商品参数/OCR/VLM证据】"
    reviews = trim(review_text(pair_rec), 1800) or "【无可用消费者评论片段】"
    actions = ", ".join(manifest.get("actions") or [])
    noise = ", ".join(rec.get("_attribute_noise_flags") or [])
    return f"""你是直播电商虚假宣传数据集的质量复核员。请判断这个 (商品, 属性) 样本是否适合进入训练集，或需要重生成。

重要约束：
- 你不能使用外部知识、销量、弱标签、样本权重或数据划分。
- 你只能基于主播话术、商品证据、消费者评论片段判断。
- 商品证据不足时要明确说不足；不要用评论替代商品证据。
- 消费者评论只用于判断消费者是否在回应主播关于该属性的说法，不能直接当作商品事实证据。

商品类目：{rec.get("category", "")}
属性：{rec.get("attribute_name", "")}
属性噪声标记：{noise or "无"}
当前待修动作：{actions or "无"}

主播话术：
{claim}

商品证据：
{evidence}

消费者评论片段（已做属性级极性抽取；方括号为 polarity/强度/是否显式提到宣传或承诺）：
{reviews}

请输出严格 JSON：
{{
  "claim_quality": "clear|mixed|garbled|no_claim",
  "attribute_quality": "product_attribute|service_or_process|subjective_or_noisy|wrong_attribute",
  "product_evidence_state": "supported|contradicted|insufficient|not_verifiable",
  "consumer_signal": "refutes_claim|supports_claim|mixed|irrelevant|insufficient",
  "label_recommendation": "positive_risk|negative_clean|drop_or_regenerate",
  "confidence": "high|medium|low",
  "keep_for_training": true/false,
  "recommended_actions": ["keep_clean|keep_silver|drop_bad_claim|drop_bad_attribute|rerun_claim_extraction|rerun_product_evidence|schema_repair_review|needs_human_review"],
  "key_claim": "原文摘录，不超过40字",
  "key_evidence": "证据摘录，无则空字符串，不超过60字",
  "key_review": "评论摘录，无则空字符串，不超过60字",
  "rationale": "一句话说明，不超过100字"
}}"""


def clean_result(obj: dict[str, Any], rec: dict[str, Any], manifest: dict[str, Any], model: str) -> dict[str, Any]:
    actions = [str(a) for a in (obj.get("recommended_actions") or [])]
    actions = [a for a in actions if a in VALID_ACTIONS][:8]
    label_rec = str(obj.get("label_recommendation", ""))[:32]
    if label_rec not in VALID_LABEL_RECOMMENDATIONS:
        ev_state = str(obj.get("product_evidence_state", ""))
        consumer = str(obj.get("consumer_signal", ""))
        keep = bool(obj.get("keep_for_training", False))
        if ev_state == "contradicted" or consumer == "refutes_claim":
            label_rec = "positive_risk"
        elif keep and ev_state == "supported" and consumer in {"supports_claim", "irrelevant", "insufficient"}:
            label_rec = "negative_clean"
        else:
            label_rec = "drop_or_regenerate"
    return {
        "pair_id": pair_id(rec),
        "product_id": rec.get("product_id"),
        "attribute_id": rec.get("attribute_id"),
        "category": rec.get("category"),
        "attribute_name": rec.get("attribute_name"),
        "manifest_priority": manifest.get("priority"),
        "manifest_actions": manifest.get("actions", []),
        "claim_quality": str(obj.get("claim_quality", ""))[:32],
        "attribute_quality": str(obj.get("attribute_quality", ""))[:32],
        "product_evidence_state": str(obj.get("product_evidence_state", ""))[:32],
        "consumer_signal": str(obj.get("consumer_signal", ""))[:32],
        "label_recommendation": label_rec,
        "confidence": str(obj.get("confidence", ""))[:32],
        "keep_for_training": bool(obj.get("keep_for_training", False)),
        "recommended_actions": actions,
        "key_claim": str(obj.get("key_claim", ""))[:80],
        "key_evidence": str(obj.get("key_evidence", ""))[:120],
        "key_review": str(obj.get("key_review", ""))[:120],
        "rationale": str(obj.get("rationale", ""))[:180],
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_product_train_v2.jsonl")
    ap.add_argument("--pair_records", default="data/final/repaired_v1/pair_records_attrpol_v1.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/regeneration_manifest_v1.jsonl")
    ap.add_argument("--recovery_file", default="data/final/repaired_v1/raw_text_evidence_recovery_hq_product_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/llm_regeneration_adjudication_v1.jsonl")
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--priority_max", type=int, default=2)
    ap.add_argument("--recovered_only", action="store_true",
                    help="Only adjudicate pairs with recovered raw text evidence.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=500)
    args = ap.parse_args()

    dataset = {pair_id(r): r for r in read_jsonl(args.dataset)}
    pair_records = {pair_id(r): r for r in read_jsonl(args.pair_records)}
    recovered_ids: set[str] = set()
    if Path(args.recovery_file).exists():
        for r in read_jsonl(args.recovery_file):
            if r.get("recovered_params") or r.get("recovered_ocr"):
                recovered_ids.add(pair_id(r))

    if args.recovered_only:
        manifest_rows = [
            {
                "pair_id": pid,
                "priority": 1,
                "actions": ["llm_recovered_evidence_filter"],
            }
            for pid in sorted(recovered_ids)
            if pid in dataset
        ]
    else:
        manifest_rows = [
            r for r in read_jsonl(args.manifest)
            if int(r.get("priority", 9)) <= args.priority_max
            and pair_id(r) in dataset
            and (not recovered_ids or pair_id(r) in recovered_ids or not args.recovered_only)
        ]
    if args.offset:
        manifest_rows = manifest_rows[args.offset:]
    if args.limit > 0:
        manifest_rows = manifest_rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in manifest_rows if pair_id(r) not in done]

    def fn(mrow: dict[str, Any]) -> dict[str, Any]:
        pid = pair_id(mrow)
        rec = dataset[pid]
        pair_rec = pair_records.get(pid, {})
        try:
            obj = chat_json(
                make_prompt(rec, pair_rec, mrow),
                system="你是严谨的数据质量复核员，只输出 JSON。",
                model=args.model,
                temperature=0.0,
                namespace="regeneration_adjudication_v1",
                max_tokens=args.max_tokens,
            )
            if not isinstance(obj, dict):
                raise ValueError("LLM output is not a JSON object")
            return clean_result(obj, rec, mrow, args.model)
        except Exception as exc:  # noqa: BLE001
            return {"pair_id": pid, "__error__": repr(exc)[:300], "model": args.model}

    results = run_many(todo, fn, concurrency=args.concurrency, desc="regen_adjudicate")
    with open(out_path, "a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[llm_regeneration_adjudicate_v1] new={len(results)} done={len(done) + len(results)} out={out_path}")


if __name__ == "__main__":
    main()
