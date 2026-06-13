"""Two-stage LLM/VLM verification for missing-claim regeneration rows.

`missing_claim_srt_first` rows are different from ordinary source-absent
records: the current dataset has consumer risk signals but no extracted live
claim. We therefore verify three atomic facts separately:

1. whether the SRT contains an exact live claim for the target attribute;
2. whether product-side materials contain evidence for that same attribute;
3. whether the consumer comment refutes or supports the live/product claim.

The script intentionally does not write training labels. A downstream builder
should derive labels from these atomic fields with deterministic rules.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import config
from common import llm
from common import product_index as pidx
from data_quality.llm_verify_regeneration_queue_v2 import (
    choose_images,
    load_done,
    ocr_context,
    product_context,
    read_jsonl,
    srt_context,
)


VALID_EVIDENCE_STATE = {"supported", "contradicted", "insufficient", "not_found"}
VALID_CONSUMER_SIGNAL = {"refutes_claim", "supports_claim", "mixed", "irrelevant"}
VALID_ACTION = {"risk_candidate", "clean_candidate", "rerun_more_evidence", "drop"}


def make_prompt(row: dict[str, Any], product_ctx: str, ocr_ctx: str, srt_ctx: str, image_paths: list[str]) -> str:
    image_list = "\n".join(f"{i + 1}. {Path(p).name}" for i, p in enumerate(image_paths)) or "无"
    return f"""你是直播电商虚假宣传数据重生成的三元证据核验员。请只基于给定材料判断，不使用外部知识。

任务：从缺失主播声明但消费者有风险评论的样本中，核验是否可以重生成高质量训练样本。

商品类目：{row.get("category")}
商品标题：{row.get("product_title")}
目标属性：{row.get("attribute_name")} ({row.get("attribute_id")})
属性类型：{row.get("attribute_objectivity")} / {row.get("expected_value_type")}
消费者风险评论样例：{row.get("risk_comment_example") or "无"}
风险评论命中数：{row.get("missing_claim_hits")}

主播/SRT候选片段：
{srt_ctx or "【无SRT候选】"}

商品标题与参数：
{product_ctx or "【无参数】"}

OCR候选：
{ocr_ctx or "【无OCR候选】"}

输入图片顺序：
{image_list}

核验原则：
- live_claim_found 只能来自主播/SRT，不能来自商品标题、参数或详情图。
- product_evidence_found 只能来自商品标题、参数、OCR 或详情图视觉观察，不能来自主播/SRT。
- product_evidence_state 判断商品证据与主播声明之间的关系：支持/矛盾/不足/未找到。
- consumer_signal 判断消费者评论相对主播声明或商品证据是反驳、支持、混合还是无关。
- 如果主播声明不是目标属性，或消费者评论不是目标属性，必须降为 drop 或 rerun_more_evidence。
- 对数值、材质、尺码、容量、厚度等属性，必须尽量抽取规格化值。

请严格输出 JSON：
{{
  "live_claim_found": true/false,
  "live_claim_text": "SRT中可回溯的主播原话，不超过100字；无则空",
  "live_claim_value": "主播声明中的规格化属性值；无则空",
  "claim_clip": "SRT文件名或clip；无则空",
  "claim_timestamp": "时间戳；无则空",
  "product_evidence_found": true/false,
  "product_source_type": "params|product_title|detail_image_ocr|detail_image_vlm|none",
  "product_evidence_text": "商品证据原文或图片观察，不超过120字；无则空",
  "product_value": "商品证据中的规格化属性值；无则空",
  "product_path_or_image": "参数名/图片名/图片序号；无则空",
  "product_evidence_state": "supported|contradicted|insufficient|not_found",
  "consumer_signal": "refutes_claim|supports_claim|mixed|irrelevant",
  "consumer_anchor": "消费者评论中对应属性的短语，不超过80字；无则空",
  "confidence": "high|medium|low",
  "reject_reason": "不能入训练的原因；否则空",
  "curation_action": "risk_candidate|clean_candidate|rerun_more_evidence|drop"
}}"""


def _as_bool(value: Any) -> bool:
    return bool(value) if not isinstance(value, str) else value.strip().lower() in {"true", "1", "yes", "是"}


def clean_result(obj: dict[str, Any], row: dict[str, Any], model: str) -> dict[str, Any]:
    live_claim_found = _as_bool(obj.get("live_claim_found"))
    product_evidence_found = _as_bool(obj.get("product_evidence_found"))
    source_type = str(obj.get("product_source_type", "none"))
    if source_type not in {"params", "product_title", "detail_image_ocr", "detail_image_vlm", "none"}:
        source_type = "none"
    state = str(obj.get("product_evidence_state", "insufficient"))
    if state not in VALID_EVIDENCE_STATE:
        state = "insufficient"
    consumer = str(obj.get("consumer_signal", "mixed"))
    if consumer not in VALID_CONSUMER_SIGNAL:
        consumer = "mixed"
    confidence = str(obj.get("confidence", "low"))[:32]

    if not live_claim_found:
        action = "drop"
    elif state == "contradicted" and product_evidence_found and source_type != "none":
        action = "risk_candidate"
    elif state == "supported" and consumer == "refutes_claim" and product_evidence_found and source_type != "none":
        action = "risk_candidate"
    elif state == "supported" and consumer == "supports_claim" and product_evidence_found and source_type != "none":
        action = "clean_candidate"
    elif live_claim_found and (product_evidence_found or consumer in {"refutes_claim", "supports_claim"}):
        action = "rerun_more_evidence"
    else:
        action = "drop"
    if confidence == "low" and action in {"risk_candidate", "clean_candidate"}:
        action = "rerun_more_evidence"

    return {
        "pair_id": row.get("pair_id"),
        "queue_type": row.get("queue_type"),
        "priority": row.get("priority"),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "attribute_objectivity": row.get("attribute_objectivity"),
        "expected_value_type": row.get("expected_value_type"),
        "missing_claim_hits": row.get("missing_claim_hits"),
        "risk_comment_example": row.get("risk_comment_example"),
        "live_claim_found": live_claim_found,
        "live_claim_text": str(obj.get("live_claim_text", ""))[:200],
        "live_claim_value": str(obj.get("live_claim_value", ""))[:120],
        "claim_clip": str(obj.get("claim_clip", ""))[:160],
        "claim_timestamp": str(obj.get("claim_timestamp", ""))[:80],
        "product_evidence_found": product_evidence_found,
        "product_source_type": source_type,
        "product_evidence_text": str(obj.get("product_evidence_text", ""))[:240],
        "product_value": str(obj.get("product_value", ""))[:120],
        "product_path_or_image": str(obj.get("product_path_or_image", ""))[:160],
        "product_evidence_state": state,
        "consumer_signal": consumer,
        "consumer_anchor": str(obj.get("consumer_anchor", ""))[:160],
        "confidence": confidence,
        "reject_reason": str(obj.get("reject_reason", ""))[:240],
        "curation_action": action,
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/regeneration_queue_v2.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/regeneration_queue_v2_missing_claim_verify_v1.jsonl")
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--max_images", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=800)
    args = ap.parse_args()

    rows = read_jsonl(args.queue)
    priorities = {p.strip() for p in re.split(r"[, ]+", args.priority) if p.strip()}
    rows = [
        r for r in rows
        if str(r.get("queue_type")) == "missing_claim_srt_first"
        and (not priorities or str(r.get("priority")) in priorities)
    ]
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if str(r.get("pair_id")) not in done]
    bundles = pidx.build_bundles()

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        product_ctx = product_context(row, bundles)
        ocr_ctx = ocr_context(row)
        srt_ctx = srt_context(row)
        images = choose_images(row, args.max_images)
        data_urls = [u for p in images if (u := llm.encode_image(p))]
        obj = llm.chat_json(
            make_prompt(row, product_ctx, ocr_ctx, srt_ctx, images),
            system="你是严谨的电商三元证据核验员，只输出 JSON。",
            model=args.model,
            images=data_urls or None,
            temperature=0.0,
            namespace="missing_claim_srt_first_verify_v1",
            max_tokens=args.max_tokens,
        )
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not an object")
        return clean_result(obj, row, args.model)

    results = llm.run_many(todo, verify, concurrency=args.concurrency, desc="missing_claim_verify_v1")
    with out_path.open("a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[llm_verify_missing_claim_srt_first_v1] new={len(results)} done={len(done) + len(results)} out={out_path}")


if __name__ == "__main__":
    main()
