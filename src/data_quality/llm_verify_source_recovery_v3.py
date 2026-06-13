"""LLM/VLM verification for source-recovery queue v3.

Compared with the older queue verifier, this prompt explicitly separates:

1. whether the target attribute is a valid objective product attribute;
2. whether the livestream span actually makes a claim about that attribute;
3. whether product evidence is grounded in params/OCR/VLM;
4. whether grounded product evidence supports or contradicts the claim.

The output is evidence curation metadata, not a direct training label oracle.
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
from common.io_utils import normalize, read_json
from data_quality.llm_verify_regeneration_queue_v2 import (
    clean,
    load_done,
    product_context,
    read_text_file,
    read_jsonl,
    split_terms,
    text_score,
)


VALID_ATTRIBUTE_QUALITY = {
    "objective_product_attribute",
    "commercial_process_or_price",
    "consumer_subjective_or_eval",
    "service_or_after_sales",
    "wrong_or_overbroad_attribute",
}
VALID_CLAIM_STATE = {
    "attribute_claim_found",
    "claim_about_other_attribute",
    "commercial_or_process_claim",
    "no_usable_claim",
}
VALID_EVIDENCE_STATE = {
    "supports_claim",
    "contradicts_claim",
    "evidence_only",
    "claim_only",
    "insufficient",
}
VALID_SOURCE = {"params", "product_title", "detail_image_ocr", "detail_image_vlm", "none"}


def load_ocr(pid: str) -> dict[str, str]:
    obj = read_json(config.STAGE_C / "ocr_text" / f"{pid}.json", default={}) or {}
    return {str(k): str(v or "") for k, v in obj.items()}


def ocr_context(row: dict[str, Any], max_lines: int = 24) -> str:
    pid = str(row.get("product_id", ""))
    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"), row.get("claim_preview"))
    candidates = []
    for img, text in load_ocr(pid).items():
        if Path(img).name.startswith("._"):
            continue
        for line in re.split(r"[\n\r]+", text or ""):
            line = line.strip()
            if not line:
                continue
            score = text_score(line, terms)
            if score > 0:
                candidates.append((score, img, line[:220]))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    rows = []
    seen = set()
    for _, img, line in candidates:
        key = normalize(line)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"[detail_image_ocr] {Path(img).name}: {line}")
        if len(rows) >= max_lines:
            break
    return "\n".join(rows)


def srt_context(row: dict[str, Any], max_lines: int = 20) -> str:
    existing = row.get("claim_segments") or []
    if existing:
        return "\n".join(
            f"[srt] {seg.get('clip_id', '')} {seg.get('start_ts', '')}-{seg.get('end_ts', '')}: {seg.get('text', '')}"
            for seg in existing[:max_lines]
        )[:3200]

    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"), row.get("claim_preview"))
    candidates = []
    for srt in row.get("srt_files") or []:
        text = read_text_file(srt)
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            score = text_score(line, terms)
            if score <= 0:
                continue
            lo, hi = max(0, i - 2), min(len(lines), i + 3)
            candidates.append((score, srt, " ".join(lines[lo:hi])[:280]))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    rows = []
    seen = set()
    for _, srt, text in candidates:
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"[srt] {Path(srt).name}: {text}")
        if len(rows) >= max_lines:
            break
    return "\n".join(rows)[:3200]


def choose_images(row: dict[str, Any], max_images: int) -> list[str]:
    if max_images <= 0:
        return []
    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"), row.get("claim_preview"))
    ocr_by_name: dict[str, int] = {}
    for img, text in load_ocr(str(row.get("product_id", ""))).items():
        ocr_by_name[Path(img).name] = text_score(text, terms)
    candidates = []
    for rel in row.get("detail_images") or []:
        path = pidx.resolve(rel)
        if path.exists() and not path.name.startswith("._"):
            candidates.append((ocr_by_name.get(path.name, 0), str(path)))
    candidates.sort(key=lambda x: (-x[0], Path(x[1]).name))
    out = []
    seen = set()
    for _, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= max_images:
            break
    return out


def make_prompt(row: dict[str, Any], product_ctx: str, ocr_ctx: str, srt_ctx: str, image_paths: list[str]) -> str:
    image_list = "\n".join(f"{i + 1}. {Path(p).name}" for i, p in enumerate(image_paths)) or "无"
    return f"""你是直播电商 ClaimArc 数据集的严格证据复核员。请只基于给定原始材料判断，不使用外部知识。

目标：判断这个 (商品, 属性) 训练样本到底是：
1) 可补充商品证据；
2) claim 与属性绑定错误；
3) 属性本身不是客观商品属性；
4) 商品证据仍不足。

商品类目：{row.get("category")}
商品标题：{row.get("product_title")}
目标属性：{row.get("attribute_name")} ({row.get("attribute_id")})
当前属性启发式类型：{row.get("attribute_objectivity")} / {row.get("expected_value_type")}
当前标签（仅用于优先级说明，不可作为判断依据）：{row.get("current_label")}

主播/SRT候选片段：
{srt_ctx or "【无SRT候选】"}

商品标题与参数：
{product_ctx or "【无参数】"}

OCR候选：
{ocr_ctx or "【无OCR候选】"}

输入图片顺序：
{image_list}

判断规则：
- 目标属性必须是商品自身可客观核验的属性；“满意度/回购/好看/购买体验/主播服务/发货/抽奖活动”等不是主任务属性。
- 价格/优惠/赠品/库存/活动规则属于 commercial_process_or_price，除非详情页中有可回溯商品规格证据，否则不进入主 claim-evidence 任务。
- claim_found 只有在主播片段确实围绕【目标属性】做了可核验陈述时才为 true。
- product evidence 必须来自商品标题、参数、详情图 OCR 或 VLM 观察；不能把 SRT 当商品证据。
- 若商品证据存在但与该 claim 无法比较，输出 evidence_only 或 insufficient，不要强行 supports。
- 对图片观察必须给出 image_index_or_name；对 OCR/参数必须复制原文短 span。

请严格输出 JSON：
{{
  "attribute_quality": "objective_product_attribute|commercial_process_or_price|consumer_subjective_or_eval|service_or_after_sales|wrong_or_overbroad_attribute",
  "claim_state": "attribute_claim_found|claim_about_other_attribute|commercial_or_process_claim|no_usable_claim",
  "claim_text": "可回溯主播原话，不超过90字；无则空",
  "product_evidence_found": true/false,
  "product_source_type": "params|product_title|detail_image_ocr|detail_image_vlm|none",
  "product_evidence_text": "证据原文或视觉观察，不超过120字；无则空",
  "normalized_value": "规格化属性值；无则空",
  "path_or_image": "参数key/图片名/图片序号；无则空",
  "evidence_state": "supports_claim|contradicts_claim|evidence_only|claim_only|insufficient",
  "training_action": "promote_clean|promote_risk|drop_bad_attribute|drop_bad_claim|rerun_more_evidence|keep_for_auxiliary",
  "confidence": "high|medium|low",
  "rationale": "一句话说明，不超过100字"
}}"""


def clean_result(obj: dict[str, Any], row: dict[str, Any], model: str) -> dict[str, Any]:
    attr_q = str(obj.get("attribute_quality", "wrong_or_overbroad_attribute"))
    if attr_q not in VALID_ATTRIBUTE_QUALITY:
        attr_q = "wrong_or_overbroad_attribute"
    claim_state = str(obj.get("claim_state", "no_usable_claim"))
    if claim_state not in VALID_CLAIM_STATE:
        claim_state = "no_usable_claim"
    ev_state = str(obj.get("evidence_state", "insufficient"))
    if ev_state not in VALID_EVIDENCE_STATE:
        ev_state = "insufficient"
    src = str(obj.get("product_source_type", "none"))
    if src not in VALID_SOURCE:
        src = "none"
    found = bool(obj.get("product_evidence_found", False))
    conf = str(obj.get("confidence", "low"))[:32]

    action = str(obj.get("training_action", "rerun_more_evidence"))
    if attr_q != "objective_product_attribute":
        action = "keep_for_auxiliary" if attr_q == "commercial_process_or_price" else "drop_bad_attribute"
    elif claim_state != "attribute_claim_found":
        action = "drop_bad_claim"
    elif not found or src == "none":
        action = "rerun_more_evidence"
    elif ev_state == "supports_claim":
        action = "promote_clean"
    elif ev_state == "contradicts_claim":
        action = "promote_risk"
    elif ev_state in {"evidence_only", "claim_only", "insufficient"}:
        action = "rerun_more_evidence"

    return {
        "pair_id": row.get("pair_id"),
        "queue_type": row.get("queue_type"),
        "priority": row.get("priority"),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "current_label": row.get("current_label"),
        "current_source_count": row.get("current_source_count"),
        "attribute_objectivity": row.get("attribute_objectivity"),
        "attribute_quality": attr_q,
        "claim_state": claim_state,
        "claim_text": str(obj.get("claim_text", ""))[:180],
        "product_evidence_found": found,
        "product_source_type": src,
        "product_evidence_text": str(obj.get("product_evidence_text", ""))[:260],
        "normalized_value": str(obj.get("normalized_value", ""))[:120],
        "path_or_image": str(obj.get("path_or_image", ""))[:160],
        "evidence_state": ev_state,
        "training_action": action,
        "confidence": conf,
        "rationale": str(obj.get("rationale", ""))[:220],
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/source_recovery_queue_v3.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/source_recovery_queue_v3_llm_verify.jsonl")
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--max_images", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=750)
    args = ap.parse_args()

    rows = read_jsonl(args.queue)
    priorities = {p.strip() for p in re.split(r"[, ]+", args.priority) if p.strip()}
    if priorities:
        rows = [r for r in rows if str(r.get("priority")) in priorities]
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
            system="你是严谨的电商证据核验员，只输出 JSON。",
            model=args.model,
            images=data_urls or None,
            temperature=0.0,
            namespace="source_recovery_v3_verify",
            max_tokens=args.max_tokens,
        )
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not an object")
        return clean_result(obj, row, args.model)

    results = llm.run_many(todo, verify, concurrency=args.concurrency, desc="source_recovery_v3")
    with out_path.open("a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[llm_verify_source_recovery_v3] new={len(results)} done={len(done) + len(results)} out={out_path}")


if __name__ == "__main__":
    main()
