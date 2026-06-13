"""LLM/VLM review for the mechanism-driven repair queue.

The prompt withholds the current label and model predictions. The review is a
data-quality signal about claim concreteness, product-evidence support, and
exact value alignment. Downstream scripts can combine this review with weak
consumer labels and current y/c to decide keep/relabel/drop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import llm


VALID_RELATION = {"supports", "contradicts", "insufficient", "not_verifiable"}
VALID_VALUE = {
    "exact_match",
    "compatible",
    "contradiction",
    "ambiguous",
    "not_applicable",
}
VALID_SOURCE = {"params", "ocr", "vlm", "detail_image", "mixed", "none"}
VALID_ACTION = {
    "keep_relation",
    "recover_more_evidence",
    "drop_bad_claim",
    "review_consumer_signal",
}


PROMPT = """你是直播电商 claim-evidence 数据质量裁决员。请只根据给定材料判断，不使用外部搜索，不猜测商品没有给出的信息。

商品类目：{category}
目标属性：{attribute_name} ({attribute_id})

主播 claim：
{claim}

当前抽取到的商品证据摘要：
[PARAMS]
{params}

[OCR]
{ocr}

[VLM]
{vlm}

可用原始详情图文件名：
{image_names}

裁决标准：
1. claim 必须是具体商品属性事实、承诺、数值、材质、功能、颜色、尺码、价格等；泛泛夸张或无上下文短语不是合格 claim。
2. 商品证据只能来自商品参数、详情图 OCR/VLM 或详情图本身；主播话术不能当商品证据。
3. 对数字、材质、尺码、重量、颜色、价格/券后价等，必须判断值是否精确兼容；ASR/OCR 小错可规范化，但不能硬凑。
4. 若证据只是重复属性名、标题泛词、营销词，不能算支持；标为 insufficient 或 not_verifiable。
5. 你不知道消费者评论和当前训练标签，因此不要输出最终风险标签，只输出 claim-evidence 关系和数据修复建议。

严格输出 JSON：
{{
  "claim_quality": "clear|mixed|garbled|no_claim",
  "claim_type": "numeric|material|size_weight|price_coupon|color_visual|function|brand|subjective|other",
  "key_claim": "最关键主播原话，不超过80字；无则空",
  "evidence_found": true/false,
  "evidence_source": "params|ocr|vlm|detail_image|mixed|none",
  "key_evidence": "最关键商品证据原文/图像观察，不超过120字；无则空",
  "relation_to_claim": "supports|contradicts|insufficient|not_verifiable",
  "value_alignment": "exact_match|compatible|contradiction|ambiguous|not_applicable",
  "likely_issue": "none|generic_evidence|missing_evidence|bad_claim_span|value_mismatch|ocr_asr_noise|needs_consumer_signal",
  "repair_action": "keep_relation|recover_more_evidence|drop_bad_claim|review_consumer_signal",
  "confidence": "high|medium|low",
  "rationale": "一句话说明，不超过90字"
}}
只输出 JSON。"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("pair_id"):
                done.add(str(obj["pair_id"]))
    return done


def trim(text: Any, cap: int) -> str:
    s = str(text or "").strip()
    return s if len(s) <= cap else s[:cap] + "..."


def make_prompt(row: dict[str, Any], image_paths: list[str]) -> str:
    ev = row.get("evidence") or {}
    image_names = "\n".join(
        f"{i + 1}. {Path(p).name}" for i, p in enumerate(image_paths)
    ) or "无"
    return PROMPT.format(
        category=row.get("category", ""),
        attribute_name=row.get("attribute_name", ""),
        attribute_id=row.get("attribute_id", ""),
        claim=trim(row.get("claim", ""), 1400) or "【无】",
        params=trim(ev.get("params", ""), 1400) or "【无】",
        ocr=trim(ev.get("ocr", ""), 1800) or "【无】",
        vlm=trim(ev.get("vlm", ""), 1200) or "【无】",
        image_names=image_names,
    )


def clean_obj(obj: Any, row: dict[str, Any], model: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    relation = str(obj.get("relation_to_claim", "insufficient"))
    if relation not in VALID_RELATION:
        relation = "insufficient"
    value = str(obj.get("value_alignment", "ambiguous"))
    if value not in VALID_VALUE:
        value = "ambiguous"
    source = str(obj.get("evidence_source", "none"))
    if source not in VALID_SOURCE:
        source = "none"
    action = str(obj.get("repair_action", "review_consumer_signal"))
    if action not in VALID_ACTION:
        action = "review_consumer_signal"
    return {
        "pair_id": row.get("pair_id"),
        "product_id": row.get("product_id"),
        "row": row.get("row"),
        "priority_score": row.get("priority_score"),
        "reasons": row.get("reasons", []),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "claim_quality": str(obj.get("claim_quality", ""))[:32],
        "claim_type": str(obj.get("claim_type", ""))[:32],
        "key_claim": str(obj.get("key_claim", ""))[:160],
        "evidence_found": bool(obj.get("evidence_found", False)),
        "evidence_source": source,
        "key_evidence": str(obj.get("key_evidence", ""))[:240],
        "relation_to_claim": relation,
        "value_alignment": value,
        "likely_issue": str(obj.get("likely_issue", ""))[:64],
        "repair_action": action,
        "confidence": str(obj.get("confidence", ""))[:32],
        "rationale": str(obj.get("rationale", ""))[:220],
        "model": model,
    }


def choose_images(row: dict[str, Any], max_images: int) -> list[str]:
    if max_images <= 0:
        return []
    paths = [p for p in row.get("raw_image_paths", []) if Path(p).exists()]
    return paths[:max_images]


def review_one(row: dict[str, Any], model: str, max_images: int, max_tokens: int) -> dict[str, Any]:
    image_paths = choose_images(row, max_images)
    images = [u for p in image_paths if (u := llm.encode_image(p))]
    obj = llm.chat_json(
        make_prompt(row, image_paths),
        system="你是严谨的电商商品证据核验员，只输出 JSON。",
        model=model,
        images=images or None,
        temperature=0.0,
        namespace="mechanism_repair_review_v1",
        max_tokens=max_tokens,
    )
    return clean_obj(obj, row, model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=650)
    ap.add_argument("--dry_run_prompt", default="")
    args = ap.parse_args()

    rows = read_jsonl(args.queue)
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    if args.dry_run_prompt:
        sample = rows[0] if rows else {}
        image_paths = choose_images(sample, args.max_images)
        Path(args.dry_run_prompt).write_text(
            make_prompt(sample, image_paths),
            encoding="utf-8",
        )
        print(f"[dry_run_prompt] -> {args.dry_run_prompt}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if str(r.get("pair_id")) not in done]

    def fn(row: dict[str, Any]) -> dict[str, Any]:
        try:
            return review_one(row, args.model, args.max_images, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            return {
                "pair_id": row.get("pair_id"),
                "__error__": repr(exc)[:300],
                "model": args.model,
            }

    results = llm.run_many(todo, fn, concurrency=args.concurrency, desc="mechanism_repair_review")
    with out_path.open("a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "queue": args.queue,
        "out": args.out,
        "model": args.model,
        "new": len(results),
        "done_before": len(done),
        "limit": args.limit,
        "offset": args.offset,
        "counts": {},
        "errors": sum(1 for r in results if isinstance(r, dict) and "__error__" in r),
    }
    for field in ("claim_quality", "relation_to_claim", "value_alignment", "repair_action", "likely_issue"):
        counts: dict[str, int] = {}
        for r in results:
            val = str((r or {}).get(field, ""))
            counts[val] = counts.get(val, 0) + 1
        report["counts"][field] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
