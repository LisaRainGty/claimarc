"""LLM/VLM runner for full pair reconstruction.

This runner consumes `full_pair_reconstruction_queue_v1`.  It asks a model to
repair the claim/evidence materials and rebuild the consumer-perception label
from comments aligned to the repaired livestream claim.

The output is an audit artifact, not a direct training dataset.  A later gate
must verify source provenance and rebuild weights before promotion.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common import llm
from common.io_utils import read_jsonl, write_json
from data_quality.llm_verify_regeneration_queue_v2 import (
    choose_images,
    load_done,
    ocr_context,
    product_context,
    srt_context,
)
from common import product_index as pidx


VALID_REL = {"support", "refute", "mixed", "unclear", "not_aligned"}
VALID_ACTION = {
    "promote_candidate",
    "silver_review",
    "rerun_claim",
    "rerun_evidence",
    "rerun_joint",
    "drop_no_reconstructable_claim",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def comments_block(row: dict[str, Any], max_comments: int) -> str:
    rows = row.get("consumer_mentions") or []
    lines = []
    for i, c in enumerate(rows[:max_comments], 1):
        lines.append(
            f"{i}. [{c.get('polarity')}/{c.get('mention_strength')}/"
            f"{'explicit' if c.get('explicit_fact_hit') else 'implicit'}] "
            f"{clean(c.get('evidence_span'))}"
        )
    return "\n".join(lines) or "无属性级评论片段"


def load_prefilter(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = clean(row.get("pair_id"))
        if pid:
            out[pid] = row
    return out


def prefilter_srt_context(row: dict[str, Any], prefilter: dict[str, dict[str, Any]], max_candidates: int) -> str:
    item = prefilter.get(clean(row.get("pair_id")))
    if not item:
        return ""
    lines = []
    for i, cand in enumerate((item.get("claim_candidates") or [])[:max_candidates], 1):
        why = cand.get("why") or {}
        hits = []
        for key in ("attribute_hits", "value_hits", "comment_hits"):
            vals = [clean(x) for x in (why.get(key) or []) if clean(x)]
            if vals:
                hits.append(f"{key}={','.join(vals[:6])}")
        lines.append(
            f"[prefilter_srt #{i} score={cand.get('score')} "
            f"{Path(clean(cand.get('srt_file'))).name} "
            f"{clean(cand.get('start_ts'))}-{clean(cand.get('end_ts'))} "
            f"{'; '.join(hits)}] {clean(cand.get('text'))}"
        )
    return "\n".join(lines)[:3600]


def make_prompt(row: dict[str, Any], product_ctx: str, ocr_ctx: str, srt_ctx: str, images: list[str], max_comments: int) -> str:
    image_list = "\n".join(f"{i + 1}. {Path(p).name}" for i, p in enumerate(images)) or "无"
    current_evidence = json.dumps(row.get("current_evidence_preview") or [], ensure_ascii=False)[:1600]
    current_claim = clean(row.get("claim_preview"))[:1200] or "无"
    return f"""你是直播电商虚假宣传数据重构审查员。请只基于给定原始材料工作，不使用外部知识。

目标：为一个 (商品, 属性) pair 重构完整训练样本：
1. 从主播 SRT 中找该属性的最小连续 claim 原话；
2. 从商品标题/参数/详情图 OCR/VLM 找同属性 product evidence；
3. 判断属性级消费者评论是否在同一命题上支持或反驳该 claim；
4. 只有评论反驳 claim 时，才把消费者感知虚假宣传标签 new_y 置为 1。

商品标题：{row.get("product_title")}
商品类目：{row.get("category")} / {row.get("subcategory")}
目标属性：{row.get("attribute_name")} ({row.get("attribute_id")})
属性值类型：{row.get("expected_value_type")}
旧抽取状态（仅用于定位修复任务，不用于判断标签）：claim={row.get("claim_state")}；evidence={row.get("evidence_state")}

当前自动抽取 claim 预览（仅供核验，可替换或忽略）：
{current_claim}

主播/SRT 候选：
{srt_ctx or "【无SRT候选】"}

商品标题与参数：
{product_ctx or "【无参数】"}

OCR 候选：
{ocr_ctx or "【无OCR候选】"}

旧 evidence 预览：
{current_evidence or "[]"}

输入图片顺序：
{image_list}

属性级消费者评论片段：
{comments_block(row, max_comments)}

严格规则：
- claim_text 必须是主播/SRT 中可回溯的连续原话；不要编写或改写。
- product evidence 只能来自商品标题、参数、详情图 OCR 或详情图视觉观察；不能来自主播/SRT 或评论。
- 评论是否构成标签，只看它是否和 repaired claim 的同一具体命题形成支持/反驳关系。
- 评论泛泛说质量差、体验差、大小不合适，但没有反驳 claim 的具体事实，判 not_aligned 或 unclear，不得触发 new_y=1。
- product evidence 与 claim 客观矛盾本身不能直接触发 new_y=1；必须存在消费者评论 refute claim。
- 如果 claim 缺失，输出 new_y=0 且 action=rerun_claim 或 drop_no_reconstructable_claim。

请严格输出 JSON：
{{
  "claim_found": true/false,
  "claim_text": "SRT连续原话，<=120字",
  "claim_source": "clip文件名或空",
  "claim_timestamp": "起止时间或空",
  "product_evidence_found": true/false,
  "evidence_source_type": "product_title|params|detail_image_ocr|detail_image_vlm|none",
  "evidence_text": "商品证据原文或客观视觉描述，<=140字",
  "evidence_source": "参数名/图片名/标题/空",
  "claim_evidence_relation": "supports_claim|contradicts_claim|insufficient",
  "comment_judgments": [
    {{"cid": 1, "aligned_to_claim": true/false, "relation": "support|refute|mixed|unclear|not_aligned", "reason": "<=20字"}}
  ],
  "new_y": 0或1,
  "label_basis": "为什么该 pair 是/不是消费者感知虚假宣传，<=80字",
  "confidence": "high|medium|low",
  "action": "promote_candidate|silver_review|rerun_claim|rerun_evidence|rerun_joint|drop_no_reconstructable_claim"
}}
只输出 JSON。"""


def clean_comment_judgments(obj: dict[str, Any], max_comments: int) -> list[dict[str, Any]]:
    out = []
    for item in obj.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        if cid < 1 or cid > max_comments:
            continue
        rel = clean(item.get("relation"))
        if rel not in VALID_REL:
            rel = "unclear"
        aligned = bool(item.get("aligned_to_claim")) and rel in {"support", "refute", "mixed"}
        out.append({
            "cid": cid,
            "aligned_to_claim": aligned,
            "relation": rel if aligned else "not_aligned",
            "reason": clean(item.get("reason"))[:60],
        })
    return out


def clean_result(obj: dict[str, Any], row: dict[str, Any], model: str, max_comments: int) -> dict[str, Any]:
    judgments = clean_comment_judgments(obj, max_comments)
    has_refute = any(j["aligned_to_claim"] and j["relation"] == "refute" for j in judgments)
    claim_found = bool(obj.get("claim_found"))
    evidence_found = bool(obj.get("product_evidence_found"))
    new_y = 1 if claim_found and has_refute else 0
    action = clean(obj.get("action"))
    if action not in VALID_ACTION:
        if not claim_found:
            action = "rerun_claim"
        elif not evidence_found:
            action = "rerun_evidence"
        elif clean(obj.get("confidence")).lower() == "high":
            action = "promote_candidate"
        else:
            action = "silver_review"
    return {
        "pair_id": row.get("pair_id"),
        "queue_type": row.get("queue_type"),
        "priority": row.get("priority"),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "old_y": row.get("old_y"),
        "old_c": row.get("old_c"),
        "claim_found": claim_found,
        "claim_text": clean(obj.get("claim_text"))[:180],
        "claim_source": clean(obj.get("claim_source"))[:180],
        "claim_timestamp": clean(obj.get("claim_timestamp"))[:80],
        "product_evidence_found": evidence_found,
        "evidence_source_type": clean(obj.get("evidence_source_type"))[:40],
        "evidence_text": clean(obj.get("evidence_text"))[:240],
        "evidence_source": clean(obj.get("evidence_source"))[:180],
        "claim_evidence_relation": clean(obj.get("claim_evidence_relation"))[:40],
        "comment_judgments": judgments,
        "new_y": new_y,
        "raw_new_y": int(obj.get("new_y", 0) or 0) if str(obj.get("new_y", "0")).isdigit() else obj.get("new_y"),
        "label_basis": clean(obj.get("label_basis"))[:160],
        "confidence": clean(obj.get("confidence")).lower()[:20],
        "action": action,
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.report.json")
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--queue_type", default="")
    ap.add_argument("--srt_prefilter", default="data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.jsonl")
    ap.add_argument("--max_srt_candidates", type=int, default=6)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_comments", type=int, default=10)
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--dry_run", action="store_true", help="Only report selected rows and context coverage; do not call LLM.")
    args = ap.parse_args()

    rows = list(read_jsonl(args.queue))
    priorities = {p.strip() for p in args.priority.replace(",", " ").split() if p.strip()}
    if priorities:
        rows = [r for r in rows if str(r.get("priority")) in priorities]
    if args.queue_type:
        rows = [r for r in rows if str(r.get("queue_type")) == args.queue_type]
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if str(r.get("pair_id")) not in done]
    bundles = pidx.build_bundles()
    srt_prefilter = load_prefilter(args.srt_prefilter)

    if args.dry_run:
        report = {
            "queue": args.queue,
            "out": args.out,
            "dry_run": True,
            "selected_rows": len(rows),
            "already_done": len(done),
            "todo": len(todo),
            "priority": dict(Counter(str(r.get("priority")) for r in rows)),
            "queue_type": dict(Counter(str(r.get("queue_type")) for r in rows)),
            "claim_state": dict(Counter(str(r.get("claim_state")) for r in rows)),
            "prefilter_available": sum(1 for r in rows if clean(r.get("pair_id")) in srt_prefilter),
            "examples": [
                {
                    "pair_id": r.get("pair_id"),
                    "priority": r.get("priority"),
                    "queue_type": r.get("queue_type"),
                    "claim_state": r.get("claim_state"),
                    "attribute_name": r.get("attribute_name"),
                    "has_prefilter": clean(r.get("pair_id")) in srt_prefilter,
                }
                for r in rows[:20]
            ],
        }
        write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not (os.environ.get("MATPOOL_API_KEY") or config.MATPOOL_API_KEY):
        raise RuntimeError("MATPOOL_API_KEY is not set. Use --dry_run for local checks without LLM calls.")

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        product_ctx = product_context(row, bundles)
        ocr_ctx = ocr_context(row, max_lines=18)
        srt_ctx = prefilter_srt_context(row, srt_prefilter, args.max_srt_candidates) or srt_context(row, max_lines=24)
        image_paths = choose_images(row, args.max_images)
        data_urls = [u for p in image_paths if (u := llm.encode_image(p))]
        obj = llm.chat_json(
            make_prompt(row, product_ctx, ocr_ctx, srt_ctx, image_paths, args.max_comments),
            system="你是严谨的直播电商虚假宣传数据重构审查员，只输出 JSON。",
            model=args.model,
            images=data_urls or None,
            temperature=0.0,
            namespace="full_pair_reconstruction_v1",
            max_tokens=args.max_tokens,
        )
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not a JSON object")
        return clean_result(obj, row, args.model, args.max_comments)

    results = llm.run_many(todo, verify, concurrency=args.concurrency, desc="full_pair_reconstruct_v1")
    with out_path.open("a", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    all_rows = list(read_jsonl(out_path))
    report = {
        "queue": args.queue,
        "out": args.out,
        "new": len(results),
        "total": len(all_rows),
        "action": dict(Counter(str(r.get("action")) for r in all_rows)),
        "new_y": dict(Counter(str(r.get("new_y")) for r in all_rows)),
        "claim_found": dict(Counter(str(bool(r.get("claim_found"))) for r in all_rows)),
        "evidence_found": dict(Counter(str(bool(r.get("product_evidence_found"))) for r in all_rows)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
