"""Re-extract missed livestream claims from a product-v2 repair queue.

The input queue is built from product-v2 attributes and comment triggers.  The
comments are only hints for what to search in raw SRT; accepted outputs must be
exact contiguous substrings from the livestream transcript.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import llm
from common import srt as S
from common.io_utils import normalize, read_jsonl, write_json, write_jsonl


TASK = """你是电商直播商品事实抽取员。
下面给出一个商品、一个目标属性、若干消费者评论触发信号，以及一段主播字幕。
若提供了商品详情证据，它只用于锚定目标属性维度；主播说法可以与详情证据一致，也可以矛盾。

你的任务：只从主播字幕中寻找是否存在针对目标属性的商品事实陈述。

硬规则：
1. 只输出 JSON 数组，不要解释。
2. extraction_text 必须是字幕中真实存在的连续原文子串，不能改写、概括、拼接。
3. 评论只用于提示要找什么，严禁把评论内容当作主播陈述输出。
4. 如果字幕中没有直接讲目标属性，输出 []。
5. 忽略链接号、下单、库存、优惠券、价格、主播闲聊、纯推荐语。
6. 目标属性必须语义精确：不要把相近但不相同的属性混用。
7. 输出最短但完整的事实子串，优先包含明确属性值。
8. 颜色必须含颜色词；材质/成分必须含材料词；尺寸/重量/容量/数量必须含数字、范围或明确规格词；功效/功能必须是商品声称的具体功能。
9. 不要输出疑问句、反问、让观众“看一下”的引导句；必须是主播对商品属性的陈述。
10. 电源容量/电池容量/净含量/重量/尺寸等数值属性必须包含具体数字或单位；“快充”“好用”“大包”等不等于容量或尺寸。
11. 材质属性必须出现具体材料值，如棉、羊毛、真皮、聚酯纤维、腈纶、混纺等；“看材质吗”“什么材质”不算。
12. 品牌/型号/货号/产地/保质期/条形码等身份规格属性必须显式说出对应身份词或具体值；“官方旗舰店”“正品”“链接号”“价格”“收到货”都不能替代这些属性。
13. 商品条形码、商品价格、链接、优惠、发货、售后、库存等交易信息通常不是商品事实属性；除非字幕逐字说出目标属性名和具体值，否则输出 []。
14. 商品详情证据只用于判断“同一属性维度”：不要要求主播原文与证据值一致，但必须是在谈同一属性。例：证据是“容量: 20000mAh”，字幕“支持快充”不是容量；字幕“20000毫安”才是容量。
15. 若目标属性是风味/口味，必须出现口感、味道、香型或具体口味词；热量、价格、规格不算风味。若目标属性是类型/产品名称/品牌，必须出现明确类型词、商品名或品牌名，不能只输出数字、链接号或“我们家”。

输出格式：
[
  {"extraction_text": "字幕原文连续子串", "reason": "为什么它直接对应目标属性，20字以内"}
]
"""


def clean(value: Any) -> str:
    return str(value or "").strip()


def chunk_ranges(concat: S.ConcatResult, chunk_chars: int) -> list[tuple[int, int]]:
    if chunk_chars <= 0 or len(concat.text) <= chunk_chars or not concat.spans:
        return [(0, len(concat.text))]
    out: list[tuple[int, int]] = []
    start = concat.spans[0].char_start
    end = start
    for sp in concat.spans:
        if end > start and sp.char_end - start > chunk_chars:
            out.append((start, end))
            start = sp.char_start
        end = sp.char_end
    if end > start:
        out.append((start, end))
    return out


def as_rows(obj: Any) -> list[dict[str, str]]:
    if isinstance(obj, dict):
        obj = obj.get("claims") or obj.get("extractions") or []
    if not isinstance(obj, list):
        return []
    rows: list[dict[str, str]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        txt = clean(item.get("extraction_text") or item.get("text"))
        reason = clean(item.get("reason"))
        if txt:
            rows.append({"extraction_text": txt, "reason": reason[:80]})
    return rows


def load_fact_records(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in read_jsonl(p):
        pid = str(rec.get("product_id") or "")
        aid = str(rec.get("attribute_id") or "")
        if pid and aid:
            out[(pid, aid)] = rec
    return out


def evidence_snippets(fact: dict[str, Any] | None, cap: int = 10) -> list[str]:
    if not fact:
        return []
    snippets: list[str] = []
    for item in fact.get("evidence_params") or []:
        key = clean(item.get("param_key"))
        text = clean(item.get("raw_text"))
        if text:
            snippets.append(f"[params] {key}: {text}" if key else f"[params] {text}")
    for item in fact.get("evidence_ocr") or []:
        text = clean(item.get("raw_text"))
        if text:
            snippets.append(f"[ocr] {text}")
    for item in fact.get("evidence_vlm") or []:
        text = clean(item.get("raw_quote") or item.get("raw_text"))
        if text:
            snippets.append(f"[vlm] {text}")
    out: list[str] = []
    seen: set[str] = set()
    for s in snippets:
        s = s[:180]
        key = normalize(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def prompt_for(row: dict[str, Any], chunk: str) -> str:
    reviews = []
    for c in (row.get("triggered_reviews") or [])[:6]:
        reviews.append(
            f"- polarity={c.get('polarity','')} triggers={','.join(c.get('trigger_hits') or [])} "
            f"text={clean(c.get('text'))[:160]}"
        )
    aliases = "、".join(clean(x) for x in (row.get("aliases") or [])[:16] if clean(x))
    snippets = row.get("_fact_snippets") or []
    evidence_block = "\n".join(f"- {s}" for s in snippets[:10]) if snippets else "（无可用详情证据；只按目标属性和字幕判断）"
    return (
        TASK
        + "\n商品标题："
        + clean(row.get("product_title"))
        + "\n目标属性："
        + clean(row.get("attribute_id"))
        + " | "
        + clean(row.get("attribute_name"))
        + "\n属性别名："
        + aliases
        + "\n属性类型："
        + clean(row.get("source_family"))
        + " / "
        + clean(row.get("expected_value_type"))
        + "\n消费者触发信号（只作为搜索提示，不可输出）：\n"
        + "\n".join(reviews)
        + "\n\n商品详情证据（只用于锚定目标属性维度，主播说法可与其一致或矛盾）：\n"
        + evidence_block
        + "\n\n主播字幕窗口：\n"
        + chunk
    )


def extract_row(
    row: dict[str, Any],
    *,
    chunk_chars: int,
    model: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    files = [str(Path(p)) for p in (row.get("srt_files") or [])]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return []
    concat = S.concat_product_srt(files)
    if not concat.text.strip():
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seq = 0
    for ci, (start, end) in enumerate(chunk_ranges(concat, chunk_chars), 1):
        chunk = concat.text[start:end]
        try:
            obj = llm.chat_json(
                prompt_for(row, chunk),
                model=model,
                temperature=0.0,
                namespace="b1_productv2_reextract_queue",
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] reextract failed {row.get('pair_id')} chunk={ci}: {exc!r}", flush=True)
            continue
        for item in as_rows(obj):
            quote = item["extraction_text"]
            local = chunk.find(quote)
            if local < 0:
                continue
            gstart = start + local
            gend = gstart + len(quote)
            if normalize(quote) and normalize(quote) not in normalize(concat.text[gstart:gend]):
                continue
            spans = concat.lookup_range(gstart, gend)
            if not spans:
                continue
            first, last = spans[0], spans[-1]
            key = (normalize(quote), first.srt_file)
            if key in seen:
                continue
            seen.add(key)
            seq += 1
            out.append({
                "claim_id": f"{row.get('product_id')}_re{seq}",
                "product_id": str(row.get("product_id")),
                "pair_id": row.get("pair_id"),
                "attribute_id": row.get("attribute_id"),
                "attribute_name": row.get("attribute_name"),
                "claim_text": quote,
                "srt_file": os.path.basename(first.srt_file),
                "srt_path": first.srt_file,
                "start_ts": first.start_ts,
                "end_ts": last.end_ts,
                "char_start": gstart,
                "char_end": gend,
                "cue_span_count": len(spans),
                "_b1_backend": "productv2_comment_triggered_reextract",
                "_queue_priority": row.get("priority"),
                "_queue_priority_score": row.get("priority_score"),
                "_queue_reason": item.get("reason", ""),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/productv2_comment_triggered_claim_reextract_queue_20260613_strict.jsonl")
    ap.add_argument("--out", default="data/processed/stageB_product_v2/claim_reextract_productv2_strict_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/claim_reextract_productv2_strict_20260613_report.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--priority", action="append", default=None)
    ap.add_argument("--chunk_chars", type=int, default=2500)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--fact_records", default="")
    ap.add_argument("--min_fact_coverage", type=int, default=0)
    args = ap.parse_args()

    rows = list(read_jsonl(args.queue))
    if args.priority:
        wanted = {str(x) for x in args.priority}
        rows = [r for r in rows if str(r.get("priority")) in wanted]
    facts = load_fact_records(args.fact_records)
    if facts:
        enriched = []
        for row in rows:
            fact = facts.get((str(row.get("product_id") or ""), str(row.get("attribute_id") or "")))
            coverage = int((fact or {}).get("coverage") or 0)
            if coverage < args.min_fact_coverage:
                continue
            out = dict(row)
            out["_fact_coverage"] = coverage
            out["_fact_snippets"] = evidence_snippets(fact)
            enriched.append(out)
        rows = enriched
    if args.limit:
        rows = rows[: args.limit]
    model = args.model or os.environ.get("CLAIMARC_TEXT_MODEL", "Qwen-Flash")

    print(f"[b1_reextract_from_queue] queue_rows={len(rows)} model={model}")

    def job(row: dict[str, Any]):
        claims = extract_row(
            row,
            chunk_chars=args.chunk_chars,
            model=model,
            max_tokens=args.max_tokens,
        )
        return {"row": row, "claims": claims}

    results = llm.run_many(
        rows,
        job,
        concurrency=int(os.environ.get("CLAIMARC_CONCURRENCY", "4")),
        desc="B1-reextract",
    )

    claims: list[dict[str, Any]] = []
    errors = 0
    for res in results:
        if isinstance(res, dict) and "__error__" in res:
            errors += 1
            continue
        if not isinstance(res, dict):
            continue
        claims.extend(res.get("claims") or [])
    write_jsonl(args.out, claims)
    report = {
        "queue": args.queue,
        "out": args.out,
        "queue_rows": len(rows),
        "claims": len(claims),
        "pairs_with_claims": len({c.get("pair_id") for c in claims}),
        "products_with_claims": len({c.get("product_id") for c in claims}),
        "errors": errors,
        "priority": args.priority or [],
        "model": model,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
