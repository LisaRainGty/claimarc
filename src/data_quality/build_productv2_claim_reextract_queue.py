"""Build a product-v2 claim re-extraction queue from comment triggers.

The legacy repair queue is useful, but it was built over the old broad
``A_cmt(p)``.  This script starts from the contracted product-v2 schema and
collects only pairs whose current validated B1 output has no direct livestream
claim, while product-v2 consumer mentions still suggest a missed seller claim.

It is deterministic: no model calls, no mutation of existing datasets.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import read_json, read_jsonl, write_json, write_jsonl


TRIGGER_PATTERNS = [
    "主播", "直播", "宣传", "说是", "说的", "说好", "承诺", "标的", "标注",
    "写的", "详情", "页面", "虚标", "不符", "不一样", "不一致", "缩水",
    "夸大", "骗人", "假", "不是", "没说", "没宣传", "跟.*不符", "和.*不一样",
]

NEGATIVE_HINTS = {
    "不符", "不一样", "不一致", "缩水", "虚标", "骗人", "假", "不是",
    "没", "差", "刺", "硬", "薄", "小", "大", "贵", "漏", "掉", "坏",
}

SOURCE_PRIORITY = {
    "numeric": 18,
    "material": 16,
    "visual_or_boolean": 14,
    "identity_or_spec": 13,
    "direct_text_match": 10,
    "objective_name_only": 6,
}

UNSPOKEN_OR_PROMO_ATTR_TERMS = {
    "商品条形码", "商品条码", "条形码", "条码", "商家编码", "商品编码",
    "价格", "商品价格", "单价", "价位", "到手价", "优惠", "优惠券",
    "链接", "物流", "发货", "售后", "库存",
}


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def pair_id(pid: str, aid: str) -> str:
    return f"p{pid}__{aid}"


def trigger_hits(text: str) -> list[str]:
    out: list[str] = []
    for pat in TRIGGER_PATTERNS:
        try:
            if re.search(pat, text):
                out.append(pat)
        except re.error:
            if pat in text:
                out.append(pat)
    return out


def negative_hint_hits(text: str) -> list[str]:
    return sorted(t for t in NEGATIVE_HINTS if t and t in text)


def bundle_fields(bundle: pidx.ProductBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {
            "category": "",
            "subcategory": "",
            "room_id": "UNKNOWN",
            "product_title": "",
            "raw_image_dir": "",
            "detail_images": [],
            "srt_files": [],
            "comment_files": [],
        }
    detail_images = [str(pidx.resolve(p)) for p in bundle.detail_images]
    raw_image_dir = str(Path(detail_images[0]).parent) if detail_images else ""
    return {
        "category": bundle.category,
        "subcategory": bundle.subcategory,
        "room_id": bundle.room_id,
        "product_title": bundle.title,
        "raw_image_dir": raw_image_dir,
        "detail_images": detail_images,
        "srt_files": [str(pidx.resolve(p)) for p in bundle.srt_files],
        "comment_files": [str(pidx.resolve(p)) for p in bundle.comment_files],
    }


def target_sources(source_family: str) -> list[str]:
    if source_family == "numeric":
        return ["srt", "product_title", "params", "detail_image_ocr"]
    if source_family == "material":
        return ["srt", "params", "detail_image_ocr", "detail_image_vlm"]
    if source_family == "visual_or_boolean":
        return ["srt", "detail_image_ocr", "detail_image_vlm", "params"]
    if source_family == "identity_or_spec":
        return ["srt", "product_title", "params", "detail_image_ocr"]
    return ["srt", "product_title", "params", "detail_image_ocr"]


def value_type(source_family: str) -> str:
    return {
        "numeric": "number_or_range",
        "material": "material_or_ingredient",
        "visual_or_boolean": "visual_or_boolean",
        "identity_or_spec": "identity_or_spec",
        "direct_text_match": "text_match",
    }.get(source_family, "attribute_value")


def is_srt_reextractable_attribute(meta: dict[str, Any]) -> tuple[bool, str]:
    """Whether a product-v2 pair is suitable for comment-triggered SRT repair."""
    source_family = clean(meta.get("source_family"))
    name = clean(meta.get("canonical_name"))
    aliases = {clean(x) for x in (meta.get("aliases") or []) if clean(x)}
    attr_terms = {name, *aliases}
    if attr_terms & UNSPOKEN_OR_PROMO_ATTR_TERMS:
        return False, "unspoken_or_promo_attribute"
    if source_family == "identity_or_spec" and not (meta.get("direct_raw_hits") or []):
        return False, "identity_without_product_raw_hit"
    return True, ""


def load_legacy_reviews(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    reviews_by_pair: dict[str, list[dict[str, Any]]] = {}
    p = Path(path)
    if not p.exists():
        return reviews_by_pair
    for rec in read_jsonl(p):
        pid = str(rec.get("product_id") or "")
        aid = str(rec.get("attribute_id") or "")
        if not pid or not aid:
            continue
        rows = []
        for c in rec.get("reviews") or []:
            text = clean(c.get("text") or c.get("review_text") or c.get("evidence_span"))
            if not text:
                continue
            rows.append({
                "comment_id": c.get("comment_id") or c.get("review_id"),
                "text": text,
                "polarity": c.get("polarity") or c.get("review_polarity"),
                "review_polarity": c.get("review_polarity"),
                "explicit_fact_hit": bool(c.get("explicit_fact_hit", False)),
                "mention_strength": c.get("mention_strength"),
                "review_time": c.get("review_time", ""),
                "_review_source": "legacy_pair_records",
            })
        if rows:
            reviews_by_pair[pair_id(pid, aid)] = rows
    return reviews_by_pair


def load_productv2_mentions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in read_jsonl(path):
        pid = str(r.get("product_id") or "")
        aid = str(r.get("attribute_id") or "")
        text = clean(r.get("review_text") or r.get("text") or r.get("evidence_span"))
        if not pid or not aid or not text:
            continue
        mentions[pair_id(pid, aid)].append({
            "comment_id": r.get("comment_id") or r.get("review_id"),
            "text": text,
            "polarity": r.get("polarity"),
            "review_polarity": r.get("review_polarity"),
            "explicit_fact_hit": bool(r.get("explicit_fact_hit", False)),
            "mention_strength": r.get("mention_strength"),
            "review_time": r.get("review_time", ""),
            "_was_free": r.get("_was_free", ""),
            "_review_source": "productv2_stagea_mentions",
        })
    return mentions


def load_pair_skeleton(path: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        out[str(r.get("pair_id") or "")] = r
    return out


def dedup_reviews(reviews: list[dict[str, Any]], cap: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in reviews:
        text = clean(r.get("text"))
        if not text:
            continue
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        row = dict(r)
        row["text"] = text[:300]
        row["trigger_hits"] = trigger_hits(text)
        row["negative_hint_hits"] = negative_hint_hits(text)
        out.append(row)
        if len(out) >= cap:
            break
    return out


def triggered_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = []
    for r in reviews:
        text = clean(r.get("text"))
        hits = trigger_hits(text)
        if r.get("explicit_fact_hit") or hits:
            row = dict(r)
            row["trigger_hits"] = hits
            row["negative_hint_hits"] = negative_hint_hits(text)
            keep.append(row)
    keep.sort(key=lambda r: (
        not bool(r.get("explicit_fact_hit")),
        str(r.get("polarity")) != "neg",
        -len(r.get("trigger_hits") or []),
        -len(r.get("negative_hint_hits") or []),
        str(r.get("comment_id")),
    ))
    return dedup_reviews(keep)


def strict_triggered_count(triggered: list[dict[str, Any]]) -> int:
    return sum(1 for r in triggered if r.get("trigger_hits"))


def priority_score(
    *,
    source_family: str,
    triggered: list[dict[str, Any]],
    meta: dict[str, Any],
) -> int:
    score = SOURCE_PRIORITY.get(source_family, 6)
    score += min(24, len(triggered) * 4)
    score += 16 if any(r.get("polarity") == "neg" for r in triggered) else 0
    score += 14 if any(r.get("explicit_fact_hit") for r in triggered) else 0
    score += min(12, sum(len(r.get("trigger_hits") or []) for r in triggered) * 2)
    score += min(10, sum(len(r.get("negative_hint_hits") or []) for r in triggered))
    score += 8 if meta.get("direct_raw_hits") else 0
    score += 4 if float(meta.get("selection_score") or 0) >= 10 else 0
    return int(score)


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acmt = read_json(args.acmt, default={}) or {}
    skeleton = load_pair_skeleton(args.pair_skeleton)
    bundles = pidx.build_bundles()
    legacy_reviews = load_legacy_reviews(args.legacy_pair_records)
    mention_reviews = load_productv2_mentions(args.productv2_mentions)
    rows: list[dict[str, Any]] = []
    skipped = Counter()

    for pid, attrs in acmt.items():
        bfields = bundle_fields(bundles.get(str(pid)))
        for aid, meta in attrs.items():
            pid_s = str(pid)
            aid_s = str(aid)
            rid = pair_id(pid_s, aid_s)
            sk = skeleton.get(rid, {})
            if sk.get("has_claim_srt") and not args.include_claimful:
                skipped["has_direct_claim"] += 1
                continue
            ok_reextract, skip_reason = is_srt_reextractable_attribute(meta)
            if not ok_reextract:
                skipped[skip_reason] += 1
                continue
            reviews = legacy_reviews.get(rid) or mention_reviews.get(rid) or []
            trig = triggered_reviews(reviews)
            if len(trig) < args.min_trigger_hits:
                skipped["no_comment_trigger"] += 1
                continue
            n_text_triggers = strict_triggered_count(trig)
            if n_text_triggers < args.min_text_trigger_hits:
                skipped["no_text_trigger"] += 1
                continue
            source_family = clean(meta.get("source_family"))
            score = priority_score(source_family=source_family, triggered=trig, meta=meta)
            if score < args.min_priority_score:
                skipped["low_priority_score"] += 1
                continue
            priority = "P0" if score >= 64 else "P1" if score >= 44 else "P2"
            rows.append({
                "queue_type": "productv2_comment_triggered_srt_reextract",
                "priority": priority,
                "priority_score": score,
                "pair_id": rid,
                "product_id": pid_s,
                "attribute_id": aid_s,
                "attribute_name": meta.get("canonical_name") or sk.get("attribute_canonical") or aid_s,
                "aliases": list(meta.get("aliases") or [])[:30],
                "source_family": source_family,
                "expected_value_type": value_type(source_family),
                "target_sources": target_sources(source_family),
                "has_current_direct_claim": bool(sk.get("has_claim_srt")),
                "current_claim_preview": clean(sk.get("passage"))[:500],
                "selection_score": meta.get("selection_score"),
                "direct_raw_hits": list(meta.get("direct_raw_hits") or [])[:20],
                **bfields,
                "trigger_count": len(trig),
                "text_trigger_count": n_text_triggers,
                "triggered_reviews": trig,
                "accept_rule": {
                    "claim": "must find an exact SRT substring that states this attribute for the product",
                    "attribute": "must match the product-v2 attribute, not a nearby broad/evaluative attribute",
                    "product_evidence": "must later be verified from title/params/OCR/VLM, never from reviews",
                    "consumer": "comment must be aligned to the same atomic claim before becoming a clean label",
                },
            })

    rows.sort(key=lambda r: (
        {"P0": 0, "P1": 1, "P2": 2}.get(str(r.get("priority")), 9),
        -int(r.get("priority_score") or 0),
        -int(r.get("trigger_count") or 0),
        str(r.get("pair_id")),
    ))
    if args.limit:
        rows = rows[: args.limit]

    report = {
        "acmt": args.acmt,
        "pair_skeleton": args.pair_skeleton,
        "productv2_mentions": args.productv2_mentions,
        "legacy_pair_records": args.legacy_pair_records,
        "n": len(rows),
        "products": len({r["product_id"] for r in rows}),
        "pairs": len({r["pair_id"] for r in rows}),
        "priority": dict(Counter(str(r.get("priority")) for r in rows)),
        "source_family": dict(Counter(str(r.get("source_family")) for r in rows)),
        "category": dict(Counter(str(r.get("category")) for r in rows).most_common(20)),
        "top_attributes": Counter(str(r.get("attribute_name")) for r in rows).most_common(30),
        "skipped": dict(skipped),
        "min_trigger_hits": args.min_trigger_hits,
        "min_text_trigger_hits": args.min_text_trigger_hits,
        "min_priority_score": args.min_priority_score,
    }
    return rows, report


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Product-v2 Comment-triggered Claim Re-extraction Queue",
        "",
        "## Summary",
        f"- rows: `{report['n']}`",
        f"- products: `{report['products']}`",
        f"- pairs: `{report['pairs']}`",
        f"- priority: `{report['priority']}`",
        f"- source_family: `{report['source_family']}`",
        f"- skipped: `{report['skipped']}`",
        "",
        "## Use",
        "Use this queue before expanding the clean benchmark. Each row requires an",
        "SRT-grounded claim re-extraction pass, then product-evidence verification,",
        "then atomic consumer-signal alignment. Rows should not enter clean training",
        "until all three states are available.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default="data/processed/stageB_product_v2/acmt_product_v2.json")
    ap.add_argument("--pair_skeleton", default="data/processed/stageB_product_v2/pair_skeleton_product_v2.jsonl")
    ap.add_argument("--productv2_mentions", default="data/processed/stageB_product_v2/resolved_aspects_product_v2.jsonl")
    ap.add_argument("--legacy_pair_records", default="data/processed/stageB/pair_records.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/productv2_comment_triggered_claim_reextract_queue_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/productv2_comment_triggered_claim_reextract_queue_20260613_report.json")
    ap.add_argument("--md", default="docs/PRODUCTV2_COMMENT_TRIGGERED_CLAIM_REEXTRACT_20260613.md")
    ap.add_argument("--min_trigger_hits", type=int, default=1)
    ap.add_argument(
        "--min_text_trigger_hits",
        type=int,
        default=0,
        help=(
            "Minimum reviews containing explicit text triggers such as "
            "直播/宣传/虚标/不符. Use 1 for strict clean expansion."
        ),
    )
    ap.add_argument("--min_priority_score", type=int, default=34)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include_claimful", action="store_true")
    args = ap.parse_args()

    rows, report = build(args)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    write_markdown(report, args.md)
    print(json.dumps({
        "out": args.out,
        "rows": report["n"],
        "products": report["products"],
        "priority": report["priority"],
        "source_family": report["source_family"],
        "skipped": report["skipped"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
