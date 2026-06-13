"""Build schema-remap and comment-triggered claim re-extraction queues.

These queues address upstream errors that source recovery cannot fix:

- overly broad A_cmt(p) candidate attributes;
- evaluative/service/live-process attributes leaking into product attributes;
- pairs with no extracted livestream claim despite review text explicitly
  mentioning a seller promise, false advertising, mismatch, or overstatement.

The script is deterministic and does not call any model.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from common.io_utils import read_jsonl
from data_quality.build_regeneration_queues_v2 import bundle_fields, clean


LEAK_TERMS = set(getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])) | {
    "主播", "直播", "讲解", "宣传", "购买渠道", "客户关系", "售后", "客服",
    "物流", "发货", "退换", "活动信息", "抽奖", "赠品", "喜好", "喜好度",
    "满意", "满意度", "推荐", "推荐度", "回购", "复购", "购买体验",
    "购买价值", "价廉物美", "物有所值", "值得购买", "客观属性名词短语",
    "建议类属性",
}

EXPLICIT_CLAIM_PATTERNS = [
    "主播", "直播", "宣传", "说是", "说的", "说好", "承诺", "标的", "标注",
    "写的", "虚标", "不符", "不一样", "不一致", "缩水", "骗人", "假",
    "不是", "没说", "没宣传", "跟.*不符", "和.*不一样",
]


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def attr_blob(attribute_id: str, meta: dict[str, Any]) -> str:
    aliases = " ".join(str(x) for x in meta.get("aliases", [])[:30])
    return f"{attribute_id} {meta.get('canonical_name', '')} {aliases}"


def leak_hits(text: str) -> list[str]:
    return sorted({t for t in LEAK_TERMS if t and t in text})


def review_trigger(text: str) -> list[str]:
    hits = []
    for pat in EXPLICIT_CLAIM_PATTERNS:
        try:
            if re.search(pat, text):
                hits.append(pat)
        except re.error:
            if pat in text:
                hits.append(pat)
    return hits


def schema_rows(acmt: dict[str, dict[str, Any]], bundles: dict[str, pidx.ProductBundle], high_cardinality: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, attrs in acmt.items():
        bfields = bundle_fields(str(pid), bundles)
        card = len(attrs)
        for aid, meta in attrs.items():
            text = attr_blob(aid, meta)
            hits = leak_hits(text)
            if not hits and card < high_cardinality:
                continue
            priority = "P0" if hits else "P2"
            if hits and card >= high_cardinality:
                priority = "P0"
            elif hits:
                priority = "P1"
            rows.append({
                "queue_type": "schema_remap_review",
                "priority": priority,
                "product_id": str(pid),
                "category": bundles[str(pid)].category if str(pid) in bundles else "",
                "attribute_id": aid,
                "canonical_name": meta.get("canonical_name"),
                "aliases": list(meta.get("aliases", []) or [])[:30],
                "acmt_cardinality": card,
                "leak_hits": hits,
                "product_title": bfields.get("product_title", ""),
                "repair_question": (
                    "drop_or_remap_evaluative_attribute"
                    if hits else "reduce_overwide_product_attribute_set"
                ),
            })
    rows.sort(key=lambda r: (
        {"P0": 0, "P1": 1, "P2": 2}.get(str(r["priority"]), 9),
        -int(r["acmt_cardinality"]),
        str(r["product_id"]),
        str(r["attribute_id"]),
    ))
    return rows


def claim_reextract_rows(pair_records: list[dict[str, Any]], bundles: dict[str, pidx.ProductBundle], min_hits: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in pair_records:
        claim = rec.get("claim") or {}
        if claim.get("has_claim_srt") and (claim.get("segments") or claim.get("passage")):
            continue
        reviews = rec.get("reviews") or []
        triggered = []
        for c in reviews:
            text = str(c.get("text") or c.get("evidence_span") or "")
            hits = review_trigger(text)
            if c.get("explicit_fact_hit") or hits:
                triggered.append({
                    "comment_id": c.get("comment_id"),
                    "text": text[:240],
                    "polarity": c.get("polarity"),
                    "explicit_fact_hit": bool(c.get("explicit_fact_hit")),
                    "hits": hits,
                })
        if len(triggered) < min_hits:
            continue
        pid = str(rec.get("product_id"))
        bfields = bundle_fields(pid, bundles)
        rows.append({
            "queue_type": "comment_triggered_srt_reextract",
            "priority": "P0" if any(t.get("polarity") == "neg" for t in triggered) else "P1",
            "pair_id": rec.get("pair_id"),
            "product_id": pid,
            "category": rec.get("category"),
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": rec.get("attribute_canonical"),
            **bfields,
            "trigger_count": len(triggered),
            "triggered_reviews": triggered[:12],
            "repair_question": "re-extract livestream claim using review-triggered attribute anchors",
        })
    rows.sort(key=lambda r: (
        {"P0": 0, "P1": 1}.get(str(r["priority"]), 9),
        -int(r["trigger_count"]),
        str(r["pair_id"]),
    ))
    return rows


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Schema And Claim Repair Queues v1",
        "",
        "## Summary",
        f"- schema queue rows: `{report['schema_queue']['n']}`",
        f"- claim re-extract rows: `{report['claim_reextract_queue']['n']}`",
        f"- A_cmt cardinality: `{report['acmt_cardinality']}`",
        f"- schema priorities: `{report['schema_queue']['priority']}`",
        f"- claim priorities: `{report['claim_reextract_queue']['priority']}`",
        "",
        "## Interpretation",
        "Use the schema queue to contract overly broad or evaluative attributes",
        "before rebuilding Stage B/C. Use the claim re-extraction queue only for",
        "pairs where review text explicitly suggests a missed livestream claim.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default="data/processed/stageB/acmt.json")
    ap.add_argument("--pair_records", default="data/processed/stageB/pair_records.jsonl")
    ap.add_argument("--schema_out", default="data/final/repaired_v1/schema_remap_review_queue_v1.jsonl")
    ap.add_argument("--claim_out", default="data/final/repaired_v1/comment_triggered_srt_reextract_queue_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/schema_claim_repair_queues_v1_report.json")
    ap.add_argument("--md", default="docs/SCHEMA_CLAIM_REPAIR_QUEUES_V1.md")
    ap.add_argument("--high_cardinality", type=int, default=50)
    ap.add_argument("--min_trigger_hits", type=int, default=1)
    args = ap.parse_args()

    bundles = pidx.build_bundles()
    acmt = read_json(args.acmt, default={}) or {}
    pairs = read_jsonl(args.pair_records)
    srows = schema_rows(acmt, bundles, args.high_cardinality)
    crows = claim_reextract_rows(pairs, bundles, args.min_trigger_hits)
    write_jsonl(args.schema_out, srows)
    write_jsonl(args.claim_out, crows)

    card = [len(v) for v in acmt.values()]
    card_sorted = sorted(card)
    report = {
        "acmt": args.acmt,
        "pair_records": args.pair_records,
        "acmt_cardinality": {
            "products": len(card),
            "mean": round(sum(card) / len(card), 2) if card else 0,
            "p50": card_sorted[int(0.50 * (len(card_sorted) - 1))] if card_sorted else 0,
            "p90": card_sorted[int(0.90 * (len(card_sorted) - 1))] if card_sorted else 0,
            "max": max(card) if card else 0,
            "high_cardinality_threshold": args.high_cardinality,
            "products_ge_threshold": sum(1 for x in card if x >= args.high_cardinality),
        },
        "schema_queue": {
            "path": args.schema_out,
            "n": len(srows),
            "priority": dict(Counter(str(r.get("priority")) for r in srows)),
            "top_leak_hits": Counter(h for r in srows for h in r.get("leak_hits", [])).most_common(30),
            "top_attributes": Counter(str(r.get("canonical_name")) for r in srows).most_common(30),
        },
        "claim_reextract_queue": {
            "path": args.claim_out,
            "n": len(crows),
            "priority": dict(Counter(str(r.get("priority")) for r in crows)),
            "top_attributes": Counter(str(r.get("attribute_name")) for r in crows).most_common(30),
        },
    }
    write_json(args.report, report)
    write_markdown(report, args.md)
    print("[build_schema_claim_repair_queues_v1] wrote queues")
    print(json.dumps({
        "schema_n": len(srows),
        "claim_reextract_n": len(crows),
        "acmt_cardinality": report["acmt_cardinality"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
