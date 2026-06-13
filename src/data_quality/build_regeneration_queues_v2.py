"""Build prioritized regeneration queues from raw-backed CLAIMARC records.

The queue is intentionally narrow. It targets:

1. source-absent product-attribute pairs that still look like objective product
   attributes and should be re-checked against params/OCR/detail images;
2. missing-claim-risk pairs that should first re-extract SRT claims before
   product evidence verification.

This script does not call any model and does not alter training data.
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
from common.io_utils import normalize
from data_quality.audit_dataset_quality import has_claim, read_jsonl, source_count


DROP_TERMS = {
    "售后", "客服", "物流", "发货", "退换", "保价", "赠品", "试用", "试用装",
    "主播", "直播", "讲解", "宣传", "购买价值", "性价比", "喜好", "好看",
    "颜值", "满意", "满意度", "舒适", "舒适度", "质量", "功能", "主要功能",
    "工艺", "款式", "风格", "是否瑕疵", "瑕疵", "商品真实性", "真实性",
    "客观属性名词短语", "推荐", "回购", "穿搭", "适合肤质",
}

PRICE_TERMS = {"价格", "券后价", "到手价", "优惠", "折扣"}

NUMERIC_TERMS = {
    "尺码", "尺寸", "规格", "净含量", "容量", "重量", "数量", "件数", "口袋数量",
    "功率", "输出功率", "电池容量", "保质期", "长度", "宽度", "高度", "厚度",
    "克重", "含量", "比例", "浓度", "蓬松度",
}

MATERIAL_TERMS = {
    "材质", "面料", "面料成分", "面料成分含量", "成分", "含绒量", "充绒量",
    "绒子", "纯棉", "真皮", "杯具材质",
}

VISUAL_TERMS = {
    "颜色", "颜色分类", "图案", "内兜", "口袋", "闭合方式", "是否加绒",
    "加绒", "是否防水", "防水", "商品等级",
}

BOOLEAN_TERMS = {
    "是否", "有无", "能否", "可调节", "防水", "加绒", "内兜",
}


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


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def attr_text(rec: dict[str, Any]) -> str:
    return f"{clean(rec.get('attribute_name'))} {rec.get('attribute_id', '')}"


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def classify_attribute(rec: dict[str, Any]) -> tuple[str, str]:
    text = attr_text(rec)
    if contains_any(text, PRICE_TERMS):
        return "drop_price_dynamic", "price"
    if contains_any(text, DROP_TERMS):
        return "drop_subjective_or_service", "other"
    if contains_any(text, MATERIAL_TERMS):
        return "objective_material", "material"
    if contains_any(text, NUMERIC_TERMS):
        if contains_any(text, BOOLEAN_TERMS):
            return "objective_boolean", "boolean"
        if any(t in text for t in {"尺码", "尺寸"}):
            return "objective_numeric", "size"
        if any(t in text for t in {"件数", "数量", "口袋数量"}):
            return "objective_numeric", "count"
        return "objective_numeric", "number"
    if contains_any(text, VISUAL_TERMS):
        if any(t in text for t in {"是否", "有无", "能否", "防水", "加绒", "内兜"}):
            return "objective_boolean", "boolean"
        if "颜色" in text:
            return "objective_visual", "color"
        return "objective_visual", "visual"
    return "drop_uncertain_attribute", "other"


def claim_preview(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    text = str(claim.get("passage", "") or "")
    if not text:
        text = " ".join(str(s.get("text", "") or "") for s in claim.get("segments", [])[:3])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def claim_segments(rec: dict[str, Any]) -> list[dict[str, Any]]:
    segs = (rec.get("claim") or {}).get("segments") or []
    out = []
    for seg in segs[:8]:
        out.append({
            "claim_id": seg.get("claim_id"),
            "clip_id": seg.get("clip_id"),
            "start_ts": seg.get("start_ts"),
            "end_ts": seg.get("end_ts"),
            "text": seg.get("text"),
        })
    return out


def resolved_paths(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        rp = pidx.resolve(p)
        try:
            out.append(str(rp.relative_to(config.ROOT)))
        except ValueError:
            out.append(str(rp))
    return out


def bundle_fields(pid: str, bundles: dict[str, pidx.ProductBundle]) -> dict[str, Any]:
    bundle = bundles.get(pid)
    if not bundle:
        image_dir = config.RAW / "product_images" / pid
        return {
            "product_title": "",
            "raw_image_dir": str(image_dir.relative_to(config.ROOT)),
            "detail_images": [],
            "srt_files": [],
            "comment_files": [],
        }
    image_dir = config.RAW / "product_images" / pid
    return {
        "product_title": bundle.title,
        "raw_image_dir": str(image_dir.relative_to(config.ROOT)),
        "detail_images": resolved_paths(bundle.detail_images[:20]),
        "srt_files": resolved_paths(bundle.srt_files),
        "comment_files": resolved_paths(bundle.comment_files),
    }


def target_sources(queue_type: str, objectivity: str, value_type: str) -> list[str]:
    if queue_type == "missing_claim_srt_first":
        base = ["srt", "params", "detail_image_ocr"]
        if objectivity in {"objective_visual", "objective_material", "objective_boolean"}:
            base.append("detail_image_vlm")
        return base
    if objectivity == "objective_numeric":
        return ["params", "product_title", "detail_image_ocr", "srt"]
    if objectivity in {"objective_visual", "objective_material", "objective_boolean"}:
        return ["detail_image_vlm", "detail_image_ocr", "params", "srt"]
    return ["params", "detail_image_ocr", "srt"]


def audit_value(rec: dict[str, Any], key: str) -> float:
    return float((rec.get("label_audit") or {}).get(key, 0) or 0)


def direct_priority(rec: dict[str, Any], objectivity: str) -> str:
    y = int(rec.get("y", 0) or 0)
    n_neg = audit_value(rec, "n_neg_aligned")
    n_aligned = audit_value(rec, "n_aligned")
    if objectivity in {"objective_material", "objective_numeric", "objective_boolean"} and (y == 1 or n_neg >= 1):
        return "P0"
    if objectivity == "objective_visual" and (y == 1 or n_neg >= 1):
        return "P1"
    if n_aligned >= 2:
        return "P2"
    return "P3"


def missing_priority(item: dict[str, Any]) -> str:
    hits = int(item.get("missing_claim_hits", 0) or 0)
    if hits >= 7:
        return "P0"
    if hits >= 3:
        return "P1"
    if hits >= 2:
        return "P2"
    return "P3"


def accept_rule() -> dict[str, str]:
    return {
        "required_fields": "evidence_found, source_type, raw_text, normalized_value, path_or_clip_id, timestamp_or_image, relation_to_claim, confidence, reject_reason",
        "positive_evidence": "raw_text must be copied from params/OCR/SRT or be a concise VLM observation grounded in an image path",
        "reject": "reject service, subjective, price-only dynamic coupon, unsupported visual inference, or evidence not tied to the requested attribute",
    }


def make_direct_rows(dataset: list[dict[str, Any]], bundles: dict[str, pidx.ProductBundle]) -> list[dict[str, Any]]:
    rows = []
    for rec in dataset:
        if not has_claim(rec) or source_count(rec) != 0:
            continue
        objectivity, value_type = classify_attribute(rec)
        if objectivity.startswith("drop_"):
            continue
        pid = str(rec.get("product_id"))
        bfields = bundle_fields(pid, bundles)
        rows.append({
            "queue_type": "direct_product_source0",
            "priority": direct_priority(rec, objectivity),
            "pair_id": rec.get("pair_id"),
            "product_id": pid,
            "category": rec.get("category"),
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": clean(rec.get("attribute_name")),
            **bfields,
            "attribute_objectivity": objectivity,
            "expected_value_type": value_type,
            "target_sources": target_sources("direct_product_source0", objectivity, value_type),
            "claim_segments": claim_segments(rec),
            "claim_preview": claim_preview(rec),
            "risk_comment_example": "",
            "current_source_count": source_count(rec),
            "current_confidence": rec.get("confidence"),
            "n_aligned": int(audit_value(rec, "n_aligned")),
            "n_neg_aligned": int(audit_value(rec, "n_neg_aligned")),
            "missing_claim_hits": 0,
            "accept_rule": accept_rule(),
        })
    return rows


def make_missing_rows(manifest: list[dict[str, Any]], bundles: dict[str, pidx.ProductBundle]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in manifest:
        if item.get("quality_bucket") != "missing_claim_risk":
            continue
        objectivity, value_type = classify_attribute(item)
        if objectivity.startswith("drop_"):
            continue
        pair_id = str(item.get("pair_id"))
        if pair_id in seen:
            continue
        seen.add(pair_id)
        pid = str(item.get("product_id"))
        bfields = bundle_fields(pid, bundles)
        rows.append({
            "queue_type": "missing_claim_srt_first",
            "priority": missing_priority(item),
            "pair_id": pair_id,
            "product_id": pid,
            "category": item.get("category"),
            "attribute_id": item.get("attribute_id"),
            "attribute_name": clean(item.get("attribute_name")),
            **bfields,
            "attribute_objectivity": objectivity,
            "expected_value_type": value_type,
            "target_sources": target_sources("missing_claim_srt_first", objectivity, value_type),
            "claim_segments": [],
            "claim_preview": str(item.get("claim_preview", "") or "")[:300],
            "risk_comment_example": str(item.get("missing_claim_example", "") or "")[:300],
            "current_source_count": int(item.get("source_count", 0) or 0),
            "current_confidence": item.get("confidence"),
            "n_aligned": 0,
            "n_neg_aligned": 0,
            "missing_claim_hits": int(item.get("missing_claim_hits", 0) or 0),
            "accept_rule": accept_rule(),
        })
    return rows


def sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    p_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(str(row.get("priority")), 9)
    type_rank = 0 if row.get("queue_type") == "direct_product_source0" else 1
    strength = int(row.get("missing_claim_hits", 0) or 0) + int(row.get("n_neg_aligned", 0) or 0) * 5
    return (p_rank, type_rank, -strength, str(row.get("pair_id", "")))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "queue_type": dict(Counter(str(r.get("queue_type")) for r in rows)),
        "priority": dict(Counter(str(r.get("priority")) for r in rows)),
        "objectivity": dict(Counter(str(r.get("attribute_objectivity")) for r in rows)),
        "value_type": dict(Counter(str(r.get("expected_value_type")) for r in rows)),
        "target_sources": dict(Counter("+".join(r.get("target_sources") or []) for r in rows)),
        "top_p0": [
            {
                "pair_id": r.get("pair_id"),
                "queue_type": r.get("queue_type"),
                "attribute_name": r.get("attribute_name"),
                "product_title": r.get("product_title"),
                "target_sources": r.get("target_sources"),
            }
            for r in rows if r.get("priority") == "P0"
        ][:30],
    }


def write_markdown(report: dict[str, Any], out: str | Path, queue_path: str) -> None:
    lines = [
        "# Regeneration Queues v2",
        "",
        f"- queue: `{queue_path}`",
        "",
        "## Summary",
        f"- n: `{report['n']}`",
        f"- queue_type: `{report['queue_type']}`",
        f"- priority: `{report['priority']}`",
        f"- objectivity: `{report['objectivity']}`",
        f"- value_type: `{report['value_type']}`",
        "",
        "## P0 Examples",
        "| pair_id | type | attribute | product | target sources |",
        "|---|---|---|---|---|",
    ]
    for r in report["top_p0"][:20]:
        lines.append(
            f"| `{r['pair_id']}` | {r['queue_type']} | {r['attribute_name']} | "
            f"{r['product_title']} | `{r['target_sources']}` |"
        )
    lines += [
        "",
        "## Use",
        "Run P0/P1 first. For `missing_claim_srt_first`, extract an exact SRT",
        "claim span before calling product evidence verification. For",
        "`direct_product_source0`, verify params/OCR/detail-image evidence before",
        "changing labels.",
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_v1.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/regeneration_manifest_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/regeneration_queue_v2.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/regeneration_queue_v2_report.json")
    ap.add_argument("--md", default="docs/REGENERATION_QUEUE_V2.md")
    args = ap.parse_args()

    dataset = read_jsonl(args.dataset)
    manifest = read_jsonl(args.manifest)
    bundles = pidx.build_bundles()
    rows = make_direct_rows(dataset, bundles) + make_missing_rows(manifest, bundles)
    rows.sort(key=sort_key)
    report = summarize(rows)

    write_jsonl(args.out, rows)
    write_json(args.report, report)
    write_markdown(report, args.md, args.out)
    print(f"[build_regeneration_queues_v2] wrote {args.out}")
    print(json.dumps({k: report[k] for k in ("n", "queue_type", "priority", "objectivity")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
