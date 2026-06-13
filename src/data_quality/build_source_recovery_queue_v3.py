"""Build a broad source-recovery queue for the current strict dataset.

Version 2 intentionally targeted a narrow high-precision subset.  This queue is
broader: it covers every claimful pair whose product evidence is absent (and,
optionally, weak single-source pairs) so that LLM/VLM verification can decide
whether the issue is missing evidence, an out-of-scope attribute, or an
unusable claim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from data_quality.audit_dataset_quality import has_claim, read_jsonl, source_count
from data_quality.build_regeneration_queues_v2 import (
    accept_rule,
    bundle_fields,
    claim_preview,
    claim_segments,
    clean,
    target_sources,
)


EVAL_TERMS = set(getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])) | {
    "好看",
    "满意",
    "推荐",
    "回购",
    "复购",
    "性价比",
    "值得",
}

OBJECTIVE_NUMERIC = {
    "尺码", "尺寸", "规格", "净含量", "容量", "重量", "数量", "件数", "功率",
    "输出功率", "电池容量", "保质期", "长度", "宽度", "高度", "厚度", "克重",
    "含量", "比例", "浓度", "蓬松度", "价格", "到手价", "券后价",
}

OBJECTIVE_MATERIAL = {
    "材质", "面料", "面料成分", "成分", "含绒量", "充绒量", "绒子", "纯棉",
    "真皮", "杯具材质", "配料", "原料",
}

OBJECTIVE_VISUAL = {
    "颜色", "颜色分类", "图案", "款式", "版型", "风格", "闭合方式", "口袋",
    "内兜", "工艺", "包装", "包装方式", "包装类型", "适用季节", "主要功能",
    "功能", "是否加绒", "是否防水", "防水", "加绒",
}

BOOLEAN_HINTS = {"是否", "有无", "能否", "可", "防水", "加绒", "内兜"}


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


def attr_text(rec: dict[str, Any]) -> str:
    return f"{clean(rec.get('attribute_name'))} {clean(rec.get('attribute_id'))}"


def contains(text: str, terms: set[str]) -> bool:
    return any(term and term in text for term in terms)


def classify_scope(rec: dict[str, Any]) -> tuple[str, str, str]:
    text = attr_text(rec)
    if contains(text, EVAL_TERMS):
        return "scope_review_eval_leakage", "other", "attribute may encode evaluation rather than product fact"
    if contains(text, OBJECTIVE_MATERIAL):
        return "objective_material", "material", ""
    if contains(text, OBJECTIVE_NUMERIC):
        if contains(text, BOOLEAN_HINTS):
            return "objective_boolean", "boolean", ""
        if "尺码" in text or "尺寸" in text:
            return "objective_numeric", "size", ""
        if "件数" in text or "数量" in text or "口袋数量" in text:
            return "objective_numeric", "count", ""
        return "objective_numeric", "number", ""
    if contains(text, OBJECTIVE_VISUAL):
        if contains(text, BOOLEAN_HINTS):
            return "objective_boolean", "boolean", ""
        if "颜色" in text:
            return "objective_visual", "color", ""
        return "objective_visual", "visual", ""
    return "scope_review_uncertain_attribute", "other", "attribute needs product-attribute validity review"


def priority(rec: dict[str, Any], scope: str, src_count: int) -> str:
    y = int(rec.get("y", 0) or 0)
    audit = rec.get("label_audit") or {}
    n_neg = int(float(audit.get("n_neg_aligned", 0) or 0))
    n_aligned = int(float(audit.get("n_aligned", 0) or 0))
    if src_count == 0 and y == 1 and scope.startswith("objective"):
        return "P0"
    if src_count == 0 and n_neg > 0:
        return "P0"
    if src_count == 0 and scope.startswith("objective"):
        return "P1"
    if src_count == 0:
        return "P2"
    if src_count == 1 and (y == 1 or n_neg > 0 or n_aligned >= 2):
        return "P3"
    return "P4"


def make_rows(dataset: list[dict[str, Any]], bundles: dict[str, pidx.ProductBundle], include_source1: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in dataset:
        if not has_claim(rec):
            continue
        sc = source_count(rec)
        if sc > 0 and not include_source1:
            continue
        if sc > 1:
            continue
        scope, value_type, scope_note = classify_scope(rec)
        pid = str(rec.get("product_id"))
        bfields = bundle_fields(pid, bundles)
        queue_type = "direct_product_source0" if sc == 0 else "weak_source_expand"
        srcs = target_sources("direct_product_source0", scope if scope.startswith("objective") else "objective_visual", value_type)
        row = {
            "queue_type": queue_type,
            "priority": priority(rec, scope, sc),
            "pair_id": rec.get("pair_id"),
            "product_id": pid,
            "category": rec.get("category"),
            "subcategory": rec.get("subcategory"),
            "room_id": rec.get("room_id"),
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": clean(rec.get("attribute_name")),
            **bfields,
            "attribute_objectivity": scope,
            "expected_value_type": value_type,
            "attribute_scope_note": scope_note,
            "target_sources": srcs,
            "claim_segments": claim_segments(rec),
            "claim_preview": claim_preview(rec),
            "risk_comment_example": "",
            "current_source_count": sc,
            "current_confidence": rec.get("confidence"),
            "current_label": int(rec.get("y", 0) or 0),
            "current_weight": rec.get("c"),
            "n_aligned": int(float((rec.get("label_audit") or {}).get("n_aligned", 0) or 0)),
            "n_neg_aligned": int(float((rec.get("label_audit") or {}).get("n_neg_aligned", 0) or 0)),
            "accept_rule": accept_rule(),
        }
        rows.append(row)
    rows.sort(key=lambda r: (
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(str(r.get("priority")), 9),
        0 if r.get("queue_type") == "direct_product_source0" else 1,
        -int(r.get("current_label", 0) or 0),
        str(r.get("pair_id", "")),
    ))
    return rows


def summarize(rows: list[dict[str, Any]], dataset_path: str) -> dict[str, Any]:
    return {
        "dataset": dataset_path,
        "n": len(rows),
        "queue_type": dict(Counter(str(r.get("queue_type")) for r in rows)),
        "priority": dict(Counter(str(r.get("priority")) for r in rows)),
        "objectivity": dict(Counter(str(r.get("attribute_objectivity")) for r in rows)),
        "value_type": dict(Counter(str(r.get("expected_value_type")) for r in rows)),
        "label": dict(Counter(str(r.get("current_label")) for r in rows)),
        "source_count": dict(Counter(str(r.get("current_source_count")) for r in rows)),
        "top_attributes": Counter(str(r.get("attribute_name")) for r in rows).most_common(40),
        "top_p0": [
            {
                "pair_id": r.get("pair_id"),
                "attribute": r.get("attribute_name"),
                "label": r.get("current_label"),
                "scope": r.get("attribute_objectivity"),
                "product": r.get("product_title"),
                "claim": r.get("claim_preview"),
                "target_sources": r.get("target_sources"),
            }
            for r in rows if r.get("priority") == "P0"
        ][:50],
    }


def write_markdown(report: dict[str, Any], path: str | Path, queue_path: str) -> None:
    lines = [
        "# Source Recovery Queue v3",
        "",
        f"- source dataset: `{report['dataset']}`",
        f"- queue: `{queue_path}`",
        f"- rows: `{report['n']}`",
        f"- queue_type: `{report['queue_type']}`",
        f"- priority: `{report['priority']}`",
        f"- objectivity: `{report['objectivity']}`",
        f"- labels: `{report['label']}`",
        "",
        "## Interpretation",
        "This broad queue is intended to separate three failure modes: true missing",
        "product evidence, attribute-scope leakage, and weak single-source product",
        "evidence. P0/P1 rows should be verified first with product params, OCR, and",
        "detail-image VLM grounding before changing labels or filtering rows.",
        "",
        "## P0 Examples",
        "| pair_id | label | scope | attribute | product |",
        "|---|---:|---|---|---|",
    ]
    for r in report["top_p0"][:25]:
        lines.append(
            f"| `{r['pair_id']}` | {r['label']} | {r['scope']} | {r['attribute']} | {r['product']} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0p1adjudicated_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/source_recovery_queue_v3.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/source_recovery_queue_v3_report.json")
    ap.add_argument("--md", default="docs/SOURCE_RECOVERY_QUEUE_V3.md")
    ap.add_argument("--include_source1", action="store_true")
    args = ap.parse_args()

    rows = make_rows(read_jsonl(args.dataset), pidx.build_bundles(), args.include_source1)
    report = summarize(rows, args.dataset)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    write_markdown(report, args.md, args.out)
    print(f"[build_source_recovery_queue_v3] wrote {args.out}")
    print(json.dumps({k: report[k] for k in ("n", "queue_type", "priority", "objectivity", "label")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
