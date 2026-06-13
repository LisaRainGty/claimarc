"""Build deterministic candidate pools for the next high-quality expansion.

This script does not call any model and does not mutate training data.  It
collects the highest-value records for the next LLM/VLM pass:

1. review-triggered SRT re-extraction candidates whose comments explicitly
   mention a seller promise/mismatch but the current pipeline missed claims;
2. source recovery rows not covered by the strict P0 verification pass;
3. schema-remap rows that explain why A_cmt(p) became too broad;
4. auxiliary hard-negative/noisy rows that should not be used as clean labels
   but are useful for contrastive diagnostics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common.io_utils import read_jsonl
from data_quality.audit_dataset_quality import source_count


LEAK_OR_PROCESS_TERMS = {
    "物流", "发货", "客服", "售后", "退换", "主播", "直播", "讲解", "宣传",
    "购买渠道", "购买体验", "购买价值", "赠品", "抽奖", "推荐", "推荐度",
    "满意", "满意度", "喜好", "喜好度", "回购", "复购", "性价比", "划算",
    "值得购买", "整体评价", "总体评价", "客观属性名词短语",
}

OBJECTIVE_STRONG_TERMS = {
    "尺码", "尺寸", "规格", "净含量", "容量", "重量", "数量", "件数",
    "厚度", "长度", "宽度", "高度", "克重", "保质期", "材质", "面料",
    "成分", "含量", "含绒量", "充绒量", "绒子", "颜色", "颜色分类",
    "图案", "口袋", "内兜", "闭合方式", "是否加绒", "防水", "产地",
    "配料", "功率", "电池容量", "货号", "商品条形码", "型号",
}

NUMERIC_TERMS = {
    "尺码", "尺寸", "规格", "净含量", "容量", "重量", "数量", "件数",
    "厚度", "长度", "宽度", "高度", "克重", "保质期", "功率", "电池容量",
}

MATERIAL_TERMS = {"材质", "面料", "成分", "含量", "含绒量", "充绒量", "绒子", "配料"}

VISUAL_TERMS = {"颜色", "颜色分类", "图案", "款式", "包装", "包装方式", "闭合方式", "口袋", "内兜"}

BOOLEAN_TERMS = {"是否", "有无", "能否", "加绒", "防水", "内兜", "进口", "临期"}

CLAIM_TRIGGER_TERMS = {
    "主播", "直播", "宣传", "说是", "说的", "承诺", "标的", "标注",
    "写的", "虚标", "不符", "不一样", "不一致", "缩水", "骗人",
    "假", "没说", "没宣传",
}


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


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def attr_blob(row: dict[str, Any]) -> str:
    vals = [
        row.get("attribute_name"),
        row.get("canonical_name"),
        row.get("attribute_id"),
        " ".join(str(x) for x in (row.get("aliases") or [])[:20]),
    ]
    return " ".join(clean(v) for v in vals)


def hits(text: str, terms: set[str]) -> list[str]:
    return sorted(t for t in terms if t and t in text)


def attr_scope(row: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    blob = attr_blob(row)
    leak = hits(blob, LEAK_OR_PROCESS_TERMS)
    obj = hits(blob, OBJECTIVE_STRONG_TERMS)
    if leak and not obj:
        return "process_or_evaluation_noise", leak, obj
    if leak and obj:
        return "mixed_needs_remap", leak, obj
    if obj:
        return "objective_product_attribute", leak, obj
    return "uncertain_attribute", leak, obj


def evidence_type(row: dict[str, Any]) -> tuple[str, str]:
    blob = attr_blob(row)
    if hits(blob, MATERIAL_TERMS):
        return "objective_material", "material"
    if hits(blob, NUMERIC_TERMS):
        if hits(blob, BOOLEAN_TERMS):
            return "objective_boolean", "boolean"
        if any(t in blob for t in {"尺码", "尺寸"}):
            return "objective_numeric", "size"
        if any(t in blob for t in {"件数", "数量"}):
            return "objective_numeric", "count"
        return "objective_numeric", "number"
    if hits(blob, BOOLEAN_TERMS):
        return "objective_boolean", "boolean"
    if hits(blob, VISUAL_TERMS):
        if "颜色" in blob:
            return "objective_visual", "color"
        return "objective_visual", "visual"
    return "uncertain_attribute", "other"


def target_sources(objectivity: str) -> list[str]:
    if objectivity == "objective_numeric":
        return ["srt", "params", "product_title", "detail_image_ocr"]
    if objectivity in {"objective_material", "objective_boolean", "objective_visual"}:
        return ["srt", "params", "detail_image_ocr", "detail_image_vlm"]
    return ["srt", "params", "detail_image_ocr"]


def claim_verify_queue_rows(claim_rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in claim_rows:
        if row.get("already_in_clean_main"):
            continue
        if row.get("scope") == "process_or_evaluation_noise":
            continue
        if int(row.get("priority_score", 0) or 0) < 30:
            continue
        objectivity, value_type = evidence_type(row)
        triggered = row.get("triggered_reviews") or []
        neg_text = next((str(r.get("text") or "") for r in triggered if r.get("polarity") == "neg"), "")
        if not neg_text and triggered:
            neg_text = str(triggered[0].get("text") or "")
        priority = "P0" if row.get("has_negative_trigger") else "P1"
        out.append({
            "queue_type": "missing_claim_srt_first",
            "priority": priority,
            "pair_id": row.get("pair_id"),
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "product_title": row.get("product_title"),
            "detail_images": row.get("detail_images") or [],
            "srt_files": row.get("srt_files") or [],
            "comment_files": row.get("comment_files") or [],
            "attribute_objectivity": objectivity,
            "expected_value_type": value_type,
            "target_sources": target_sources(objectivity),
            "claim_segments": [],
            "claim_preview": "",
            "risk_comment_example": neg_text[:300],
            "current_source_count": 0,
            "current_confidence": "missing_claim",
            "n_aligned": 0,
            "n_neg_aligned": 1 if row.get("has_negative_trigger") else 0,
            "missing_claim_hits": row.get("trigger_count"),
            "trigger_hits": row.get("trigger_hits") or [],
            "priority_score": row.get("priority_score"),
            "accept_rule": {
                "claim": "live_claim_found must come from SRT only and match the target attribute",
                "product_evidence": "product evidence must come from params/title/OCR/detail-image VLM, not SRT",
                "consumer": "risk comment must refute or support the same attribute-level claim",
            },
        })
        if limit and len(out) >= limit:
            break
    return out


def row_pair_id(row: dict[str, Any]) -> str:
    pair_id = str(row.get("pair_id") or "")
    if pair_id:
        return pair_id
    return f"{row.get('product_id')}__{row.get('attribute_id')}"


def load_done_verify(path: str | Path) -> set[str]:
    done: set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("pair_id"):
                done.add(str(obj["pair_id"]))
    return done


def claim_reextract_candidates(rows: list[dict[str, Any]], existing_clean: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scope, leak, obj = attr_scope(row)
        triggered = row.get("triggered_reviews") or []
        trigger_blob = " ".join(str(r.get("text") or "") for r in triggered)
        trigger_hits = hits(trigger_blob, CLAIM_TRIGGER_TERMS)
        has_neg = any(r.get("polarity") == "neg" for r in triggered)
        strong_count = int(row.get("trigger_count", 0) or len(triggered))
        priority_score = 0
        priority_score += 20 if has_neg else 0
        priority_score += min(20, strong_count * 4)
        priority_score += 10 if scope in {"objective_product_attribute", "mixed_needs_remap"} else 0
        priority_score += min(10, len(trigger_hits) * 2)
        priority_score -= 12 if scope == "process_or_evaluation_noise" else 0
        pair_id = row_pair_id(row)
        out.append({
            "pool": "claim_reextract",
            "pair_id": pair_id,
            "already_in_clean_main": pair_id in existing_clean,
            "priority": row.get("priority"),
            "priority_score": priority_score,
            "scope": scope,
            "leak_hits": leak,
            "objective_hits": obj,
            "trigger_hits": trigger_hits,
            "has_negative_trigger": has_neg,
            "trigger_count": strong_count,
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "product_title": row.get("product_title"),
            "claim_preview": row.get("claim_preview"),
            "triggered_reviews": triggered[:8],
            "srt_files": row.get("srt_files") or [],
            "detail_images": row.get("detail_images") or [],
            "comment_files": row.get("comment_files") or [],
            "recommended_llm_task": "extract_grounded_srt_claim_then_verify_product_evidence",
        })
    out.sort(key=lambda r: (
        r["already_in_clean_main"],
        -int(r["priority_score"]),
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(str(r.get("priority")), 9),
        str(r.get("pair_id")),
    ))
    return out


def evidence_recovery_candidates(rows: list[dict[str, Any]], verified_pairs: set[str], existing_clean: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pair_id = row_pair_id(row)
        if pair_id in verified_pairs:
            continue
        scope, leak, obj = attr_scope(row)
        y = int(row.get("current_label", 0) or 0)
        n_neg = int(row.get("n_neg_aligned", 0) or 0)
        source_n = int(row.get("current_source_count", 0) or 0)
        score = 0
        score += 25 if source_n == 0 and y == 1 else 0
        score += 15 if n_neg else 0
        score += 12 if scope == "objective_product_attribute" else 0
        score += 6 if scope == "mixed_needs_remap" else 0
        score -= 10 if scope == "process_or_evaluation_noise" else 0
        out.append({
            "pool": "evidence_recovery",
            "pair_id": pair_id,
            "already_in_clean_main": pair_id in existing_clean,
            "priority": row.get("priority"),
            "priority_score": score,
            "scope": scope,
            "leak_hits": leak,
            "objective_hits": obj,
            "current_label": y,
            "current_source_count": source_n,
            "n_aligned": row.get("n_aligned"),
            "n_neg_aligned": n_neg,
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "product_title": row.get("product_title"),
            "claim_preview": row.get("claim_preview"),
            "claim_segments": row.get("claim_segments") or [],
            "target_sources": row.get("target_sources") or [],
            "detail_images": row.get("detail_images") or [],
            "recommended_llm_task": "verify_missing_product_evidence_or_drop_bad_pair",
        })
    out.sort(key=lambda r: (
        r["already_in_clean_main"],
        -int(r["priority_score"]),
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(str(r.get("priority")), 9),
        str(r.get("pair_id")),
    ))
    return out


def schema_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scope, leak, obj = attr_scope(row)
        card = int(row.get("acmt_cardinality", 0) or 0)
        score = card
        score += 20 if leak else 0
        score += 8 if scope == "mixed_needs_remap" else 0
        score -= 8 if scope == "objective_product_attribute" else 0
        out.append({
            "pool": "schema_remap",
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "canonical_name": row.get("canonical_name"),
            "aliases": row.get("aliases") or [],
            "product_title": row.get("product_title"),
            "priority": row.get("priority"),
            "priority_score": score,
            "scope": scope,
            "leak_hits": leak,
            "objective_hits": obj,
            "acmt_cardinality": card,
            "repair_question": row.get("repair_question"),
            "recommended_llm_task": "drop_merge_or_remap_attribute_before_rebuilding_acmt",
        })
    out.sort(key=lambda r: (
        {"P0": 0, "P1": 1, "P2": 2}.get(str(r.get("priority")), 9),
        -int(r["priority_score"]),
        str(r.get("product_id")),
        str(r.get("attribute_id")),
    ))
    return out


def aux_candidates(aux_rows: list[dict[str, Any]], removed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in aux_rows:
        pair_id = row_pair_id(row)
        out.append({
            "pool": "aux_hard_or_process",
            "pair_id": pair_id,
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "y": row.get("y"),
            "c": row.get("c"),
            "source_count": source_count(row) if isinstance(row, dict) else None,
            "recommended_use": "auxiliary_contrast_or_process-risk_analysis_not_clean_main",
        })
    for row in removed_rows:
        out.append({
            "pool": "removed_noise_review",
            "pair_id": row_pair_id(row),
            "product_id": row.get("product_id"),
            "category": row.get("category"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "y": row.get("y"),
            "reason": row.get("_source_recovery_v3", {}).get("training_action")
            or row.get("_source_recovery_v3", {}).get("drop_reason")
            or "removed_from_main",
            "recommended_use": "audit_noise_only",
        })
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "priority": dict(Counter(str(r.get("priority")) for r in rows if r.get("priority") is not None)),
        "scope": dict(Counter(str(r.get("scope")) for r in rows if r.get("scope") is not None)),
        "category": dict(Counter(str(r.get("category")) for r in rows if r.get("category") is not None).most_common(20)),
        "top_attributes": Counter(
            clean(r.get("attribute_name") or r.get("canonical_name")) for r in rows
        ).most_common(30),
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Expansion Candidate Pools v4",
        "",
        "## Summary",
    ]
    for name, item in report["pools"].items():
        lines.append(f"- `{name}`: `{item['n']}` rows; priority={item.get('priority', {})}; scope={item.get('scope', {})}")
    lines += [
        "",
        "## Recommended Order",
        "1. Verify top claim-reextract rows that are objective or mixed-remap and have negative trigger comments.",
        "2. Verify remaining evidence-recovery rows not covered by strict P0 source recovery.",
        "3. Use schema-remap rows to contract A_cmt(p) before a full Stage B/C rebuild.",
        "4. Keep aux/noise rows outside clean supervision; use them only for contrastive diagnostics or appendix analysis.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current_clean", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl")
    ap.add_argument("--claim_queue", default="data/final/repaired_v1/comment_triggered_srt_reextract_queue_v1.jsonl")
    ap.add_argument("--source_queue", default="data/final/repaired_v1/source_recovery_queue_v3.jsonl")
    ap.add_argument("--verified", default="data/final/repaired_v1/source_recovery_queue_v3_llm_verify_p0_v3strict_merged.jsonl")
    ap.add_argument("--schema_queue", default="data/final/repaired_v1/schema_remap_review_queue_v1.jsonl")
    ap.add_argument("--aux_manifest", default="data/final/repaired_v1/source_recovery_v3_dropunresolved_auxiliary_manifest.jsonl")
    ap.add_argument("--source_report", default="data/final/repaired_v1/source_recovered_v3_dropunresolved_report.json")
    ap.add_argument("--out_dir", default="data/final/repaired_v1/expansion_candidate_pools_v4")
    ap.add_argument("--claim_verify_limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    clean_rows = list(read_jsonl(args.current_clean))
    existing_clean = {row_pair_id(r) for r in clean_rows}
    verified_pairs = load_done_verify(args.verified)

    claim_rows = claim_reextract_candidates(list(read_jsonl(args.claim_queue)), existing_clean)
    evidence_rows = evidence_recovery_candidates(list(read_jsonl(args.source_queue)), verified_pairs, existing_clean)
    schema_rows_out = schema_candidates(list(read_jsonl(args.schema_queue)))
    source_report = read_json(args.source_report, default={}) or {}
    aux_rows = aux_candidates(
        list(read_jsonl(args.aux_manifest)) if Path(args.aux_manifest).exists() else [],
        source_report.get("removed_rows", []) or [],
    )
    claim_verify_rows = claim_verify_queue_rows(claim_rows, limit=args.claim_verify_limit)

    paths = {
        "claim_reextract": out_dir / "claim_reextract_candidates.jsonl",
        "claim_verify_queue": out_dir / "claim_reextract_verify_queue.jsonl",
        "evidence_recovery": out_dir / "evidence_recovery_candidates.jsonl",
        "schema_remap": out_dir / "schema_remap_candidates.jsonl",
        "auxiliary_or_noise": out_dir / "auxiliary_or_noise_candidates.jsonl",
    }
    write_jsonl(paths["claim_reextract"], claim_rows)
    write_jsonl(paths["claim_verify_queue"], claim_verify_rows)
    write_jsonl(paths["evidence_recovery"], evidence_rows)
    write_jsonl(paths["schema_remap"], schema_rows_out)
    write_jsonl(paths["auxiliary_or_noise"], aux_rows)

    report = {
        "current_clean": args.current_clean,
        "current_clean_n": len(clean_rows),
        "current_clean_pairs": len(existing_clean),
        "verified_pairs": len(verified_pairs),
        "pools": {
            "claim_reextract": {**summarize(claim_rows), "path": str(paths["claim_reextract"])},
            "claim_verify_queue": {**summarize(claim_verify_rows), "path": str(paths["claim_verify_queue"])},
            "evidence_recovery": {**summarize(evidence_rows), "path": str(paths["evidence_recovery"])},
            "schema_remap": {**summarize(schema_rows_out), "path": str(paths["schema_remap"])},
            "auxiliary_or_noise": {**summarize(aux_rows), "path": str(paths["auxiliary_or_noise"])},
        },
        "high_value_estimate": {
            "claim_reextract_score_ge_30": sum(1 for r in claim_rows if int(r["priority_score"]) >= 30 and not r["already_in_clean_main"]),
            "claim_reextract_objective_or_mixed": sum(
                1 for r in claim_rows
                if r["scope"] in {"objective_product_attribute", "mixed_needs_remap"}
                and not r["already_in_clean_main"]
            ),
            "evidence_recovery_score_ge_25": sum(1 for r in evidence_rows if int(r["priority_score"]) >= 25),
            "schema_p0_p1": sum(1 for r in schema_rows_out if r.get("priority") in {"P0", "P1"}),
        },
    }
    write_json(out_dir / "report.json", report)
    write_markdown(report, "docs/EXPANSION_CANDIDATE_POOLS_V4.md")
    print(json.dumps({
        "current_clean_n": report["current_clean_n"],
        "claim_reextract": report["pools"]["claim_reextract"]["n"],
        "claim_verify_queue": report["pools"]["claim_verify_queue"]["n"],
        "evidence_recovery": report["pools"]["evidence_recovery"]["n"],
        "schema_remap": report["pools"]["schema_remap"]["n"],
        "auxiliary_or_noise": report["pools"]["auxiliary_or_noise"]["n"],
        "high_value_estimate": report["high_value_estimate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
