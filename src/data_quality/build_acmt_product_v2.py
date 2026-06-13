"""Build a product-only A_cmt(p) view for raw-stage regeneration.

The original Stage B0 directly aggregates Stage A resolved aspects into
A_cmt(p).  That is too permissive for a product-fact verification benchmark:
service/process/evaluation aspects expand the candidate schema and then drag
claim extraction, evidence extraction, and labels off target.

This script writes a parallel directory and never overwrites current Stage B:

  data/processed/stageB_product_v2/

Outputs:
- acmt_product_v2.json: product_id -> kept product attributes
- resolved_aspects_product_v2.jsonl: Stage A mentions kept for clean rebuild
- resolved_aspects_aux_v2.jsonl: price/service/process/perception mentions
- acmt_product_v2_drop_audit.jsonl: dropped/remapped audit rows
- acmt_product_v2_report.json and docs/ACMT_PRODUCT_V2.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from common.io_utils import normalize, read_json, read_jsonl, write_json, write_jsonl


SERVICE_TERMS = {
    "客服", "售后", "物流", "快递", "配送", "发货", "到货", "退货", "退款",
    "退换", "保价", "补发", "服务", "店铺", "商家", "卖家",
}

PROCESS_TERMS = {
    "主播", "直播", "讲解", "宣传", "话术", "购买渠道", "下单", "链接",
    "库存", "秒杀", "活动", "抽奖", "赠品", "试用", "优惠券",
}
HARD_PROCESS_TERMS = {
    "下单", "链接", "库存", "秒杀", "抽奖", "赠品", "赠送", "赠", "优惠券",
    "购买渠道", "小黄车", "发货", "物流", "售后",
}

PERCEPTION_TERMS = set(getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])) | {
    "满意", "满意度", "推荐", "推荐度", "回购", "复购", "喜欢", "喜好",
    "喜好度", "值得购买", "值得入手", "整体评价", "总体评价", "购买体验",
    "购买价值", "物有所值", "性价比", "划算", "好评", "差评", "品质",
    "产品品质", "质量好坏", "客观属性名词短语", "建议类属性",
}

PRICE_TERMS = {"价格", "券后价", "到手价", "优惠", "折扣", "便宜", "贵", "实惠"}

OBJECTIVE_TERMS = {
    "品牌", "产品名称", "名称", "型号", "货号", "条形码", "规格", "规格型号",
    "尺码", "尺寸", "大小", "长度", "宽度", "高度", "厚度", "克重",
    "重量", "净含量", "容量", "数量", "件数", "组合件数", "套餐", "包装",
    "包装方式", "包装类型", "材质", "材料", "面料", "面料材质", "成分",
    "配料", "原料", "含量", "含绒量", "充绒量", "绒子", "产地", "原产地",
    "保质期", "生产日期", "颜色", "颜色分类", "图案", "款式", "版型",
    "闭合方式", "口袋", "内兜", "是否加绒", "加绒", "是否防水", "防水",
    "功率", "输出功率", "电池容量", "续航", "接口", "适用型号", "适用人群",
    "适用季节", "功能", "功效", "香型", "口味", "风味", "口感", "气味",
    "形状", "商品等级", "等级", "执行标准", "许可证", "证书",
    "弹力", "弹性", "袖长", "衣长", "裤长", "裤型", "鞋底材质",
    "鞋面材质", "内里材质", "帽子深度", "帽深", "颜色数", "颜色数量",
    "类型", "类别", "电源容量", "电池类型", "质地", "质感",
}

CANONICAL_ALLOW = {
    "品牌", "产品名称", "包装", "包装方式", "包装类型", "适用人群",
    "适用季节", "功能", "功效", "口感", "风味", "气味", "款式", "版型",
}

PRICE_POLICY = "aux_price_dynamic"


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def text_hits(text: str, terms: set[str]) -> list[str]:
    return sorted(t for t in terms if t and t in text)


def read_ocr_text(pid: str, cap: int = 12000) -> str:
    path = config.STAGE_C / "ocr_text" / f"{pid}.json"
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    chunks = []
    for key, val in obj.items():
        if Path(str(key)).name.startswith("._"):
            continue
        if val:
            chunks.append(str(val))
    return "\n".join(chunks)[:cap]


def load_cas(stage_a_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(stage_a_dir.glob("CAS+_*.json")):
        if path.name.startswith("._"):
            continue
        cat = path.stem.replace("CAS+_", "")
        obj = read_json(path, default={"attributes": []}) or {"attributes": []}
        out[cat] = {str(a.get("attribute_id")): a for a in obj.get("attributes", [])}
    return out


def attr_meta(cas: dict[str, dict[str, dict[str, Any]]], category: str, aid: str) -> dict[str, Any]:
    a = cas.get(category, {}).get(aid, {})
    return {
        "canonical_name": clean(a.get("canonical_name") or aid),
        "aliases": [clean(x) for x in (a.get("aliases") or []) if clean(x)][:50],
        "source": a.get("source"),
        "value_type": a.get("value_type"),
    }


def attr_blob(meta: dict[str, Any], aid: str) -> str:
    return " ".join([clean(meta.get("canonical_name")), aid] + [clean(x) for x in meta.get("aliases", [])])


def classify_attribute(meta: dict[str, Any], aid: str, *, row_types: Counter[str]) -> tuple[str, list[str]]:
    blob = attr_blob(meta, aid)
    canonical = clean(meta.get("canonical_name"))
    service = text_hits(blob, SERVICE_TERMS)
    process = text_hits(blob, PROCESS_TERMS)
    hard_process = text_hits(blob, HARD_PROCESS_TERMS)
    perception = text_hits(blob, PERCEPTION_TERMS)
    price = text_hits(blob, PRICE_TERMS)
    objective = text_hits(blob, OBJECTIVE_TERMS)

    if price and not (objective and canonical not in PRICE_TERMS):
        return PRICE_POLICY, price
    if hard_process:
        return "drop_process", hard_process
    if row_types and row_types.get("service", 0) > row_types.get("attribute", 0) and not objective:
        return "drop_service_stagea_type", ["stagea_type=service"]
    if service and not objective:
        return "drop_service", service
    if process and not objective:
        return "drop_process", process
    if perception and not objective:
        return "drop_perception", perception
    if objective or canonical in CANONICAL_ALLOW:
        return "keep_product_objective", objective or [canonical]
    if service or process or perception:
        return "aux_mixed_scope_review", service + process + perception + objective
    return "drop_uncertain_not_verifiable", []


def source_family(meta: dict[str, Any], aid: str, product_text: str, ocr_text: str) -> tuple[str, list[str]]:
    terms = [clean(meta.get("canonical_name"))] + [clean(x) for x in meta.get("aliases", [])[:20]]
    terms = [t for t in dict.fromkeys(terms) if len(t) >= 2]
    n_product = normalize(product_text)
    n_ocr = normalize(ocr_text)
    direct = []
    for term in terms:
        nt = normalize(term)
        if nt and nt in n_product:
            direct.append(f"param_or_title:{term}")
        elif nt and nt in n_ocr:
            direct.append(f"ocr:{term}")
    blob = attr_blob(meta, aid)
    if any(t in blob for t in {"材质", "面料", "成分", "配料", "含绒量", "充绒量"}):
        return "material", direct
    if any(t in blob for t in {"颜色", "图案", "款式", "版型", "闭合方式", "口袋", "内兜", "加绒", "防水"}):
        return "visual_or_boolean", direct
    if any(t in blob for t in {"尺码", "尺寸", "规格", "净含量", "容量", "重量", "数量", "件数", "厚度", "长度", "宽度", "高度"}):
        return "numeric", direct
    if any(t in blob for t in {"品牌", "产品名称", "型号", "货号", "条形码", "产地", "保质期", "执行标准"}):
        return "identity_or_spec", direct
    if direct:
        return "direct_text_match", direct
    return "objective_name_only", direct


def product_text(bundle: pidx.ProductBundle) -> str:
    parts = [bundle.title]
    for k, v in (bundle.params or {}).items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def score_group(rows: list[dict[str, Any]], meta: dict[str, Any], direct_hits: list[str], source: str) -> float:
    score = 0.0
    score += len(rows)
    score += sum(2.5 for r in rows if r.get("explicit_fact_hit"))
    score += sum(1.5 for r in rows if r.get("polarity") == "neg")
    score += sum(0.7 for r in rows if r.get("mention_strength") == "strong")
    score += min(4.0, len(direct_hits) * 1.5)
    score += 1.0 if source in {"numeric", "material", "identity_or_spec"} else 0.0
    if meta.get("source") in {"param", "both"}:
        score += 2.0
    return round(score, 3)


def summarize_cardinality(acmt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = sorted(len(v) for v in acmt.values())
    if not sizes:
        return {"products": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0, "pairs": 0}
    return {
        "products": len(sizes),
        "mean": round(sum(sizes) / len(sizes), 2),
        "p50": sizes[int(0.50 * (len(sizes) - 1))],
        "p90": sizes[int(0.90 * (len(sizes) - 1))],
        "max": max(sizes),
        "pairs": sum(sizes),
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# A_cmt Product v2",
        "",
        "## Summary",
        f"- input mentions: `{report['mentions']['input']}`",
        f"- clean mentions: `{report['mentions']['clean']}`",
        f"- auxiliary mentions: `{report['mentions']['auxiliary']}`",
        f"- dropped mentions: `{report['mentions']['dropped']}`",
        f"- original cardinality: `{report['cardinality_original']}`",
        f"- product-v2 cardinality: `{report['cardinality_product_v2']}`",
        f"- auxiliary pairs: `{report['auxiliary_pairs']}`",
        "",
        "## Interpretation",
        "This view contracts A_cmt(p) to product-fact attributes before rebuilding",
        "Stage B/C. Price, service, live-process, and perception-only mentions are",
        "kept out of the clean schema and can be used as auxiliary/process analyses.",
        "",
        "## Top Drop Reasons",
    ]
    for reason, n in report["drop_reasons"].items():
        lines.append(f"- `{reason}`: `{n}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a_dir", default="data/processed/stageA_repaired_v1")
    ap.add_argument("--resolved", default="data/processed/stageA_repaired_v1/resolved_aspects_schema_clean_v1.jsonl")
    ap.add_argument("--out_dir", default="data/processed/stageB_product_v2")
    ap.add_argument("--max_attrs_per_product", type=int, default=24)
    ap.add_argument("--keep_direct_overflow", action="store_true",
                    help="Keep over-cap attributes when raw title/param/OCR directly hits an alias.")
    ap.add_argument("--overflow_min_score", type=float, default=10.0)
    ap.add_argument("--report", default="data/final/repaired_v1/acmt_product_v2_report.json")
    ap.add_argument("--md", default="docs/ACMT_PRODUCT_V2.md")
    args = ap.parse_args()

    stage_a = Path(args.stage_a_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = pidx.build_bundles()
    cas = load_cas(stage_a)
    rows = list(read_jsonl(args.resolved))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r.get("product_id")), str(r.get("attribute_id")))].append(r)

    acmt: dict[str, dict[str, Any]] = defaultdict(dict)
    clean_mentions: list[dict[str, Any]] = []
    aux_mentions: list[dict[str, Any]] = []
    drop_audit: list[dict[str, Any]] = []
    pair_candidates: dict[str, list[tuple[str, dict[str, Any], float, str, list[str], list[str]]]] = defaultdict(list)
    original_by_pid: dict[str, set[str]] = defaultdict(set)
    reason_counter: Counter[str] = Counter()
    aux_pair_counter: Counter[str] = Counter()

    for (pid, aid), group in grouped.items():
        original_by_pid[pid].add(aid)
        category = str(group[0].get("category") or "")
        meta = attr_meta(cas, category, aid)
        row_types = Counter(str(g.get("type") or "") for g in group)
        decision, reasons = classify_attribute(meta, aid, row_types=row_types)
        b = bundles.get(pid)
        ptext = product_text(b) if b else ""
        otext = read_ocr_text(pid)
        source, direct_hits = source_family(meta, aid, ptext, otext)
        score = score_group(group, meta, direct_hits, source)
        if decision == "keep_product_objective":
            pair_candidates[pid].append((aid, meta, score, source, direct_hits, reasons))
        elif decision.startswith("aux_") or decision == PRICE_POLICY:
            aux_pair_counter[decision] += 1
            for r in group:
                out = dict(r)
                out["_acmt_product_v2_decision"] = decision
                out["_acmt_product_v2_reasons"] = reasons
                aux_mentions.append(out)
        else:
            reason_counter[decision] += 1
            drop_audit.append({
                "product_id": pid,
                "category": category,
                "attribute_id": aid,
                "canonical_name": meta.get("canonical_name"),
                "aliases": meta.get("aliases", [])[:20],
                "decision": decision,
                "reasons": reasons,
                "mention_count": len(group),
                "type_counts": dict(row_types),
                "sample_spans": [g.get("evidence_span") for g in group[:5]],
            })

    over_cap_rows: list[dict[str, Any]] = []
    overflow_direct_hit_count = 0
    overflow_high_score_count = 0
    for pid, candidates in pair_candidates.items():
        candidates.sort(key=lambda x: (-x[2], x[1].get("canonical_name", ""), x[0]))
        keep = candidates[: args.max_attrs_per_product]
        overflow = candidates[args.max_attrs_per_product :]
        if args.keep_direct_overflow and overflow:
            rescued = [
                item for item in overflow
                if item[4] or float(item[2]) >= float(args.overflow_min_score)
            ]
            if rescued:
                rescue_ids = {item[0] for item in rescued}
                keep = keep + rescued
                overflow = [item for item in overflow if item[0] not in rescue_ids]
        for aid, meta, score, source, direct_hits, reasons in keep:
            acmt[pid][aid] = {
                "canonical_name": meta.get("canonical_name"),
                "aliases": meta.get("aliases", []),
                "source": meta.get("source"),
                "value_type": meta.get("value_type"),
                "source_family": source,
                "direct_raw_hits": direct_hits[:20],
                "selection_score": score,
                "selection_reasons": reasons,
            }
            for r in grouped[(pid, aid)]:
                out = dict(r)
                out["_acmt_product_v2_score"] = score
                out["_acmt_product_v2_source_family"] = source
                out["_acmt_product_v2_direct_raw_hits"] = direct_hits[:10]
                clean_mentions.append(out)
        for aid, meta, score, source, direct_hits, reasons in overflow:
            reason_counter["drop_over_product_attr_cap"] += 1
            if direct_hits:
                overflow_direct_hit_count += 1
            if float(score) >= float(args.overflow_min_score):
                overflow_high_score_count += 1
            over_cap_rows.append({
                "product_id": pid,
                "attribute_id": aid,
                "canonical_name": meta.get("canonical_name"),
                "score": score,
                "source_family": source,
                "direct_raw_hits": direct_hits[:10],
                "reason": "over_product_attr_cap",
            })
            for r in grouped[(pid, aid)]:
                out = dict(r)
                out["_acmt_product_v2_decision"] = "aux_over_product_attr_cap"
                out["_acmt_product_v2_score"] = score
                aux_mentions.append(out)

    write_json(out_dir / "acmt_product_v2.json", acmt)
    write_jsonl(out_dir / "resolved_aspects_product_v2.jsonl", clean_mentions)
    write_jsonl(out_dir / "resolved_aspects_aux_v2.jsonl", aux_mentions)
    write_jsonl(out_dir / "acmt_product_v2_drop_audit.jsonl", drop_audit + over_cap_rows)

    original_card = {
        pid: {aid: {} for aid in attrs}
        for pid, attrs in original_by_pid.items()
    }
    report = {
        "stage_a_dir": str(stage_a),
        "resolved": args.resolved,
        "out_dir": str(out_dir),
        "max_attrs_per_product": args.max_attrs_per_product,
        "keep_direct_overflow": args.keep_direct_overflow,
        "overflow_min_score": args.overflow_min_score,
        "mentions": {
            "input": len(rows),
            "clean": len(clean_mentions),
            "auxiliary": len(aux_mentions),
            "dropped": sum(len(grouped[(r["product_id"], r["attribute_id"])]) for r in drop_audit if "attribute_id" in r),
        },
        "cardinality_original": summarize_cardinality(original_card),
        "cardinality_product_v2": summarize_cardinality(acmt),
        "auxiliary_pairs": dict(aux_pair_counter),
        "drop_reasons": dict(reason_counter),
        "source_family": dict(Counter(m.get("source_family") for attrs in acmt.values() for m in attrs.values())),
        "top_clean_attributes": Counter(m.get("canonical_name") for attrs in acmt.values() for m in attrs.values()).most_common(40),
        "top_dropped_attributes": Counter(r.get("canonical_name") for r in drop_audit).most_common(40),
        "overflow_audit": {
            "dropped_over_cap_pairs": len(over_cap_rows),
            "dropped_with_direct_raw_hits": overflow_direct_hit_count,
            "dropped_high_score": overflow_high_score_count,
        },
    }
    write_json(args.report, report)
    write_markdown(report, args.md)
    print(json.dumps({
        "input_mentions": report["mentions"]["input"],
        "clean_mentions": report["mentions"]["clean"],
        "aux_mentions": report["mentions"]["auxiliary"],
        "cardinality_original": report["cardinality_original"],
        "cardinality_product_v2": report["cardinality_product_v2"],
        "drop_reasons": report["drop_reasons"],
        "auxiliary_pairs": report["auxiliary_pairs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
