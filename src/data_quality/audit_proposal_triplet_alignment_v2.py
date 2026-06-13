"""Audit proposal-complete rows for claim-attribute-evidence alignment.

The previous proposal-faithful audit checks whether a row has a claim, product
evidence, and proposal label.  This script adds the next gate required by the
proposal: the claim and evidence must both be about the same target attribute.

It is intentionally a quality audit, not a learnability filter.  Rows that fail
the gate are written to a repair queue with concrete reasons; labels are never
changed here.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import normalize, read_jsonl, write_json, write_jsonl


PROMO_TERMS = {
    "下单", "拍", "秒杀", "库存", "链接", "小黄车", "优惠", "优惠券", "券",
    "福利", "赠", "包邮", "发货", "物流", "售后", "客服", "到手价", "价格",
    "卖了", "销量", "抢", "补货", "现货", "排队",
}
INVENTORY_TERMS = {"库存", "只有", "剩", "拍到", "补货", "卖完", "抢", "发货"}
QUESTION_RE = re.compile(r"[?？]|有没有|是不是|能不能|可以吗|行吗|看一下|看下|谁要")
PRICE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|块钱)|[一二三四五六七八九十两百千]+块(?:钱)?|到手价|售价|价格")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|mm|kg|g|ml|l|mah|w|瓦|克|斤|升|毫升|厘米|毫米|"
    r"米|寸|码|%|天|月|年|个|只|件|包|袋|盒|瓶|双|支|片|条|度|℃|°C|℉|°F)",
    re.I,
)
PURE_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")

MATERIAL_TERMS = {
    "材质", "面料", "材料", "成分", "配料", "含", "不含", "绒", "棉", "羊毛",
    "羊绒", "鸭绒", "鹅绒", "羽绒", "真皮", "牛皮", "皮革", "聚酯", "锦纶",
    "腈纶", "氨纶", "涤纶", "粘纤", "莱赛尔", "亚麻", "桑蚕丝", "不锈钢",
    "陶瓷", "玻璃", "塑料", "硅胶", "乳胶", "EVA", "eva", "橡胶", "发泡",
}
STRUCTURE_TERMS = {
    "双层", "单层", "三层", "加厚", "薄款", "厚款", "高腰", "低腰", "直筒",
    "阔腿", "修身", "宽松", "圆领", "V领", "v领", "松紧", "弹力", "短袖",
    "长袖", "七分袖", "九分裤", "连帽", "翻领", "厚底", "低帮", "高帮",
    "尖头", "圆头", "方头", "平底", "坡跟", "短款", "中长款", "长款",
    "拉链", "口袋", "内兜", "防水", "透气", "加绒", "保暖",
}
COLOR_RE = re.compile(r"(黑|白|红|蓝|绿|黄|紫|灰|棕|咖|粉|橙|金|银|杏|米色|卡其|藏青|藕粉|驼色)")
IDENTITY_TERMS = {"品牌", "型号", "货号", "名称", "产地", "保质期", "生产日期", "执行标准", "条码", "真伪", "正版"}
SEASON_TERMS = {"春夏", "春秋", "秋冬", "春天", "夏天", "秋天", "冬天", "四季", "换季", "零下", "保暖", "抗寒", "防寒", "过冬", "冬季", "夏季"}
AUDIENCE_TERMS = {"男生", "女生", "男士", "女士", "男女", "通用", "儿童", "宝宝", "婴儿", "老人", "学生", "孕妇", "小个子", "大个子", "胖", "瘦"}
FUNCTION_TERMS = {"功能", "功效", "效果", "防晒", "防水", "防风", "透气", "保暖", "除螨", "杀菌", "控油", "补水", "美白", "清洁", "收纳", "剃毛", "脱毛", "冷却", "降温", "快充", "续航"}
STYLE_TERMS = {"风格", "百搭", "复古", "港风", "通勤", "休闲", "运动", "氛围", "显白", "修饰", "搭配", "穿搭", "设计", "原创", "经典", "高级"}
GENERIC_WORDS = {
    "属性", "商品", "产品", "情况", "相关", "信息", "是否", "一般", "其他",
    "类型", "款式", "风格", "功能", "效果", "质量", "品质",
}


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def family(attribute_name: str, attribute_id: str) -> str:
    text = f"{attribute_name} {attribute_id}"
    if any(t in text for t in ("价格", "价位", "售价", "到手价")):
        return "price"
    if any(t in text for t in ("材质", "面料", "成分", "配料", "绒", "棉", "羊毛", "皮", "橡胶")):
        return "material"
    if any(t in text for t in ("尺码", "尺寸", "大小", "重量", "容量", "净含量", "含量", "厚度", "长度", "宽度", "高度", "数量", "件数", "电源容量", "电池容量")):
        return "numeric"
    if any(t in text for t in ("适用季节", "季节", "上市时间", "保暖", "抗寒")):
        return "season"
    if any(t in text for t in ("适用对象", "适用人群", "适用性别", "适用年龄", "人群", "对象", "性别", "年龄")):
        return "audience"
    if any(t in text for t in ("功能", "功效", "见效", "效果", "卖点", "防晒", "脱毛", "快充")):
        return "function_effect"
    if any(t in text for t in ("风格", "搭配", "穿搭")):
        return "style"
    if any(t in text for t in ("颜色", "款式", "版型", "图案", "外观", "形状", "鞋头", "帽顶", "结构", "是否", "防水", "透气", "加绒")):
        return "visual_or_boolean"
    if any(t in text for t in IDENTITY_TERMS):
        return "identity"
    return "attribute_value"


def expanded_attribute_terms(attribute_name: str, attribute_id: str, aliases: list[str]) -> list[str]:
    text = f"{attribute_name} {attribute_id}"
    extra: list[str] = []
    if any(t in text for t in ("适用季节", "季节")):
        extra.extend(["适用季节", "上市时间", "季节", "春夏", "春秋", "秋冬", "冬季", "夏季", "四季", "保暖", "抗寒"])
    if any(t in text for t in ("适用对象", "适用人群", "适用性别", "适用年龄", "人群", "对象", "性别", "年龄")):
        extra.extend(["适用对象", "适用人群", "适用性别", "适用年龄", "人群", "对象", "性别", "年龄", "男", "女", "通用"])
    if any(t in text for t in ("功能", "功效", "见效", "效果", "卖点")):
        extra.extend(["功能", "功效", "效果", "见效", "卖点", "防晒", "防水", "透气", "剃毛", "脱毛", "快充"])
    if any(t in text for t in ("风格", "搭配", "穿搭")):
        extra.extend(["风格", "搭配", "穿搭", "百搭", "复古", "港风", "休闲", "通勤", "设计"])
    if any(t in text for t in ("衣长", "裤长", "袖长", "版型", "款式")):
        extra.extend(["衣长", "裤长", "袖长", "长款", "短款", "中长款", "版型", "款式", "宽松", "修身"])
    return terms_from_values(attribute_name, attribute_id, aliases, extra)


def terms_from_values(*values: Any) -> list[str]:
    raw: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            raw.extend(str(x) for x in value)
        else:
            raw.append(str(value))
    out: list[str] = []
    for text in raw:
        for tok in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text):
            tok = tok.strip()
            if not tok:
                continue
            if re.fullmatch(r"[A-Za-z0-9]+", tok):
                if len(tok) >= 2:
                    out.append(tok.lower())
                continue
            if 2 <= len(tok) <= 8 and tok not in GENERIC_WORDS:
                out.append(tok)
            elif len(tok) > 8:
                out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
                out.extend(tok[i:i + 3] for i in range(len(tok) - 2))
    seen: set[str] = set()
    dedup = []
    for term in out:
        n = normalize(term)
        if n and n not in seen and n not in {normalize(x) for x in GENERIC_WORDS}:
            seen.add(n)
            dedup.append(term)
    return dedup[:80]


def text_hits(text: str, terms: list[str]) -> list[str]:
    ntext = normalize(text)
    hits = []
    for term in terms:
        nterm = normalize(term)
        if len(nterm) >= 2 and nterm in ntext:
            hits.append(term)
    return hits[:20]


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ev in row.get("evidence_params") or []:
        items.append({
            "source": "params",
            "key": str(ev.get("param_key", "") or ""),
            "text": str(ev.get("raw_text", "") or ""),
            "path": "",
        })
    for ev in row.get("evidence_ocr") or []:
        items.append({
            "source": "ocr",
            "key": "",
            "text": str(ev.get("raw_text", "") or ""),
            "path": str(ev.get("image_path", "") or ""),
        })
    for ev in row.get("evidence_vlm") or []:
        items.append({
            "source": "vlm",
            "key": "",
            "text": str(ev.get("raw_quote", "") or ev.get("raw_text", "") or ""),
            "path": str(ev.get("image_path", "") or ""),
        })
    return items


def claim_segments(row: dict[str, Any]) -> list[dict[str, Any]]:
    claim = row.get("claim") or {}
    segs = [s for s in (claim.get("segments") or []) if isinstance(s, dict)]
    if segs:
        return segs
    passage = str(claim.get("passage", "") or "").strip()
    if not passage:
        return []
    return [{"text": p.strip()} for p in passage.split("\n---\n") if p.strip()]


def has_family_signal(text: str, fam: str) -> tuple[bool, list[str]]:
    hits: list[str] = []
    if fam == "price":
        if PRICE_RE.search(text):
            hits.append("price_value")
    elif fam == "material":
        hits.extend(t for t in MATERIAL_TERMS if t in text)
    elif fam == "numeric":
        if MEASUREMENT_RE.search(text):
            hits.append("measurement")
        elif NUMBER_RE.search(text):
            hits.append("number")
        hits.extend(t for t in STRUCTURE_TERMS if t in text and t in {"双层", "单层", "三层", "加厚", "薄款", "厚款", "弹力", "松紧", "加绒"})
    elif fam == "visual_or_boolean":
        if COLOR_RE.search(text):
            hits.append("color")
        hits.extend(t for t in STRUCTURE_TERMS if t in text)
    elif fam == "identity":
        hits.extend(t for t in IDENTITY_TERMS if t in text)
    elif fam == "season":
        hits.extend(t for t in SEASON_TERMS if t in text)
        if "零下" in text or re.search(r"\d+(?:\.\d+)?\s*(?:度|℃|°C|℉|°F)", text, re.I):
            hits.append("temperature")
    elif fam == "audience":
        hits.extend(t for t in AUDIENCE_TERMS if t in text)
        if re.search(r"\d+\s*(?:岁|周岁)", text):
            hits.append("age")
    elif fam == "function_effect":
        hits.extend(t for t in FUNCTION_TERMS if t in text)
        if re.search(r"\d+\s*(?:天|周|月)", text):
            hits.append("time_to_effect")
    elif fam == "style":
        hits.extend(t for t in STYLE_TERMS if t in text)
    else:
        hits.extend(t for t in MATERIAL_TERMS | STRUCTURE_TERMS | IDENTITY_TERMS if t in text)
        if MEASUREMENT_RE.search(text):
            hits.append("measurement")
    return bool(hits), list(dict.fromkeys(hits))[:20]


def validate_claim_segment(text: str, attr_terms: list[str], evidence_terms: list[str], fam: str, attr_name: str) -> dict[str, Any]:
    alias_hits = text_hits(text, attr_terms)
    ev_hits = text_hits(text, evidence_terms)
    fam_ok, fam_hits = has_family_signal(text, fam)
    promo_hits = sorted(t for t in PROMO_TERMS if t in text)
    question = bool(QUESTION_RE.search(text))
    inventory_shape = bool(INVENTORY_TERMS & set(promo_hits)) and any(t in attr_name for t in ("件数", "数量"))

    valid_signal = bool(alias_hits or ev_hits or fam_ok)
    if question:
        status = "review_question_like"
    elif inventory_shape and not ev_hits:
        status = "review_inventory_or_order_not_pack_quantity"
    elif promo_hits and not valid_signal:
        status = "reject_promo_or_order"
    elif fam == "numeric" and not (MEASUREMENT_RE.search(text) or ev_hits or alias_hits) and NUMBER_RE.search(text):
        status = "review_bare_number_claim"
    elif valid_signal:
        status = "valid_attribute_claim"
    else:
        status = "review_low_attribute_specificity"
    return {
        "text": text[:220],
        "status": status,
        "alias_hits": alias_hits,
        "evidence_value_hits": ev_hits,
        "family_hits": fam_hits,
        "promo_hits": promo_hits,
        "question_like": question,
    }


def validate_evidence_item(item: dict[str, Any], attr_terms: list[str], fam: str, attr_name: str) -> dict[str, Any]:
    text = str(item.get("text", "") or "")
    key = str(item.get("key", "") or "")
    blob = f"{key}: {text}" if key else text
    alias_hits = text_hits(blob, attr_terms)
    fam_ok, fam_hits = has_family_signal(blob, fam)
    ntext = normalize(text)
    too_short = len(ntext) < 2 or bool(PURE_NUMBER_RE.fullmatch(text)) or text.strip() in {"有", "无", "是", "否"}
    source = str(item.get("source", ""))

    if source in {"ocr", "vlm"} and too_short and not alias_hits:
        status = "reject_too_short_unkeyed_evidence"
    elif source == "params" and (alias_hits or fam_ok):
        status = "valid_product_evidence"
    elif source in {"ocr", "vlm"} and (alias_hits or fam_ok):
        status = "valid_product_evidence"
    elif fam == "numeric" and NUMBER_RE.search(text) and not too_short and any(t in attr_name for t in ("尺码", "尺寸", "容量", "重量", "净含量", "厚度")):
        status = "review_numeric_evidence_without_key"
    else:
        status = "review_low_attribute_specificity"
    return {
        "source": source,
        "key": key[:80],
        "text": text[:180],
        "status": status,
        "alias_hits": alias_hits,
        "family_hits": fam_hits,
        "too_short_unkeyed": too_short and source in {"ocr", "vlm"},
    }


def label_quality(row: dict[str, Any]) -> dict[str, Any]:
    pq = row.get("_proposal_quality") or {}
    state = (pq.get("label") or {}).get("state")
    if not state:
        audit = row.get("label_audit") or {}
        y = int(row.get("y", 0) or 0)
        n_aligned = int(audit.get("n_aligned", 0) or 0)
        n_neg = int(audit.get("n_neg_aligned", 0) or 0)
        if y == 1 and n_neg > 0:
            state = "label_positive_claim_aligned_neg"
        elif y == 0 and n_aligned > 0:
            state = "label_negative_claim_aligned_nonneg"
        else:
            state = "label_negative_no_aligned_review"
    supported = state in {"label_positive_claim_aligned_neg", "label_negative_claim_aligned_nonneg"}
    return {
        "state": state,
        "supported_by_aligned_review": supported,
        "c": row.get("c"),
        "y": row.get("y"),
    }


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    attr_name = str(row.get("attribute_name", "") or "")
    attr_id = str(row.get("attribute_id", "") or "")
    aliases = []
    pq = row.get("_proposal_quality") or {}
    aliases.extend(((pq.get("claim") or {}).get("claim_alias_hits") or []))
    attr_terms = expanded_attribute_terms(attr_name, attr_id, aliases)
    fam = family(attr_name, attr_id)

    ev_items = evidence_items(row)
    evidence_terms = terms_from_values(
        [e.get("key", "") for e in ev_items],
        [e.get("text", "") for e in ev_items],
    )
    claim_checks = [
        validate_claim_segment(str(seg.get("text", "") or ""), attr_terms, evidence_terms, fam, attr_name)
        for seg in claim_segments(row)
    ]
    evidence_checks = [validate_evidence_item(item, attr_terms, fam, attr_name) for item in ev_items]

    valid_claim = any(c["status"] == "valid_attribute_claim" for c in claim_checks)
    valid_evidence = any(e["status"] == "valid_product_evidence" for e in evidence_checks)
    review_claim = bool(claim_checks) and not valid_claim
    review_evidence = bool(evidence_checks) and not valid_evidence
    label = label_quality(row)

    issues: list[str] = []
    if not claim_checks:
        issues.append("missing_claim_after_gate")
    elif review_claim:
        issues.append("claim_attribute_alignment_review")
    if not evidence_checks:
        issues.append("missing_evidence_after_gate")
    elif review_evidence:
        issues.append("product_evidence_alignment_review")
    if not label["supported_by_aligned_review"]:
        issues.append("proposal_low_confidence_negative_label")

    status = "triplet_aligned"
    if not valid_claim or not valid_evidence:
        status = "needs_repair_before_training"
    elif not label["supported_by_aligned_review"]:
        status = "triplet_aligned_low_confidence_label"

    return {
        "pair_id": pair_id(row),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": attr_id,
        "attribute_name": attr_name,
        "attribute_family": fam,
        "y": row.get("y"),
        "c": row.get("c"),
        "status": status,
        "issues": issues,
        "claim_valid": valid_claim,
        "evidence_valid": valid_evidence,
        "label": label,
        "claim_checks": claim_checks[:12],
        "evidence_checks": evidence_checks[:12],
    }


def write_md(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Proposal Triplet Alignment Audit v2",
        "",
        "## Principle",
        "This gate checks whether an already complete record is a natural",
        "`(attribute-grounded claim, product evidence, proposal label)` triplet.",
        "It does not remove hard samples by score and it does not relabel rows.",
        "Rows that fail the gate are routed back to repair queues.",
        "",
        "## Outputs",
        f"- audit: `{report['audit_out']}`",
        f"- aligned training pool: `{report['aligned_out']}`",
        f"- aligned label-supported core: `{report['label_supported_out']}`",
        f"- repair queue: `{report['repair_out']}`",
        "",
        "## Summary",
        f"- input rows: `{report['input_n']}`",
        f"- status: `{report['status']}`",
        f"- labels by status: `{report['labels_by_status']}`",
        f"- issue counts: `{report['issues']}`",
        f"- attribute families: `{report['families']}`",
        "",
        "## Interpretation",
        "The aligned training pool is suitable for the next controlled model run",
        "because every retained row has at least one attribute-specific SRT claim",
        "and one attribute-specific product evidence item.  Low-confidence",
        "proposal negatives are kept with their original sample weights rather",
        "than discarded.  The label-supported core is only a robustness/audit view.",
        "",
        "## Top Repair Examples",
    ]
    for ex in report.get("repair_examples", [])[:12]:
        lines.append(
            f"- `{ex['pair_id']}` {ex['attribute_name']} y={ex['y']} "
            f"issues={ex['issues']} claim={ex['claim_excerpt']} evidence={ex['evidence_excerpt']}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_proposal_complete_claim_evidence_v1_20260613.jsonl")
    ap.add_argument("--audit_out", default="data/final/repaired_v1/proposal_triplet_alignment_audit_v2_20260613.jsonl")
    ap.add_argument("--aligned_out", default="data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_v2_20260613.jsonl")
    ap.add_argument("--label_supported_out", default="data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_label_supported_v2_20260613.jsonl")
    ap.add_argument("--repair_out", default="data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/proposal_triplet_alignment_audit_v2_20260613.report.json")
    ap.add_argument("--md", default="docs/PROPOSAL_TRIPLET_ALIGNMENT_AUDIT_V2_20260613.md")
    args = ap.parse_args()

    rows = list(read_jsonl(args.dataset))
    audits = [audit_row(row) for row in rows]
    by_id = {a["pair_id"]: a for a in audits}

    aligned_rows = []
    label_supported_rows = []
    repair_rows = []
    for row in rows:
        a = by_id[pair_id(row)]
        enriched = dict(row)
        enriched["_proposal_triplet_alignment_v2"] = a
        if a["claim_valid"] and a["evidence_valid"]:
            aligned_rows.append(enriched)
            if a["label"]["supported_by_aligned_review"]:
                label_supported_rows.append(enriched)
        else:
            repair_rows.append(enriched)

    write_jsonl(args.audit_out, audits)
    write_jsonl(args.aligned_out, aligned_rows)
    write_jsonl(args.label_supported_out, label_supported_rows)
    write_jsonl(args.repair_out, repair_rows)

    labels_by_status: dict[str, dict[str, int]] = {}
    for a in audits:
        labels_by_status.setdefault(str(a["status"]), Counter())
        labels_by_status[str(a["status"])][str(a["y"])] += 1
    report = {
        "dataset": args.dataset,
        "audit_out": args.audit_out,
        "aligned_out": args.aligned_out,
        "label_supported_out": args.label_supported_out,
        "repair_out": args.repair_out,
        "input_n": len(rows),
        "aligned_n": len(aligned_rows),
        "label_supported_n": len(label_supported_rows),
        "repair_n": len(repair_rows),
        "status": dict(Counter(str(a["status"]) for a in audits)),
        "labels_by_status": {k: dict(v) for k, v in labels_by_status.items()},
        "issues": dict(Counter(issue for a in audits for issue in a["issues"])),
        "families": dict(Counter(str(a["attribute_family"]) for a in audits)),
        "repair_examples": [
            {
                "pair_id": a["pair_id"],
                "attribute_name": a["attribute_name"],
                "y": a["y"],
                "issues": a["issues"],
                "claim_excerpt": " | ".join(c["text"] for c in a["claim_checks"][:2])[:180],
                "evidence_excerpt": " | ".join(
                    (f"{e['source']}:{e['key']}:{e['text']}").strip(":")
                    for e in a["evidence_checks"][:2]
                )[:180],
            }
            for a in audits
            if a["status"] == "needs_repair_before_training"
        ][:80],
    }
    write_json(args.report, report)
    write_md(report, args.md)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
