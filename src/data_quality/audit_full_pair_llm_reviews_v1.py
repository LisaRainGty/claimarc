"""Audit full-pair LLM reconstruction reviews before dataset promotion.

This script is a local quality gate.  It checks whether each reconstructed row
obeys the proposal-level label definition:

1. a recoverable livestream claim,
2. product-side evidence from title/params/OCR/VLM, and
3. a consumer comment relation judged against that same claim.

It does not drop rows or optimize benchmark difficulty.  Rows with problems are
flagged for rerun, silver handling, or manual inspection before promotion.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


VALID_ACTION = {
    "promote_candidate",
    "silver_review",
    "rerun_claim",
    "rerun_evidence",
    "rerun_joint",
    "drop_no_reconstructable_claim",
}
VALID_CONFIDENCE = {"high", "medium", "low", ""}
VALID_EVIDENCE_SOURCE = {
    "product_title",
    "params",
    "detail_image_ocr",
    "detail_image_vlm",
    "none",
    "",
}
VALID_CLAIM_EVIDENCE_REL = {
    "supports_claim",
    "contradicts_claim",
    "insufficient",
    "",
}
VALID_COMMENT_REL = {"support", "refute", "mixed", "unclear", "not_aligned", ""}
IDENTITY_VALUE_ATTRS = {"品牌", "型号", "货号", "条码", "条形码", "执行标准"}
PRICE_ATTR_TERMS = {"价格", "售价", "到手价"}
QUANTITY_ATTR_TERMS = {"净含量", "数量", "包数", "袋数", "件数", "重量", "容量", "尺寸"}
PRICE_VALUE_JUDGMENT_TERMS = {"太贵", "偏贵", "不便宜", "小贵", "不值", "物没价廉", "价格合理", "合理性"}
PRICE_OVERCHARGE_CUES = {"多收", "贵了", "贵", "涨价", "不是这个价", "价格不符", "实付", "到手不是", "付款"}
QUANTITY_VALUE_JUDGMENT_TERMS = {"太少", "少的可怜", "量少", "分量少", "不多", "最少也得", "应该", "应为", "不够"}
NUMERIC_CONFLICT_CUES = {"不是", "不符", "少发", "少给", "只有", "收到少", "实付", "到手不是", "降价", "买成"}
COMMERCIAL_PROMISE_ATTRS = {"售卖方式", "购买渠道", "广告宣传", "宣传", "活动信息"}
SUBJECTIVE_EVAL_ATTR_TERMS = {"智商税", "虚假宣传", "商品质量", "真实性评价", "性价比", "体验", "感受", "评价", "推荐"}
COLOR_ATTR_TERMS = {"颜色", "包装颜色", "色"}
EXPECTATION_MISMATCH_CUES = {"以为", "以爲", "没看清", "本来是买", "本来想买", "结果来一看"}
COUNT_UNIT_CUES = {"个", "颗", "件", "瓶", "袋", "双", "盒", "片", "支", "只", "排"}
BATTERY_CAPACITY_UNIT_CUES = {"mah", "毫安", "安时", "ah"}
NON_TEXTILE_PRODUCT_TERMS = {"洗发", "沐浴", "含片", "豆奶", "粉条", "电池", "眼镜", "隔离霜", "脱毛仪"}
EXHAUSTIVE_ENUM_CUES = {
    "两个颜色",
    "两种颜色",
    "2个颜色",
    "2种颜色",
    "只有",
    "只",
    "一共",
    "总共",
    "就这",
}
COLOR_VALUES = {
    "白色": "白",
    "白底": "白",
    "红色": "红",
    "红底": "红",
    "绿色": "绿",
    "绿底": "绿",
    "蓝色": "蓝",
    "蓝底": "蓝",
    "黑色": "黑",
    "黑底": "黑",
    "黄色": "黄",
    "黄底": "黄",
    "橙色": "橙",
    "橙底": "橙",
    "粉色": "粉",
    "粉底": "粉",
    "紫色": "紫",
    "紫底": "紫",
    "灰色": "灰",
    "灰底": "灰",
    "银色": "银",
    "金色": "金",
    "棕色": "棕",
    "咖啡色": "咖啡",
    "米色": "米",
    "透明": "透明",
    "卡其": "卡其",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return clean(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def compact(text: Any) -> str:
    return "".join(clean(text).split()).lower()


def boolish(value: Any) -> bool:
    return bool(value)


def int01(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        return 0


def identity_expected_values(queue_row: dict[str, Any]) -> list[str]:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if attr not in IDENTITY_VALUE_ATTRS:
        return []
    raw_params = queue_row.get("raw_params") or {}
    values: list[str] = []
    if isinstance(raw_params, dict):
        for key, val in raw_params.items():
            key_s = clean(key).strip("<>")
            if attr == key_s or attr in key_s or key_s in attr:
                text = clean(val)
                if 1 < len(text) <= 80:
                    values.append(text)
    return list(dict.fromkeys(values))


def claim_contains_identity_value(queue_row: dict[str, Any], claim_text: str) -> bool | None:
    vals = identity_expected_values(queue_row)
    if not vals:
        return None
    claim_norm = compact(claim_text)
    return any(compact(v) and compact(v) in claim_norm for v in vals)


def numeric_value_judgment_refutes(queue_row: dict[str, Any], review: dict[str, Any]) -> list[str]:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    is_price = any(t in attr for t in PRICE_ATTR_TERMS)
    is_quantity = any(t in attr for t in QUANTITY_ATTR_TERMS)
    if not is_price and not is_quantity:
        return []
    mentions = queue_row.get("consumer_mentions") or []
    out: list[str] = []
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        text = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        reason = clean(item.get("reason"))
        blob = text + " " + reason
        if any(cue in blob for cue in NUMERIC_CONFLICT_CUES):
            continue
        if is_price and any(term in blob for term in PRICE_VALUE_JUDGMENT_TERMS):
            out.append(f"cid={cid}:price_value_judgment")
        if is_quantity and any(term in blob for term in QUANTITY_VALUE_JUDGMENT_TERMS):
            out.append(f"cid={cid}:quantity_value_judgment")
    return out[:5]


def extract_price_values(text: Any) -> list[float]:
    blob = clean(text)
    vals: list[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:元|块|¥|￥)", blob):
        vals.append(float(m.group(1)))
    for m in re.finditer(r"(\d+)\s*(?:元|块)\s*(\d)", blob):
        vals.append(float(f"{m.group(1)}.{m.group(2)}"))
    return vals


def price_comment_not_refuting_claim_value(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if not any(t in attr for t in PRICE_ATTR_TERMS) or rel.get("refute", 0) <= 0:
        return False
    claim_prices = extract_price_values(review.get("claim_text"))
    if not claim_prices:
        return False
    claim_price = max(claim_prices)
    mentions = queue_row.get("consumer_mentions") or []
    comment_prices: list[float] = []
    overcharge_text = False
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        span = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        blob = span + " " + clean(item.get("reason"))
        comment_prices.extend(extract_price_values(blob))
        if any(cue in blob for cue in PRICE_OVERCHARGE_CUES):
            overcharge_text = True
    if not comment_prices:
        return False
    if max(comment_prices) <= claim_price * 1.05 and not overcharge_text:
        return True
    if max(comment_prices) <= claim_price * 1.05 and min(comment_prices) < claim_price * 0.8:
        return True
    return False


def commercial_promise_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return attr in COMMERCIAL_PROMISE_ATTRS


def subjective_eval_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return any(term in attr for term in SUBJECTIVE_EVAL_ATTR_TERMS)


def consumer_expectation_mismatch(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> bool:
    if not (rel.get("support", 0) > 0 and rel.get("refute", 0) > 0):
        return False
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if not any(term in attr for term in {"数量", "规格", "包装", "产品名称"}):
        return False
    mentions = queue_row.get("consumer_mentions") or []
    refute = 0
    expectation = 0
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        refute += 1
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        span = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        blob = span + " " + clean(item.get("reason"))
        if any(cue in blob for cue in EXPECTATION_MISMATCH_CUES):
            expectation += 1
    return bool(refute and expectation / refute >= 0.5)


def attribute_semantic_drift(queue_row: dict[str, Any], review: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>").lower()
    if "电池容量" not in attr:
        if "面料" in attr:
            title = clean(queue_row.get("product_title"))
            return any(term in title for term in NON_TEXTILE_PRODUCT_TERMS)
        return False
    blob = (clean(review.get("claim_text")) + " " + clean(review.get("evidence_text"))).lower()
    has_count_unit = any(cue in blob for cue in COUNT_UNIT_CUES)
    has_capacity_unit = any(cue in blob for cue in BATTERY_CAPACITY_UNIT_CUES)
    return bool(has_count_unit and not has_capacity_unit)


def conflicting_comment_relation(rel: Counter) -> bool:
    return bool(rel.get("support", 0) > 0 and rel.get("refute", 0) > 0)


def color_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return any(term in attr for term in COLOR_ATTR_TERMS)


def color_values(text: Any) -> set[str]:
    blob = clean(text)
    return {value for token, value in COLOR_VALUES.items() if token in blob}


def exhaustive_enum_claim(text: Any) -> bool:
    blob = clean(text)
    return any(cue in blob for cue in EXHAUSTIVE_ENUM_CUES)


def enumeration_claim_evidence_extra_values(queue_row: dict[str, Any], review: dict[str, Any]) -> list[str]:
    if not color_attr(queue_row):
        return []
    claim_text = clean(review.get("claim_text"))
    evidence_text = clean(review.get("evidence_text"))
    if not exhaustive_enum_claim(claim_text):
        return []
    claim_vals = color_values(claim_text)
    evidence_vals = color_values(evidence_text)
    extra = sorted(evidence_vals - claim_vals)
    return extra if claim_vals and evidence_vals and extra else []


def read_reviews(path: str | Path) -> tuple[dict[str, dict[str, Any]], Counter]:
    path = Path(path)
    if not path.exists():
        return {}, Counter()
    out: dict[str, dict[str, Any]] = {}
    duplicates: Counter = Counter()
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid in out:
            duplicates[pid] += 1
        out[pid] = row
    return out, duplicates


def srt_candidate_texts(queue_row: dict[str, Any]) -> list[str]:
    pref = queue_row.get("srt_prefilter") or {}
    return [clean(c.get("text")) for c in (pref.get("claim_candidates") or []) if clean(c.get("text"))]


def claim_in_prefilter(queue_row: dict[str, Any], claim: str) -> bool | None:
    claim_norm = compact(claim)
    if len(claim_norm) < 4:
        return None
    candidates = [compact(x) for x in srt_candidate_texts(queue_row)]
    if not candidates:
        return None
    return any(claim_norm in cand or cand in claim_norm for cand in candidates)


def add_flag(flags: list[dict[str, str]], severity: str, code: str, detail: str = "") -> None:
    flags.append({"severity": severity, "code": code, "detail": detail})


def comment_stats(review: dict[str, Any], max_comments: int, flags: list[dict[str, str]]) -> Counter:
    stats: Counter = Counter()
    judgments = review.get("comment_judgments")
    if not isinstance(judgments, list):
        add_flag(flags, "high", "comment_judgments_not_list")
        return stats
    seen: Counter = Counter()
    for item in judgments:
        if not isinstance(item, dict):
            add_flag(flags, "high", "comment_judgment_not_object")
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        if cid < 1 or cid > max_comments:
            add_flag(flags, "high", "comment_cid_out_of_range", str(cid))
            continue
        seen[cid] += 1
        rel = clean(item.get("relation"))
        aligned = boolish(item.get("aligned_to_claim"))
        if rel not in VALID_COMMENT_REL:
            add_flag(flags, "high", "invalid_comment_relation", rel)
        if aligned and rel not in {"support", "refute", "mixed"}:
            add_flag(flags, "high", "aligned_comment_without_valid_relation", f"cid={cid}, rel={rel}")
        if (not aligned) and rel in {"support", "refute", "mixed"}:
            add_flag(flags, "medium", "relation_present_but_not_aligned", f"cid={cid}, rel={rel}")
        if aligned:
            stats[rel] += 1
    for cid, count in seen.items():
        if count > 1:
            add_flag(flags, "low", "duplicate_comment_judgment", str(cid))
    return stats


def promotion_state(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> str:
    if review.get("__error__"):
        return "llm_error"
    claim_found = boolish(review.get("claim_found"))
    evidence_found = boolish(review.get("product_evidence_found"))
    claim_evidence_relation = clean(review.get("claim_evidence_relation"))
    if not claim_found:
        return "repair_missing_claim"
    if not evidence_found:
        if rel.get("refute", 0) > 0:
            return "silver_refute_missing_product_evidence"
        return "repair_missing_evidence"
    if claim_evidence_relation in {"", "insufficient"}:
        if rel.get("refute", 0) > 0:
            return "silver_refute_insufficient_product_evidence"
        return "repair_insufficient_product_evidence"
    if claim_contains_identity_value(queue_row, clean(review.get("claim_text"))) is False:
        return "repair_identity_claim_value"
    if numeric_value_judgment_refutes(queue_row, review):
        return "repair_numeric_value_judgment"
    if price_comment_not_refuting_claim_value(queue_row, review, rel):
        return "silver_price_value_not_direct_refute"
    if commercial_promise_attr(queue_row):
        return "silver_commercial_promise_attribute"
    if subjective_eval_attr(queue_row):
        return "silver_subjective_eval_attribute"
    if consumer_expectation_mismatch(queue_row, review, rel):
        return "silver_consumer_expectation_mismatch"
    if attribute_semantic_drift(queue_row, review):
        return "silver_attribute_semantic_drift"
    if conflicting_comment_relation(rel):
        return "silver_conflicting_comment_relation"
    if enumeration_claim_evidence_extra_values(queue_row, review):
        return "silver_enumeration_evidence_extra_values"
    if rel.get("refute", 0) > 0:
        return "main_positive_refute"
    if rel.get("support", 0) > 0 and rel.get("mixed", 0) == 0:
        return "main_negative_support"
    if rel.get("mixed", 0) > 0:
        return "silver_mixed_comment_relation"
    return "lowinfo_no_aligned_comment"


def audit_one(queue_row: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    flags: list[dict[str, str]] = []
    if review is None:
        add_flag(flags, "missing", "missing_review")
        return {
            "pair_id": pair_id(queue_row),
            "priority": queue_row.get("priority"),
            "queue_type": queue_row.get("queue_type"),
            "category": queue_row.get("category"),
            "attribute_name": queue_row.get("attribute_name"),
            "new_y": None,
            "promotion_state": "missing_review",
            "flags": flags,
        }

    max_comments = len(queue_row.get("consumer_mentions") or [])
    rel = comment_stats(review, max_comments, flags)
    state = promotion_state(queue_row, review, rel)

    claim_found = boolish(review.get("claim_found"))
    claim_text = clean(review.get("claim_text"))
    evidence_found = boolish(review.get("product_evidence_found"))
    evidence_text = clean(review.get("evidence_text"))
    evidence_source_type = clean(review.get("evidence_source_type"))
    claim_evidence_relation = clean(review.get("claim_evidence_relation"))
    confidence = clean(review.get("confidence")).lower()
    action = clean(review.get("action"))
    new_y = int01(review.get("new_y"))
    raw_new_y = review.get("raw_new_y")

    if review.get("__error__"):
        add_flag(flags, "high", "llm_error", clean(review.get("__error__"))[:80])
    if claim_found and not claim_text:
        add_flag(flags, "high", "claim_found_empty_text")
    if (not claim_found) and claim_text:
        add_flag(flags, "low", "claim_text_present_but_claim_found_false")
    identity_hit = claim_contains_identity_value(queue_row, claim_text) if claim_found else None
    if identity_hit is False:
        severity = "high" if state in {"main_positive_refute", "main_negative_support"} or action == "promote_candidate" else "medium"
        add_flag(
            flags,
            severity,
            "identity_attribute_claim_lacks_value",
            f"attr={clean(queue_row.get('attribute_name'))}",
        )
    hit = claim_in_prefilter(queue_row, claim_text)
    if hit is False:
        add_flag(flags, "medium", "claim_not_in_top_srt_prefilter", "manual source check needed")

    if evidence_found and evidence_source_type in {"none", ""}:
        add_flag(flags, "high", "evidence_found_without_source_type")
    if evidence_found and not evidence_text:
        add_flag(flags, "high", "evidence_found_empty_text")
    if (not evidence_found) and evidence_text:
        add_flag(flags, "low", "evidence_text_present_but_evidence_found_false")
    if evidence_source_type not in VALID_EVIDENCE_SOURCE:
        add_flag(flags, "high", "invalid_evidence_source_type", evidence_source_type)
    if claim_evidence_relation not in VALID_CLAIM_EVIDENCE_REL:
        add_flag(flags, "high", "invalid_claim_evidence_relation", claim_evidence_relation)
    if confidence not in VALID_CONFIDENCE:
        add_flag(flags, "medium", "invalid_confidence", confidence)
    if action not in VALID_ACTION:
        add_flag(flags, "medium", "invalid_action", action)
    if action == "promote_candidate" and claim_evidence_relation in {"", "insufficient"}:
        add_flag(flags, "high", "promote_with_insufficient_claim_evidence_relation", claim_evidence_relation or "empty")
    numeric_judgments = numeric_value_judgment_refutes(queue_row, review)
    if numeric_judgments:
        severity = "high" if state in {"main_positive_refute", "main_negative_support"} or action == "promote_candidate" else "medium"
        add_flag(flags, severity, "numeric_value_judgment_used_as_refute", ";".join(numeric_judgments))
    if price_comment_not_refuting_claim_value(queue_row, review, rel):
        add_flag(flags, "medium", "price_value_not_direct_refute_requires_silver")
    if commercial_promise_attr(queue_row) and state in {"main_positive_refute", "main_negative_support"}:
        add_flag(flags, "medium", "commercial_promise_attribute_requires_manual")
    if subjective_eval_attr(queue_row):
        add_flag(flags, "medium", "subjective_eval_attribute_requires_silver")
    if consumer_expectation_mismatch(queue_row, review, rel):
        add_flag(flags, "medium", "consumer_expectation_mismatch_requires_silver")
    if attribute_semantic_drift(queue_row, review):
        add_flag(flags, "medium", "attribute_semantic_drift_requires_silver")
    if conflicting_comment_relation(rel):
        add_flag(flags, "medium", "conflicting_comment_relation_requires_silver")
    extra_enum_values = enumeration_claim_evidence_extra_values(queue_row, review)
    if extra_enum_values:
        severity = "high" if action == "promote_candidate" else "medium"
        add_flag(
            flags,
            severity,
            "enumeration_claim_evidence_extra_values",
            ",".join(extra_enum_values),
        )

    expected_y = 1 if claim_found and rel.get("refute", 0) > 0 else 0
    if new_y != expected_y:
        add_flag(flags, "high", "new_y_inconsistent_with_claim_comment_relation", f"new_y={new_y}, expected={expected_y}")
    if raw_new_y not in {None, ""} and int01(raw_new_y) != expected_y:
        add_flag(flags, "low", "raw_llm_y_overridden_by_clean_rule", f"raw_new_y={raw_new_y}, expected={expected_y}")
    if new_y == 1 and not evidence_found:
        add_flag(flags, "medium", "positive_label_missing_product_evidence_for_main")
    if claim_evidence_relation == "contradicts_claim" and rel.get("refute", 0) == 0:
        add_flag(flags, "info", "mechanism_contradiction_without_consumer_refute")
    if action == "promote_candidate" and state not in {"main_positive_refute", "main_negative_support"}:
        add_flag(flags, "medium", "promote_action_but_not_main_ready", state)

    return {
        "pair_id": pair_id(queue_row),
        "priority": queue_row.get("priority"),
        "queue_type": queue_row.get("queue_type"),
        "category": queue_row.get("category"),
        "attribute_name": queue_row.get("attribute_name"),
        "new_y": new_y,
        "confidence": confidence,
        "promotion_state": state,
        "comment_relation_counts": dict(rel),
        "flags": flags,
    }


def summarize(audits: list[dict[str, Any]], review_rows: int, duplicates: Counter) -> dict[str, Any]:
    flag_code: Counter = Counter()
    flag_severity: Counter = Counter()
    flagged_rows = 0
    high_rows = 0
    medium_or_high_rows = 0
    for row in audits:
        flags = row.get("flags") or []
        if flags:
            flagged_rows += 1
        severities = {f.get("severity") for f in flags}
        if "high" in severities:
            high_rows += 1
        if "high" in severities or "medium" in severities:
            medium_or_high_rows += 1
        for f in flags:
            flag_code[clean(f.get("code"))] += 1
            flag_severity[clean(f.get("severity"))] += 1
    return {
        "queue_rows": len(audits),
        "review_rows": review_rows,
        "matched_reviews": sum(1 for r in audits if r.get("promotion_state") != "missing_review"),
        "missing_reviews": sum(1 for r in audits if r.get("promotion_state") == "missing_review"),
        "duplicate_review_pairs": len(duplicates),
        "duplicate_review_events": sum(duplicates.values()),
        "flagged_rows": flagged_rows,
        "high_flag_rows": high_rows,
        "medium_or_high_flag_rows": medium_or_high_rows,
        "flag_severity": dict(flag_severity),
        "flag_code": dict(flag_code),
        "promotion_state": dict(Counter(clean(r.get("promotion_state")) for r in audits)),
        "new_y": dict(Counter(str(r.get("new_y")) for r in audits)),
        "confidence": dict(Counter(clean(r.get("confidence")) for r in audits if r.get("confidence") is not None)),
        "category": dict(Counter(clean(r.get("category")) for r in audits)),
    }


def write_markdown(path: str | Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Full Pair LLM Review Audit v1",
        "",
        "This report audits LLM/VLM reconstruction reviews before promotion.",
        "It checks label-definition consistency rather than benchmark separability.",
        "",
        "## Inputs",
        "",
        f"- queue: `{args.queue}`",
        f"- reviews: `{args.reviews}`",
        "",
        "## Outputs",
        "",
        f"- report json: `{args.out}`",
        f"- flagged rows: `{args.flagged}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Gate Interpretation",
        "",
        "- `high` flags block main promotion until rerun or manual repair.",
        "- `medium` flags require manual sampling or silver routing.",
        "- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.",
        "- Missing reviews are expected before the LLM pilot has been executed.",
        "",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json")
    ap.add_argument("--flagged", default="data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_LLM_REVIEW_AUDIT_20260614.md")
    args = ap.parse_args()

    queue_rows = list(read_jsonl(args.queue))
    reviews, duplicates = read_reviews(args.reviews)
    audits = [audit_one(row, reviews.get(pair_id(row))) for row in queue_rows]
    flagged = [row for row in audits if row.get("flags")]
    report = summarize(audits, len(reviews), duplicates)
    report.update({
        "queue": args.queue,
        "reviews": args.reviews,
        "flagged": args.flagged,
        "out": args.out,
    })
    write_json(args.out, report)
    write_jsonl(args.flagged, flagged)
    write_markdown(args.markdown, report, args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
