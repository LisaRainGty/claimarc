"""Validate existing Stage-B claims against product-only A_cmt v2.

This is a deterministic bridge before a full LLM rerun of B1.  It reuses the
current grounded claim_list files, but only accepts claims whose attribute is
kept by `build_acmt_product_v2.py` and whose text is plausibly a product-fact
claim rather than a promo/order/interaction utterance.

Outputs:
- pair_skeleton_product_v2.jsonl
- claim_attribute_validation_v2.jsonl
- claim_attribute_validation_v2_report.json
- docs/CLAIM_ATTRIBUTE_VALIDATION_V2.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common import srt as S
from common.io_utils import char_jaccard, normalize, read_json, read_jsonl, write_json, write_jsonl


PROMO_TERMS = {
    "下单", "拍", "秒", "秒杀", "库存", "链接", "小黄车", "优惠", "优惠券",
    "券", "福利", "赠", "赠品", "包邮", "邮费", "加购", "领券", "到手价", "价格",
    "直播间", "主播", "客服", "售后", "发货", "物流", "退换", "保价",
    "块钱", "卖了", "销量",
}
HARD_PROMO_TERMS = {
    "下单", "拍", "秒", "秒杀", "库存", "链接", "小黄车", "优惠", "优惠券",
    "券", "福利", "赠", "赠品", "包邮", "邮费", "加购", "领券", "到手价", "价格",
    "客服", "售后", "发货", "物流", "退换", "保价",
    "块钱", "卖了", "销量",
}
PROMO_RESCUE_TERMS = {
    "材质", "面料", "材料", "成分", "配料", "含", "不含", "棉", "羊毛",
    "真皮", "牛皮", "聚酯", "腈纶", "混纺", "规格", "尺寸", "尺码",
    "容量", "净含量", "重量", "厚", "薄", "功率", "电池", "毫安",
    "mah", "防水", "加绒", "保质期", "产地", "型号", "货号", "标准",
}

GENERIC_PRAISE_TERMS = {
    "闭眼", "放心", "值得", "好看", "不错", "高级", "舒服", "喜欢", "满意",
    "划算", "性价比", "质量好", "品质好", "推荐",
}

FACT_CUE_TERMS = {
    "是", "有", "采用", "使用", "含", "不含", "材质", "面料", "成分", "配料",
    "规格", "尺寸", "尺码", "容量", "净含量", "重量", "厚", "薄", "颜色",
    "色", "款", "版型", "产地", "保质期", "功率", "电池", "防水", "加绒",
    "口袋", "内兜", "包装", "型号", "货号", "标准", "弹力", "袖长", "裤型",
}
MATERIAL_VALUE_TERMS = {
    "羊毛", "绵羊毛", "棉", "纯棉", "全棉", "真皮", "牛皮", "皮革",
    "聚酯", "聚酯纤维", "锦纶", "腈纶", "氨纶", "涤纶", "粘纤",
    "莱赛尔", "莫代尔", "羽绒", "白鸭绒", "鸭绒", "鹅绒", "混纺",
    "羊绒", "亚麻", "桑蚕丝", "桑残丝", "醋酸", "三醋", "粘胶", "pu", "pvc", "乳胶",
    "硅胶", "不锈钢", "陶瓷", "玻璃", "碳钢", "塑料", "树脂",
}
STRUCTURE_VALUE_TERMS = {
    "双层", "单层", "三层", "加厚", "薄款", "厚款", "高腰", "低腰",
    "直筒", "阔腿", "修身", "宽松", "a字", "A字", "圆领", "v领", "V领",
    "松紧", "弹力", "短袖", "长袖", "七分袖", "九分裤", "连帽", "翻领",
    "厚底", "低帮", "高帮", "尖头", "圆头", "方头", "平底", "坡跟",
    "短款", "中长款", "长款", "超长款", "拼接款", "贝雷", "渔夫帽",
    "棒球帽", "鸭舌帽", "包头帽", "A版", "a版",
}
STYLE_VALUE_RE = re.compile(
    r"(尖头|圆头|方头|厚底|平底|坡跟|低帮|高帮|短款|中长款|长款|超长款|"
    r"拼接款|连帽|翻领|圆领|v领|V领|A字|a字|修身|宽松|直筒|阔腿|高腰|低腰|"
    r"贝雷帽?|渔夫帽?|棒球帽?|鸭舌帽?|包头帽?|小披肩)"
)
SEASON_VALUE_RE = re.compile(r"(春夏|春秋|秋冬|冬天|夏天|四季|换季|零下|保暖|抗寒|防寒)")
AUDIENCE_VALUE_RE = re.compile(r"(男生|女生|男女|男士|女士|儿童|宝宝|婴儿|老人|学生|孕妇|小个子|大个子|胖|瘦)")
FUNCTION_VALUE_RE = re.compile(r"(快充|防水|防风|三防|透气|保暖|抗寒|防晒|除螨|杀菌|控油|补水|美白|清洁|收纳)")
GENERIC_FACT_CUES = {"是", "有", "个", "款"}
VISUAL_GENERIC_CUES = {"色", "颜色", "款", "款式"}
IDENTITY_CUE_TERMS = {"品牌", "型号", "货号", "名称", "产地", "保质期", "生产日期", "执行标准", "标准", "条形码", "条码"}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?", re.I)
MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|mm|kg|g|ml|l|mah|w|瓦|克|斤|升|毫升|厘米|毫米|"
    r"米|寸|码|%|天|月|年|个|只|件|包|袋|盒|瓶)",
    re.I,
)
RANGE_SIZE_RE = re.compile(r"\d+(?:\.\d+)?\s*[-~到至]\s*\d+(?:\.\d+)?\s*(?:cm|mm|厘米|毫米|米|寸|码|斤|克|kg|g|ml|毫升)?", re.I)
CAPACITY_RE = re.compile(r"(mah|毫安|电量|容量|mAh|MAH)", re.I)
LINK_RE = re.compile(r"(?:\d+|[一二三四五六七八九十两]+)号(?:链接|链|令节|款|色)?")
PROMO_ONLY_NUMBER_RE = re.compile(r"^(?:\d+|[一二三四五六七八九十两]+)号(?:链接|链|令节|款|色)?")
QUESTION_RE = re.compile(r"[?？]|什么|哪[个款种]?|有没有|是不是|能不能|可以吗|行吗|看一下|看下")
PRICE_OR_SALES_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:元|块|块钱)|[一二三四五六七八九十两百千]+块(?:钱)?|到手价|价格|售价|起步价|卖了\d+|销量)")
FLAVOR_RE = re.compile(r"(味道|口味|口感|风味|香|辣|甜|酸|咸|原味|奶味|茶味|麻辣|酸辣|花香|果香)")
PACKAGE_AMOUNT_RE = re.compile(r"(净含量|净重|规格|装|一包|一袋|一盒|一箱|一瓶|每包|每袋|每盒|每瓶|克|g|kg|斤|毫升|ml|升)")
NUTRITION_PER_RE = re.compile(r"(每\s*100\s*(?:克|g)|含有|营养成分|蛋白质|脂肪|碳水|能量|热量|毫克)")
TYPE_CUE_RE = re.compile(r"(类型|品类|款型|型号|车|山地|公路|钢化膜|水光机|洗发水|面包|豆奶|豆浆|茶|粉|片|鞋|帽|袜|裤|裙|包)")
SCREEN_SIZE_RE = re.compile(r"(屏幕尺寸|屏幕大小|\d+(?:\.\d+)?\s*(?:寸|英寸|inch))", re.I)
FILM_THICKNESS_RE = re.compile(r"(\d+(?:\.\d+)?\s*mm|毫米).*(保护|加高|镜头|钢化膜)|(?:保护|加高|镜头|钢化膜).*(\d+(?:\.\d+)?\s*mm|毫米)", re.I)
COLOR_VALUE_RE = re.compile(r"(黑|白|红|蓝|绿|黄|紫|灰|棕|咖|粉|橙|金|银|杏|米色|自然色|冷白|暖白|墨色|卡其|驼色|藏青|藕粉|安可拉红)")
GENERIC_COLOR_RE = re.compile(r"^(?:这个|这款|这种|它的|颜色|这个颜色|这颜色|颜色呢|颜色的话|一定要去收|很好看|好看|显白|百搭|高级|耐看|去收|收)")
NUMERIC_ONLY_STYLE_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:的|码|寸|号|米)?\s*$")
BRAND_GENERIC_TERMS = {"厂家", "品牌", "牌子", "大品牌", "第一个", "下方"}
BRAND_GENERIC_RE = re.compile(r"^(?:厂家监督|直护大品牌的|第一个下方的品牌|下方的品牌|这个品牌|大品牌)$")


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def token_terms(meta: dict[str, Any]) -> list[str]:
    terms = [clean(meta.get("canonical_name"))]
    terms.extend(clean(x) for x in (meta.get("aliases") or [])[:30])
    out: list[str] = []
    for term in terms:
        if not term or len(term) < 2:
            continue
        if term in {"商品", "产品", "属性", "情况", "相关", "信息", "一般"}:
            continue
        out.append(term)
    return list(dict.fromkeys(out))


def term_score(text: str, terms: list[str]) -> tuple[int, list[str]]:
    nt = normalize(text)
    hits: list[str] = []
    score = 0
    for term in terms:
        nterm = normalize(term)
        if nterm and nterm in nt:
            hits.append(term)
            score += 3 if len(term) >= 3 else 2
    return score, hits[:20]


def source_family_cues(source_family: str) -> set[str]:
    if source_family == "material":
        return {"材质", "面料", "材料", "成分", "配料", "含", "不含", "绒", "棉", "皮", "纤维"} | MATERIAL_VALUE_TERMS
    if source_family == "numeric":
        return {"规格", "尺寸", "尺码", "容量", "净含量", "重量", "厚", "薄", "长度", "宽度", "高度", "件", "包"} | STRUCTURE_VALUE_TERMS
    if source_family == "visual_or_boolean":
        return {"颜色", "色", "款式", "版型", "图案", "口袋", "内兜", "闭合", "拉链", "防水", "加绒"}
    if source_family == "identity_or_spec":
        return IDENTITY_CUE_TERMS
    return FACT_CUE_TERMS


def has_any(text: str, terms: set[str]) -> list[str]:
    return sorted(t for t in terms if t and t in text)


def attribute_value_hits(attr_name: str, text: str, source_family: str) -> list[str]:
    """Narrow rescue for short but concrete attribute values."""
    hits: list[str] = []
    if source_family == "material":
        hits.extend(has_any(text, MATERIAL_VALUE_TERMS))
    if any(k in attr_name for k in ("款式", "版型", "鞋头", "帽顶", "衣长", "裤型", "裤长", "袖长")):
        hits.extend(STYLE_VALUE_RE.findall(text))
    if any(k in attr_name for k in ("风味", "香味", "口味", "味道", "口感")) and FLAVOR_RE.search(text):
        hits.append("flavor_value")
    if any(k in attr_name for k in ("适用季节", "季节")):
        hits.extend(SEASON_VALUE_RE.findall(text))
    if any(k in attr_name for k in ("适用人群", "适用对象", "人群", "对象")):
        hits.extend(AUDIENCE_VALUE_RE.findall(text))
    if any(k in attr_name for k in ("功能", "功效", "卖点")):
        hits.extend(FUNCTION_VALUE_RE.findall(text))
    if any(k in attr_name for k in ("是否加绒", "加绒", "厚度", "结构")):
        hits.extend(has_any(text, STRUCTURE_VALUE_TERMS))
    return list(dict.fromkeys(str(h) for h in hits if h))


def has_link_or_promo_shape(text: str) -> bool:
    return bool(LINK_RE.search(text) or PROMO_ONLY_NUMBER_RE.search(text))


def strong_product_signal(
    *,
    text: str,
    attr_name: str,
    own_hits: list[str],
    meaningful_fact_hits: list[str],
    number_hit: bool,
    measurement_hit: bool,
    source_family: str,
) -> bool:
    link_or_promo_shape = has_link_or_promo_shape(text)
    if source_family == "material":
        if has_any(text, MATERIAL_VALUE_TERMS):
            return True
        return bool(own_hits and any(h not in {"材质", "面料", "材料", "质地"} for h in own_hits))
    if source_family == "numeric":
        if measurement_hit or RANGE_SIZE_RE.search(text):
            return True
        if number_hit and not link_or_promo_shape and has_any(text, {"规格", "尺寸", "尺码", "容量", "净含量", "重量", "厚", "薄"}):
            return True
        return bool(has_any(text, {"厚", "薄", "加厚", "薄款", "厚款", "双层", "单层", "三层", "弹力", "松紧"}))
    if source_family == "visual_or_boolean":
        if COLOR_VALUE_RE.search(text) or has_any(text, STRUCTURE_VALUE_TERMS):
            return True
    if attribute_value_hits(attr_name, text, source_family):
        return True
    if own_hits:
        return True
    if meaningful_fact_hits and any(h not in VISUAL_GENERIC_CUES for h in meaningful_fact_hits):
        return True
    if has_any(text, PROMO_RESCUE_TERMS):
        return True
    if measurement_hit and not link_or_promo_shape:
        return True
    return False


def attribute_specific_reject(attr_name: str, text: str, own_hits: list[str]) -> str:
    if PRICE_OR_SALES_RE.search(text):
        return "promo_or_order"
    if LINK_RE.search(text) and attr_name in {"产品名称", "商品名称", "品牌", "型号", "货号", "条形码", "产品类型", "商品类型", "类型"}:
        return "promo_or_order"
    if "品牌" in attr_name:
        has_specific_hit = any(h and h not in BRAND_GENERIC_TERMS for h in own_hits)
        if BRAND_GENERIC_RE.search(text) or not has_specific_hit:
            return "wrong_attribute"
    if "颜色" in attr_name:
        if not COLOR_VALUE_RE.search(text):
            return "wrong_attribute" if GENERIC_COLOR_RE.search(text) else "too_vague"
    if "风味" in attr_name and not (own_hits or FLAVOR_RE.search(text)):
        return "wrong_attribute"
    if "净含量" in attr_name:
        if NUTRITION_PER_RE.search(text) and not any(t in text for t in ("净含量", "净重", "一包", "一袋", "一盒", "一瓶", "每包", "每袋", "每盒", "每瓶")):
            return "wrong_attribute"
        if not (own_hits or PACKAGE_AMOUNT_RE.search(text)):
            return "wrong_attribute"
    if "屏幕尺寸" in attr_name:
        if FILM_THICKNESS_RE.search(text):
            return "wrong_attribute"
        if not (own_hits or SCREEN_SIZE_RE.search(text)):
            return "wrong_attribute"
    if attr_name in {"款式", "产品款式", "商品款式"}:
        if NUMERIC_ONLY_STYLE_RE.match(text) and not own_hits:
            return "wrong_attribute"
    if attr_name in {"类型", "产品类型", "商品类型"} and not (own_hits or TYPE_CUE_RE.search(text)):
        return "wrong_attribute"
    return ""


def classify_claim(claim: dict[str, Any], aid: str, attrs: dict[str, Any]) -> dict[str, Any]:
    text = clean(claim.get("claim_text"))
    meta = attrs[aid]
    source_family = str(meta.get("source_family", ""))
    attr_name = clean(meta.get("canonical_name") or aid)
    attr_blob_text = f"{aid} {attr_name}"
    own_score, own_hits = term_score(text, token_terms(meta))
    promo_hits = has_any(text, PROMO_TERMS)
    hard_promo_hits = has_any(text, HARD_PROMO_TERMS)
    hard_promo_non_link_hits = [h for h in hard_promo_hits if h != "链接"]
    praise_hits = has_any(text, GENERIC_PRAISE_TERMS)
    fact_hits = sorted(set(has_any(text, source_family_cues(source_family)) + has_any(text, FACT_CUE_TERMS)))
    meaningful_fact_hits = [h for h in fact_hits if h not in GENERIC_FACT_CUES]
    number_hit = bool(NUMBER_RE.search(text))
    measurement_hit = bool(MEASUREMENT_RE.search(text) or RANGE_SIZE_RE.search(text))
    link_or_promo_shape = has_link_or_promo_shape(text)
    question_like = bool(QUESTION_RE.search(text))
    strong_signal = strong_product_signal(
        text=text,
        attr_name=attr_name,
        own_hits=own_hits,
        meaningful_fact_hits=meaningful_fact_hits,
        number_hit=number_hit,
        measurement_hit=measurement_hit,
        source_family=source_family,
    )

    best_other = ("", 0, [])
    for other_aid, other_meta in attrs.items():
        if other_aid == aid:
            continue
        score, hits = term_score(text, token_terms(other_meta))
        if score > best_other[1]:
            best_other = (other_aid, score, hits)

    identity_weak = (
        source_family == "identity_or_spec"
        and not own_hits
        and not has_any(text, IDENTITY_CUE_TERMS)
    )
    capacity_mismatch = (
        any(t in attr_blob_text for t in {"电池容量", "电源容量", "容量"})
        and any(t in attr_blob_text for t in {"电池", "电源"})
        and not bool(CAPACITY_RE.search(text))
    )
    specific_reject = attribute_specific_reject(attr_name, text, own_hits)

    if question_like:
        status = "too_vague"
    elif specific_reject:
        status = specific_reject
    elif identity_weak:
        status = "too_vague"
    elif capacity_mismatch:
        status = "wrong_attribute"
    elif hard_promo_hits and (hard_promo_non_link_hits or not strong_signal):
        status = "promo_or_order"
    elif promo_hits and not strong_signal:
        status = "promo_or_order"
    elif link_or_promo_shape and not strong_signal and not (meaningful_fact_hits and any(h not in VISUAL_GENERIC_CUES for h in meaningful_fact_hits)):
        status = "promo_or_order"
    elif best_other[1] >= own_score + 4 and best_other[1] >= 4:
        status = "wrong_attribute"
    elif strong_signal:
        status = "direct"
    elif praise_hits:
        status = "too_vague"
    else:
        status = "too_vague"

    # For broad visual attributes, direct alias hits like 颜色/款式 are often
    # enough.  Identity/spec attributes are not rescued here; otherwise generic
    # visual words such as "配色" can incorrectly validate barcode/brand pairs.
    if (
        status == "too_vague"
        and not question_like
        and str(meta.get("source_family")) == "visual_or_boolean"
        and meaningful_fact_hits
        and (
            own_hits
            or attr_name in {"颜色", "颜色分类", "款式", "版型", "图案"}
            or any(h in {"颜色", "色", "款式", "款", "版型", "图案"} for h in meaningful_fact_hits)
        )
    ):
        status = "direct"

    return {
        "claim_id": claim.get("claim_id"),
        "attribute_id": aid,
        "claim_text": text,
        "validation_status": status,
        "own_score": own_score,
        "own_hits": own_hits,
        "fact_hits": fact_hits,
        "meaningful_fact_hits": meaningful_fact_hits,
        "attribute_value_hits": attribute_value_hits(attr_name, text, source_family),
        "number_hit": number_hit,
        "measurement_hit": measurement_hit,
        "link_or_promo_shape": link_or_promo_shape,
        "question_like": question_like,
        "identity_weak": identity_weak,
        "capacity_mismatch": capacity_mismatch,
        "attribute_specific_reject": specific_reject,
        "strong_product_signal": strong_signal,
        "promo_hits": promo_hits,
        "hard_promo_hits": hard_promo_hits,
        "hard_promo_non_link_hits": hard_promo_non_link_hits,
        "praise_hits": praise_hits,
        "best_other_attribute_id": best_other[0],
        "best_other_score": best_other[1],
        "best_other_hits": best_other[2],
        "srt_file": claim.get("srt_file"),
        "srt_path": claim.get("srt_path"),
        "start_ts": claim.get("start_ts"),
        "end_ts": claim.get("end_ts"),
    }


def build_passage(claims: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    claims = sorted(claims, key=lambda c: (c.get("srt_file", ""), S.ts_to_seconds(c.get("start_ts", "00:00:00,000"))))
    dedup: list[dict[str, Any]] = []
    for c in claims:
        if dedup and char_jaccard(clean(c.get("claim_text")), clean(dedup[-1].get("claim_text"))) >= config.B3_CLAIM_JACCARD_DEDUP:
            continue
        dedup.append(c)
    passage = "\n---\n".join(clean(c.get("claim_text")) for c in dedup)
    segments = [{
        "claim_id": c.get("claim_id"),
        "clip_id": c.get("srt_file", ""),
        "t_start": S.ts_to_seconds(c.get("start_ts", "00:00:00,000")),
        "t_end": S.ts_to_seconds(c.get("end_ts", "00:00:00,000")),
        "start_ts": c.get("start_ts", ""),
        "end_ts": c.get("end_ts", ""),
        "text": clean(c.get("claim_text")),
        "_claim_attribute_validation": c.get("_claim_attribute_validation"),
    } for c in dedup]
    return passage, segments


def read_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Claim Attribute Validation v2",
        "",
        "## Summary",
        f"- products: `{report['products']}`",
        f"- product-v2 pairs: `{report['pairs']}`",
        f"- pairs with direct claim: `{report['pairs_with_direct_claim']}`",
        f"- existing claims read: `{report['claims_read']}`",
        f"- validation status: `{report['validation_status']}`",
        "",
        "## Interpretation",
        "This is a deterministic bridge over the old B1 claim extraction.  It is",
        "not a substitute for rerunning B1 with product-v2 A_cmt, but it shows how",
        "many current claims survive a product-only schema and a basic relation gate.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default="data/processed/stageB_product_v2/acmt_product_v2.json")
    ap.add_argument("--claim_dir", default="data/processed/stageB/claim_list")
    ap.add_argument("--out_dir", default="data/processed/stageB_product_v2")
    ap.add_argument("--report", default="data/final/repaired_v1/claim_attribute_validation_v2_report.json")
    ap.add_argument("--md", default="docs/CLAIM_ATTRIBUTE_VALIDATION_V2.md")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={}) or {}
    claim_dir = Path(args.claim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    claims_read = 0
    claims_kept = 0
    status_counter: Counter[str] = Counter()
    by_product_attr: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for pid, attrs in acmt.items():
        claims = read_claims(claim_dir / f"{pid}.jsonl")
        claims_read += len(claims)
        for claim in claims:
            aid = str(claim.get("attribute_id") or "")
            if aid not in attrs:
                continue
            val = classify_claim(claim, aid, attrs)
            val["product_id"] = pid
            val["attribute_name"] = attrs[aid].get("canonical_name")
            val["source_family"] = attrs[aid].get("source_family")
            validation_rows.append(val)
            status_counter[val["validation_status"]] += 1
            if val["validation_status"] == "direct":
                out_claim = dict(claim)
                out_claim["_claim_attribute_validation"] = {
                    "status": "direct",
                    "own_hits": val["own_hits"],
                    "fact_hits": val["fact_hits"],
                    "number_hit": val["number_hit"],
                }
                by_product_attr[(pid, aid)].append(out_claim)
                claims_kept += 1

    for pid, attrs in acmt.items():
        for aid, meta in attrs.items():
            direct_claims = by_product_attr.get((pid, aid), [])
            if direct_claims:
                passage, segments = build_passage(direct_claims)
            else:
                passage, segments = "", []
            pair_rows.append({
                "pair_id": f"p{pid}__{aid}",
                "product_id": pid,
                "attribute_id": aid,
                "attribute_canonical": meta.get("canonical_name", aid),
                "aliases": meta.get("aliases", []),
                "source_family": meta.get("source_family"),
                "selection_score": meta.get("selection_score"),
                "has_claim_srt": bool(direct_claims),
                "passage": passage,
                "segments": segments,
            })

    write_jsonl(out_dir / "claim_attribute_validation_v2.jsonl", validation_rows)
    write_jsonl(out_dir / "pair_skeleton_product_v2.jsonl", pair_rows)
    report = {
        "acmt": args.acmt,
        "claim_dir": args.claim_dir,
        "products": len(acmt),
        "pairs": sum(len(v) for v in acmt.values()),
        "claims_read": claims_read,
        "claims_validated_in_product_schema": len(validation_rows),
        "claims_kept_direct": claims_kept,
        "pairs_with_direct_claim": sum(1 for r in pair_rows if r["has_claim_srt"]),
        "validation_status": dict(status_counter),
        "top_invalid_examples": [
            {
                "product_id": r.get("product_id"),
                "attribute": r.get("attribute_name"),
                "status": r.get("validation_status"),
                "claim": r.get("claim_text"),
                "promo_hits": r.get("promo_hits"),
                "best_other_attribute_id": r.get("best_other_attribute_id"),
                "best_other_hits": r.get("best_other_hits"),
            }
            for r in validation_rows if r.get("validation_status") != "direct"
        ][:80],
    }
    write_json(args.report, report)
    write_markdown(report, args.md)
    print(json.dumps({
        "pairs": report["pairs"],
        "claims_read": claims_read,
        "claims_validated": len(validation_rows),
        "claims_kept_direct": claims_kept,
        "pairs_with_direct_claim": report["pairs_with_direct_claim"],
        "validation_status": report["validation_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
