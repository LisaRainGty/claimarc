"""Deterministic precision filters for atomic alignment records.

LLM alignment is useful but still over-maps generic consumer experience to
specific factual claims.  These filters demote common high-risk cases without
dropping the record, so the claim can remain as a low-weight negative example.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from typing import Any

from common.io_utils import normalize, read_jsonl, write_json, write_jsonl


BAD_REASON = re.compile(r"(无关|未提|没提|没有提|未指明|未明确|不明确|无法判断)")
COLOR_RE = re.compile(
    r"(黑|白|灰|绿|蓝|红|粉|紫|黄|橙|咖|棕|米|杏|青|银|金|卡其|驼|藏青|香槟|透明|原木|奶油|焦糖|墨|沙漠|夜脉).{0,3}色"
    r"|黑白|灰黑|蓝白|粉白|紫灰|咖色|米色|白色|黑色|灰色|绿色|蓝色|红色|粉色|紫色|黄色"
    r"|发黄|偏黄|太黄|发黑|太黑|偏黑|发灰|偏灰|暗一点|比较暗|假白"
    r"|深色|浅色|太深|太浅|偏深|偏浅|不深|不浅|很暗|太暗|偏暗|没有想象中的深"
    r"|冷白|瓷白|自然色|黄调|黑壳|黑色壳|[黑白灰绿蓝红粉紫黄橙咖棕米杏青银金]"
)
# Keep numeric filtering conservative.  Chinese ASR often contains filler such
# as "一个/一号"; treating every Chinese numeral as exact numeric evidence
# wrongly demotes color/material claims.
NUM_RE = re.compile(
    r"\d+|[一二两三四五六七八九十百千万半]+(?:克|斤|码|岁|支|片|颗|米|厘米|毫升|寸|小时|分钟|天|度)"
)
TIME_RE = re.compile(r"(小时|分钟|秒|天|全天|半天|持久|续航|待机|时间|一整天|不脱|不掉|不晕)")
EFFECT_RE = re.compile(
    r"(效果|功效|有用|没用|无效|没效果|没有效果|作用|没作用|没有作用|改善|解决|温和|刺激|"
    r"止痒|去屑|头皮|水油|清爽|蓬松|顺滑|清洁|清新|口气|口臭|异味|治疗|"
    r"皱纹|泪沟|松弛|下垂|提亮|洁净|养护|保暖|防水|防晒|抗疲劳|防蓝光|快充|续航)"
)
COLOR_COMPARE_RE = re.compile(
    r"(色差|颜色.*(?:不一样|不符|不对|差|偏|错)|发错|错色|不是.{0,6}色|"
    r"(?:不一样|不符|偏色|掉色|褪色|发黄|发黑|发灰))"
)
EXPECTATION_GAP_RE = re.compile(
    r"(说是|说的|主播说|直播间说|宣传|描述|详情|视频里|视频的|视频上|图片上|图上|实物|收到)"
    r".{0,12}(不符|不一样|不一致|不对|没[有]?|不是|差|小|大|薄|厚|浅|深|少|短|长|低|弱|无)"
    r"|(?:不符|不一样|不一致|货不对版|实物.*差|和直播间.*(?:不同|不一样|不符)|不是视频)"
)
FILM_FEATURE_RE = re.compile(r"(贴膜|膜|指纹|手汗|气泡|贴合|碎|裂|遮挡|无尘|防窥|高清|顺滑|钢化)")
CERT_OR_SAFETY_RE = re.compile(r"(3C|三C|ccc|认证|标示|标识|过敏|安全|不含|放心)")
GENERIC_BAD = {
    "颜色很怪", "颜色不好看", "商品尺寸非常合适", "尺寸非常合适",
    "版型很正", "款式不错", "款式可以", "瓶子有点小",
}


def _is_question_or_noise(claim: str) -> bool:
    n = normalize(claim)
    if not n or len(n) < 2:
        return True
    if "某" in claim:
        return True
    if any(x in n for x in ("搭什么", "怎么搭", "怎么选", "好看吗", "要不要", "可以吗")):
        return True
    if n in {"它的厚度", "这个颜色", "年款", "商场的包装"}:
        return True
    return False


def _has_color(text: str) -> bool:
    return bool(COLOR_RE.search(text or ""))


def _has_color_comparison(text: str) -> bool:
    return _has_color(text) or bool(COLOR_COMPARE_RE.search(text or ""))


def _has_num(text: str) -> bool:
    return bool(NUM_RE.search(text or ""))


def _amount_cue(text: str) -> bool:
    return any(x in text for x in ("少", "多", "不够", "不足", "量小", "分量", "份量", "克", "g", "毫升", "ml", "只有", "少了", "小了", "有点小", "感觉小", "量大", "量足"))


def _spec_cue(text: str) -> bool:
    """Package/spec claims are often contradicted by comparative size wording."""
    return any(
        x in text
        for x in (
            "大包", "小包", "中包", "大号", "小号", "中号", "太小", "很小", "偏小", "偏大",
            "这么小", "有点小", "感觉小", "比较小", "小的", "小了", "大了", "变小", "变成了小",
            "大小", "尺寸", "规格", "层", "抽", "包", "箱",
            "件", "片", "只", "瓶", "不是视频", "视频里", "视频里的",
        )
    )


def _size_cue(text: str) -> bool:
    return any(x in text for x in ("偏大", "偏小", "太大", "太小", "有点大", "有点小", "买大了", "买小了", "大了点", "小了点", "大了", "小了", "码", "尺码", "尺寸", "长", "短", "松", "紧", "合身", "腰围", "体重", "斤", "厘米", "cm"))


def _thick_cue(text: str) -> bool:
    return any(x in text for x in ("厚", "薄", "绒", "保暖", "冷", "暖", "轻薄", "厚实"))


def _weight_cue(text: str) -> bool:
    return any(x in text for x in ("重", "轻", "斤", "克", "沉"))


def _explicit_gap_cue(text: str) -> bool:
    return bool(EXPECTATION_GAP_RE.search(text or ""))


def _keep_numeric_alignment(attr: str, claim: str, comment: str) -> bool:
    if _has_num(comment):
        return True
    if _explicit_gap_cue(comment):
        return True
    if any(x in attr for x in ("尺码", "尺寸", "码数", "腰围")):
        return _size_cue(comment)
    if any(x in attr for x in ("规格", "净含量", "容量", "份量", "分量", "数量", "件数", "包数", "套餐")):
        if "规格" in attr and _spec_cue(comment):
            return True
        return _amount_cue(comment)
    if any(x in attr for x in ("厚度", "充绒", "绒子")):
        return _thick_cue(comment)
    if "重量" in attr:
        return _weight_cue(comment)
    if any(x in attr for x in ("功效", "主要功能")) and re.search(r"(小时|分钟|天|零下|度)", claim):
        return bool(TIME_RE.search(comment)) or bool(re.search(r"(零下|度|冷|暖|保暖|防寒)", comment))
    if any(x in attr for x in ("功效", "功能", "主要功能", "附加功能")):
        return bool(EFFECT_RE.search(comment))
    if any(x in attr for x in ("特点", "特性", "卖点")):
        return bool(EFFECT_RE.search(comment)) or bool(FILM_FEATURE_RE.search(comment))
    if any(x in attr for x in ("安全", "认证", "证书", "CCC", "标识", "标示")):
        return bool(CERT_OR_SAFETY_RE.search(comment))
    return False


def _should_demote(rec: dict[str, Any], c: dict[str, Any]) -> str:
    if int(c.get("y_supportability", 0) or 0) != 1:
        return ""
    claim = (rec.get("claim") or {}).get("passage", "")
    attr = str(rec.get("attribute_name") or rec.get("attribute_id") or "")
    comment = str(c.get("text") or "")
    reason = str(c.get("alignment_rationale") or "")
    sf = str(rec.get("source_family") or "")
    if BAD_REASON.search(reason):
        return "bad_reason"
    if normalize(comment) in {normalize(x) for x in GENERIC_BAD}:
        return "generic_comment"
    if "颜色" in attr and _has_color(claim) and not _has_color_comparison(comment):
        return "color_without_color"
    if "颜色" not in attr and _has_num(claim) and not _keep_numeric_alignment(attr, claim, comment):
        # Exact numeric claims are high-risk: generic sentiment cannot verify
        # the number.  The attribute-specific cue exceptions above preserve
        # size/thickness/amount consumer-perception signals.
        return "numeric_without_comparable_cue"
    if sf == "objective_name_only":
        cn = set(normalize(claim))
        tn = set(normalize(comment))
        if len(cn & tn) <= 1:
            return "objective_name_low_overlap"
    return ""


def _stats(reviews: list[dict[str, Any]]) -> dict[str, int]:
    aligned = [c for c in reviews if int(c.get("y_supportability", 0) or 0) == 1]
    rel = Counter(str(c.get("relation") or "unclear") for c in aligned)
    return {
        "N_total": len(reviews),
        "N_aligned": len(aligned),
        "N_support": rel.get("support", 0),
        "N_refute": rel.get("refute", 0),
        "N_mixed": rel.get("mixed", 0),
        "N_unclear": rel.get("unclear", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--drop_bad_claim_records", action="store_true")
    args = ap.parse_args()

    rows = []
    demoted = Counter()
    dropped = Counter()
    examples = []
    for r in read_jsonl(args.atomic_records):
        r = copy.deepcopy(r)
        claim = (r.get("claim") or {}).get("passage", "")
        if args.drop_bad_claim_records and _is_question_or_noise(claim):
            dropped["bad_claim"] += 1
            continue
        for c in r.get("reviews", []) or []:
            why = _should_demote(r, c)
            if not why:
                continue
            before = (int(c.get("y_supportability", 0) or 0), c.get("relation"))
            c["y_supportability"] = 0
            c["relation"] = "unclear"
            c["_atomic_post_filter"] = why
            c["alignment_rationale"] = why
            demoted[why] += 1
            if len(examples) < 80:
                examples.append({
                    "atomic_id": r.get("atomic_id"),
                    "attribute_name": r.get("attribute_name"),
                    "claim": claim,
                    "comment": c.get("text", ""),
                    "before": before,
                    "filter": why,
                })
        r["stats"] = _stats(r.get("reviews", []) or [])
        rows.append(r)
    write_jsonl(args.out, rows)
    report = {
        "atomic_records": args.atomic_records,
        "out": args.out,
        "n_out": len(rows),
        "demoted_comments": dict(demoted),
        "dropped_records": dict(dropped),
        "record_status": dict(Counter(
            "refute" if r["stats"]["N_refute"] else
            "support" if r["stats"]["N_support"] else
            "aligned_other" if r["stats"]["N_aligned"] else "unaligned"
            for r in rows
        )),
        "examples": examples,
    }
    write_json(args.report or args.out.replace(".jsonl", "_filter_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
