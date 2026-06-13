"""Deterministic product-evidence prefilter for proposal repair rows.

This is a high-recall, non-labeling step.  It scans product title, params,
current evidence previews, and OCR caches for target-attribute evidence
candidates.  The output is intended to focus later LLM/VLM verification; it
does not promote rows into training and never changes proposal labels.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common.io_utils import normalize, read_json, read_jsonl, write_json, write_jsonl


STOP_TERMS = {
    "属性", "商品", "产品", "情况", "说明", "是否", "相关", "支持", "这个", "那个",
    "大家", "咱家", "我们", "宝贝", "链接", "准备", "直播间", "所有女生",
}
GENERIC_OCR = {
    "欢乐", "开怀", "美味", "优质", "精选", "品质", "好物", "热卖", "新品",
    "详情", "特点", "卖点", "专业", "专属", "严选", "好评",
}
PRICE_WORDS = {"价格", "价", "元", "块", "¥", "￥", "市场价", "到手", "售价", "优惠"}
SIZE_WORDS = {"尺码", "尺寸", "大小", "码", "cm", "厘米", "公分", "适合", "身高", "体重"}
MATERIAL_WORDS = {"材质", "面料", "成分", "聚酯", "棉", "羊毛", "绒", "牛皮", "玻璃", "不锈钢"}
FUNCTION_WORDS = {"功能", "功效", "防", "抗", "护", "不", "无", "认证", "质保", "保价"}
STYLE_WORDS = {"款式", "版型", "鞋头", "圆头", "尖头", "宽松", "修身", "颜色", "风味"}


NUM_RE = re.compile(
    r"(?:¥|￥)?\d+(?:\.\d+)?\s*(?:cm|CM|mm|MM|mAh|mah|g|G|kg|KG|ml|ML|%|元|块|年|天|抽|张|支|包|斤|克|毫升|公分|厘米|码|号)?"
)
CHUNK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def clean_text(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def claim_text(row: dict[str, Any]) -> str:
    verdict = row.get("previous_triplet_verdict") or {}
    return clean_text(verdict.get("claim_text")) or clean_text(row.get("claim_preview"))


def term_chunks(*values: Any) -> list[str]:
    text = " ".join(clean_text(v) for v in values)
    terms: list[str] = []
    for m in NUM_RE.findall(text):
        m = clean_text(m)
        if m:
            terms.append(m)
    for ch in CHUNK_RE.findall(text):
        ch = ch.strip()
        if not ch:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", ch):
            if len(ch) >= 2:
                terms.append(ch.lower())
            continue
        if 2 <= len(ch) <= 8:
            terms.append(ch)
        elif len(ch) > 8:
            terms.extend(ch[i:i + 2] for i in range(len(ch) - 1))
            terms.extend(ch[i:i + 3] for i in range(len(ch) - 2))
    out: list[str] = []
    for t in terms:
        nt = normalize(t)
        if not nt or nt in {normalize(x) for x in STOP_TERMS}:
            continue
        if t not in out:
            out.append(t)
    return out


def family_terms(row: dict[str, Any]) -> set[str]:
    text = f"{row.get('attribute_id', '')} {row.get('attribute_name', '')} {row.get('expected_value_type', '')}"
    terms: set[str] = set()
    if any(x in text for x in PRICE_WORDS):
        terms |= PRICE_WORDS
    if any(x in text for x in SIZE_WORDS):
        terms |= SIZE_WORDS
    if any(x in text for x in MATERIAL_WORDS):
        terms |= MATERIAL_WORDS
    if any(x in text for x in FUNCTION_WORDS):
        terms |= FUNCTION_WORDS
    if any(x in text for x in STYLE_WORDS):
        terms |= STYLE_WORDS
    return terms


def load_ocr(pid: str) -> dict[str, str]:
    obj = read_json(config.STAGE_C / "ocr_text" / f"{pid}.json", default={}) or {}
    return {str(k): str(v or "") for k, v in obj.items()}


def split_ocr_lines(text: str) -> list[str]:
    lines = [x.strip() for x in re.split(r"[\n\r]+", text or "") if x.strip()]
    windows = []
    for i, line in enumerate(lines):
        windows.append(line)
        if i + 1 < len(lines) and len(line) <= 24:
            windows.append(f"{line} {lines[i + 1]}")
    return windows


def score_candidate(
    row: dict[str, Any],
    *,
    source: str,
    key: str,
    text: str,
    attr_terms: list[str],
    claim_terms: list[str],
    fam_terms: set[str],
) -> tuple[int, list[str]]:
    nt = normalize(text)
    nk = normalize(key)
    reasons: list[str] = []
    score = 0
    for t in attr_terms:
        z = normalize(t)
        if z and (z in nt or z in nk):
            score += 5
            reasons.append(f"attr:{t}")
    for t in claim_terms:
        z = normalize(t)
        if z and z in nt:
            score += 3 if NUM_RE.fullmatch(t) else 2
            reasons.append(f"claim:{t}")
    for t in fam_terms:
        z = normalize(t)
        if z and (z in nt or z in nk):
            score += 2
            reasons.append(f"family:{t}")
    if source == "params" and any(normalize(t) in nk for t in attr_terms + list(fam_terms)):
        score += 4
        reasons.append("param_key_hit")
    if source == "title" and score > 0:
        score += 1
    if len(normalize(text)) <= 2 and source != "params":
        score -= 4
        reasons.append("too_short")
    if any(normalize(x) == nt or normalize(x) in nt for x in GENERIC_OCR):
        score -= 2
        reasons.append("generic_ocr")
    return score, reasons


def relation_hint(claim: str, text: str) -> str:
    c_nums = {normalize(x) for x in NUM_RE.findall(claim)}
    e_nums = {normalize(x) for x in NUM_RE.findall(text)}
    if c_nums and e_nums:
        return "value_overlap" if c_nums & e_nums else "value_compare_needed"
    c_terms = set(term_chunks(claim))
    e_terms = set(term_chunks(text))
    if c_terms and e_terms and {normalize(x) for x in c_terms} & {normalize(x) for x in e_terms}:
        return "lexical_overlap"
    return "attribute_relevant"


def build_candidates(row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    attr_terms = term_chunks(row.get("attribute_name"), row.get("attribute_id"))
    claim = claim_text(row)
    claim_terms = term_chunks(claim)
    fam_terms = family_terms(row)
    candidates: list[dict[str, Any]] = []

    def add(source: str, key: str, text: str) -> None:
        text = clean_text(text)
        if not text:
            return
        score, reasons = score_candidate(
            row,
            source=source,
            key=key,
            text=text,
            attr_terms=attr_terms,
            claim_terms=claim_terms,
            fam_terms=fam_terms,
        )
        if score <= 0:
            return
        candidates.append({
            "source": source,
            "key": key,
            "text": text[:260],
            "score": score,
            "reasons": reasons[:12],
            "relation_hint": relation_hint(claim, text),
        })

    add("title", "product_title", row.get("product_title", ""))
    for k, v in (row.get("raw_params") or {}).items():
        add("params", str(k), f"{k}: {v}")
    for ev in row.get("current_evidence_preview") or []:
        add(str(ev.get("source") or "current_evidence"), str(ev.get("key") or ""), str(ev.get("text") or ""))
    for path, text in load_ocr(str(row.get("product_id", ""))).items():
        if Path(path).name.startswith("._"):
            continue
        for line in split_ocr_lines(text):
            add("ocr_cache", Path(path).name, line)

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["source"], c["key"], normalize(c["text"]))
        if key not in dedup or c["score"] > dedup[key]["score"]:
            dedup[key] = c
    out = sorted(dedup.values(), key=lambda x: (-int(x["score"]), x["source"], x["key"], x["text"]))
    return out[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_refresh.jsonl",
    )
    ap.add_argument(
        "--out",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_prefilter_v1.jsonl",
    )
    ap.add_argument(
        "--report",
        default="data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/product_evidence_prefilter_v1.report.json",
    )
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    rows = list(read_jsonl(args.queue))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        cands = build_candidates(row, args.top_k)
        if cands and int(cands[0]["score"]) >= 5:
            state = "strong_candidate"
        elif cands:
            state = "weak_candidate"
        else:
            state = "no_candidate"
        out_rows.append({
            "pair_id": pair_id(row),
            "product_id": row.get("product_id"),
            "attribute_id": row.get("attribute_id"),
            "attribute_name": row.get("attribute_name"),
            "current_y": row.get("current_y"),
            "current_c": row.get("current_c"),
            "claim_text": claim_text(row),
            "previous_relation": (row.get("previous_triplet_verdict") or {}).get("relation_to_claim"),
            "candidate_count": len(cands),
            "top_score": cands[0]["score"] if cands else 0,
            "prefilter_state": state,
            "candidates": cands,
        })
    write_jsonl(args.out, out_rows)
    report = {
        "queue": args.queue,
        "out": args.out,
        "n": len(out_rows),
        "states": dict(Counter(r["prefilter_state"] for r in out_rows)),
        "top_score_hist": dict(Counter(str(min(20, int(r["top_score"]))) for r in out_rows)),
        "source_hits": dict(Counter(c["source"] for r in out_rows for c in r["candidates"][:3])),
        "relation_hints": dict(Counter(c["relation_hint"] for r in out_rows for c in r["candidates"][:3])),
        "top_examples": [
            {
                "pair_id": r["pair_id"],
                "attribute_id": r["attribute_id"],
                "top_score": r["top_score"],
                "claim_text": r["claim_text"][:120],
                "candidates": r["candidates"][:3],
            }
            for r in sorted(out_rows, key=lambda x: -int(x["top_score"]))[:12]
        ],
        "no_candidate_examples": [
            {
                "pair_id": r["pair_id"],
                "attribute_id": r["attribute_id"],
                "claim_text": r["claim_text"][:120],
            }
            for r in out_rows if r["prefilter_state"] == "no_candidate"
        ][:12],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
