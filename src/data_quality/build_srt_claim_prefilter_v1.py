"""Build high-recall SRT claim candidates for full-pair reconstruction.

This deterministic prefilter does not label data and does not promote claims.
It searches raw SRT cues for target-attribute snippets so later LLM/VLM reviews
do not have to blindly scan long livestream transcripts.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from common import product_index as pidx
from common.io_utils import normalize, read_jsonl, write_json, write_jsonl
from common.srt import Cue, parse_srt


STOP_TERMS = {
    "商品", "产品", "属性", "是否", "情况", "相关", "信息", "客观", "名词", "短语",
    "适用", "使用", "方法", "类型", "详情", "产品名称", "颜色分类",
}

ALIAS_TERMS = {
    "尺码": ("尺码", "码数", "大码", "小码", "加大码", "均码", "紧", "肥", "合脚"),
    "尺寸": ("尺寸", "大小", "长宽", "高度", "宽度", "长度", "大号", "小号"),
    "颜色": ("颜色", "色", "黑", "白", "灰", "蓝", "红", "粉", "绿", "黄", "棕", "米"),
    "颜色分类": ("颜色", "色", "黑", "白", "灰", "蓝", "红", "粉", "绿", "黄", "棕", "米"),
    "材质": ("材质", "材料", "面料", "成分", "棉", "聚酯", "羊毛", "真皮", "橡胶", "塑料"),
    "面料材质": ("材质", "面料", "成分", "棉", "聚酯", "羊毛", "绒", "真皮"),
    "面料成分": ("材质", "面料", "成分", "棉", "聚酯", "羊毛", "绒", "真皮"),
    "厚度": ("厚度", "厚", "薄", "加厚", "轻薄", "薄款", "厚款"),
    "重量": ("重量", "克", "斤", "公斤", "千克", "g", "kg", "重"),
    "容量": ("容量", "毫升", "升", "ml", "l", "大容量"),
    "净含量": ("净含量", "含量", "克", "斤", "毫升", "ml", "g"),
    "数量": ("数量", "个", "件", "包", "盒", "瓶", "支", "片", "双"),
    "件数": ("件数", "件", "个", "双", "套", "组"),
    "包数": ("包数", "包", "袋", "盒", "箱"),
    "张数": ("张数", "张", "抽", "片"),
    "价格": ("价格", "价", "块", "元", "券", "到手", "便宜", "保价"),
    "<价格>": ("价格", "价", "块", "元", "券", "到手", "便宜", "保价"),
    "生产日期": ("生产日期", "日期", "新鲜", "保质期", "临期", "过期", "月份"),
    "保质期": ("保质期", "日期", "新鲜", "临期", "过期", "月份"),
    "是否临期": ("临期", "日期", "新鲜", "保质期", "过期", "月份"),
    "功效": ("功效", "效果", "防水", "防油", "防污", "保暖", "清洁", "祛", "除"),
    "功能": ("功能", "效果", "防水", "防油", "防污", "保暖", "清洁", "祛", "除"),
    "主要功能": ("功能", "效果", "防水", "防油", "防污", "保暖", "清洁", "祛", "除"),
    "品牌": ("品牌", "牌", "正品", "正版", "官方", "授权"),
    "商品真伪": ("正品", "正版", "假一", "授权", "官方", "假货"),
    "是否加绒": ("加绒", "绒", "内里", "保暖", "厚"),
    "保价承诺": ("保价", "承诺", "天", "年", "退差", "差价"),
    "<保价承诺>": ("保价", "承诺", "天", "年", "退差", "差价"),
}

GENERIC_CLAIM_CUES = (
    "给大家", "我们", "这个", "这款", "它是", "都是", "就是", "可以", "能够",
    "不是", "没有", "不会", "包", "保", "承诺", "送", "赠", "拍", "发",
)

UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:cm|mm|kg|g|ml|l|斤|克|千克|公斤|毫升|升|元|块|件|个|包|袋|盒|瓶|支|片|双|张|抽|码|天|年|月)", re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def split_terms(*values: Any) -> list[str]:
    text = " ".join(clean(v) for v in values)
    chunks = TOKEN_RE.findall(text)
    terms: list[str] = []
    for ch in chunks:
        if re.fullmatch(r"[A-Za-z0-9]+", ch):
            if len(ch) >= 2:
                terms.append(ch.lower())
            continue
        if 2 <= len(ch) <= 8:
            terms.append(ch)
        elif len(ch) > 8:
            terms.extend(ch[i:i + 2] for i in range(len(ch) - 1))
            terms.extend(ch[i:i + 3] for i in range(len(ch) - 2))
    out = []
    for t in terms:
        if t and t not in STOP_TERMS and t not in out:
            out.append(t)
    return out


def value_terms(*values: Any) -> list[str]:
    text = " ".join(clean(v) for v in values)
    out: list[str] = []
    for m in UNIT_RE.findall(text):
        out.append(m.replace(" ", ""))
    for m in NUM_RE.findall(text):
        out.append(m)
    for color in ("黑", "白", "灰", "蓝", "红", "粉", "绿", "黄", "棕", "紫", "米", "咖"):
        if color in text:
            out.append(color)
    for mat in ("棉", "绒", "羊毛", "聚酯", "涤纶", "真皮", "牛皮", "橡胶", "塑料", "不锈钢"):
        if mat in text:
            out.append(mat)
    return list(dict.fromkeys(out))


def aliases(attr_name: str, attr_id: str) -> list[str]:
    terms: list[str] = []
    base = clean(attr_name)
    tail = clean(str(attr_id).split("_", 1)[-1])
    for key in (base, tail):
        terms.extend(ALIAS_TERMS.get(key, ()))
    for key, vals in ALIAS_TERMS.items():
        if key and (key in base or key in tail):
            terms.extend(vals)
    terms.extend(split_terms(base, tail))
    return list(dict.fromkeys(t for t in terms if t and t not in STOP_TERMS))


def compact_comment_text(row: dict[str, Any], cap: int) -> str:
    mentions = row.get("consumer_mentions") or []
    return " ".join(clean(m.get("evidence_span")) for m in mentions[:cap])


@lru_cache(maxsize=4096)
def load_cues(path_str: str) -> tuple[Cue, ...]:
    path = pidx.resolve(path_str)
    if not path.exists():
        return tuple()
    try:
        return tuple(parse_srt(path))
    except Exception:
        return tuple()


def hit_terms(text: str, terms: list[str]) -> list[str]:
    nt = normalize(text)
    hits: list[str] = []
    for term in terms:
        if normalize(term) and normalize(term) in nt:
            hits.append(term)
    return hits


def score_window(text: str, attr_terms: list[str], title_terms: list[str], comment_terms: list[str], values: list[str]) -> tuple[int, dict[str, Any]]:
    attr_hits = hit_terms(text, attr_terms)
    title_hits = hit_terms(text, title_terms)
    comment_hits = hit_terms(text, comment_terms)
    value_hits = hit_terms(text, values)
    cue_hits = [cue for cue in GENERIC_CLAIM_CUES if cue in text]
    score = 0
    score += 6 * len(set(attr_hits[:8]))
    score += 2 * len(set(title_hits[:8]))
    score += 3 * len(set(comment_hits[:8]))
    score += 4 * len(set(value_hits[:8]))
    score += min(6, 2 * len(set(cue_hits[:4])))
    if attr_hits and value_hits:
        score += 6
    if attr_hits and comment_hits:
        score += 4
    if value_hits and comment_hits:
        score += 3
    return score, {
        "attribute_hits": list(dict.fromkeys(attr_hits))[:12],
        "title_hits": list(dict.fromkeys(title_hits))[:12],
        "comment_hits": list(dict.fromkeys(comment_hits))[:12],
        "value_hits": list(dict.fromkeys(value_hits))[:12],
        "claim_cue_hits": list(dict.fromkeys(cue_hits))[:8],
    }


def candidate_windows(row: dict[str, Any], max_candidates: int, comment_cap: int) -> list[dict[str, Any]]:
    attr_terms = aliases(clean(row.get("attribute_name")), clean(row.get("attribute_id")))
    title_terms = split_terms(row.get("product_title"))[:80]
    comment_text = compact_comment_text(row, comment_cap)
    comment_terms = split_terms(comment_text)[:80]
    values = value_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"), comment_text)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for srt in row.get("srt_files") or []:
        cues = load_cues(str(srt))
        if not cues:
            continue
        for i, cue in enumerate(cues):
            lo = max(0, i - 1)
            hi = min(len(cues), i + 2)
            window_cues = cues[lo:hi]
            text = " ".join(c.text for c in window_cues).strip()
            if not text:
                continue
            score, why = score_window(text, attr_terms, title_terms, comment_terms, values)
            if score <= 0:
                continue
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "score": score,
                "srt_file": str(pidx.resolve(srt)),
                "clip_name": Path(str(srt)).name,
                "cue_start": lo,
                "cue_end": hi - 1,
                "start_ts": window_cues[0].start_ts,
                "end_ts": window_cues[-1].end_ts,
                "text": text[:320],
                "center_text": cue.text[:180],
                "why": why,
            })
    candidates.sort(key=lambda r: (-int(r["score"]), r["clip_name"], r["start_ts"], r["text"]))
    return candidates[:max_candidates]


def prefilter_state(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "no_srt_candidate"
    top = int(candidates[0].get("score", 0))
    if top >= 20:
        return "strong_srt_candidate"
    if top >= 8:
        return "weak_srt_candidate"
    return "very_weak_srt_candidate"


def build_item(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "product_id": row.get("product_id"),
        "room_id": row.get("room_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "product_title": row.get("product_title"),
        "priority": row.get("priority"),
        "queue_type": row.get("queue_type"),
        "claim_state": row.get("claim_state"),
        "evidence_state": row.get("evidence_state"),
        "old_label_state": row.get("old_label_state"),
        "consumer_mentions_total": row.get("consumer_mentions_total"),
        "consumer_mentions_neg": row.get("consumer_mentions_neg"),
        "consumer_mentions_explicit": row.get("consumer_mentions_explicit"),
        "srt_file_count": len(row.get("srt_files") or []),
        "prefilter_state": prefilter_state(candidates),
        "top_score": int(candidates[0]["score"]) if candidates else 0,
        "claim_candidates": candidates,
        "note": "High-recall SRT candidates only; do not promote without LLM/provenance validation.",
    }


def write_markdown(path: str, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pair SRT Claim Prefilter v1",
        "",
        "This report summarizes deterministic SRT candidate retrieval for full-pair reconstruction.",
        "It does not label rows or promote claims.",
        "",
        "## Summary",
        "",
        f"- input queue: `{report['queue']}`",
        f"- output: `{report['out']}`",
        f"- rows processed: `{report['n']}`",
        f"- skipped: `{report['skipped']}`",
        f"- prefilter state: `{report['prefilter_state']}`",
        f"- queue type: `{report['queue_type']}`",
        f"- claim state: `{report['claim_state']}`",
        f"- category: `{report['category']}`",
        "",
        "## Cross Tabs",
        "",
        f"- claim state by prefilter: `{report['claim_state_by_prefilter']}`",
        f"- queue type by prefilter: `{report['queue_type_by_prefilter']}`",
        "",
        "## Interpretation",
        "",
        "- `strong_srt_candidate` means raw SRT likely contains attribute-related claim material for later LLM review.",
        "- `weak_srt_candidate` means SRT has lexical overlap but needs careful review.",
        "- `no_srt_candidate` means either no SRT file was available or no useful lexical candidate was found.",
        "",
        "## Top Examples",
        "",
    ]
    for ex in report.get("examples", []):
        cand = (ex.get("claim_candidates") or [{}])[0]
        lines.append(
            f"- `{ex.get('pair_id')}` state={ex.get('prefilter_state')} "
            f"score={ex.get('top_score')} attr={ex.get('attribute_name')} "
            f"text={json.dumps(cand.get('text', ''), ensure_ascii=False)[:260]}"
        )
    lines.extend(["", "## No-Candidate Examples", ""])
    for ex in report.get("no_candidate_examples", []):
        lines.append(
            f"- `{ex.get('pair_id')}` attr={ex.get('attribute_name')} "
            f"category={ex.get('category')} srt_files={ex.get('srt_file_count')}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_CLAIM_SRT_PREFILTER_20260614.md")
    ap.add_argument("--claim_state", default="claim_missing,claim_present_review_needed")
    ap.add_argument("--priority", default="")
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--comment_cap", type=int, default=12)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    states = {x.strip() for x in args.claim_state.replace(",", " ").split() if x.strip()}
    priorities = {x.strip() for x in args.priority.replace(",", " ").split() if x.strip()}
    rows = []
    skipped = Counter()
    for row in read_jsonl(args.queue):
        if states and str(row.get("claim_state")) not in states:
            skipped["claim_state_filter"] += 1
            continue
        if priorities and str(row.get("priority")) not in priorities:
            skipped["priority_filter"] += 1
            continue
        rows.append(row)
    if args.max_items > 0:
        rows = rows[:args.max_items]

    out_rows = []
    for row in rows:
        cands = candidate_windows(row, args.max_candidates, args.comment_cap)
        out_rows.append(build_item(row, cands))
    write_jsonl(args.out, out_rows)

    report = {
        "queue": args.queue,
        "out": args.out,
        "n": len(out_rows),
        "skipped": dict(skipped),
        "prefilter_state": dict(Counter(str(r.get("prefilter_state")) for r in out_rows)),
        "queue_type": dict(Counter(str(r.get("queue_type")) for r in out_rows)),
        "claim_state": dict(Counter(str(r.get("claim_state")) for r in out_rows)),
        "category": dict(Counter(str(r.get("category")) for r in out_rows)),
        "claim_state_by_prefilter": {
            f"{claim}|{state}": count
            for (claim, state), count in Counter(
                (str(r.get("claim_state")), str(r.get("prefilter_state"))) for r in out_rows
            ).items()
        },
        "queue_type_by_prefilter": {
            f"{qt}|{state}": count
            for (qt, state), count in Counter(
                (str(r.get("queue_type")), str(r.get("prefilter_state"))) for r in out_rows
            ).items()
        },
        "top_score_bins": {
            "score_ge_40": sum(1 for r in out_rows if int(r.get("top_score", 0)) >= 40),
            "score_20_39": sum(1 for r in out_rows if 20 <= int(r.get("top_score", 0)) < 40),
            "score_8_19": sum(1 for r in out_rows if 8 <= int(r.get("top_score", 0)) < 20),
            "score_1_7": sum(1 for r in out_rows if 1 <= int(r.get("top_score", 0)) < 8),
            "score_0": sum(1 for r in out_rows if int(r.get("top_score", 0)) == 0),
        },
        "examples": [
            r for r in sorted(out_rows, key=lambda x: (-int(x.get("top_score", 0)), str(x.get("pair_id"))))[:20]
        ],
        "no_candidate_examples": [
            r for r in out_rows if r.get("prefilter_state") == "no_srt_candidate"
        ][:20],
    }
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
