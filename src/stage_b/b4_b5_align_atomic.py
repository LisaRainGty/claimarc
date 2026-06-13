"""Stage B4/B5 atomic — single-claim consumer-signal alignment.

The legacy B4 aligns all comments to a concatenated `(product, attribute)`
passage.  This variant keeps each validated direct claim as the unit of
alignment so downstream labels can distinguish "this exact claim is refuted"
from generic attribute sentiment.

Example:
  python -m stage_b.b4_b5_align_atomic \
    --atomic_skeleton data/processed/stageB_product_v2/atomic_claim_skeleton_productv2_direct_strict_stratified120.jsonl \
    --resolved data/processed/stageB_product_v2/resolved_aspects_product_v2.jsonl \
    --out data/processed/stageB_product_v2/atomic_records_productv2_direct_strict_stratified120.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from common import llm
from common.io_utils import read_jsonl, write_json, write_jsonl


ATOMIC_B4_PROMPT = """角色：直播电商虚假宣传审查员。
你要判断消费者评论是否在回应主播的"单条具体口播 claim"，不是泛泛评价商品。

属性：{attr}（同义词：{aliases}）
主播单条 claim：
{claim}

判定规则：
1. aligned=1：评论必须与这条 claim 的具体值/事实形成可比较关系，包含同义、省略、反话、数值/颜色/材质/规格/功能的直接回应。
2. aligned=0：评论只是在泛泛谈该属性或商品体验，无法判断是否回应这条 claim。
3. relation 只在 aligned=1 时有意义：
   - support：评论支持或确认该 claim。
   - refute：评论否定、抱怨、质疑或给出与 claim 冲突的体验/事实。
   - mixed：同一评论同时支持和反驳，或只部分匹配。
   - unclear：能看出相关但方向不清。
4. 不要因为评论情绪正/负就强行 aligned=1；必须和 claim 的具体内容有关。
5. 仅有使用效果、喜好、搭配效果、泛泛体验，不足以支持/反驳客观 claim。

负例边界：
- claim="灰黑色"，评论="灰色有点显黑"：aligned=0（显黑是使用效果，不反驳颜色）。
- claim="加高一点的"，评论="很厚实"：aligned=0（厚实与加高不是同一事实）。
- claim="羊毛混纺"，评论="有点扎"：aligned=0（扎不等于否定羊毛混纺）。
- claim="125ml"，评论="容量太小"：aligned=0（只评价小，没有确认或否定 125ml）。
正例边界：
- claim="防水"，评论="一点都不防水"：aligned=1, relation=refute。
- claim="羊毛混纺"，评论="不是羊毛，像腈纶"：aligned=1, relation=refute。
- claim="125ml"，评论="确实只有125ml"：aligned=1, relation=support。
- claim="灰黑色"，评论="实物不是灰黑，是绿色"：aligned=1, relation=refute。

评论（编号 1..K）：
{reviews}

严格输出 JSON 数组，每条格式：
{{"cid": <编号int>, "aligned": 0或1, "relation": "support/refute/mixed/unclear", "rationale": "不超过18字"}}
只输出 JSON 数组。"""


def _comment_key(c: dict[str, Any]) -> str:
    return str(c.get("review_id") or c.get("comment_id") or c.get("text") or "")


def _comments_by_pair(resolved_path: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(resolved_path):
        pid = str(r.get("product_id") or "")
        aid = str(r.get("attribute_id") or "")
        if not pid or not aid:
            continue
        rid = _comment_key(r)
        text = str(r.get("review_text") or r.get("evidence_span") or "").strip()
        if not rid or not text:
            continue
        cur = {
            "comment_id": rid,
            "text": text,
            "polarity": r.get("polarity") or r.get("review_polarity", "neu"),
            "review_polarity": r.get("review_polarity", "neu"),
            "mention_strength": r.get("mention_strength", "weak"),
            "explicit_fact_hit": bool(r.get("explicit_fact_hit", False)),
            "evidence_span": r.get("evidence_span", ""),
            "review_time": r.get("review_time", ""),
        }
        prev = by_pair[(pid, aid)].get(rid)
        prev_score = _comment_priority(prev) if prev else -1
        if prev is None or _comment_priority(cur) > prev_score:
            by_pair[(pid, aid)][rid] = cur
    return {k: list(v.values()) for k, v in by_pair.items()}


def _comment_priority(c: dict[str, Any] | None) -> float:
    if not c:
        return -1.0
    score = 0.0
    if c.get("explicit_fact_hit"):
        score += 4.0
    if c.get("mention_strength") == "strong":
        score += 2.0
    if c.get("polarity") == "neg":
        score += 1.0
    score += min(len(str(c.get("text") or "")), 80) / 200.0
    return score


def _select_comments(comments: list[dict[str, Any]], max_comments: int) -> list[dict[str, Any]]:
    if max_comments <= 0 or len(comments) <= max_comments:
        return comments
    # Keep strong/explicit/negative evidence first, while preserving deterministic order.
    return sorted(comments, key=lambda c: (-_comment_priority(c), str(c.get("comment_id", ""))))[:max_comments]


def _align_atomic(atom: dict[str, Any], comments: list[dict[str, Any]], model: str | None = None) -> list[dict[str, Any]]:
    if not comments:
        return []
    review_lines = "\n".join(
        f"{i + 1}. {str(c.get('text') or '')[:160]}" for i, c in enumerate(comments)
    )
    prompt = ATOMIC_B4_PROMPT.format(
        attr=atom.get("attribute_name") or atom.get("attribute_id", ""),
        aliases="、".join((atom.get("aliases") or [])[:8]),
        claim=(atom.get("claim") or {}).get("passage", "")[:500],
        reviews=review_lines,
    )
    try:
        arr = llm.chat_json(prompt, namespace="b4_atomic", model=model, max_tokens=1536)
    except Exception:
        return []
    by_cid: dict[int, dict[str, Any]] = {}
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item.get("cid"))
            except (TypeError, ValueError):
                continue
            rel = str(item.get("relation") or "unclear").strip().lower()
            if rel not in {"support", "refute", "mixed", "unclear"}:
                rel = "unclear"
            aligned = 1 if int(item.get("aligned", 0) or 0) == 1 else 0
            if not aligned:
                rel = "unclear"
            by_cid[cid] = {
                "y_supportability": aligned,
                "relation": rel,
                "alignment_rationale": str(item.get("rationale") or "")[:40],
            }
    out = []
    for i, c in enumerate(comments, start=1):
        nr = dict(c)
        nr.update(by_cid.get(i, {"y_supportability": 0, "relation": "unclear", "alignment_rationale": ""}))
        out.append(nr)
    return out


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
    ap.add_argument("--atomic_skeleton", required=True)
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--max_comments", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    atoms = list(read_jsonl(args.atomic_skeleton))
    if args.limit:
        atoms = atoms[:args.limit]
    comments_map = _comments_by_pair(args.resolved)
    print(f"[B4-atomic] atoms={len(atoms)} max_comments={args.max_comments}", flush=True)

    def _job(atom: dict[str, Any]) -> dict[str, Any]:
        comments = comments_map.get((str(atom["product_id"]), str(atom["attribute_id"])), [])
        comments = _select_comments(comments, args.max_comments)
        reviews = _align_atomic(atom, comments, model=args.model)
        return {
            "atomic_id": atom["atomic_id"],
            "pair_id": atom["pair_id"],
            "product_id": atom["product_id"],
            "category": atom.get("category", ""),
            "subcategory": atom.get("subcategory", ""),
            "room_id": atom.get("room_id", "UNKNOWN"),
            "attribute_id": atom["attribute_id"],
            "attribute_name": atom.get("attribute_name", atom["attribute_id"]),
            "source_family": atom.get("source_family", ""),
            "claim": atom.get("claim", {}),
            "reviews": reviews,
            "stats": _stats(reviews),
            "_claim_attribute_validation": atom.get("_claim_attribute_validation", {}),
        }

    records = llm.run_many(atoms, _job, desc="B4-atomic")
    records = [r for r in records if isinstance(r, dict) and "__error__" not in r]
    write_jsonl(args.out, records)

    report = {
        "atomic_skeleton": args.atomic_skeleton,
        "resolved": args.resolved,
        "out": args.out,
        "n_records": len(records),
        "n_pairs": len({r["pair_id"] for r in records}),
        "n_products": len({r["product_id"] for r in records}),
        "stats": dict(Counter(
            "refute" if r["stats"]["N_refute"] else
            "support" if r["stats"]["N_support"] else
            "aligned_other" if r["stats"]["N_aligned"] else "unaligned"
            for r in records
        )),
    }
    report_path = args.report or args.out.replace(".jsonl", "_report.json")
    write_json(report_path, report)
    print(f"[B4-atomic] records={len(records)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
