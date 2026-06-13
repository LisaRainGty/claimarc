"""Second-pass verifier for suspicious atomic alignments.

This is deliberately narrow: it re-checks only aligned claim/comment pairs
whose surface overlap is low, the common failure mode where generic attribute
sentiment is mistaken for claim-specific evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from typing import Any

from common import llm
from common.io_utils import bigram_jaccard, normalize, read_jsonl, write_json, write_jsonl


VERIFY_PROMPT = """你是严格的数据质检员。请判断消费者评论是否真的能支持/反驳单条主播 claim。

属性：{attr}
claim：{claim}
评论：{comment}
初判关系：{relation}

严格规则：
- aligned=1 只有在评论明确谈到 claim 的同一个具体事实/数值/颜色/材质/功能/规格，或明确给出不同事实。
- 泛泛正负评价、使用效果、喜好、搭配感、"尺寸合适/偏大/颜色怪/版型正/厚实" 等，不能自动支持/反驳具体 claim。
- 如果 claim 指向某个选项/链接/尺码，评论也必须指向同一个选项/尺码，除非语义明确。
- 评论谈的是同一部位的另一个属性，也判 aligned=0。

参考：
claim="灰黑色"，评论="清新的绿色" -> aligned=1, relation=refute（评论给出不同实际颜色）
claim="灰黑色"，评论="颜色很怪" -> aligned=0
claim="110斤以下拍S"，评论="商品尺寸非常合适" -> aligned=0
claim="110斤以下拍S"，评论="我100斤拍S刚好" -> aligned=1, relation=support
claim="三号链接会更薄"，评论="很厚实" -> aligned=0
claim="三号链接会更薄"，评论="三号一点也不薄，很厚" -> aligned=1, relation=refute
claim="羊毛混纺"，评论="硬得很" -> aligned=0
claim="羊毛混纺"，评论="不是羊毛，像腈纶" -> aligned=1, relation=refute
claim="内里是羊皮"，评论="内里跟鞋底都是米色，收到是黑色" -> aligned=0

输出 JSON 对象：
{{"aligned": 0或1, "relation": "support/refute/mixed/unclear", "reason": "不超过18字"}}
只输出 JSON。"""


def _overlap(claim: str, text: str) -> tuple[int, float]:
    cn = normalize(claim)
    tn = normalize(text)
    return len(set(cn) & set(tn)), bigram_jaccard(cn, tn)


def _needs_verify(claim: str, comment: str, relation: str, char_max: int, bigram_min: float) -> bool:
    char_ov, bj = _overlap(claim, comment)
    if relation not in {"support", "refute", "mixed"}:
        return False
    return char_ov <= char_max or bj < bigram_min


def _verify(attr: str, claim: str, comment: str, relation: str, model: str | None) -> dict[str, Any]:
    prompt = VERIFY_PROMPT.format(attr=attr, claim=claim[:300], comment=comment[:300], relation=relation)
    try:
        obj = llm.chat_json(prompt, namespace="b4_atomic_verify", model=model, max_tokens=320)
    except Exception as e:  # noqa: BLE001
        return {"aligned": 0, "relation": "unclear", "reason": f"verify_error:{repr(e)[:40]}"}
    if not isinstance(obj, dict):
        return {"aligned": 0, "relation": "unclear", "reason": "non_object"}
    aligned = 1 if int(obj.get("aligned", 0) or 0) == 1 else 0
    rel = str(obj.get("relation") or "unclear").lower()
    if rel not in {"support", "refute", "mixed", "unclear"}:
        rel = "unclear"
    if not aligned:
        rel = "unclear"
    return {"aligned": aligned, "relation": rel, "reason": str(obj.get("reason") or "")[:40]}


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
    ap.add_argument("--char_max", type=int, default=1)
    ap.add_argument("--bigram_min", type=float, default=0.08)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    rows = [copy.deepcopy(r) for r in read_jsonl(args.atomic_records)]
    tasks: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        claim = (r.get("claim") or {}).get("passage", "")
        for j, c in enumerate(r.get("reviews", []) or []):
            if int(c.get("y_supportability", 0) or 0) != 1:
                continue
            if _needs_verify(claim, c.get("text", ""), str(c.get("relation") or ""), args.char_max, args.bigram_min):
                tasks.append((i, j))
    print(f"[refine-atomic] records={len(rows)} verify_tasks={len(tasks)}", flush=True)

    def _job(idx: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        i, j = idx
        r = rows[i]
        c = r["reviews"][j]
        claim = (r.get("claim") or {}).get("passage", "")
        res = _verify(r.get("attribute_name", r.get("attribute_id", "")), claim, c.get("text", ""), c.get("relation", ""), args.model)
        return i, j, res

    results = llm.run_many(tasks, _job, desc="verify-atomic") if tasks else []
    changes = Counter()
    examples = []
    for item in results:
        if not isinstance(item, tuple):
            continue
        i, j, res = item
        c = rows[i]["reviews"][j]
        before = (int(c.get("y_supportability", 0) or 0), c.get("relation"))
        c["_atomic_verify"] = res
        c["y_supportability"] = int(res["aligned"])
        c["relation"] = res["relation"]
        c["alignment_rationale"] = res["reason"] or c.get("alignment_rationale", "")
        after = (int(c.get("y_supportability", 0) or 0), c.get("relation"))
        if before != after:
            changes[f"{before}->{after}"] += 1
            if len(examples) < 50:
                examples.append({
                    "atomic_id": rows[i].get("atomic_id"),
                    "attribute_name": rows[i].get("attribute_name"),
                    "claim": (rows[i].get("claim") or {}).get("passage", ""),
                    "comment": c.get("text", ""),
                    "before": before,
                    "after": after,
                    "reason": res.get("reason", ""),
                })
    for r in rows:
        r["stats"] = _stats(r.get("reviews", []) or [])
    write_jsonl(args.out, rows)
    report = {
        "atomic_records": args.atomic_records,
        "out": args.out,
        "verify_tasks": len(tasks),
        "changes": dict(changes),
        "post_record_status": dict(Counter(
            "refute" if r["stats"]["N_refute"] else
            "support" if r["stats"]["N_support"] else
            "aligned_other" if r["stats"]["N_aligned"] else "unaligned"
            for r in rows
        )),
        "examples": examples,
    }
    write_json(args.report or args.out.replace(".jsonl", "_refine_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
