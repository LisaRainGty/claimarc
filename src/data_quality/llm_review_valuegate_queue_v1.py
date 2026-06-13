"""LLM review for value-gated recovered atomic rows.

The value gate intentionally recovers short seller claims such as flavor,
style, season, and function values.  These rows need a separate review because
short claims are easy to align to generic consumer sentiment.  This script
reviews only a small priority queue rather than relabeling the full dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from common import llm
from common.io_utils import read_jsonl, write_json, write_jsonl


PROMPT = """你是直播电商数据质检员。请审查这条从短值 claim 恢复出的训练样本是否可以保留。

属性：{attr}
主播 claim：{claim}
证据片段：
{evidence}
已对齐消费者评论：
{reviews}

判断标准：
1. claim 必须是该属性下的具体商品事实或明确承诺；泛泛修辞、适用对象口号、过短无上下文词，不能保留。
2. 证据片段至少不能明显反驳 claim；如果证据缺失但评论很明确，可标 evidence_ok=uncertain。
3. risk_label_valid=1 只有在评论明确反驳 claim 或说明实际体验与 claim 不一致时成立。
4. 泛泛好/不好、好看/不好看、舒服/不舒服、效果一般，不能支持或反驳短值 claim，除非评论复述了同一具体事实。

输出 JSON：
{{
  "claim_concrete": 0或1,
  "evidence_ok": "yes/no/uncertain",
  "risk_label_valid": 0或1,
  "recommended_action": "keep/drop/review",
  "reason": "不超过30字"
}}
只输出 JSON。"""


def fmt_items(items: list[dict[str, Any]], key: str = "text", limit: int = 6) -> str:
    lines = []
    for i, item in enumerate(items[:limit], start=1):
        text = str(item.get(key) or "")[:220]
        if not text:
            continue
        prefix = item.get("source") or item.get("relation") or ""
        lines.append(f"{i}. [{prefix}] {text}" if prefix else f"{i}. {text}")
    return "\n".join(lines) if lines else "（无）"


def review_row(row: dict[str, Any], model: str | None) -> dict[str, Any]:
    prompt = PROMPT.format(
        attr=row.get("attribute_name", ""),
        claim=row.get("claim_text", ""),
        evidence=fmt_items(row.get("evidence_snippets") or []),
        reviews=fmt_items(row.get("aligned_reviews") or []),
    )
    try:
        obj = llm.chat_json(prompt, namespace="valuegate_review_v1", model=model, max_tokens=360)
    except Exception as e:  # noqa: BLE001
        obj = {
            "claim_concrete": 0,
            "evidence_ok": "uncertain",
            "risk_label_valid": 0,
            "recommended_action": "review",
            "reason": f"error:{repr(e)[:40]}",
        }
    if not isinstance(obj, dict):
        obj = {"recommended_action": "review", "reason": "non_object"}
    out = dict(row)
    out["llm_review"] = {
        "claim_concrete": 1 if int(obj.get("claim_concrete", 0) or 0) == 1 else 0,
        "evidence_ok": str(obj.get("evidence_ok") or "uncertain").lower(),
        "risk_label_valid": 1 if int(obj.get("risk_label_valid", 0) or 0) == 1 else 0,
        "recommended_action": str(obj.get("recommended_action") or "review").lower(),
        "reason": str(obj.get("reason") or "")[:80],
    }
    if out["llm_review"]["recommended_action"] not in {"keep", "drop", "review"}:
        out["llm_review"]["recommended_action"] = "review"
    if out["llm_review"]["evidence_ok"] not in {"yes", "no", "uncertain"}:
        out["llm_review"]["evidence_ok"] = "uncertain"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--priority", action="append", default=["P0"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    wanted = set(args.priority or [])
    rows = [r for r in read_jsonl(args.queue) if str(r.get("priority")) in wanted]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[valuegate-review] rows={len(rows)} priority={sorted(wanted)}", flush=True)

    def _job(row: dict[str, Any]) -> dict[str, Any]:
        return review_row(row, args.model)

    reviewed = llm.run_many(rows, _job, desc="valuegate-review")
    reviewed = [r for r in reviewed if isinstance(r, dict) and "__error__" not in r]
    write_jsonl(args.out, reviewed)
    actions = Counter((r.get("llm_review") or {}).get("recommended_action", "") for r in reviewed)
    valid = Counter(str((r.get("llm_review") or {}).get("risk_label_valid", 0)) for r in reviewed)
    report = {
        "queue": args.queue,
        "out": args.out,
        "priority": sorted(wanted),
        "n": len(reviewed),
        "actions": dict(actions),
        "risk_label_valid": dict(valid),
        "by_attribute": Counter(str(r.get("attribute_name") or "") for r in reviewed).most_common(30),
        "examples": [
            {
                "attribute": r.get("attribute_name"),
                "claim": r.get("claim_text"),
                "action": (r.get("llm_review") or {}).get("recommended_action"),
                "valid": (r.get("llm_review") or {}).get("risk_label_valid"),
                "reason": (r.get("llm_review") or {}).get("reason"),
            }
            for r in reviewed[:30]
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
