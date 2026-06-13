"""Audit CLAIMARC data against the proposal and build completion queues.

The paper's supervised task requires a complete triplet:

1. an attribute-grounded livestream claim;
2. product-side evidence for the same attribute;
3. a consumer-perception label produced by claim-aligned reviews.

This script does not relabel rows and does not remove hard-but-complete cases
to inflate separability.  It instead:

- annotates every A_cmt(p) row with claim/evidence/label quality states;
- writes a strict complete candidate set for main training/evaluation;
- writes queues for claim re-extraction, product-evidence recovery, and label
  alignment review so incomplete rows can be repaired from raw data before
  being promoted.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io_utils import normalize, read_jsonl, write_json, write_jsonl
from data_quality.audit_dataset_quality import has_claim, source_count


GENERIC_CLAIM_TERMS = (
    "一号链接",
    "二号链接",
    "三号链接",
    "四号链接",
    "五号链接",
    "给大家",
    "这个",
    "那个",
    "可以的",
    "好看的",
    "很好看",
    "蛮好",
    "自留",
    "闭眼",
    "冲",
)

CLAIM_REFERENCE_RE = re.compile(
    r"(直播|主播|宣传|介绍|页面|详情|描述|写着|写的|标着|标的|标注|标明|"
    r"说是|说的|说过|承诺|号称|展示|图片|链接|以为)"
)

MISMATCH_TERMS = (
    "不是",
    "并不是",
    "根本不是",
    "没有",
    "不符",
    "不符合",
    "不一致",
    "不一样",
    "差别",
    "差距",
    "虚假",
    "骗人",
    "欺骗",
    "挂羊头卖狗肉",
    "货不对板",
    "实物不符",
    "实际",
    "实际上",
    "缩水",
    "少了",
    "不存在",
)

SERVICE_OR_DEFECT_TERMS = (
    "售后",
    "客服",
    "物流",
    "快递",
    "发货",
    "退货",
    "换货",
    "退款",
    "破损",
    "破了",
    "坏了",
    "瑕疵",
    "污渍",
    "划痕",
    "开线",
    "脱线",
)


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def key(rec: dict[str, Any]) -> tuple[str, str]:
    return str(rec.get("product_id", "")), str(rec.get("attribute_id", ""))


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    segs = claim.get("segments") or []
    parts = [str(s.get("text", "") or "") for s in segs if isinstance(s, dict)]
    text = "\n".join(p for p in parts if p).strip()
    return text or str(claim.get("passage", "") or "").strip()


def aliases_for(base: dict[str, Any], pair_record: dict[str, Any] | None) -> list[str]:
    vals = [
        str(base.get("attribute_name", "") or ""),
        str(base.get("attribute_id", "") or ""),
    ]
    if pair_record:
        vals.append(str(pair_record.get("attribute_canonical", "") or ""))
        vals.extend(str(x) for x in (pair_record.get("aliases") or []))
    out = []
    seen = set()
    for val in vals:
        val = val.strip()
        if val and val not in seen:
            out.append(val)
            seen.add(val)
    return out


def claim_state(base: dict[str, Any], pair_record: dict[str, Any] | None) -> dict[str, Any]:
    if not has_claim(base):
        return {
            "state": "claim_missing",
            "claim_char_len": 0,
            "claim_alias_hits": [],
            "generic_terms": [],
            "claim_has_digit": False,
            "review_needed": True,
            "reasons": ["no_attribute_grounded_srt_claim"],
        }
    text = claim_text(base)
    norm_text = normalize(text)
    aliases = aliases_for(base, pair_record)
    alias_hits = [a for a in aliases if len(normalize(a)) >= 2 and normalize(a) in norm_text]
    generic_hits = [t for t in GENERIC_CLAIM_TERMS if t in text]
    has_digit = any(ch.isdigit() for ch in text)
    reasons: list[str] = []
    if len(norm_text) < 10:
        reasons.append("very_short_claim")
    if generic_hits and not alias_hits and not has_digit:
        reasons.append("generic_anchor_speech_without_attribute_token")
    if not alias_hits and len(norm_text) < 24 and not has_digit:
        reasons.append("low_attribute_specificity")
    state = "claim_present_specific"
    if reasons:
        state = "claim_present_review_needed"
    return {
        "state": state,
        "claim_char_len": len(norm_text),
        "claim_alias_hits": alias_hits[:8],
        "generic_terms": generic_hits,
        "claim_has_digit": has_digit,
        "review_needed": bool(reasons),
        "reasons": reasons,
    }


def evidence_state(rec: dict[str, Any]) -> dict[str, Any]:
    count = source_count(rec)
    cov = int(rec.get("coverage", 0) or 0)
    if count <= 0 or cov <= 0:
        state = "evidence_missing"
    elif cov == 1:
        state = "evidence_single_source"
    else:
        state = "evidence_multi_source"
    return {
        "state": state,
        "source_count": count,
        "coverage": cov,
        "needs_recovery": state == "evidence_missing",
    }


def label_state(rec: dict[str, Any]) -> dict[str, Any]:
    audit = rec.get("label_audit") or {}
    y = int(rec.get("y", 0) or 0)
    n_aligned = int(audit.get("n_aligned", 0) or 0)
    n_neg = int(audit.get("n_neg_aligned", 0) or 0)
    n_pos = int(audit.get("n_pos_aligned", 0) or 0)
    suspected_fake = bool(audit.get("suspected_fake"))
    if y == 1 and n_neg > 0:
        state = "label_positive_claim_aligned_neg"
        supported = True
    elif y == 0 and n_aligned > 0 and n_neg == 0:
        state = "label_negative_claim_aligned_nonneg"
        supported = not suspected_fake
    elif y == 0 and n_aligned == 0:
        state = "label_negative_no_aligned_review"
        supported = False
    else:
        state = "label_inconsistent_audit"
        supported = False
    return {
        "state": state,
        "supported_by_alignment": supported,
        "n_aligned": n_aligned,
        "n_neg_aligned": n_neg,
        "n_pos_aligned": n_pos,
        "suspected_fake": suspected_fake,
    }


def direct_consumer_signal(pair_record: dict[str, Any] | None, aliases: list[str]) -> list[dict[str, Any]]:
    if not pair_record:
        return []
    hits = []
    norm_aliases = [normalize(a) for a in aliases if len(normalize(a)) >= 2]
    for c in pair_record.get("reviews") or []:
        text = str(c.get("text", "") or c.get("evidence_span", "") or "")
        norm_text = normalize(text)
        if str(c.get("polarity", "")) != "neg" or str(c.get("mention_strength", "")) != "strong":
            continue
        has_ref = bool(CLAIM_REFERENCE_RE.search(text))
        has_mismatch = any(term in text for term in MISMATCH_TERMS)
        has_alias = any(a and a in norm_text for a in norm_aliases)
        service_only = any(term in text for term in SERVICE_OR_DEFECT_TERMS) and not has_ref
        if service_only:
            continue
        if has_ref or (has_mismatch and has_alias) or (c.get("explicit_fact_hit") and has_alias):
            hits.append(c)
    return hits


def annotate_row(base: dict[str, Any], pair_record: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    cstate = claim_state(out, pair_record)
    estate = evidence_state(out)
    lstate = label_state(out)
    aliases = aliases_for(out, pair_record)
    direct_hits = direct_consumer_signal(pair_record, aliases)

    issues: list[str] = []
    if cstate["state"] == "claim_missing":
        issues.append("missing_claim")
    elif cstate["review_needed"]:
        issues.append("claim_specificity_review")
    if estate["needs_recovery"]:
        issues.append("missing_product_evidence")
    if not lstate["supported_by_alignment"]:
        issues.append("weak_or_unsupported_consumer_label")
    if direct_hits and cstate["state"] == "claim_missing":
        issues.append("no_claim_but_direct_consumer_claim_reference")

    complete_main = (
        str(out.get("_attribute_scope", "")) == "product_attribute"
        and cstate["state"] == "claim_present_specific"
        and estate["state"] in {"evidence_single_source", "evidence_multi_source"}
    )
    complete_after_claim_review = (
        str(out.get("_attribute_scope", "")) == "product_attribute"
        and has_claim(out)
        and estate["state"] in {"evidence_single_source", "evidence_multi_source"}
    )
    out["_proposal_quality"] = {
        "version": "proposal_faithful_quality_audit_v1_20260613",
        "claim": cstate,
        "evidence": estate,
        "label": lstate,
        "direct_consumer_claim_reference_count": len(direct_hits),
        "direct_consumer_claim_reference_examples": [
            str(c.get("text", "") or c.get("evidence_span", ""))[:160] for c in direct_hits[:5]
        ],
        "issues": issues,
        "complete_claim_evidence_main_candidate": complete_main,
        "complete_after_claim_review_candidate": complete_after_claim_review,
        "label_high_reliability_slice": lstate["supported_by_alignment"],
    }
    return out


def queue_item(row: dict[str, Any], action: str, priority: int) -> dict[str, Any]:
    pq = row.get("_proposal_quality") or {}
    return {
        "pair_id": pair_id(row),
        "product_id": row.get("product_id"),
        "room_id": row.get("room_id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "y": row.get("y"),
        "c": row.get("c"),
        "split": row.get("split"),
        "priority": priority,
        "recommended_action": action,
        "claim_state": (pq.get("claim") or {}).get("state"),
        "evidence_state": (pq.get("evidence") or {}).get("state"),
        "label_state": (pq.get("label") or {}).get("state"),
        "issues": pq.get("issues", []),
        "claim_preview": claim_text(row)[:240],
        "direct_consumer_claim_reference_examples": pq.get("direct_consumer_claim_reference_examples", []),
    }


def build_queues(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    queues = {
        "claim_completion": [],
        "evidence_completion": [],
        "label_alignment_review": [],
        "attribute_schema_review": [],
    }
    for row in rows:
        pq = row.get("_proposal_quality") or {}
        claim = pq.get("claim") or {}
        evidence = pq.get("evidence") or {}
        label = pq.get("label") or {}
        direct_refs = int(pq.get("direct_consumer_claim_reference_count", 0) or 0)
        scope = str(row.get("_attribute_scope", ""))
        if scope != "product_attribute":
            queues["attribute_schema_review"].append(
                queue_item(row, "review_attribute_scope_against_proposal", 3)
            )
        if claim.get("state") == "claim_missing" or claim.get("review_needed"):
            priority = 1 if direct_refs else 2
            queues["claim_completion"].append(
                queue_item(row, "rerun_or_review_attribute_grounded_claim_extraction_from_raw_srt", priority)
            )
        if evidence.get("needs_recovery") and has_claim(row):
            priority = 1 if int(row.get("y", 0) or 0) == 1 else 2
            queues["evidence_completion"].append(
                queue_item(row, "rerun_attribute_conditioned_product_evidence_extraction_from_raw_details", priority)
            )
        if not label.get("supported_by_alignment") and has_claim(row):
            queues["label_alignment_review"].append(
                queue_item(row, "rerun_comment_claim_alignment_or_manual_audit_prompt", 2)
            )
    for vals in queues.values():
        vals.sort(key=lambda r: (int(r.get("priority", 9)), str(r.get("category", "")), str(r.get("pair_id", ""))))
    return queues


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq = [r.get("_proposal_quality") or {} for r in rows]
    claim = [(x.get("claim") or {}).get("state", "") for x in pq]
    evidence = [(x.get("evidence") or {}).get("state", "") for x in pq]
    label = [(x.get("label") or {}).get("state", "") for x in pq]
    issues = [issue for x in pq for issue in (x.get("issues") or [])]
    return {
        "n": len(rows),
        "labels": dict(Counter(int(r.get("y", 0) or 0) for r in rows)),
        "pos_rate": round(sum(int(r.get("y", 0) or 0) for r in rows) / max(1, len(rows)), 4),
        "claim_state": dict(Counter(claim)),
        "evidence_state": dict(Counter(evidence)),
        "label_state": dict(Counter(label)),
        "issues": dict(Counter(issues)),
        "coverage": dict(Counter(str(r.get("coverage", "")) for r in rows)),
        "attribute_scope": dict(Counter(str(r.get("_attribute_scope", "")) for r in rows)),
        "split": dict(Counter(str(r.get("split", "")) for r in rows)),
    }


def examples(rows: list[dict[str, Any]], issue: str, k: int = 8) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        pq = row.get("_proposal_quality") or {}
        if issue not in (pq.get("issues") or []):
            continue
        out.append({
            "pair_id": pair_id(row),
            "attribute_name": row.get("attribute_name"),
            "y": row.get("y"),
            "c": row.get("c"),
            "claim_preview": claim_text(row)[:160],
            "issue_examples": pq.get("direct_consumer_claim_reference_examples", []),
            "claim_state": (pq.get("claim") or {}).get("state"),
            "evidence_state": (pq.get("evidence") or {}).get("state"),
            "label_state": (pq.get("label") or {}).get("state"),
        })
        if len(out) >= k:
            break
    return out


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Proposal-Faithful Data Quality Audit v1",
        "",
        "## Principle",
        "The main supervised dataset should contain complete claim, evidence, and consumer-perception labels. Incomplete rows are not relabeled for training; they are routed to completion queues.",
        "",
        "## Outputs",
    ]
    for name, out_path in report["outputs"].items():
        lines.append(f"- `{name}`: `{out_path}`")
    lines += ["", "## Summaries"]
    for name, stats in report["summaries"].items():
        lines.append(f"### {name}")
        for key, value in stats.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines += [
        "## Queue Sizes",
    ]
    for name, stats in report["queues"].items():
        lines.append(f"- `{name}`: `{stats}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dataset", default="data/final/repaired_v1/dataset_attrpol_all_v1.jsonl")
    ap.add_argument("--pair_records", default="data/final/repaired_v1/pair_records_attrpol_v1.jsonl")
    ap.add_argument("--out_all", default="data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl")
    ap.add_argument("--out_complete", default="data/final/repaired_v1/dataset_attrpol_proposal_complete_claim_evidence_v1_20260613.jsonl")
    ap.add_argument("--out_complete_broad", default="data/final/repaired_v1/dataset_attrpol_proposal_complete_after_claim_review_v1_20260613.jsonl")
    ap.add_argument("--queue_dir", default="data/final/repaired_v1/proposal_completion_queues_v1_20260613")
    ap.add_argument("--report", default="data/final/repaired_v1/proposal_quality_audit_v1_20260613.report.json")
    ap.add_argument("--md", default="docs/PROPOSAL_FAITHFUL_DATA_QUALITY_AUDIT_20260613.md")
    args = ap.parse_args()

    pair_by_key = {key(r): r for r in read_jsonl(args.pair_records)}
    rows = [annotate_row(base, pair_by_key.get(key(base))) for base in read_jsonl(args.base_dataset)]
    rows.sort(key=lambda r: (str(r.get("room_id", "")), pair_id(r)))
    complete = [
        r for r in rows
        if (r.get("_proposal_quality") or {}).get("complete_claim_evidence_main_candidate")
    ]
    complete_broad = [
        r for r in rows
        if (r.get("_proposal_quality") or {}).get("complete_after_claim_review_candidate")
    ]
    queues = build_queues(rows)

    write_jsonl(args.out_all, rows)
    write_jsonl(args.out_complete, complete)
    write_jsonl(args.out_complete_broad, complete_broad)
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_paths = {}
    for name, vals in queues.items():
        path = queue_dir / f"{name}.jsonl"
        write_jsonl(path, vals)
        queue_paths[name] = str(path)

    report = {
        "inputs": {
            "base_dataset": args.base_dataset,
            "pair_records": args.pair_records,
        },
        "outputs": {
            "audit_all": args.out_all,
            "complete_claim_evidence_main": args.out_complete,
            "complete_after_claim_review": args.out_complete_broad,
            **{f"queue_{k}": v for k, v in queue_paths.items()},
            "markdown": args.md,
        },
        "summaries": {
            "all": summarize(rows),
            "complete_claim_evidence_main": summarize(complete),
            "complete_after_claim_review": summarize(complete_broad),
        },
        "queues": {
            name: {
                "n": len(vals),
                "priority": dict(Counter(int(v.get("priority", 9)) for v in vals)),
                "label": dict(Counter(int(v.get("y", 0) or 0) for v in vals)),
                "claim_state": dict(Counter(str(v.get("claim_state", "")) for v in vals)),
            }
            for name, vals in queues.items()
        },
        "examples": {
            "missing_claim": examples(rows, "missing_claim"),
            "claim_specificity_review": examples(rows, "claim_specificity_review"),
            "missing_product_evidence": examples(rows, "missing_product_evidence"),
            "weak_or_unsupported_consumer_label": examples(rows, "weak_or_unsupported_consumer_label"),
            "no_claim_but_direct_consumer_claim_reference": examples(
                rows, "no_claim_but_direct_consumer_claim_reference"
            ),
        },
        "interpretation": [
            "Use complete_claim_evidence_main as the current supervised benchmark: claim and product evidence are present, while label reliability remains encoded by c.",
            "Use complete_after_claim_review only after claim-specificity review, not as an automatic clean benchmark.",
            "Use queues to recover data from raw SRT/product details/images and then rebuild the complete set.",
            "Do not treat no-claim rows as training positives or easy negatives before claim completion.",
        ],
    }
    write_json(args.report, report)
    write_markdown(report, args.md)
    print(json.dumps({
        "complete_claim_evidence_main": report["summaries"]["complete_claim_evidence_main"],
        "complete_after_claim_review": report["summaries"]["complete_after_claim_review"],
        "queues": report["queues"],
        "report": args.report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
