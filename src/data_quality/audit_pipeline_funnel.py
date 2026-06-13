"""Audit raw-to-final CLAIMARC data construction funnels.

The script quantifies whether the methodology's intended pair generation and
labeling logic matches the actual artifacts. It is deterministic and does not
call any model.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_count(rec: dict[str, Any]) -> int:
    ev = rec.get("evidence_count") or {}
    if isinstance(ev, dict):
        return sum(int(ev.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))
    return int(ev or 0)


def has_claim(rec: dict[str, Any]) -> bool:
    claim = rec.get("claim") or {}
    return bool(claim.get("has_claim_srt") and (claim.get("segments") or claim.get("passage")))


def product_index_stats(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    clips = obj.get("clips", []) if isinstance(obj, dict) else obj
    products = defaultdict(list)
    for row in clips:
        products[str(row.get("product_id", ""))].append(row)
    by_cat = Counter()
    neg_products = set()
    zero_comment = 0
    for pid, vals in products.items():
        cat = vals[0].get("直播间一级分类", "")
        by_cat[str(cat)] += 1
        if any(str(v.get("是否为负样本", "")) == "是" for v in vals):
            neg_products.add(pid)
        comments = max(int(v.get("评论总数", 0) or 0) for v in vals)
        if comments == 0:
            zero_comment += 1
    return {
        "clips": len(clips),
        "products": len(products),
        "products_marked_negative": len(neg_products),
        "products_zero_comment": zero_comment,
        "products_by_category": dict(by_cat),
    }


def stage_a_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_product = defaultdict(set)
    for r in rows:
        by_product[str(r.get("product_id", ""))].add(str(r.get("attribute_id", "")))
    return {
        "aspect_mentions": len(rows),
        "products_with_aspects": len(by_product),
        "unique_product_attribute_pairs": sum(len(v) for v in by_product.values()),
        "polarity": dict(Counter(str(r.get("polarity", "")) for r in rows)),
        "type": dict(Counter(str(r.get("type", "")) for r in rows)),
        "explicit_fact_hit": dict(Counter(bool(r.get("explicit_fact_hit")) for r in rows)),
        "mention_strength": dict(Counter(str(r.get("mention_strength", "")) for r in rows)),
        "top_attributes": Counter(str(r.get("attribute_id", "")) for r in rows).most_common(30),
    }


def claim_list_stats(claim_dir: Path) -> dict[str, Any]:
    files = [p for p in claim_dir.glob("*.jsonl") if not p.name.startswith("._")]
    rows = []
    products_with_rows = set()
    empty_files = 0
    for p in files:
        try:
            part = read_jsonl(p)
        except Exception:
            continue
        if part:
            products_with_rows.add(str(part[0].get("product_id") or p.stem))
            rows.extend(part)
        else:
            empty_files += 1
    by_product = Counter(str(r.get("product_id") or r.get("claim_id", "").split("_", 1)[0]) for r in rows)
    by_attr = Counter(str(r.get("attribute_id", "")) for r in rows)
    return {
        "claim_files": len(files),
        "claim_files_nonempty": len(products_with_rows),
        "claim_files_empty": empty_files,
        "atomic_claims": len(rows),
        "products_with_claims": len(products_with_rows),
        "products_with_claims_from_rows": len([k for k in by_product if k]),
        "top_claim_attributes": by_attr.most_common(30),
    }


def pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claimful = [r for r in rows if has_claim(r)]
    aligned_pos = [r for r in rows if (r.get("stats") or {}).get("N_aligned_neg", 0)]
    return {
        "pairs": len(rows),
        "claimful_pairs": len(claimful),
        "no_claim_pairs": len(rows) - len(claimful),
        "pairs_with_aligned_negative_review": len(aligned_pos),
        "category": dict(Counter(str(r.get("category", "")) for r in rows)),
        "claimful_by_category": dict(Counter(str(r.get("category", "")) for r in claimful)),
        "stats_N_total": {
            "zero": sum(1 for r in rows if int((r.get("stats") or {}).get("N_total", 0) or 0) == 0),
            "one": sum(1 for r in rows if int((r.get("stats") or {}).get("N_total", 0) or 0) == 1),
            "ge2": sum(1 for r in rows if int((r.get("stats") or {}).get("N_total", 0) or 0) >= 2),
        },
    }


def final_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claimful = [r for r in rows if has_claim(r)]
    sourceful = [r for r in rows if source_count(r) > 0]
    claim_source = [r for r in rows if has_claim(r) and source_count(r) > 0]
    return {
        "records": len(rows),
        "labels": dict(Counter(int(r.get("y", 0)) for r in rows)),
        "claimful": len(claimful),
        "sourceful": len(sourceful),
        "claimful_sourceful": len(claim_source),
        "confidence": dict(Counter(str(r.get("confidence", "")) for r in rows)),
        "claimful_labels": dict(Counter(int(r.get("y", 0)) for r in claimful)),
        "claimful_sourceful_labels": dict(Counter(int(r.get("y", 0)) for r in claim_source)),
        "split": dict(Counter(str(r.get("split", "")) for r in rows)),
    }


def write_markdown(report: dict[str, Any], out: Path) -> None:
    lines = ["# CLAIMARC Pipeline Funnel Audit", ""]
    for section, data in report.items():
        lines.append(f"## {section}")
        for k, v in data.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product_index", default="data/index/product_index.json")
    ap.add_argument("--stage_a", default="data/processed/stageA/resolved_aspects.jsonl")
    ap.add_argument("--claim_dir", default="data/processed/stageB/claim_list")
    ap.add_argument("--pair_records", default="data/processed/stageB/pair_records.jsonl")
    ap.add_argument("--final_dataset", default="data/final/dataset.jsonl")
    ap.add_argument("--out_json", default="data/final/pipeline_funnel_audit_20260612.json")
    ap.add_argument("--out_md", default="docs/PIPELINE_FUNNEL_AUDIT_20260612.md")
    args = ap.parse_args()

    report = {
        "product_index": product_index_stats(Path(args.product_index)),
        "stage_a": stage_a_stats(read_jsonl(args.stage_a)),
        "stage_b_claims": claim_list_stats(Path(args.claim_dir)),
        "stage_b_pairs": pair_stats(read_jsonl(args.pair_records)),
        "final_dataset": final_stats(read_jsonl(args.final_dataset)),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.out_md))
    print(f"[audit_pipeline_funnel] wrote {out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
