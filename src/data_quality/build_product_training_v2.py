"""Build a product-scope training pool and regeneration manifest.

This script is deterministic and does not call an external model. It turns the
repaired claim-bearing product-attribute pool into:

- a 3000+ product-only training candidate with quality-aware sample weights;
- a manifest of pairs that should be regenerated/adjudicated by LLM/VLM before
  being promoted to the clean benchmark.

The clean benchmark remains `dataset_attrpol_hq_product_v1.jsonl`; this file is
for training augmentation and targeted data repair.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import has_claim, quality_bucket, source_count
from data_quality.rebuild_repaired_datasets_v1 import attribute_scope, split_leakage
from common import product_index as pidx


BUCKET_WEIGHT = {
    "pos_core": 1.20,
    "pos_silver": 0.95,
    "pos_weak": 0.50,
    "neg_core": 1.15,
    "neg_silver_sourceful": 0.85,
    "neg_silver_comment_only": 0.55,
    "neg_context_sourceful": 0.45,
    "neg_weak": 0.18,
    "neg_suspect_fake": 0.20,
}

SOURCE0_WEIGHT_MULT = {
    1: 0.75,
    0: 0.55,
}

HIGH_PRIORITY_BUCKETS = {
    "pos_weak",
    "neg_weak",
    "neg_suspect_fake",
}

ATTRIBUTE_NOISE_TERMS = (
    "直播",
    "视频",
    "展示",
    "宣传真实性",
    "产品真实性",
    "真实性",
    "使用方法",
    "购买",
    "运费",
    "包退",
    "售后",
    "客服",
    "店铺",
    "价格波动",
    "品质",
    "产品质量",
    "商品质量",
    "假货",
    "仿版",
    "效果",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def attribute_noise_flags(rec: dict[str, Any]) -> list[str]:
    text = f"{rec.get('attribute_name', '')} {rec.get('attribute_id', '')}"
    return [term for term in ATTRIBUTE_NOISE_TERMS if term in text]


def train_weight(rec: dict[str, Any], bucket: str, n_source: int, noise_flags: list[str]) -> tuple[float, float]:
    bucket_w = BUCKET_WEIGHT.get(bucket, 0.10)
    mult = SOURCE0_WEIGHT_MULT.get(int(rec.get("y", 0)), 0.60) if n_source == 0 else 1.0
    if noise_flags:
        mult *= 0.75
    c0 = float(rec.get("c", 0.05) or 0.05)
    return bucket_w, round(max(0.03, min(1.0, c0 * bucket_w * mult)), 4)


def regeneration_actions(rec: dict[str, Any], bucket: str, n_source: int) -> list[str]:
    actions: list[str] = []
    if rec.get("_attribute_noise_flags"):
        actions.append("schema_repair_review")
    if n_source == 0:
        actions.append("rerun_product_evidence")
    if bucket in HIGH_PRIORITY_BUCKETS:
        actions.append("llm_claim_comment_adjudication")
    if str(rec.get("confidence", "")) in {"absent", "medium"}:
        actions.append("evidence_sufficiency_check")
    if int(rec.get("y", 0)) == 1 and n_source == 0:
        actions.append("search_missing_counterevidence")
    if int(rec.get("y", 0)) == 0 and bucket in {"neg_suspect_fake", "neg_weak"}:
        actions.append("negative_label_verification")
    return list(dict.fromkeys(actions))


def annotate_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in rows:
        if not has_claim(rec):
            continue
        if str(rec.get("_attribute_scope", "")) != "product_attribute":
            continue
        row = dict(rec)
        bucket = quality_bucket(row)
        n_source = source_count(row)
        noise_flags = attribute_noise_flags(row)
        row["_attribute_noise_flags"] = noise_flags
        bucket_w, new_c = train_weight(row, bucket, n_source, noise_flags)
        row["_quality_bucket"] = bucket
        row["_source_count"] = n_source
        row["_c_original"] = float(row.get("c", 0.05) or 0.05)
        row["_quality_weight"] = bucket_w
        row["_source0_weight_multiplier"] = (
            SOURCE0_WEIGHT_MULT.get(int(row.get("y", 0)), 0.60) if n_source == 0 else 1.0
        )
        row["_train_policy"] = "product_claimful_quality_weighted_v2"
        row["_regeneration_actions"] = regeneration_actions(row, bucket, n_source)
        row["_needs_regeneration"] = bool(row["_regeneration_actions"])
        row["c"] = new_c
        out.append(row)
    out.sort(key=lambda r: (str(r.get("room_id", "")), pair_id(r)))
    return out


def build_manifest(train_rows: list[dict[str, Any]], missing_claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    bundles = pidx.build_bundles()

    for rec in train_rows:
        actions = rec.get("_regeneration_actions") or []
        if not actions:
            continue
        priority = 3
        bucket = str(rec.get("_quality_bucket", ""))
        if source_count(rec) == 0 and int(rec.get("y", 0)) == 1:
            priority = 1
        elif bucket in {"pos_weak", "neg_suspect_fake", "neg_weak"}:
            priority = 2
        manifest[pair_id(rec)] = {
            "pair_id": pair_id(rec),
            "product_id": rec.get("product_id"),
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": rec.get("attribute_name"),
            "category": rec.get("category"),
            "split": rec.get("split"),
            "current_y": rec.get("y"),
            "current_c": rec.get("c"),
            "quality_bucket": bucket,
            "source_count": source_count(rec),
            "confidence": rec.get("confidence"),
            "actions": actions,
            "priority": priority,
            "claim_available": has_claim(rec),
            "claim_preview": str((rec.get("claim") or {}).get("passage", ""))[:160],
        }

    for rec in missing_claim_rows:
        pid = str(rec.get("pair_id", ""))
        if not pid:
            continue
        scope_rec = {
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": rec.get("attribute_name"),
        }
        if attribute_scope(scope_rec) != "product_attribute":
            continue
        product_id = str(rec.get("product_id", ""))
        category = bundles[product_id].category if product_id in bundles else rec.get("category", "")
        item = manifest.setdefault(pid, {
            "pair_id": pid,
            "product_id": product_id,
            "attribute_id": rec.get("attribute_id"),
            "attribute_name": rec.get("attribute_name"),
            "category": category,
            "split": "",
            "current_y": None,
            "current_c": None,
            "quality_bucket": "missing_claim_risk",
            "source_count": 0,
            "confidence": "unknown",
            "actions": [],
            "priority": 1,
            "claim_available": False,
            "claim_preview": "",
        })
        item["actions"] = list(dict.fromkeys(item["actions"] + [
            "rerun_claim_extraction",
            "llm_claim_comment_adjudication",
        ]))
        item["priority"] = min(int(item.get("priority", 3)), 1)
        item["missing_claim_hits"] = rec.get("n_hits")
        item["missing_claim_example"] = rec.get("example")

    rows = list(manifest.values())
    rows.sort(key=lambda r: (int(r.get("priority", 3)), str(r.get("category", "")), str(r.get("pair_id", ""))))
    return rows


def summarize(train_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "train_n": len(train_rows),
        "train_labels": dict(Counter(int(r.get("y", 0)) for r in train_rows)),
        "train_split": dict(Counter(str(r.get("split", "")) for r in train_rows)),
        "train_split_leakage": split_leakage(train_rows),
        "train_quality_bucket": dict(Counter(str(r.get("_quality_bucket", "")) for r in train_rows)),
        "train_confidence": dict(Counter(str(r.get("confidence", "")) for r in train_rows)),
        "train_source0": sum(1 for r in train_rows if source_count(r) == 0),
        "train_needs_regeneration": sum(1 for r in train_rows if r.get("_needs_regeneration")),
        "train_attribute_noise": sum(1 for r in train_rows if r.get("_attribute_noise_flags")),
        "manifest_n": len(manifest_rows),
        "manifest_priority": dict(Counter(int(r.get("priority", 3)) for r in manifest_rows)),
        "manifest_actions": dict(Counter(a for r in manifest_rows for a in (r.get("actions") or []))),
    }


def write_markdown(report: dict[str, Any], path: str | Path, train_path: str, manifest_path: str) -> None:
    lines = [
        "# Product Training v2",
        "",
        "## Outputs",
        f"- training dataset: `{train_path}`",
        f"- regeneration manifest: `{manifest_path}`",
        "",
        "## Summary",
    ]
    for key, value in report.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Interpretation",
        "This is a training-augmentation pool, not the clean evaluation benchmark.",
        "Rows with absent evidence, weak labels, or suspected fake positive comments are kept for scale but down-weighted and queued for regeneration.",
        "The clean main benchmark should remain the HQ product set until these manifest items are re-adjudicated.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claimful", default="data/final/repaired_v1/dataset_attrpol_claimful_v1.jsonl")
    ap.add_argument("--missing_claim", default="data/final/repaired_v1/missing_claim_risk_pairs_v1.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_attrpol_product_train_v2.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/regeneration_manifest_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/product_training_v2_report.json")
    ap.add_argument("--md", default="docs/PRODUCT_TRAINING_V2_AND_REGENERATION.md")
    args = ap.parse_args()

    train_rows = annotate_training_rows(read_jsonl(args.claimful))
    missing_rows = read_jsonl(args.missing_claim) if Path(args.missing_claim).exists() else []
    manifest_rows = build_manifest(train_rows, missing_rows)
    report = summarize(train_rows, manifest_rows)

    write_jsonl(args.out, train_rows)
    write_jsonl(args.manifest, manifest_rows)
    write_json(args.report, report)
    write_markdown(report, args.md, args.out, args.manifest)
    print(f"[build_product_training_v2] train={len(train_rows)} manifest={len(manifest_rows)}")
    print(f"[build_product_training_v2] wrote {args.out}")
    print(f"[build_product_training_v2] wrote {args.manifest}")


if __name__ == "__main__":
    main()
