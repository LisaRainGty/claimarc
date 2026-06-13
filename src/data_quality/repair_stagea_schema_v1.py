"""Create a versioned cleaned Stage A schema for future B/C regeneration.

The current pipeline should not overwrite Stage A artifacts while experiments
are in progress. This script therefore writes a parallel directory:

  data/processed/stageA_repaired_v1/

Repairs are deterministic:
- strip angle-bracket placeholders from aliases/canonical names
- remove service/process and subjective/personal-evaluation aliases
- drop attributes whose canonical name is clearly service/evaluation
- merge duplicate canonical names inside each category
- write a product-scope resolved_aspects file mapped to kept attribute ids
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common.io_utils import read_json, read_jsonl, write_json, write_jsonl


ANGLE = re.compile(r"^<(.+)>$")
SERVICE_TERMS = (
    "客服", "售后", "物流", "发货", "快递", "配送", "退货", "退款", "退换",
    "下单", "店铺", "卖家", "商家", "服务",
)
EVAL_TERMS = tuple(getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])) + (
    "体验", "感受", "评价", "满意", "推荐", "回购", "复购", "性价比",
    "划算", "喜欢", "购买意愿", "个人口味", "小孩喜好", "喜好度",
)

ALLOW_CANONICAL = {
    "价格",
    "品牌",
    "口感",
    "风味",
    "气味",
    "异味",
    "适用人群",
    "包装",
    "包装类型",
    "包装规格",
}


def clean_token(value: Any) -> str:
    text = str(value or "").strip()
    m = ANGLE.match(text)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"\s+", "", text)
    return text


def is_bad_phrase(text: str, *, canonical: bool = False) -> bool:
    if not text:
        return True
    if canonical and text in ALLOW_CANONICAL:
        return False
    return any(term in text for term in SERVICE_TERMS) or any(term in text for term in EVAL_TERMS)


def clean_aliases(canonical: str, aliases: list[Any]) -> list[str]:
    out = []
    seen = set()
    for raw in [canonical] + list(aliases or []):
        alias = clean_token(raw)
        if not alias or alias in seen:
            continue
        if alias != canonical and is_bad_phrase(alias):
            continue
        seen.add(alias)
        out.append(alias)
    return out


def clean_cas_file(path: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    obj = read_json(path, default={"attributes": []}) or {"attributes": []}
    by_name: dict[str, dict[str, Any]] = {}
    id_map: dict[str, str] = {}
    dropped = []
    duplicate_merges = []
    placeholder_aliases = 0
    bad_aliases = 0

    for attr in obj.get("attributes", []):
        old_id = str(attr.get("attribute_id", ""))
        canonical = clean_token(attr.get("canonical_name", ""))
        aliases_raw = list(attr.get("aliases", []) or [])
        text_raw = " ".join([str(attr.get("canonical_name", ""))] + [str(a) for a in aliases_raw])
        placeholder_aliases += len(re.findall(r"<[^>]+>", text_raw))
        if is_bad_phrase(canonical, canonical=True):
            dropped.append({"attribute_id": old_id, "canonical_name": canonical, "reason": "bad_canonical"})
            continue
        aliases = clean_aliases(canonical, aliases_raw)
        bad_aliases += max(0, len([canonical] + aliases_raw) - len(aliases))
        if canonical in by_name:
            keep = by_name[canonical]
            keep_id = str(keep["attribute_id"])
            old_aliases = list(keep.get("aliases", []))
            merged = []
            seen = set()
            for item in old_aliases + aliases:
                if item and item not in seen:
                    seen.add(item)
                    merged.append(item)
            keep["aliases"] = merged
            id_map[old_id] = keep_id
            duplicate_merges.append({"dropped_id": old_id, "kept_id": keep_id, "canonical_name": canonical})
            continue
        new_attr = dict(attr)
        new_attr["canonical_name"] = canonical
        new_attr["aliases"] = aliases
        by_name[canonical] = new_attr
        id_map[old_id] = old_id

    cleaned = dict(obj)
    cleaned["attributes"] = list(by_name.values())
    report = {
        "input_attributes": len(obj.get("attributes", [])),
        "output_attributes": len(cleaned["attributes"]),
        "dropped_attributes": len(dropped),
        "duplicate_merges": len(duplicate_merges),
        "placeholder_alias_hits": placeholder_aliases,
        "bad_aliases_removed_est": bad_aliases,
        "dropped_examples": dropped[:50],
        "duplicate_examples": duplicate_merges[:50],
    }
    return cleaned, id_map, report


def clean_resolved(
    stage_a: Path,
    out_dir: Path,
    id_maps_by_category: dict[str, dict[str, str]],
) -> dict[str, Any]:
    rows = []
    stats = Counter()
    for row in read_jsonl(stage_a / "resolved_aspects.jsonl"):
        cat = str(row.get("category", ""))
        aid = str(row.get("attribute_id", ""))
        id_map = id_maps_by_category.get(cat, {})
        if aid not in id_map:
            stats["dropped_unmapped_or_bad_attr"] += 1
            continue
        out = dict(row)
        new_id = id_map[aid]
        if new_id != aid:
            stats["remapped_duplicate_attr"] += 1
            out["_attribute_id_before_schema_repair"] = aid
            out["attribute_id"] = new_id
        rows.append(out)
    write_jsonl(out_dir / "resolved_aspects_schema_clean_v1.jsonl", rows)
    stats["input_resolved"] = sum(1 for _ in read_jsonl(stage_a / "resolved_aspects.jsonl"))
    stats["output_resolved"] = len(rows)
    stats["unique_pairs"] = len({(r.get("product_id"), r.get("attribute_id")) for r in rows})
    return dict(stats)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Stage A Schema Repair v1", ""]
    lines.append("## Summary")
    for k, v in report.get("summary", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Categories")
    for cat, rep in sorted(report.get("categories", {}).items()):
        lines.append(f"### {cat}")
        for k, v in rep.items():
            if isinstance(v, list):
                lines.append(f"- `{k}`: {len(v)} examples")
            else:
                lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default=str(config.STAGE_A))
    ap.add_argument("--out_dir", default=str(config.PROCESSED / "stageA_repaired_v1"))
    ap.add_argument("--report_json", default=str(config.FINAL / "stageA_schema_repair_v1_report.json"))
    ap.add_argument("--report_md", default=str(config.ROOT / "docs" / "STAGEA_SCHEMA_REPAIR_V1.md"))
    args = ap.parse_args()

    stage_a = Path(args.stage_a)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_maps: dict[str, dict[str, str]] = {}
    category_reports = {}
    for path in sorted(stage_a.glob("CAS+_*.json")):
        if path.name.startswith("._"):
            continue
        cat = path.stem.replace("CAS+_", "")
        cleaned, id_map, rep = clean_cas_file(path)
        write_json(out_dir / path.name, cleaned)
        id_maps[cat] = id_map
        category_reports[cat] = rep

    resolved_report = clean_resolved(stage_a, out_dir, id_maps)
    summary = {
        "categories": len(category_reports),
        "input_attributes": sum(r["input_attributes"] for r in category_reports.values()),
        "output_attributes": sum(r["output_attributes"] for r in category_reports.values()),
        "dropped_attributes": sum(r["dropped_attributes"] for r in category_reports.values()),
        "duplicate_merges": sum(r["duplicate_merges"] for r in category_reports.values()),
        "placeholder_alias_hits": sum(r["placeholder_alias_hits"] for r in category_reports.values()),
        **resolved_report,
    }
    report = {"summary": summary, "categories": category_reports}
    write_json(args.report_json, report)
    write_markdown(report, Path(args.report_md))
    print(f"[repair_stagea_schema_v1] wrote {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
