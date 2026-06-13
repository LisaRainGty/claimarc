"""Recover high-recall text evidence from raw product data.

This script targets source-absent or manifest-flagged product-attribute pairs.
It does not call an LLM. It searches raw product title, product parameters, and
cached all-image OCR text to create auditable candidate evidence.

Outputs are versioned and non-destructive:

- recovered evidence records per pair;
- a diagnostic dataset with recovered text evidence appended;
- a compact JSON/Markdown report.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from common.io_utils import normalize, read_json, read_jsonl, write_json, write_jsonl
from data_quality.audit_dataset_quality import has_claim, source_count


GENERIC_TERMS = {
    "是否", "情况", "效果", "属性", "商品", "产品", "直播", "描述", "展示",
    "相关", "信息", "购买", "使用", "方法", "质量", "品质", "喜好", "价值",
    "用性", "格合", "合理", "理性", "适度", "好度", "程度",
}

NOISY_ATTRIBUTE_TERMS = {
    "喜好度", "购买价值", "购买渠道", "购买频次", "直播展示", "直播描述",
    "直播内容", "直播宣传", "搭配效果", "满意", "推荐", "回购",
}

NUMERIC_HINT_TERMS = {
    "尺码", "尺寸", "高度", "宽度", "厚度", "重量", "克重", "容量", "净含量",
    "含量", "蓬松度", "充绒量", "绒子", "比例", "浓度", "长度", "价格",
}


def read_jsonl_list(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def clean_name(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def split_terms(text: str) -> list[str]:
    text = clean_name(text)
    chunks = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    out: list[str] = []
    for ch in chunks:
        if re.fullmatch(r"[A-Za-z0-9]+", ch):
            if len(ch) >= 2:
                out.append(ch.lower())
            continue
        if len(ch) <= 4:
            out.append(ch)
        else:
            # Keep meaningful bi/tri-grams for noisy generated attribute names.
            out.extend(ch[i:i + 2] for i in range(len(ch) - 1))
            out.extend(ch[i:i + 3] for i in range(len(ch) - 2))
    return [t for t in dict.fromkeys(out) if t and t not in GENERIC_TERMS]


def claim_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?\s*(?:cm|CM|mm|g|kg|%|％|m|ml|ML|元|米|块|码)?", text or ""))


def load_cas_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((config.PROCESSED / "stageA_repaired_v1").glob("CAS+_*.json")):
        obj = read_json(path, default={"attributes": []})
        for attr in obj.get("attributes", []) or []:
            if attr.get("attribute_id"):
                out[str(attr["attribute_id"])] = attr
    for path in sorted(config.STAGE_A.glob("CAS+_*.json")):
        obj = read_json(path, default={"attributes": []})
        for attr in obj.get("attributes", []) or []:
            if attr.get("attribute_id") and str(attr["attribute_id"]) not in out:
                out[str(attr["attribute_id"])] = attr
    return out


def attribute_terms(rec: dict[str, Any], cas_meta: dict[str, dict[str, Any]]) -> list[str]:
    primary_vals = [
        rec.get("attribute_name", ""),
        rec.get("attribute_canonical", ""),
        str(rec.get("attribute_id", "")).split("_", 1)[-1],
    ]
    meta = cas_meta.get(str(rec.get("attribute_id", "")), {})
    primary_vals.append(meta.get("canonical_name", ""))
    terms: list[str] = []
    for val in primary_vals:
        terms.extend(split_terms(str(val)))
    # Aliases in CAS+ can be broad or evaluative. Keep only short complete
    # aliases; do not decompose long aliases into ambiguous bi-grams.
    for alias in (meta.get("aliases") or [])[:12]:
        al = clean_name(alias)
        if 2 <= len(al) <= 6 and al not in GENERIC_TERMS:
            terms.append(al)
    return [t for t in dict.fromkeys(terms) if t]


def claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    passage = str(claim.get("passage", "") or "")
    if passage:
        return passage
    return "\n".join(str(s.get("text", "") or "") for s in (claim.get("segments") or []))


def is_schema_noisy(rec: dict[str, Any]) -> bool:
    text = f"{clean_name(rec.get('attribute_name'))} {rec.get('attribute_id', '')}"
    return any(term in text for term in NOISY_ATTRIBUTE_TERMS)


def load_raw_products() -> dict[str, dict[str, Any]]:
    idx = pidx.load_index()
    return idx.get("products", {}) or {}


def product_price_record(pid: str, raw_products: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    prod = raw_products.get(pid, {}) or {}
    vals = []
    if prod.get("price") not in (None, ""):
        vals.append({"param_key": "raw_price", "raw_text": f"价格: {prod.get('price')}", "match": "raw_price"})
    if prod.get("sales_info"):
        vals.append({"param_key": "raw_sales_info", "raw_text": f"销量: {prod.get('sales_info')}", "match": "raw_sales_info"})
    return vals


def score_text(text: str, terms: list[str], nums: set[str]) -> int:
    ntext = normalize(text)
    score = 0
    for term in terms:
        nt = normalize(term)
        if nt and nt in ntext:
            score += 2 if len(nt) >= 3 else 1
    for num in nums:
        if num and normalize(num) in ntext:
            score += 2
    return score


def term_num_scores(text: str, terms: list[str], nums: set[str]) -> tuple[int, int]:
    ntext = normalize(text)
    term_score = 0
    for term in terms:
        nt = normalize(term)
        if nt and nt in ntext:
            term_score += 2 if len(nt) >= 3 else 1
    num_score = 0
    for num in nums:
        if num and normalize(num) in ntext:
            num_score += 2
    return term_score, num_score


def is_price_attr(rec: dict[str, Any]) -> bool:
    attr_name = clean_name(rec.get("attribute_name"))
    aid = str(rec.get("attribute_id", ""))
    return attr_name == "价格" or aid.endswith("_价格")


def has_numeric_hint(rec: dict[str, Any], terms: list[str]) -> bool:
    text = f"{clean_name(rec.get('attribute_name'))} {rec.get('attribute_id', '')} {' '.join(terms)}"
    return any(t in text for t in NUMERIC_HINT_TERMS)


def recover_from_title(rec: dict[str, Any], bundle: pidx.ProductBundle, terms: list[str], nums: set[str]) -> list[dict[str, Any]]:
    if is_price_attr(rec):
        return []
    title = bundle.title or ""
    if not title:
        return []
    term_score, num_score = term_num_scores(title, terms, nums)
    score = term_score + num_score
    # Product titles are noisy but often contain key claims like 90绒/700蓬/100%棉.
    if term_score >= 2 or (has_numeric_hint(rec, terms) and term_score >= 1 and num_score >= 2):
        return [{"param_key": "raw_title", "raw_text": f"商品标题: {title}", "match": "raw_title", "score": score}]
    return []


def recover_from_params(
    rec: dict[str, Any],
    bundle: pidx.ProductBundle,
    terms: list[str],
    nums: set[str],
    raw_products: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    attr_name = clean_name(rec.get("attribute_name"))
    for key, val in (bundle.params or {}).items():
        text = f"{key}: {val}"
        key_term_score, _ = term_num_scores(str(key), terms, set())
        value_term_score, value_num_score = term_num_scores(str(val), terms, nums)
        exact = normalize(attr_name) and normalize(attr_name) in normalize(str(key))
        if exact or key_term_score >= 1 or (key_term_score >= 1 and value_num_score >= 2) or value_term_score >= 2:
            out.append({
                "param_key": str(key),
                "raw_text": str(val),
                "match": "raw_param_relaxed",
                "score": key_term_score + value_term_score + value_num_score + (3 if exact else 0),
            })
    if is_price_attr(rec):
        out.extend(product_price_record(str(rec.get("product_id")), raw_products))
    out.sort(key=lambda x: (-int(x.get("score", 0)), str(x.get("param_key", ""))))
    return out[:6]


def ocr_cache(pid: str) -> dict[str, str]:
    path = config.STAGE_C / "ocr_text" / f"{pid}.json"
    obj = read_json(path, default={}) or {}
    return {str(k): str(v or "") for k, v in obj.items()}


def split_ocr_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text or "") if ln.strip()]
    merged: list[str] = []
    for i, line in enumerate(lines):
        ctx = line
        if i + 1 < len(lines) and len(ctx) < 16:
            ctx = f"{ctx} {lines[i + 1]}"
        merged.append(ctx)
    return merged


def recover_from_ocr(pid: str, terms: list[str], nums: set[str], limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for image_path, text in ocr_cache(pid).items():
        for line in split_ocr_lines(text):
            term_score, num_score = term_num_scores(line, terms, nums)
            score = term_score + num_score
            if term_score >= 2 or (term_score >= 1 and num_score >= 2):
                out.append({
                    "raw_text": line[:180],
                    "image_path": image_path,
                    "match": "raw_ocr_recall",
                    "score": score,
                })
    out.sort(key=lambda x: (-int(x.get("score", 0)), str(x.get("image_path", "")), str(x.get("raw_text", ""))))
    dedup = []
    seen = set()
    for item in out:
        key = normalize(item["raw_text"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= limit:
            break
    return dedup


def evidence_count(rec: dict[str, Any]) -> dict[str, int]:
    return {
        "params": len(rec.get("evidence_params") or []),
        "ocr": len(rec.get("evidence_ocr") or []),
        "vlm": len(rec.get("evidence_vlm") or []),
    }


def recover_pair(
    rec: dict[str, Any],
    bundles: dict[str, pidx.ProductBundle],
    raw_products: dict[str, dict[str, Any]],
    cas_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pid = str(rec.get("product_id", ""))
    bundle = bundles.get(pid)
    terms = attribute_terms(rec, cas_meta)
    nums = claim_numbers(claim_text(rec))
    result = {
        "pair_id": pair_id(rec),
        "product_id": pid,
        "attribute_id": rec.get("attribute_id"),
        "attribute_name": rec.get("attribute_name"),
        "category": rec.get("category"),
        "source_count_before": source_count(rec),
        "schema_noisy": is_schema_noisy(rec),
        "attribute_terms": terms[:20],
        "claim_numbers": sorted(nums),
        "recovered_params": [],
        "recovered_ocr": [],
        "recovered_vlm": [],
    }
    if not bundle or result["schema_noisy"]:
        return result
    result["recovered_params"] = (
        recover_from_title(rec, bundle, terms, nums)
        + recover_from_params(rec, bundle, terms, nums, raw_products)
    )
    result["recovered_ocr"] = [] if is_price_attr(rec) else recover_from_ocr(pid, terms, nums)
    return result


def merge_dataset(rows: list[dict[str, Any]], recovered: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in rows:
        pid = pair_id(rec)
        rr = recovered.get(pid)
        nr = dict(rec)
        if rr and not rr.get("schema_noisy"):
            params = list(nr.get("evidence_params") or [])
            ocr = list(nr.get("evidence_ocr") or [])
            params.extend(rr.get("recovered_params") or [])
            ocr.extend(rr.get("recovered_ocr") or [])
            # Deduplicate by display text.
            seen_p = set()
            dedup_p = []
            for item in params:
                key = (str(item.get("param_key", "")), str(item.get("raw_text", "")))
                if key not in seen_p:
                    seen_p.add(key)
                    dedup_p.append(item)
            seen_o = set()
            dedup_o = []
            for item in ocr:
                key = (str(item.get("image_path", "")), str(item.get("raw_text", "")))
                if key not in seen_o:
                    seen_o.add(key)
                    dedup_o.append(item)
            nr["evidence_params"] = dedup_p
            nr["evidence_ocr"] = dedup_o
            nr["_raw_text_evidence_recovery"] = {
                "params_added": len(rr.get("recovered_params") or []),
                "ocr_added": len(rr.get("recovered_ocr") or []),
                "schema_noisy": rr.get("schema_noisy", False),
            }
            cnt = evidence_count(nr)
            nr["evidence_count"] = cnt
            coverage = sum(1 for v in cnt.values() if v > 0)
            nr["coverage"] = coverage
            nr["confidence"] = config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent")
        out.append(nr)
    return out


def summarize(target_rows: list[dict[str, Any]], recovered_rows: list[dict[str, Any]], merged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    any_rec = [r for r in recovered_rows if (r.get("recovered_params") or r.get("recovered_ocr")) and not r.get("schema_noisy")]
    return {
        "targets": len(target_rows),
        "targets_source0": sum(1 for r in target_rows if source_count(r) == 0),
        "schema_noisy": sum(1 for r in recovered_rows if r.get("schema_noisy")),
        "recovered_any": len(any_rec),
        "recovered_params": sum(1 for r in recovered_rows if r.get("recovered_params")),
        "recovered_ocr": sum(1 for r in recovered_rows if r.get("recovered_ocr")),
        "source0_after": sum(1 for r in merged_rows if source_count(r) == 0),
        "confidence_after": dict(Counter(str(r.get("confidence", "")) for r in merged_rows)),
        "label_source0_after": dict(Counter(f"{int(r.get('y', 0))}:{source_count(r) == 0}" for r in merged_rows)),
        "category_recovered": dict(Counter(str(r.get("category", "")) for r in any_rec)),
    }


def write_markdown(report: dict[str, Any], path: str | Path, out_dataset: str, evidence_out: str) -> None:
    lines = [
        "# Raw Text Evidence Recovery v1",
        "",
        f"- recovered evidence: `{evidence_out}`",
        f"- diagnostic dataset: `{out_dataset}`",
        "",
        "## Summary",
    ]
    for k, v in report.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Notes",
        "This is a high-recall candidate recovery step. It should be followed by",
        "LLM/VLM adjudication before recovered evidence is promoted to a clean",
        "benchmark.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_v1.jsonl")
    ap.add_argument("--manifest", default="data/final/repaired_v1/regeneration_manifest_v1.jsonl")
    ap.add_argument("--out_evidence", default="data/final/repaired_v1/raw_text_evidence_recovery_hq_product_v1.jsonl")
    ap.add_argument("--out_dataset", default="data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_v1.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/raw_text_evidence_recovery_hq_product_v1_report.json")
    ap.add_argument("--md", default="docs/RAW_TEXT_EVIDENCE_RECOVERY_V1.md")
    ap.add_argument("--only_source0", action="store_true", default=True)
    ap.add_argument("--all_claimful", action="store_true", help="Ignore --only_source0 and process all claimful rows.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl_list(args.dataset)
    manifest_ids = set()
    if Path(args.manifest).exists():
        for item in read_jsonl(args.manifest):
            if "rerun_product_evidence" in (item.get("actions") or []):
                manifest_ids.add(str(item.get("pair_id")))

    target = [r for r in rows if has_claim(r)]
    if not args.all_claimful:
        target = [
            r for r in target
            if (source_count(r) == 0 if args.only_source0 else False) or pair_id(r) in manifest_ids
        ]
    if args.limit:
        target = target[:args.limit]

    bundles = pidx.build_bundles()
    raw_products = load_raw_products()
    cas_meta = load_cas_meta()
    recovered_rows = [recover_pair(r, bundles, raw_products, cas_meta) for r in target]
    recovered_by_pair = {str(r["pair_id"]): r for r in recovered_rows}
    merged = merge_dataset(rows, recovered_by_pair)
    report = summarize(target, recovered_rows, merged)

    write_jsonl(args.out_evidence, recovered_rows)
    write_jsonl(args.out_dataset, merged)
    write_json(args.report, report)
    write_markdown(report, args.md, args.out_dataset, args.out_evidence)
    print(f"[recover_raw_text_evidence_v1] targets={len(target)} recovered_any={report['recovered_any']}")
    print(f"[recover_raw_text_evidence_v1] wrote {args.out_evidence}")
    print(f"[recover_raw_text_evidence_v1] wrote {args.out_dataset}")


if __name__ == "__main__":
    main()
