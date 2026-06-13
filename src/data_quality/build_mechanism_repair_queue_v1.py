"""Build a targeted data-repair queue from OOF mechanism failures.

The queue is meant for offline LLM/human adjudication. It focuses on rows where
the current CLAIMARC verifier appears to over-trigger on attribute mentions or
where exact product evidence/value alignment is likely wrong.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


EXACT_VALUE_TERMS = (
    "价格", "券", "优惠", "尺码", "码", "重量", "克", "斤", "容量", "毫升",
    "升", "长度", "宽", "高", "厚", "尺寸", "材质", "面料", "成分", "牛皮",
    "羊毛", "棉", "聚酯", "羽绒", "含量", "颜色", "色", "品相", "库存",
)

HIGH_RISK_CATEGORIES = {
    "digital_and_electronics",
    "jewelry_and_collectibles",
    "shoes_and_bags",
    "sports_and_outdoor",
}


def load_oof(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def get_method(oof: dict[str, Any], method: str) -> tuple[np.ndarray, np.ndarray]:
    for p_key, yhat_key in (
        (f"p__{method}", f"yhat__{method}"),
        (f"{method}__p", f"{method}__yhat"),
    ):
        if p_key in oof and yhat_key in oof:
            return np.asarray(oof[p_key], float), np.asarray(oof[yhat_key], int)
    raise KeyError(f"method {method!r} not found")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def normalize_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    return str(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.generic):
            value = value.item()
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, np.generic):
            value = value.item()
        return int(float(value))
    except Exception:
        return default


def compare_dataset_oof_alignment(
    records: list[dict[str, Any]],
    oof: dict[str, Any],
    *,
    c_tol: float = 1e-6,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Check that saved OOF predictions still belong to this dataset view."""
    n_dataset = len(records)
    y_oof = np.asarray(oof["y"], int) if "y" in oof else np.asarray([], int)
    c_oof = np.asarray(oof["c"], float) if "c" in oof else np.asarray([], float)
    n_oof = int(len(y_oof)) if len(y_oof) else int(len(next(iter(oof.values()))))
    n = min(n_dataset, n_oof)
    dataset_ids = [normalize_scalar(row.get("pair_id")) for row in records]
    oof_ids = [normalize_scalar(v) for v in oof["pair_id"]] if "pair_id" in oof else []
    dataset_seen: dict[str, int] = {}
    oof_seen: dict[str, int] = {}
    duplicate_dataset_ids: list[str] = []
    duplicate_oof_ids: list[str] = []
    for i, pid in enumerate(dataset_ids):
        if not pid:
            continue
        if pid in dataset_seen:
            duplicate_dataset_ids.append(pid)
        dataset_seen[pid] = i
    for i, pid in enumerate(oof_ids):
        if not pid:
            continue
        if pid in oof_seen:
            duplicate_oof_ids.append(pid)
        oof_seen[pid] = i

    report: dict[str, Any] = {
        "n_dataset": n_dataset,
        "n_oof": n_oof,
        "length_match": n_dataset == n_oof,
        "missing_oof_keys": sorted(k for k in ("pair_id", "attribute_id", "y", "c") if k not in oof),
        "duplicate_dataset_pair_ids": duplicate_dataset_ids[:sample_limit],
        "duplicate_oof_pair_ids": duplicate_oof_ids[:sample_limit],
        "dataset_pair_ids_missing_in_oof_count": 0,
        "oof_pair_ids_missing_in_dataset_count": 0,
        "row_order_pair_id_mismatch_count": 0,
        "row_order_attribute_id_mismatch_count": 0,
        "row_order_category_mismatch_count": 0,
        "by_pair_attribute_id_mismatch_count": 0,
        "by_pair_y_mismatch_count": 0,
        "by_pair_c_mismatch_count": 0,
        "by_pair_category_mismatch_count": 0,
        "requires_pair_id_reindexing": False,
        "samples": {
            "row_order_pair_id": [],
            "row_order_attribute_id": [],
            "row_order_category": [],
            "dataset_missing_in_oof": [],
            "oof_missing_in_dataset": [],
            "by_pair_attribute_id": [],
            "by_pair_y": [],
            "by_pair_c": [],
            "by_pair_category": [],
        },
    }

    def add_sample(kind: str, sample: dict[str, Any]) -> None:
        if len(report["samples"][kind]) < sample_limit:
            report["samples"][kind].append(sample)

    for i in range(n):
        row = records[i]
        if "pair_id" in oof:
            row_v = normalize_scalar(row.get("pair_id"))
            oof_v = normalize_scalar(oof["pair_id"][i])
            if row_v != oof_v:
                report["row_order_pair_id_mismatch_count"] += 1
                report["requires_pair_id_reindexing"] = True
                add_sample("row_order_pair_id", {"row": i, "dataset": row_v, "oof": oof_v})
        if "attribute_id" in oof:
            row_v = normalize_scalar(row.get("attribute_id"))
            oof_v = normalize_scalar(oof["attribute_id"][i])
            if row_v != oof_v:
                report["row_order_attribute_id_mismatch_count"] += 1
                add_sample("row_order_attribute_id", {"row": i, "dataset": row_v, "oof": oof_v})
        if "category" in oof:
            row_v = normalize_scalar(row.get("category"))
            oof_v = normalize_scalar(oof["category"][i])
            if row_v != oof_v:
                report["row_order_category_mismatch_count"] += 1
                add_sample("row_order_category", {"row": i, "pair_id": row.get("pair_id"), "dataset": row_v, "oof": oof_v})

    if "pair_id" in oof:
        dataset_missing = sorted(set(dataset_seen) - set(oof_seen))
        oof_missing = sorted(set(oof_seen) - set(dataset_seen))
        report["dataset_pair_ids_missing_in_oof_count"] = len(dataset_missing)
        report["oof_pair_ids_missing_in_dataset_count"] = len(oof_missing)
        for pid in dataset_missing[:sample_limit]:
            add_sample("dataset_missing_in_oof", {"pair_id": pid})
        for pid in oof_missing[:sample_limit]:
            add_sample("oof_missing_in_dataset", {"pair_id": pid})

        if not duplicate_dataset_ids and not duplicate_oof_ids:
            for row in records:
                pid = normalize_scalar(row.get("pair_id"))
                if not pid or pid not in oof_seen:
                    continue
                j = oof_seen[pid]
                if "attribute_id" in oof:
                    row_v = normalize_scalar(row.get("attribute_id"))
                    oof_v = normalize_scalar(oof["attribute_id"][j])
                    if row_v != oof_v:
                        report["by_pair_attribute_id_mismatch_count"] += 1
                        add_sample("by_pair_attribute_id", {"pair_id": pid, "dataset": row_v, "oof": oof_v})
                if "category" in oof:
                    row_v = normalize_scalar(row.get("category"))
                    oof_v = normalize_scalar(oof["category"][j])
                    if row_v != oof_v:
                        report["by_pair_category_mismatch_count"] += 1
                        add_sample("by_pair_category", {"pair_id": pid, "dataset": row_v, "oof": oof_v})
                if "y" in oof:
                    row_y = safe_int(row.get("y", 0))
                    oof_y = safe_int(y_oof[j])
                    if row_y != oof_y:
                        report["by_pair_y_mismatch_count"] += 1
                        add_sample("by_pair_y", {"pair_id": pid, "dataset": row_y, "oof": oof_y})
                if "c" in oof:
                    row_c = safe_float(row.get("c", 0.0))
                    oof_c = safe_float(c_oof[j])
                    if abs(row_c - oof_c) > c_tol:
                        report["by_pair_c_mismatch_count"] += 1
                        add_sample("by_pair_c", {"pair_id": pid, "dataset": round(row_c, 6), "oof": round(oof_c, 6)})

    fatal_counts = (
        report["dataset_pair_ids_missing_in_oof_count"],
        report["oof_pair_ids_missing_in_dataset_count"],
        report["by_pair_attribute_id_mismatch_count"],
        report["by_pair_y_mismatch_count"],
        report["by_pair_c_mismatch_count"],
    )
    report["status"] = "pass" if (
        report["length_match"]
        and not any(fatal_counts)
        and not report["missing_oof_keys"]
        and not duplicate_dataset_ids
        and not duplicate_oof_ids
    ) else "fail"
    return report


def oof_order_for_records(records: list[dict[str, Any]], oof: dict[str, Any]) -> np.ndarray:
    if "pair_id" not in oof:
        return np.arange(len(records), dtype=int)
    index: dict[str, int] = {}
    for i, pid in enumerate(oof["pair_id"]):
        key = normalize_scalar(pid)
        if key in index:
            raise ValueError(f"duplicate pair_id in OOF file: {key}")
        index[key] = i
    order = []
    for row in records:
        key = normalize_scalar(row.get("pair_id"))
        if key not in index:
            raise ValueError(f"dataset pair_id missing from OOF file: {key}")
        order.append(index[key])
    return np.asarray(order, dtype=int)


def evidence_text(record: dict[str, Any], max_len: int = 1200) -> dict[str, str]:
    def join_items(key: str, field: str) -> str:
        vals = [
            str(item.get(field, "") or "").strip()
            for item in record.get(key, []) or []
            if str(item.get(field, "") or "").strip()
        ]
        return " | ".join(vals)[:max_len]

    return {
        "params": join_items("evidence_params", "raw_text"),
        "ocr": join_items("evidence_ocr", "raw_text"),
        "vlm": join_items("evidence_vlm", "raw_quote"),
    }


def raw_image_paths(raw_image_root: Path, product_id: str, limit: int = 40) -> list[str]:
    if not product_id:
        return []
    root = raw_image_root / product_id
    if not root.exists():
        return []
    paths = [
        str(p)
        for p in sorted(root.iterdir(), key=lambda x: x.name)
        if p.is_file() and not p.name.startswith("._")
    ]
    return paths[:limit]


def claim_text(record: dict[str, Any], max_len: int = 1200) -> str:
    claim = record.get("claim", {}) or {}
    return str(claim.get("passage", "") or "")[:max_len]


def source_count(record: dict[str, Any]) -> int:
    counts = record.get("evidence_count", {}) or {}
    return sum(int(counts.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))


def evidence_combo(record: dict[str, Any]) -> str:
    parts = []
    if record.get("evidence_params"):
        parts.append("P")
    if record.get("evidence_ocr"):
        parts.append("O")
    if record.get("evidence_vlm"):
        parts.append("V")
    return "".join(parts) if parts else "none"


def exact_value_hint(record: dict[str, Any]) -> bool:
    text = " ".join([
        str(record.get("attribute_name", "") or ""),
        claim_text(record, 2000),
        " ".join(evidence_text(record, 2000).values()),
    ])
    return any(term in text for term in EXACT_VALUE_TERMS) or bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:元|块|米|cm|厘米|mm|克|g|kg|斤|ml|毫升|L|升|码|%)", text)
    )


def priority_score(record, y, p_m, yhat_m, p_b, yhat_b) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    category = str(record.get("category", "") or "")
    combo = evidence_combo(record)
    sc = source_count(record)
    attr = str(record.get("attribute_name", "") or "")
    high_p_false_pos = y == 0 and yhat_m == 1 and p_m >= 0.75
    bge_correct_claimarc_wrong = yhat_m != y and yhat_b == y
    both_wrong = yhat_m != y and yhat_b != y

    if high_p_false_pos:
        score += 5
        reasons.append("claimarc_high_conf_false_positive")
    if bge_correct_claimarc_wrong:
        score += 4
        reasons.append("bge_correct_claimarc_wrong")
    if both_wrong and y == 0 and p_m >= 0.75:
        score += 3
        reasons.append("both_wrong_high_conf_negative")
    if category in HIGH_RISK_CATEGORIES:
        score += 2
        reasons.append(f"high_risk_category:{category}")
    if combo in {"P", "O", "PO"}:
        score += 1
        reasons.append(f"exact_source_combo:{combo}")
    if exact_value_hint(record):
        score += 2
        reasons.append("exact_value_or_material_hint")
    if sc == 0:
        score -= 2
        reasons.append("no_product_evidence")
    if attr in {"<价格>", "价格"}:
        score += 1
        reasons.append("price_or_coupon")
    return score, reasons


def build_item(i, record, y, c, p_m, yhat_m, p_b, yhat_b, score, reasons,
               raw_image_root: Path):
    product_id = str(record.get("product_id", "") or "")
    return {
        "row": int(i),
        "priority_score": int(score),
        "reasons": reasons,
        "pair_id": record.get("pair_id", ""),
        "product_id": product_id,
        "room_id": record.get("room_id", ""),
        "category": record.get("category", ""),
        "subcategory": record.get("subcategory", ""),
        "attribute_id": record.get("attribute_id", ""),
        "attribute_name": record.get("attribute_name", ""),
        "y_current": int(y),
        "c_current": round(float(c), 4),
        "claimarc_p": round(float(p_m), 4),
        "claimarc_yhat": int(yhat_m),
        "baseline_p": round(float(p_b), 4),
        "baseline_yhat": int(yhat_b),
        "source_count": source_count(record),
        "evidence_combo": evidence_combo(record),
        "confidence": record.get("confidence", ""),
        "claim": claim_text(record),
        "evidence": evidence_text(record),
        "raw_image_paths": raw_image_paths(raw_image_root, product_id),
        "label_audit": record.get("label_audit", {}),
        "llm_repair_questions": [
            "Is the live-stream claim span a concrete product-attribute claim rather than generic persuasion?",
            "Which exact product-detail evidence span supports, refutes, or fails to cover the claim?",
            "For numeric/material/size/price/color claims, are the values exactly compatible after OCR/ASR noise normalization?",
            "Does consumer evidence indicate a perceived mismatch, disappointment, or expectation gap tied to this attribute?",
            "Should the row be kept, relabeled, confidence-adjusted, regenerated with better evidence, or dropped?",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--method", default="CLAIMARC_pcls")
    ap.add_argument("--baseline", default="bge_lr")
    ap.add_argument("--min_priority", type=int, default=5)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--raw_image_root", default="data/raw/product_images")
    ap.add_argument("--allow_oof_mismatch", action="store_true",
                    help="Continue even if the OOF file no longer aligns with the dataset.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_out", required=True)
    args = ap.parse_args()

    records = read_jsonl(Path(args.dataset))
    raw_image_root = Path(args.raw_image_root)
    oof = load_oof(Path(args.oof))
    y_raw = np.asarray(oof["y"], int)
    c_raw = np.asarray(oof.get("c", np.ones_like(y_raw)), float)
    if len(records) != len(y_raw):
        raise ValueError(f"dataset/oof length mismatch: {len(records)} vs {len(y_raw)}")
    alignment = compare_dataset_oof_alignment(records, oof)
    if alignment["status"] != "pass" and not args.allow_oof_mismatch:
        failed_stats = {
            "status": "fail",
            "fail_reasons": ["dataset_oof_alignment_mismatch"],
            "dataset": args.dataset,
            "oof": args.oof,
            "method": args.method,
            "baseline": args.baseline,
            "alignment": alignment,
        }
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(
            json.dumps(failed_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(failed_stats, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(2)
    order = oof_order_for_records(records, oof)
    y = y_raw[order]
    c = c_raw[order]
    p_m_raw, yhat_m_raw = get_method(oof, args.method)
    p_b_raw, yhat_b_raw = get_method(oof, args.baseline)
    p_m, yhat_m = p_m_raw[order], yhat_m_raw[order]
    p_b, yhat_b = p_b_raw[order], yhat_b_raw[order]

    items = []
    for i, record in enumerate(records):
        score, reasons = priority_score(
            record, int(y[i]), float(p_m[i]), int(yhat_m[i]),
            float(p_b[i]), int(yhat_b[i]))
        if score >= args.min_priority:
            items.append(build_item(
                i, record, int(y[i]), float(c[i]), float(p_m[i]), int(yhat_m[i]),
                float(p_b[i]), int(yhat_b[i]), score, reasons, raw_image_root))

    items.sort(key=lambda x: (
        -x["priority_score"],
        -abs(x["claimarc_p"] - x["y_current"]),
        str(x["pair_id"]),
    ))
    if args.limit > 0:
        items = items[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "dataset": args.dataset,
        "oof": args.oof,
        "method": args.method,
        "baseline": args.baseline,
        "min_priority": args.min_priority,
        "limit": args.limit,
        "raw_image_root": args.raw_image_root,
        "n_queue": len(items),
        "alignment": alignment,
        "by_reason": {},
        "by_category": {},
        "by_combo": {},
    }
    for item in items:
        for reason in item["reasons"]:
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        stats["by_category"][item["category"]] = stats["by_category"].get(item["category"], 0) + 1
        stats["by_combo"][item["evidence_combo"]] = stats["by_combo"].get(item["evidence_combo"], 0) + 1
    for key in ("by_reason", "by_category", "by_combo"):
        stats[key] = dict(sorted(stats[key].items(), key=lambda kv: (-kv[1], kv[0])))
    json.dump(stats, open(args.stats_out, "w"), ensure_ascii=False, indent=2)
    print(f"[repair_queue] n={len(items)} -> {out_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
