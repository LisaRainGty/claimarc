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
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_out", required=True)
    args = ap.parse_args()

    records = read_jsonl(Path(args.dataset))
    raw_image_root = Path(args.raw_image_root)
    oof = load_oof(Path(args.oof))
    y = np.asarray(oof["y"], int)
    c = np.asarray(oof.get("c", np.ones_like(y)), float)
    if len(records) != len(y):
        raise ValueError(f"dataset/oof length mismatch: {len(records)} vs {len(y)}")
    p_m, yhat_m = get_method(oof, args.method)
    p_b, yhat_b = get_method(oof, args.baseline)

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
