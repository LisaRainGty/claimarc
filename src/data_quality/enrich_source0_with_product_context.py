"""Add low-confidence raw product context to claimful records with no evidence.

This is a conservative repair for source0 claimful pairs: instead of treating
"no extracted evidence" as substantive evidence of misleading risk, we attach
raw product-index context that was already available before labeling. The
context is marked as low-confidence PARAM evidence and fully auditable.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_quality.audit_dataset_quality import read_jsonl, source_count


def load_products(path: str | Path) -> dict[str, dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    products = obj.get("products", {}) if isinstance(obj, dict) else {}
    return {str(k): v for k, v in products.items() if isinstance(v, dict)}


def product_context(product: dict[str, Any], cap: int) -> str:
    parts: list[str] = []
    title = str(product.get("title") or product.get("商品名称") or "").strip()
    if title:
        parts.append(f"商品标题: {title}")
    shop = str(product.get("shop_name") or "").strip()
    if shop:
        parts.append(f"店铺: {shop}")
    price = product.get("price")
    if price not in (None, ""):
        parts.append(f"价格: {price}")
    params = product.get("产品参数") or {}
    if isinstance(params, dict) and params:
        kv = []
        for k, v in params.items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                kv.append(f"{kk}={vv}")
        if kv:
            parts.append("产品参数: " + "；".join(kv))
    text = "；".join(parts)
    return text[:cap]


def ensure_count(rec: dict[str, Any]) -> dict[str, int]:
    ev = rec.get("evidence_count") or {}
    out = {
        "params": int((ev.get("params", 0) if isinstance(ev, dict) else 0) or 0),
        "ocr": int((ev.get("ocr", 0) if isinstance(ev, dict) else 0) or 0),
        "vlm": int((ev.get("vlm", 0) if isinstance(ev, dict) else 0) or 0),
    }
    return out


def enrich(rec: dict[str, Any], products: dict[str, dict[str, Any]], cap: int) -> tuple[dict[str, Any], str]:
    out = dict(rec)
    if source_count(out) > 0:
        return out, "kept_sourceful"
    product = products.get(str(out.get("product_id", "")))
    if not product:
        out["_source0_enrichment"] = {"status": "missing_product_index"}
        return out, "missing_product_index"
    text = product_context(product, cap)
    if not text:
        out["_source0_enrichment"] = {"status": "empty_product_context"}
        return out, "empty_product_context"
    params = list(out.get("evidence_params") or [])
    params.append({
        "param_key": "__raw_product_context__",
        "raw_text": text,
        "confidence": "low",
        "source": "product_index",
    })
    out["evidence_params"] = params
    ev = ensure_count(out)
    ev["params"] = max(ev["params"], 1)
    out["evidence_count"] = ev
    out["coverage"] = max(float(out.get("coverage", 0.0) or 0.0), 0.25)
    if str(out.get("confidence", "")) == "absent":
        out["confidence"] = "low"
    out["_source0_enrichment"] = {
        "status": "raw_product_context",
        "context_chars": len(text),
        "note": "Low-confidence product-index context added before adjudication; no labels or reviews used.",
    }
    return out, "raw_product_context"


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_hq_broad_claimful_v1.jsonl")
    ap.add_argument("--product_index", default="data/index/product_index.json")
    ap.add_argument("--out", default="data/final/dataset_hq_broad_claimful_enriched_v1.jsonl")
    ap.add_argument("--report", default="data/final/dataset_hq_broad_claimful_enriched_v1_report.json")
    ap.add_argument("--cap", type=int, default=1200)
    args = ap.parse_args()

    products = load_products(args.product_index)
    rows = read_jsonl(args.dataset)
    out_rows: list[dict[str, Any]] = []
    status = Counter()
    before_source0 = 0
    after_source0 = 0
    for rec in rows:
        before_source0 += int(source_count(rec) == 0)
        enriched, s = enrich(rec, products, args.cap)
        after_source0 += int(source_count(enriched) == 0)
        status[s] += 1
        out_rows.append(enriched)

    report = {
        "input": args.dataset,
        "out": args.out,
        "n": len(out_rows),
        "before_source0": before_source0,
        "after_source0": after_source0,
        "status": dict(status),
    }
    write_jsonl(args.out, out_rows)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[enrich_source0_with_product_context] wrote {args.out} report={args.report}")
    print(report)


if __name__ == "__main__":
    main()
