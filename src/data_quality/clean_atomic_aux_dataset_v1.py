"""Build a deterministic cleaned view of atomic auxiliary datasets.

The raw atomic auxiliary pool is intentionally high-recall.  This view removes
transaction/promotion shortcuts, downgrades title-only evidence, and leaves a
small audit trail on each retained row.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl


PRICE_RE = re.compile(
    r"(?:[¥￥]\s*\d+(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s*(?:元|块|rmb|RMB))|(?:\d+块\d*)|"
    r"(?:[一二两三四五六七八九十百千万]+(?:多)?(?:块钱|块|元钱|元))"
)
TRANSACTION_RE = re.compile(
    r"(链接|下单|订单|拍下|拍[一二两三四五六七八九十\d]+|备注|库存|发货|物流|客服|售后|退货|"
    r"优惠|券|赠品|赠送|买[一二两三四五六七八九十\d]+送|加购|直播价|到手价|价格|"
    r"安装费|收费|费用|上墙|打孔|手工费|调试费)"
)
PROMO_ATTR_RE = re.compile(r"(链接|订单|物流|发货|客服|售后|价格|优惠|券|库存|备注|运费|购买)")
PRODUCT_CUE_RE = re.compile(
    r"(颜色|色|黑|白|灰|红|蓝|绿|粉|紫|黄|橙|棕|米|杏|材质|面料|棉|羊毛|羽绒|鸭绒|绒|"
    r"规格|尺寸|尺码|码|容量|净含量|分量|份量|克|斤|g|G|毫升|ml|ML|厘米|cm|寸|"
    r"包|箱|片|只|瓶|件|层|抽|大果|中果|小果|厚|薄|高|低|长|短|深|浅|"
    r"防水|保暖|清新|口气|不含|蛋白|风味|口味|香味|功效|功能|认证|3C|三C)"
)
TITLE_KEY_RE = re.compile(r"(商品标题|标题|商品名称|产品标题)")
WEAK_OCR_RE = re.compile(r"^(品牌|参数|规格|颜色|色差|材质|美味|优质|精选|详情|功能|特点|卖点|正品)$")
CONFIDENCE_BY_COVERAGE = {0: "absent", 1: "low", 2: "medium", 3: "high"}


def _claim_text(rec: dict[str, Any]) -> str:
    claim = rec.get("claim") or {}
    segs = claim.get("segments") or []
    text = " ".join(str(s.get("text") or "") for s in segs if s.get("text"))
    return text or str(claim.get("passage") or "")


def _scrub_claim_text(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    out = text or ""
    new = PRICE_RE.sub("", out)
    if new != out:
        changes.append("price_removed")
        out = new
    new = TRANSACTION_RE.sub("", out)
    if new != out:
        changes.append("transaction_terms_removed")
        out = new
    out = re.sub(r"\s+", " ", out).strip(" ,，。;；、")
    return out, changes


def _clean_claim(rec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], str]:
    attr = str(rec.get("attribute_name") or rec.get("attribute_id") or "")
    raw = _claim_text(rec)
    if PROMO_ATTR_RE.search(attr):
        return None, ["promo_attribute"], raw
    scrubbed, changes = _scrub_claim_text(raw)
    had_transaction = bool(TRANSACTION_RE.search(raw) or PRICE_RE.search(raw))
    if had_transaction and (len(scrubbed) < 4 or not PRODUCT_CUE_RE.search(scrubbed + attr)):
        return None, changes + ["transaction_only_claim"], raw
    if not scrubbed:
        return None, changes + ["empty_claim_after_scrub"], raw
    if scrubbed == raw:
        return rec, changes, raw

    out = copy.deepcopy(rec)
    claim = copy.deepcopy(out.get("claim") or {})
    claim["passage"] = scrubbed
    segs = []
    for seg in claim.get("segments") or []:
        nseg = copy.deepcopy(seg)
        st, _ = _scrub_claim_text(str(nseg.get("text") or ""))
        nseg["text"] = st or scrubbed
        segs.append(nseg)
    if segs:
        claim["segments"] = segs
    out["claim"] = claim
    return out, changes, raw


def _strong_ocr(item: dict[str, Any]) -> bool:
    text = str(item.get("raw_text") or "").strip()
    if not text:
        return False
    if len(text) <= 2:
        return False
    if WEAK_OCR_RE.match(text):
        return False
    return True


def _reweight_evidence(rec: dict[str, Any], coverage_penalty: float) -> tuple[dict[str, Any] | None, list[str]]:
    out = copy.deepcopy(rec)
    changes: list[str] = []
    params = []
    title_hints = []
    for item in out.get("evidence_params") or []:
        key = str(item.get("param_key") or "")
        if TITLE_KEY_RE.search(key):
            title_hints.append(item)
        else:
            params.append(item)
    if title_hints:
        changes.append("title_hint_not_counted")
        out["evidence_title_hints"] = title_hints
        out["evidence_params"] = params

    ocr = [it for it in (out.get("evidence_ocr") or []) if _strong_ocr(it)]
    if len(ocr) != len(out.get("evidence_ocr") or []):
        changes.append("weak_ocr_not_counted")
        out["evidence_ocr_weak"] = [
            it for it in (out.get("evidence_ocr") or []) if not _strong_ocr(it)
        ]
        out["evidence_ocr"] = ocr

    vlm = [
        it for it in (out.get("evidence_vlm") or [])
        if str(it.get("raw_quote") or it.get("raw_text") or "").strip()
    ]
    out["evidence_vlm"] = vlm
    counts = {"params": len(params), "ocr": len(ocr), "vlm": len(vlm)}
    new_cov = sum(1 for v in counts.values() if v > 0)
    old_cov = int(out.get("coverage", 0) or 0)
    out["evidence_count"] = counts
    out["coverage"] = new_cov
    out["confidence"] = CONFIDENCE_BY_COVERAGE.get(new_cov, "high")
    if new_cov != old_cov:
        changes.append(f"coverage_{old_cov}_to_{new_cov}")
        out["c"] = round(max(float(out.get("c", 0.05) or 0.05) * coverage_penalty, 0.05), 4)
    return out, changes


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = Counter()
    changed = Counter()
    examples: list[dict[str, Any]] = []

    for path in args.dataset:
        for rec in read_jsonl(path):
            rid = str(rec.get("atomic_id") or rec.get("pair_id") or "")
            if args.dedupe and rid and rid in seen:
                dropped["duplicate"] += 1
                continue
            clean, claim_changes, raw_claim = _clean_claim(rec)
            if clean is None:
                for c in claim_changes:
                    dropped[c] += 1
                if len(examples) < 80:
                    examples.append({
                        "drop": claim_changes,
                        "atomic_id": rec.get("atomic_id"),
                        "attribute_name": rec.get("attribute_name"),
                        "claim": raw_claim,
                    })
                continue
            clean, ev_changes = _reweight_evidence(clean, args.coverage_penalty)
            changes = claim_changes + ev_changes
            if int(clean.get("coverage", 0) or 0) < args.min_effective_coverage:
                dropped["low_effective_coverage"] += 1
                if len(examples) < 80:
                    examples.append({
                        "drop": ["low_effective_coverage"],
                        "atomic_id": rec.get("atomic_id"),
                        "attribute_name": rec.get("attribute_name"),
                        "claim": raw_claim,
                    })
                continue
            if changes:
                for c in changes:
                    changed[c] += 1
                clean["_quality_view"] = {
                    "name": args.view_name,
                    "changes": changes,
                    "raw_claim": raw_claim if "price_removed" in changes or "transaction_terms_removed" in changes else "",
                }
            else:
                clean["_quality_view"] = {"name": args.view_name, "changes": []}
            rows.append(clean)
            if rid:
                seen.add(rid)

    report = {
        "datasets": args.dataset,
        "out": args.out,
        "view_name": args.view_name,
        "n": len(rows),
        "dropped": dict(dropped),
        "changed": dict(changed),
        "labels": dict(Counter(str(r.get("y")) for r in rows)),
        "coverage": dict(Counter(str(r.get("coverage")) for r in rows)),
        "category": dict(Counter(str(r.get("category")) for r in rows)),
        "source": dict(Counter(str(r.get("_source", "legacy_v6clean")) for r in rows)),
        "examples": examples,
    }
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--view_name", default="clean_atomic_aux_v1")
    ap.add_argument("--min_effective_coverage", type=int, default=1)
    ap.add_argument("--coverage_penalty", type=float, default=0.75)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    rows, report = build(args)
    write_jsonl(args.out, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
