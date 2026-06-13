"""Deterministic integrity audit for raw-to-training CLAIMARC artifacts.

This audit is intentionally model-free. It checks whether generated Stage A/B/C
artifacts obey the documented constraints and highlights likely repair sets for
LLM reprocessing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common import srt as S


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def norm_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()【】\\[\\]{}<>《》]", "", text)
    return text


def source_count(rec: dict[str, Any]) -> int:
    ev = rec.get("evidence_count") or {}
    if isinstance(ev, dict):
        return sum(int(ev.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))
    return int(ev or 0)


def has_claim(rec: dict[str, Any]) -> bool:
    claim = rec.get("claim") or {}
    return bool(claim.get("has_claim_srt") and (claim.get("segments") or claim.get("passage")))


def localize_path(path: Any, root: Path) -> Path | None:
    raw = str(path or "")
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    marker = "data/raw/"
    if marker in raw:
        cand = root / raw[raw.index(marker):]
        if cand.exists():
            return cand
    cand = root / raw
    if cand.exists():
        return cand
    return None


def product_maps(product_index: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    obj = read_json(product_index, default={}) or {}
    products = {str(k): v for k, v in (obj.get("products") or {}).items() if isinstance(v, dict)}
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obj.get("clips") or []:
        clips[str(row.get("product_id", ""))].append(row)
    return products, clips


def load_cas(stage_a: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for path in stage_a.glob("CAS+_*.json"):
        if path.name.startswith("._"):
            continue
        cat = path.stem.replace("CAS+_", "")
        attrs = read_json(path, default={"attributes": []}).get("attributes", [])
        out[cat] = {str(a.get("attribute_id")): a for a in attrs if a.get("attribute_id")}
    return out


def alias_hit(attribute_id: str, meta: dict[str, Any], params: dict[str, Any]) -> list[dict[str, str]]:
    patterns = set()
    for s in [meta.get("canonical_name", "")] + list(meta.get("aliases", []) or []):
        n = norm_text(s)
        if n:
            patterns.add(n)
    hits = []
    for k, v in (params or {}).items():
        nk = norm_text(k)
        if nk and (nk in patterns or any(p and (p in nk or nk in p) for p in patterns)):
            hits.append({"attribute_id": attribute_id, "param_key": str(k), "raw_text": str(v)})
    return hits


def audit_stage_a(stage_a: Path) -> dict[str, Any]:
    resolved = read_jsonl(stage_a / "resolved_aspects.jsonl")
    cas = load_cas(stage_a)
    cas_dups = {}
    eval_kw = getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])
    leak_attrs = []
    for cat, attrs in cas.items():
        ids = [aid for aid in attrs]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            cas_dups[cat] = dup
        for aid, meta in attrs.items():
            name = str(meta.get("canonical_name", ""))
            if any(k in name for k in eval_kw):
                leak_attrs.append({"category": cat, "attribute_id": aid, "canonical_name": name})
    return {
        "resolved_aspects": len(resolved),
        "products": len({str(r.get("product_id", "")) for r in resolved}),
        "unique_product_attribute_pairs": len({(str(r.get("product_id", "")), str(r.get("attribute_id", ""))) for r in resolved}),
        "polarity": dict(Counter(str(r.get("polarity", "")) for r in resolved)),
        "type": dict(Counter(str(r.get("type", "")) for r in resolved)),
        "free_remaining": sum(1 for r in resolved if str(r.get("attribute_id", "")).startswith("FREE::")),
        "attribute_id_empty": sum(1 for r in resolved if not r.get("attribute_id")),
        "top_attributes": Counter(str(r.get("attribute_id", "")) for r in resolved).most_common(30),
        "cas_categories": len(cas),
        "cas_duplicate_ids": cas_dups,
        "eval_leakage_attrs_in_cas_plus": leak_attrs[:100],
        "eval_leakage_attr_count": len(leak_attrs),
    }


def audit_stage_b(stage_b: Path, product_index: Path, root: Path) -> dict[str, Any]:
    acmt = read_json(stage_b / "acmt.json", default={}) or {}
    products, clips = product_maps(product_index)
    claim_dir = stage_b / "claim_list"
    files = [p for p in claim_dir.glob("*.jsonl") if not p.name.startswith("._")]
    rows: list[dict[str, Any]] = []
    empty_files = 0
    claim_attr_violations = []
    claim_text_misses = []
    srt_missing = 0
    for path in files:
        pid = path.stem
        part = read_jsonl(path)
        if not part:
            empty_files += 1
        rows.extend(part)
        valid = set((acmt.get(pid) or {}).keys())
        for obj in part:
            aid = str(obj.get("attribute_id", ""))
            if valid and aid not in valid:
                claim_attr_violations.append({
                    "product_id": pid,
                    "attribute_id": aid,
                    "claim_id": obj.get("claim_id"),
                    "claim_text": obj.get("claim_text", "")[:120],
                })
            srt = localize_path(obj.get("srt_path"), root)
            if srt is None:
                srt_missing += 1
                continue
            # Validate against parsed cue text rather than the raw .srt file.
            # Many valid claims span adjacent cues; raw timestamp/index lines can
            # otherwise create false misses after normalization.
            concat_text = S.concat_product_srt([srt]).text
            if norm_text(obj.get("claim_text")) and norm_text(obj.get("claim_text")) not in norm_text(concat_text):
                claim_text_misses.append({
                    "product_id": pid,
                    "claim_id": obj.get("claim_id"),
                    "srt_path": str(obj.get("srt_path")),
                    "claim_text": obj.get("claim_text", "")[:160],
                })
    pair_rows = read_jsonl(stage_b / "pair_records.jsonl")
    claim_keys = {(str(r.get("claim_id", "")).split("_", 1)[0], str(r.get("attribute_id", ""))) for r in rows}
    pair_claim_true_no_claimlist = []
    pair_claimlist_not_true = []
    pair_keys_true = set()
    for r in pair_rows:
        pid, aid = str(r.get("product_id", "")), str(r.get("attribute_id", ""))
        key = (pid, aid)
        if has_claim(r):
            pair_keys_true.add(key)
            if key not in claim_keys:
                pair_claim_true_no_claimlist.append({"pair_id": r.get("pair_id"), "product_id": pid, "attribute_id": aid})
        elif key in claim_keys:
            pair_claimlist_not_true.append({"pair_id": r.get("pair_id"), "product_id": pid, "attribute_id": aid})
    no_srt_products = sum(1 for pid in acmt if not clips.get(str(pid)))
    return {
        "acmt_products": len(acmt),
        "acmt_candidate_pairs": sum(len(v) for v in acmt.values()),
        "claim_files": len(files),
        "claim_files_empty": empty_files,
        "claim_rows": len(rows),
        "products_with_claim_rows": len({str(r.get("claim_id", "")).split("_", 1)[0] for r in rows}),
        "claim_attr_violations": len(claim_attr_violations),
        "claim_attr_violation_examples": claim_attr_violations[:30],
        "srt_path_missing": srt_missing,
        "claim_text_not_found_in_srt": len(claim_text_misses),
        "claim_text_miss_examples": claim_text_misses[:30],
        "pair_records": len(pair_rows),
        "pair_claimful": sum(1 for r in pair_rows if has_claim(r)),
        "pairs_with_aligned_negative": sum(1 for r in pair_rows if int((r.get("stats") or {}).get("N_aligned_neg", 0) or 0) > 0),
        "pair_claim_true_no_claimlist": len(pair_claim_true_no_claimlist),
        "pair_claim_true_no_claimlist_examples": pair_claim_true_no_claimlist[:30],
        "pair_claimlist_not_true": len(pair_claimlist_not_true),
        "pair_claimlist_not_true_examples": pair_claimlist_not_true[:30],
        "products_in_acmt_without_index_product": sum(1 for pid in acmt if str(pid) not in products),
        "products_in_acmt_without_srt_clip": no_srt_products,
    }


def audit_stage_c(stage_a: Path, stage_b: Path, stage_c: Path, product_index: Path) -> dict[str, Any]:
    products, _ = product_maps(product_index)
    acmt = read_json(stage_b / "acmt.json", default={}) or {}
    cas = load_cas(stage_a)
    fact_rows = read_jsonl(stage_c / "fact_records.jsonl")
    coverage = Counter(int(r.get("coverage", 0) or 0) for r in fact_rows)
    confidence = Counter(str(r.get("confidence", "")) for r in fact_rows)
    source0_with_params = 0
    params_obvious_miss = []
    bad_param_key = []
    for r in fact_rows:
        pid = str(r.get("product_id", ""))
        aid = str(r.get("attribute_id", ""))
        product = products.get(pid) or {}
        params = product.get("产品参数") or {}
        if source_count(r) == 0 and params:
            source0_with_params += 1
        meta = (cas.get(str(r.get("category", ""))) or {}).get(aid) or (acmt.get(pid) or {}).get(aid) or {}
        if not (r.get("evidence_params") or []) and params:
            hits = alias_hit(aid, meta, params)
            if hits:
                params_obvious_miss.append({
                    "product_id": pid,
                    "attribute_id": aid,
                    "attribute_name": meta.get("canonical_name", aid),
                    "hits": hits[:5],
                })
        for item in r.get("evidence_params") or []:
            key = str(item.get("param_key", ""))
            if key.startswith("__"):
                continue
            if key and key not in params:
                bad_param_key.append({"product_id": pid, "attribute_id": aid, "param_key": key})
    return {
        "fact_records": len(fact_rows),
        "coverage": dict(sorted(coverage.items())),
        "confidence": dict(confidence),
        "source0": sum(1 for r in fact_rows if source_count(r) == 0),
        "source0_with_nonempty_product_params": source0_with_params,
        "params_obvious_alias_miss": len(params_obvious_miss),
        "params_obvious_alias_miss_examples": params_obvious_miss[:50],
        "evidence_param_key_not_in_product_params": len(bad_param_key),
        "evidence_param_key_not_in_product_params_examples": bad_param_key[:30],
        "by_category": {
            cat: {
                "n": len(vals),
                "source0": sum(1 for r in vals if source_count(r) == 0),
                "params_miss_examples": sum(1 for ex in params_obvious_miss if ex["product_id"] in {str(v.get("product_id", "")) for v in vals}),
            }
            for cat, vals in sorted(_group_by(fact_rows, "category").items())
        },
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, ""))].append(r)
    return out


def audit_final(final_dataset: Path, adjudicated: Path | None = None) -> dict[str, Any]:
    rows = read_jsonl(final_dataset)
    claimful = [r for r in rows if has_claim(r)]
    out: dict[str, Any] = {
        "records": len(rows),
        "labels": dict(Counter(int(r.get("y", 0)) for r in rows)),
        "claimful": len(claimful),
        "claimful_labels": dict(Counter(int(r.get("y", 0)) for r in claimful)),
        "source0": sum(1 for r in rows if source_count(r) == 0),
        "claimful_source0": sum(1 for r in claimful if source_count(r) == 0),
        "no_claim_y1": sum(1 for r in rows if not has_claim(r) and int(r.get("y", 0)) == 1),
        "split": dict(Counter(str(r.get("split", "")) for r in rows)),
    }
    if adjudicated and adjudicated.exists():
        adj_rows = read_jsonl(adjudicated)
        out["adjudicated_records"] = len(adj_rows)
        out["adjudicated_labels"] = dict(Counter(int(r.get("y", 0)) for r in adj_rows))
        out["adjudicated_decisions"] = dict(Counter(str(r.get("_adjudication_decision", "")) for r in adj_rows))
        out["adjudicated_source0"] = sum(1 for r in adj_rows if source_count(r) == 0)
    return out


def write_markdown(report: dict[str, Any], out: Path) -> None:
    lines = ["# CLAIMARC Raw Pipeline Integrity Audit", ""]
    for section, data in report.items():
        lines.append(f"## {section}")
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"- `{k}`: `{len(v)} items`")
                for item in v[:10]:
                    lines.append(f"  - `{item}`")
            else:
                lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(config.ROOT))
    ap.add_argument("--product_index", default=str(config.INDEX / "product_index.json"))
    ap.add_argument("--stage_a", default=str(config.STAGE_A))
    ap.add_argument("--stage_b", default=str(config.STAGE_B))
    ap.add_argument("--stage_c", default=str(config.STAGE_C))
    ap.add_argument("--final_dataset", default=str(config.FINAL / "dataset.jsonl"))
    ap.add_argument("--adjudicated", default=str(config.FINAL / "dataset_hq_broad_enriched_adjudicated_strict_v1.jsonl"))
    ap.add_argument("--out_json", default=str(config.FINAL / "raw_pipeline_integrity_audit_20260612.json"))
    ap.add_argument("--out_md", default=str(config.ROOT / "docs" / "RAW_PIPELINE_INTEGRITY_AUDIT_20260612.md"))
    args = ap.parse_args()

    root = Path(args.root)
    report = {
        "stage_a": audit_stage_a(Path(args.stage_a)),
        "stage_b": audit_stage_b(Path(args.stage_b), Path(args.product_index), root),
        "stage_c": audit_stage_c(Path(args.stage_a), Path(args.stage_b), Path(args.stage_c), Path(args.product_index)),
        "final_dataset": audit_final(Path(args.final_dataset), Path(args.adjudicated)),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.out_md))
    print(f"[audit_raw_pipeline_integrity] wrote {out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
