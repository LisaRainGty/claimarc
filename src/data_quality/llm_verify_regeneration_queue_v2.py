"""LLM/VLM verification for prioritized regeneration queue v2.

This tool consumes `regeneration_queue_v2.jsonl` and produces auditable
per-pair verification records. The prompt does not include labels, split, or
sample weights. It only exposes the target attribute, raw product evidence,
candidate SRT snippets, OCR text, and optionally detail images.

Use it for small P0/P1 batches first; outputs are append-only and resumable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import config
from common import llm
from common import product_index as pidx
from common.io_utils import normalize, read_json


VALID_REL = {"supports_claim", "contradicts_claim", "insufficient", "claim_only", "evidence_only"}
VALID_SRC = {"params", "product_title", "detail_image_ocr", "detail_image_vlm", "srt", "none"}
PRODUCT_SOURCES = {"params", "product_title", "detail_image_ocr", "detail_image_vlm"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("pair_id"):
                done.add(str(obj["pair_id"]))
    return done


def clean(value: Any) -> str:
    return str(value or "").strip().strip("<>").strip()


def split_terms(*values: Any) -> list[str]:
    text = " ".join(clean(v) for v in values)
    chunks = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    terms: list[str] = []
    for ch in chunks:
        if re.fullmatch(r"[A-Za-z0-9]+", ch):
            if len(ch) >= 2:
                terms.append(ch.lower())
        elif 2 <= len(ch) <= 8:
            terms.append(ch)
        elif len(ch) > 8:
            terms.extend(ch[i:i + 2] for i in range(len(ch) - 1))
            terms.extend(ch[i:i + 3] for i in range(len(ch) - 2))
    block = {"属性", "商品", "产品", "是否", "情况", "相关", "信息"}
    return [t for t in dict.fromkeys(terms) if t and t not in block]


def text_score(text: str, terms: list[str]) -> int:
    nt = normalize(text)
    score = 0
    for term in terms:
        if normalize(term) in nt:
            score += 2 if len(term) >= 3 else 1
    return score


def product_context(row: dict[str, Any], bundles: dict[str, pidx.ProductBundle]) -> str:
    pid = str(row.get("product_id", ""))
    b = bundles.get(pid)
    parts = []
    if b:
        if b.title:
            parts.append(f"[product_title] {b.title}")
        for key, val in list((b.params or {}).items())[:80]:
            parts.append(f"[params] {key}: {val}")
    return "\n".join(parts)[:2600]


def load_ocr(pid: str) -> dict[str, str]:
    obj = read_json(config.STAGE_C / "ocr_text" / f"{pid}.json", default={}) or {}
    return {str(k): str(v or "") for k, v in obj.items()}


def ocr_context(row: dict[str, Any], max_lines: int = 16) -> str:
    pid = str(row.get("product_id", ""))
    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"))
    candidates = []
    for img, text in load_ocr(pid).items():
        if Path(img).name.startswith("._"):
            continue
        for line in re.split(r"[\n\r]+", text or ""):
            line = line.strip()
            if not line:
                continue
            score = text_score(line, terms)
            if score > 0:
                candidates.append((score, img, line[:180]))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    rows = []
    seen = set()
    for _, img, line in candidates:
        key = normalize(line)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"[detail_image_ocr] {Path(img).name}: {line}")
        if len(rows) >= max_lines:
            break
    return "\n".join(rows)


def read_text_file(path_str: str) -> str:
    path = pidx.resolve(path_str)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def srt_context(row: dict[str, Any], max_lines: int = 18) -> str:
    existing = row.get("claim_segments") or []
    if existing:
        return "\n".join(
            f"[srt] {seg.get('clip_id', '')} {seg.get('start_ts', '')}-{seg.get('end_ts', '')}: {seg.get('text', '')}"
            for seg in existing[:max_lines]
        )[:2600]

    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("risk_comment_example"), row.get("product_title"))
    candidates = []
    for srt in row.get("srt_files") or []:
        text = read_text_file(srt)
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            score = text_score(line, terms)
            if score <= 0:
                continue
            lo, hi = max(0, i - 1), min(len(lines), i + 2)
            window = " ".join(lines[lo:hi])
            candidates.append((score, srt, window[:240]))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    rows = []
    seen = set()
    for _, srt, text in candidates:
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"[srt] {Path(srt).name}: {text}")
        if len(rows) >= max_lines:
            break
    return "\n".join(rows)[:2600]


def choose_images(row: dict[str, Any], max_images: int) -> list[str]:
    targets = set(row.get("target_sources") or [])
    if not ({"detail_image_vlm", "detail_image_ocr"} & targets):
        return []
    terms = split_terms(row.get("attribute_name"), row.get("attribute_id"), row.get("product_title"), row.get("claim_preview"))
    ocr_by_name: dict[str, int] = {}
    for img, text in load_ocr(str(row.get("product_id", ""))).items():
        ocr_by_name[Path(img).name] = text_score(text, terms)
    candidates = []
    for rel in row.get("detail_images") or []:
        path = pidx.resolve(rel)
        if path.exists() and not path.name.startswith("._"):
            candidates.append((ocr_by_name.get(path.name, 0), str(path)))
    candidates.sort(key=lambda x: (-x[0], Path(x[1]).name))
    out = []
    seen = set()
    for _, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= max_images:
            break
    return out


def make_prompt(row: dict[str, Any], product_ctx: str, ocr_ctx: str, srt_ctx: str, image_paths: list[str]) -> str:
    image_list = "\n".join(f"{i + 1}. {Path(p).name}" for i, p in enumerate(image_paths)) or "无"
    return f"""你是直播电商虚假宣传数据重生成的证据核验员。请只基于给定原始材料判断，不使用外部知识。

任务类型：{row.get("queue_type")}
商品类目：{row.get("category")}
商品标题：{row.get("product_title")}
目标属性：{row.get("attribute_name")} ({row.get("attribute_id")})
属性类型：{row.get("attribute_objectivity")} / {row.get("expected_value_type")}
风险评论样例：{row.get("risk_comment_example") or "无"}
目标来源优先级：{", ".join(row.get("target_sources") or [])}

主播/SRT候选片段：
{srt_ctx or "【无SRT候选】"}

商品标题与参数：
{product_ctx or "【无参数】"}

OCR候选：
{ocr_ctx or "【无OCR候选】"}

输入图片顺序：
{image_list}

图片核验要求：
- 若输入了详情图，请优先寻找图中文字、规格表、尺码表、材质/成分说明、结构细节。
- 对数值/材质/尺码/厚度等属性，图片中文字也可作为商品证据。
- 不要把主播/SRT 原话当作商品证据。

请严格输出 JSON：
{{
  "claim_found": true/false,
  "claim_text": "可直接回溯的主播原话，不超过80字；无则空",
  "evidence_found": true/false,
  "source_type": "params|product_title|detail_image_ocr|detail_image_vlm|srt|none",
  "raw_text": "证据原文或图片观察，不超过100字；无则空",
  "normalized_value": "规格化属性值，如90绒/20000mAh/304不锈钢/有内兜；无则空",
  "path_or_clip_id": "证据文件名/图片名/clip名；无则空",
  "timestamp_or_image": "时间戳或图片序号；无则空",
  "relation_to_claim": "supports_claim|contradicts_claim|insufficient|claim_only|evidence_only",
  "confidence": "high|medium|low",
  "reject_reason": "若不能入训练，说明原因；否则空",
  "curation_action": "keep_clean|keep_risk|keep_silver|rerun_more_evidence|drop"
}}"""


def clean_result(obj: dict[str, Any], row: dict[str, Any], model: str) -> dict[str, Any]:
    rel = str(obj.get("relation_to_claim", "insufficient"))
    if rel not in VALID_REL:
        rel = "insufficient"
    src = str(obj.get("source_type", "none"))
    if src not in VALID_SRC:
        src = "none"
    claim_found = bool(obj.get("claim_found", False))
    evidence_found = bool(obj.get("evidence_found", False))
    action = str(obj.get("curation_action", ""))[:40]

    # SRT proves the claim side only. Product evidence must come from title,
    # params, OCR, or VLM grounded in product detail images.
    if src == "srt" and row.get("queue_type") == "direct_product_source0":
        evidence_found = False
        if rel in {"supports_claim", "contradicts_claim"}:
            rel = "claim_only"
        src = "none"
    if rel in {"claim_only", "evidence_only", "insufficient"}:
        action = "rerun_more_evidence" if claim_found or evidence_found else "drop"
    elif rel == "supports_claim" and src in PRODUCT_SOURCES and claim_found and evidence_found:
        action = "keep_clean"
    elif rel == "contradicts_claim" and src in PRODUCT_SOURCES and claim_found and evidence_found:
        action = "keep_risk"
    else:
        action = "rerun_more_evidence"

    return {
        "pair_id": row.get("pair_id"),
        "queue_type": row.get("queue_type"),
        "priority": row.get("priority"),
        "product_id": row.get("product_id"),
        "category": row.get("category"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "attribute_objectivity": row.get("attribute_objectivity"),
        "expected_value_type": row.get("expected_value_type"),
        "claim_found": claim_found,
        "claim_text": str(obj.get("claim_text", ""))[:160],
        "evidence_found": evidence_found,
        "source_type": src,
        "raw_text": str(obj.get("raw_text", ""))[:220],
        "normalized_value": str(obj.get("normalized_value", ""))[:120],
        "path_or_clip_id": str(obj.get("path_or_clip_id", ""))[:160],
        "timestamp_or_image": str(obj.get("timestamp_or_image", ""))[:80],
        "relation_to_claim": rel,
        "confidence": str(obj.get("confidence", ""))[:32],
        "reject_reason": str(obj.get("reject_reason", ""))[:220],
        "curation_action": action,
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/regeneration_queue_v2.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/regeneration_queue_v2_llm_verify.jsonl")
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--queue_type", default="")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--model", default="Qwen3-VL-Plus")
    ap.add_argument("--max_images", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=600)
    args = ap.parse_args()

    rows = read_jsonl(args.queue)
    priorities = {p.strip() for p in re.split(r"[, ]+", args.priority) if p.strip()}
    if priorities:
        rows = [r for r in rows if str(r.get("priority")) in priorities]
    if args.queue_type:
        rows = [r for r in rows if str(r.get("queue_type")) == args.queue_type]
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if str(r.get("pair_id")) not in done]
    bundles = pidx.build_bundles()

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        product_ctx = product_context(row, bundles)
        ocr_ctx = ocr_context(row)
        srt_ctx = srt_context(row)
        images = choose_images(row, args.max_images)
        data_urls = [u for p in images if (u := llm.encode_image(p))]
        obj = llm.chat_json(
            make_prompt(row, product_ctx, ocr_ctx, srt_ctx, images),
            system="你是严谨的电商证据核验员，只输出 JSON。",
            model=args.model,
            images=data_urls or None,
            temperature=0.0,
            namespace="regeneration_queue_v2_verify",
            max_tokens=args.max_tokens,
        )
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not an object")
        return clean_result(obj, row, args.model)

    results = llm.run_many(todo, verify, concurrency=args.concurrency, desc="queue_verify_v2")
    with out_path.open("a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[llm_verify_regeneration_queue_v2] new={len(results)} done={len(done) + len(results)} out={out_path}")


if __name__ == "__main__":
    main()
