"""Pair-targeted claim re-extraction from full raw SRT.

This is used after the proposal completion audit finds `evidence_only` rows:
product evidence exists, but the current claim side is missing or too weak.
Unlike product-level B1, each prompt is constrained to one `(product, attribute)`
pair and may include consumer trigger examples to focus the search.  The model
must return exact source substrings; local code verifies substring grounding
and maps matches back to SRT timestamps.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import llm
from common import product_index as pidx
from common import srt as S
from common.io_utils import normalize, read_jsonl, write_json


PROMPT = """你是电商直播主播 claim 抽取员。请从字幕窗口里抽取【目标属性】相关的主播原话。

硬规则：
1. 只输出 JSON 数组，不要解释。
2. claim_text 必须是字幕窗口中真实存在的连续原文子串，不能改写、概括、拼接。
3. 只抽目标属性相关的商品事实/承诺/可比较表述；不要抽链接号、库存、下单、优惠、闲聊。
4. claim 必须能和目标属性或已找到的商品证据形成同属性比较；如果只是同商品的其它属性，输出 []。
5. 如果没有可回溯的目标属性 claim，输出 []。
6. 若消费者触发评论给出线索，可用它帮助寻找同属性话术，但不能把评论当 claim。

目标属性：{attribute_name} ({attribute_id})
属性别名：{aliases}
消费者触发评论：{consumer_examples}
已找到的商品证据：{product_evidence_hint}
当前商品标题：{product_title}

字幕窗口：
{chunk}

严格 JSON 输出：
[
  {{"claim_text": "字幕原文连续子串", "claim_type": "fact|promise|comparison|implicit_expectation", "confidence": "high|medium|low"}}
]"""


def read_queue(path: str | Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(path)]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("pair_id"):
            done.add(str(obj["pair_id"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def chunk_ranges(concat: S.ConcatResult, chunk_chars: int) -> list[tuple[int, int]]:
    if chunk_chars <= 0 or len(concat.text) <= chunk_chars or not concat.spans:
        return [(0, len(concat.text))]
    out: list[tuple[int, int]] = []
    start = concat.spans[0].char_start
    end = start
    for sp in concat.spans:
        if end > start and sp.char_end - start > chunk_chars:
            out.append((start, end))
            start = sp.char_start
        end = sp.char_end
    if end > start:
        out.append((start, end))
    return out


def as_items(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        obj = obj.get("claims") or obj.get("items") or []
    if not isinstance(obj, list):
        return []
    out = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim_text") or item.get("extraction_text") or "").strip()
        if text:
            out.append({
                "claim_text": text,
                "claim_type": str(item.get("claim_type", ""))[:40],
                "confidence": str(item.get("confidence", ""))[:20],
            })
    return out


def extract_for_pair(row: dict[str, Any], chunk_chars: int, model: str, max_tokens: int) -> dict[str, Any]:
    pid = str(row.get("product_id") or "")
    bundle = pidx.build_bundles().get(pid)
    files = [str(pidx.resolve(p)) for p in (bundle.srt_files if bundle else [])]
    if not files and row.get("srt_files"):
        files = [str(pidx.resolve(p)) for p in (row.get("srt_files") or [])]
    files = [f for f in files if Path(f).exists()]
    if not files:
        return {"pair_id": pair_id(row), "product_id": pid, "claims": [], "status": "missing_srt"}
    concat = S.concat_product_srt(files)
    aliases = row.get("aliases") or []
    if not aliases:
        aliases = [row.get("attribute_name", ""), row.get("attribute_id", "")]
    consumer_examples = row.get("risk_comment_example") or "；".join(row.get("direct_consumer_claim_reference_examples") or [])
    verify = row.get("_verify_context") or {}
    product_evidence_hint = ""
    if verify:
        product_evidence_hint = (
            f"{verify.get('source_type', '')}: {verify.get('raw_text', '')} "
            f"{verify.get('normalized_value', '')}"
        ).strip()
    seen: set[tuple[str, str]] = set()
    claims = []
    for start, end in chunk_ranges(concat, chunk_chars):
        chunk = concat.text[start:end]
        prompt = PROMPT.format(
            attribute_name=row.get("attribute_name", ""),
            attribute_id=row.get("attribute_id", ""),
            aliases="、".join(str(a) for a in aliases[:20]),
            consumer_examples=consumer_examples or "无",
            product_evidence_hint=product_evidence_hint or "无",
            product_title=(bundle.title if bundle else str(row.get("product_title", "") or "")),
            chunk=chunk,
        )
        try:
            obj = llm.chat_json(
                prompt,
                model=model,
                temperature=0.0,
                namespace="pair_claim_reextract_v1",
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            claims.append({"__error__": repr(exc)[:240]})
            continue
        for item in as_items(obj):
            quote = item["claim_text"]
            local = chunk.find(quote)
            if local < 0:
                continue
            gstart = start + local
            gend = gstart + len(quote)
            spans = concat.lookup_range(gstart, gend)
            if not spans:
                continue
            first, last = spans[0], spans[-1]
            sig = (normalize(quote), first.srt_file)
            if sig in seen:
                continue
            seen.add(sig)
            claims.append({
                "claim_text": quote,
                "claim_type": item.get("claim_type"),
                "confidence": item.get("confidence"),
                "srt_file": Path(first.srt_file).name,
                "srt_path": first.srt_file,
                "start_ts": first.start_ts,
                "end_ts": last.end_ts,
                "char_start": gstart,
                "char_end": gend,
            })
    clean_claims = [c for c in claims if "__error__" not in c]
    return {
        "pair_id": pair_id(row),
        "product_id": pid,
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "queue_type": row.get("queue_type"),
        "current_y": row.get("current_y"),
        "current_c": row.get("current_c"),
        "status": "claim_found" if clean_claims else "no_claim_found",
        "claims": clean_claims,
        "errors": [c for c in claims if "__error__" in c],
    }


def error_result(row: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "pair_id": pair_id(row),
        "product_id": row.get("product_id"),
        "attribute_id": row.get("attribute_id"),
        "attribute_name": row.get("attribute_name"),
        "queue_type": row.get("queue_type"),
        "current_y": row.get("current_y"),
        "current_c": row.get("current_c"),
        "status": "error",
        "claims": [],
        "errors": [{"__error__": repr(exc)[:240]}],
    }


def process_batch(
    batch: list[dict[str, Any]],
    *,
    out: Path,
    counts: Counter,
    progress_start: int,
    total: int,
    chunk_chars: int,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> int:
    processed = 0

    def run(row: dict[str, Any]) -> dict[str, Any]:
        return extract_for_pair(row, chunk_chars, model, max_tokens)

    if concurrency <= 1:
        for row in batch:
            res = run(row)
            counts[res.get("status", "")] += 1
            append_jsonl(out, res)
            processed += 1
            done = progress_start + processed
            print(f"[pair_claim_reextract] {done}/{total} {res['pair_id']} {res['status']} n={len(res.get('claims') or [])}", flush=True)
        return processed

    ex = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {ex.submit(run, row): row for row in batch}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = error_result(row, exc)
            counts[res.get("status", "")] += 1
            append_jsonl(out, res)
            processed += 1
            done = progress_start + processed
            print(f"[pair_claim_reextract] {done}/{total} {res['pair_id']} {res['status']} n={len(res.get('claims') or [])}", flush=True)
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        ex.shutdown(wait=True)
    return processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/proposal_llm_completion_queue_v1_20260613.jsonl")
    ap.add_argument("--verify", default="",
                    help="Optional verification file; when set, only rows with relation_to_claim=evidence_only are processed.")
    ap.add_argument("--out", default="data/final/repaired_v1/pair_claim_reextract_v1.jsonl")
    ap.add_argument("--priority", default="P0")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk_chars", type=int, default=2200)
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=20)
    args = ap.parse_args()

    rows = read_queue(args.queue)
    if args.priority:
        keep = {p.strip() for p in args.priority.split(",") if p.strip()}
        rows = [r for r in rows if str(r.get("priority")) in keep]
    if args.verify:
        verify_rows = {
            str(r.get("pair_id")): r
            for r in read_queue(args.verify)
            if str(r.get("relation_to_claim")) == "evidence_only"
        }
        evidence_only = set(verify_rows)
        rows = [r for r in rows if pair_id(r) in evidence_only]
        for r in rows:
            r["_verify_context"] = verify_rows.get(pair_id(r), {})
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]

    out = Path(args.out)
    done = load_done(out)
    todo = [r for r in rows if pair_id(r) not in done]
    counts = Counter()
    processed = 0
    batch_size = max(1, args.batch_size)
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        processed += process_batch(
            batch,
            out=out,
            counts=counts,
            progress_start=processed,
            total=len(todo),
            chunk_chars=args.chunk_chars,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=max(1, args.concurrency),
        )
    report = {
        "out": str(out),
        "processed": processed,
        "already_done": len(done),
        "status": dict(counts),
        "concurrency": max(1, args.concurrency),
        "batch_size": batch_size,
    }
    write_json(str(out) + ".report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
