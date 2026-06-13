"""Stage B1 fallback: direct JSON claim extraction with exact local grounding.

LangExtract is preferred when it is stable.  For very long SRT windows or
provider-specific hangs, this fallback asks the LLM for exact source substrings
and then performs deterministic substring alignment locally before writing the
same claim_list schema.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import config
from common import llm
from common import product_index as pidx
from common import srt as S
from common.io_utils import normalize, read_json, write_jsonl


TASK = """你是电商直播商品事实抽取员。
从给定字幕窗口中抽取针对候选商品属性的事实陈述。

硬规则：
1. 只输出 JSON 数组，不要解释。
2. extraction_text 必须是字幕窗口里真实存在的连续原文子串，不能改写、概括、拼接。
3. attribute_id 必须来自候选属性集合；如果没有语义精确匹配，宁可不输出。
4. 不要把 claim 归到最近但不相同的宽泛属性。例如头围/尺码不能归入帽顶款式，弹力/松紧不能归入厚度，适用人群不能归入款式。
5. 忽略链接编号、下单、库存、价格优惠、主播闲聊、纯推荐语。
6. extraction_text 取最短但完整的事实子串，不要整段长句；优先包含明确属性值。
7. 排除主观穿戴效果或体验词，如显瘦、好看、舒服、百搭、不影响视线，除非候选属性本身就是对应客观属性且原文给出明确值。
8. 颜色 claim 必须包含明确颜色词；材质 claim 必须包含材质/成分/面料词；尺码/尺寸/重量/容量 claim 必须包含数字、范围或明确规格词。
9. 同一属性重复口播只保留最完整的一次。
10. 如果原文只是“1号链接/2号款/拍这个/下方链接/库存/到手价/优惠券”，即便有数字也输出 []。
11. 产品名称、品牌、型号、类型不能用链接号、价格、店铺、官方、正品等交易/营销词代替。

输出格式：
[
  {"attribute_id": "候选属性ID", "extraction_text": "字幕原文连续子串"}
]
"""

PROMO_SHAPE_RE = re.compile(r"(?:\d+|[一二三四五六七八九十两]+)号(?:链接|链|令节|款|色)?")
PROMO_ONLY_RE = re.compile(r"^(?:第)?(?:\d+|[一二三四五六七八九十两]+)号(?:链接|链|令节|款|色)?(?:的|这个|这款)?$")
HARD_PROMO_RE = re.compile(r"(下单|拍下|库存|小黄车|优惠券|领券|到手价|价格|售价|包邮|发货|物流|客服|售后|退换|保价|赠品|福利|秒杀|加购)")
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?\s*(?:元|块|块钱)|[一二三四五六七八九十两百千]+块(?:钱)?|到手价|售价|价格|销量|卖了\d+)")
MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|mm|kg|g|ml|l|mah|w|瓦|克|斤|升|毫升|厘米|毫米|"
    r"米|寸|码|%|天|月|年|个|只|件|包|袋|盒|瓶)",
    re.I,
)
RANGE_SIZE_RE = re.compile(r"\d+(?:\.\d+)?\s*[-~到至]\s*\d+(?:\.\d+)?\s*(?:cm|mm|厘米|毫米|米|寸|码|斤|克|kg|g|ml|毫升)?", re.I)
COLOR_RE = re.compile(r"(黑|白|红|蓝|绿|黄|紫|灰|棕|咖|粉|橙|金|银|杏|米色|自然色|冷白|暖白|墨色|卡其|驼色|藏青|藕粉|安可拉红)")
MATERIAL_RE = re.compile(r"(材质|面料|成分|配料|羊毛|绵羊毛|羊绒|棉|纯棉|全棉|真皮|牛皮|皮革|聚酯|聚酯纤维|锦纶|腈纶|氨纶|涤纶|粘纤|粘胶|莱赛尔|莫代尔|羽绒|鸭绒|鹅绒|混纺|亚麻|桑蚕丝|醋酸|PU|PVC|pu|pvc|乳胶|硅胶|不锈钢|陶瓷|玻璃|碳钢|塑料|树脂)")
STRUCTURE_RE = re.compile(r"(双层|单层|三层|加厚|薄款|厚款|高腰|低腰|直筒|阔腿|修身|宽松|A字|a字|圆领|V领|v领|松紧|弹力|短袖|长袖|连帽|翻领|厚底)")
IDENTITY_ATTR_TERMS = {"品牌", "产品名称", "商品名称", "型号", "货号", "条形码", "类型", "产品类型", "商品类型"}


def forbidden_quote(quote: str, meta: dict[str, Any]) -> bool:
    text = str(quote or "").strip()
    if not text:
        return True
    attr_name = str(meta.get("canonical_name") or "")
    family = str(meta.get("source_family") or "")
    has_value_cue = bool(
        MEASUREMENT_RE.search(text)
        or RANGE_SIZE_RE.search(text)
        or COLOR_RE.search(text)
        or MATERIAL_RE.search(text)
        or STRUCTURE_RE.search(text)
    )
    if PRICE_RE.search(text):
        return True
    if PROMO_ONLY_RE.search(text):
        return True
    if PROMO_SHAPE_RE.search(text) and not has_value_cue:
        return True
    if HARD_PROMO_RE.search(text) and not has_value_cue:
        return True
    if attr_name in IDENTITY_ATTR_TERMS and PROMO_SHAPE_RE.search(text):
        return True
    if "颜色" in attr_name and not COLOR_RE.search(text):
        return True
    if family == "material" and not MATERIAL_RE.search(text):
        return True
    if family == "numeric" and not (MEASUREMENT_RE.search(text) or RANGE_SIZE_RE.search(text) or STRUCTURE_RE.search(text)):
        return True
    return False


def acmt_block(acmt_p: dict[str, Any]) -> str:
    lines = []
    for aid, meta in acmt_p.items():
        lines.append(
            f"- {aid} | {meta.get('canonical_name', aid)}"
            f" | family={meta.get('source_family', '')}"
            f" | value_type={meta.get('value_type', '')}"
            f" | aliases={'、'.join((meta.get('aliases') or [])[:8])}"
        )
    return "\n".join(lines)


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


def as_rows(obj: Any) -> list[dict[str, str]]:
    if isinstance(obj, dict):
        obj = obj.get("claims") or obj.get("extractions") or []
    if not isinstance(obj, list):
        return []
    rows: list[dict[str, str]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("attribute_id") or "").strip()
        txt = str(item.get("extraction_text") or item.get("text") or "").strip()
        if aid and txt:
            rows.append({"attribute_id": aid, "extraction_text": txt})
    return rows


def extract_product(
    product_id: str,
    acmt_p: dict[str, Any],
    srt_files: list[str],
    *,
    chunk_chars: int,
    model: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    files = [str(pidx.resolve(f)) for f in srt_files]
    files = [f for f in files if os.path.exists(f)]
    if not files or not acmt_p:
        return []
    concat = S.concat_product_srt(files)
    if not concat.text.strip():
        return []

    valid_ids = set(acmt_p)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    seq = 0
    schema = acmt_block(acmt_p)
    for ci, (start, end) in enumerate(chunk_ranges(concat, chunk_chars), 1):
        chunk = concat.text[start:end]
        prompt = (
            TASK
            + "\n候选属性集合：\n"
            + schema
            + "\n\n字幕窗口：\n"
            + chunk
        )
        try:
            obj = llm.chat_json(
                prompt,
                model=model,
                temperature=0.0,
                namespace="b1_direct_json",
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] B1 direct JSON failed {product_id} chunk={ci}: {exc!r}", flush=True)
            continue
        for item in as_rows(obj):
            aid = item["attribute_id"]
            quote = item["extraction_text"]
            if aid not in valid_ids:
                continue
            if forbidden_quote(quote, acmt_p[aid]):
                continue
            local = chunk.find(quote)
            if local < 0:
                continue
            gstart = start + local
            gend = gstart + len(quote)
            if normalize(quote) and normalize(quote) not in normalize(concat.text[gstart:gend]):
                continue
            spans = concat.lookup_range(gstart, gend)
            if not spans:
                continue
            first, last = spans[0], spans[-1]
            key = (aid, normalize(quote), first.srt_file, first.start_ts)
            if key in seen:
                continue
            seen.add(key)
            seq += 1
            rows.append({
                "claim_id": f"{product_id}_{seq}",
                "attribute_id": aid,
                "claim_text": quote,
                "srt_file": os.path.basename(first.srt_file),
                "srt_path": first.srt_file,
                "start_ts": first.start_ts,
                "end_ts": last.end_ts,
                "char_start": gstart,
                "char_end": gend,
                "cue_span_count": len(spans),
                "_b1_backend": "direct_json_exact",
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--out_dir", default=str(config.STAGE_B / "claim_list_direct"))
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--product_id", action="append", default=None)
    ap.add_argument("--product_id_file", default="",
                    help="Optional UTF-8 text file with one product_id per line.")
    ap.add_argument("--chunk_chars", type=int, default=2500)
    ap.add_argument("--model", default=config.TEXT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={}) or {}
    bundles = pidx.build_bundles()
    pids = [p for p in acmt if p in bundles]
    if args.product_id:
        wanted = {str(x) for x in args.product_id}
        pids = [p for p in pids if p in wanted]
    if args.product_id_file:
        wanted = {
            line.strip()
            for line in Path(args.product_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        pids = [p for p in pids if p in wanted]
    if args.category:
        pids = [p for p in pids if bundles[p].category == args.category]
    if args.limit:
        pids = pids[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[B1-direct] products={len(pids)} model={args.model} chunk_chars={args.chunk_chars}")

    def job(pid: str):
        out = out_dir / f"{pid}.jsonl"
        if out.exists() and not args.force:
            return ("skip", pid, 0)
        rows = extract_product(
            pid,
            acmt[pid],
            bundles[pid].srt_files,
            chunk_chars=args.chunk_chars,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        write_jsonl(out, rows)
        return ("done", pid, len(rows))

    results = llm.run_many(pids, job, concurrency=int(os.environ.get("CLAIMARC_CONCURRENCY", "2")), desc="B1-direct")
    done = sum(1 for r in results if isinstance(r, tuple) and r[0] == "done")
    nclaims = sum(r[2] for r in results if isinstance(r, tuple))
    print(f"[B1-direct] done products={done} total claims={nclaims}")


if __name__ == "__main__":
    main()
