"""Generate RAFTS-style support/refute arguments for claim-evidence pairs.

The generated text is intentionally label-free: the prompt never includes y,
c, label_audit, or consumer comment alignment. The goal is to create a compact
claim-evidence reasoning view that can be appended to the evidence stream.

Usage:
  source env.sh
  python -m models.argument_aug \
      --dataset data/final/dataset_verify_faithful.jsonl \
      --out data/final/dataset_verify_faithful_args.jsonl \
      --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.llm import chat_json, run_many


def claim_text(r: dict) -> str:
    claim = r.get("claim", {}) or {}
    segs = claim.get("segments", []) or []
    return " ".join(s.get("text", "") for s in segs).strip()


def evidence_text(r: dict) -> str:
    parts = []
    for label, key, field in (
        ("PARAM", "evidence_params", "raw_text"),
        ("OCR", "evidence_ocr", "raw_text"),
        ("VLM", "evidence_vlm", "raw_quote"),
    ):
        for it in r.get(key, []) or []:
            txt = (it.get(field, "") or "").strip()
            if txt:
                parts.append(f"[{label}] {txt}")
    return "\n".join(parts)


def trim(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "..."


def make_prompt(r: dict) -> str:
    claim = trim(claim_text(r), 1200)
    evidence = trim(evidence_text(r), 1800)
    if not claim:
        claim = "(无明确主播话术)"
    if not evidence:
        evidence = "(无可用商品参数/OCR/VLM证据)"
    return f"""你是直播电商宣传核验研究助手。请只基于下面给出的属性、主播话术和商品证据，生成结构化论证。

要求：
1. 不要使用消费者评论、弱标签或外部常识来下结论。
2. 分别写“支持该宣传的理由”和“反驳/风险理由”。
3. 如果证据缺失，请明确指出缺口，不要臆造事实。
4. 输出严格 JSON，不要 Markdown。

商品类目：{r.get("category", "")}
属性：{r.get("attribute_name", "")}

主播话术：
{claim}

商品证据：
{evidence}

JSON schema:
{{
  "supporting_argument": "一句到两句，说明证据如何支持主播话术；没有则写证据不足。",
  "refuting_argument": "一句到两句，说明证据如何反驳、削弱或无法覆盖主播话术；没有则写未发现直接反驳。",
  "evidence_gap": "最关键的缺失证据或不确定性。",
  "risk_cues": ["短语1", "短语2"]
}}"""


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def record_key(r: dict) -> str:
    return str(r.get("atomic_id") or r.get("pair_id") or "")


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done[record_key(r)] = r
            except Exception:
                continue
    return done


def augment_one(r: dict, model: str, max_tokens: int) -> dict:
    obj = chat_json(
        make_prompt(r),
        system="你是严谨的中文事实核验与论证压缩助手，只输出 JSON。",
        model=model,
        temperature=0.0,
        namespace="argument_aug",
        max_tokens=max_tokens,
    )
    out = dict(r)
    out["arguments"] = {
        "supporting_argument": str(obj.get("supporting_argument", ""))[:500],
        "refuting_argument": str(obj.get("refuting_argument", ""))[:500],
        "evidence_gap": str(obj.get("evidence_gap", ""))[:300],
        "risk_cues": [str(x)[:80] for x in (obj.get("risk_cues", []) or [])[:6]],
        "model": model,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/dataset_verify_faithful.jsonl")
    ap.add_argument("--out", default="data/final/dataset_verify_faithful_args.jsonl")
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=500)
    args = ap.parse_args()

    recs = load_jsonl(args.dataset)
    if args.limit > 0:
        recs = recs[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    pending = [r for r in recs if record_key(r) not in done]
    print(f"[argument_aug] pending={len(pending)} done={len(done)} out={out_path}", flush=True)

    def _job(r: dict) -> dict:
        try:
            return augment_one(r, args.model, args.max_tokens)
        except Exception as e:  # noqa: BLE001
            aug = dict(r)
            aug["arguments"] = {"__error__": repr(e)[:300], "model": args.model}
            return aug

    results = run_many(pending, _job, desc="argument_aug")
    n_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for aug in results:
            f.write(json.dumps(aug, ensure_ascii=False) + "\n")
            n_new += 1
        f.flush()
    print(f"[argument_aug] done new={n_new} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
