"""Create CLAIMARC dataset variants with explicit evidence ordering policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARG_KEYS = ("supporting_argument", "refuting_argument", "evidence_gap")


def trim_text(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def source_len(rec):
    total = 0
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for it in rec.get(key, []) or []:
            total += len(str(it.get(field, "") or ""))
    return total


def has_source(rec):
    return source_len(rec) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=["source_first", "no_args", "args_first"],
                    default="source_first")
    ap.add_argument("--arg_max_chars", type=int, default=120)
    ap.add_argument("--drop_args_without_source", action="store_true",
                    help="Clear arguments when a pair has no params/OCR/VLM evidence.")
    args = ap.parse_args()

    n = 0
    trimmed = 0
    rows_with_source = 0
    dropped_arg_rows = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_evidence_policy"] = args.policy
            row_has_source = has_source(rec)
            if args.drop_args_without_source and not row_has_source and rec.get("arguments"):
                rec["arguments"] = dict(rec["arguments"])
                for key in ARG_KEYS:
                    rec["arguments"][key] = ""
                rec["arguments"]["risk_cues"] = []
                rec["_dropped_args_without_source"] = True
                dropped_arg_rows += 1
            if args.arg_max_chars > 0 and rec.get("arguments"):
                rec["arguments"] = dict(rec["arguments"])
                for key in ARG_KEYS:
                    txt = str(rec["arguments"].get(key, "") or "")
                    new = trim_text(txt, args.arg_max_chars)
                    if new != txt:
                        trimmed += 1
                    rec["arguments"][key] = new
            rows_with_source += int(row_has_source)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({
        "input": args.input,
        "out": args.out,
        "policy": args.policy,
        "arg_max_chars": args.arg_max_chars,
        "drop_args_without_source": args.drop_args_without_source,
        "rows": n,
        "rows_with_source": rows_with_source,
        "trimmed_argument_fields": trimmed,
        "dropped_arg_rows": dropped_arg_rows,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
