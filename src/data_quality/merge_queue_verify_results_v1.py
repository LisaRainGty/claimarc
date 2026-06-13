"""Merge queue verifier JSONL outputs into a clean latest-by-pair file."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    latest: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    errors = 0
    missing_pair = 0
    for path in args.input:
        for row in read_jsonl(path):
            raw_rows += 1
            if "__error__" in row:
                errors += 1
                continue
            pid = str(row.get("pair_id") or "")
            if not pid:
                missing_pair += 1
                continue
            latest[pid] = row

    rows = sorted(latest.values(), key=lambda r: (str(r.get("priority", "")), str(r.get("pair_id", ""))))
    write_jsonl(args.out, rows)
    report = {
        "inputs": args.input,
        "raw_rows": raw_rows,
        "errors": errors,
        "missing_pair": missing_pair,
        "n_output": len(rows),
        "curation_action": dict(Counter(str(r.get("curation_action")) for r in rows)),
        "relation_to_claim": dict(Counter(str(r.get("relation_to_claim")) for r in rows)),
        "source_type": dict(Counter(str(r.get("source_type")) for r in rows)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
