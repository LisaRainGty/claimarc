"""Join atomic claim records, product facts, and atomic labels.

The output schema is intentionally compatible with `models.data`: each row has
`pair_id`, `product_id`, `attribute_id`, `claim`, evidence fields, `y`, `c`,
and `split`.  `atomic_id` is kept for diagnostics, while group splits still use
`room_id` to avoid livestream leakage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common.io_utils import read_jsonl, write_json, write_jsonl
from final.join_split import grouped_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_records", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table_prefix", required=True)
    args = ap.parse_args()

    atoms = {r["atomic_id"]: r for r in read_jsonl(args.atomic_records)}
    facts = {(str(r["product_id"]), str(r["attribute_id"])): r for r in read_jsonl(args.facts)}
    labels = {r["atomic_id"]: r for r in read_jsonl(args.labels)}
    keys = sorted(set(atoms) & set(labels))
    room_split = grouped_split([str(atoms[k].get("room_id", "UNKNOWN")) for k in keys])

    rows = []
    for aid in keys:
        ar = atoms[aid]
        lb = labels[aid]
        fr = facts.get((str(ar["product_id"]), str(ar["attribute_id"])), {})
        split = room_split.get(str(ar.get("room_id", "UNKNOWN")), "train")
        rows.append({
            "atomic_id": aid,
            "pair_id": ar["pair_id"],
            "product_id": ar["product_id"],
            "category": ar.get("category", ""),
            "subcategory": ar.get("subcategory", ""),
            "room_id": ar.get("room_id", "UNKNOWN"),
            "attribute_id": ar["attribute_id"],
            "attribute_name": ar.get("attribute_name", ar["attribute_id"]),
            "source_family": ar.get("source_family", ""),
            "claim": ar.get("claim", {}),
            "evidence_params": fr.get("evidence_params", []),
            "evidence_ocr": fr.get("evidence_ocr", []),
            "evidence_vlm": fr.get("evidence_vlm", []),
            "evidence_count": fr.get("evidence_count", {"params": 0, "ocr": 0, "vlm": 0}),
            "coverage": fr.get("coverage", 0),
            "confidence": fr.get("confidence", "absent"),
            "y": lb["y"],
            "c": lb["c"],
            "label_audit": lb.get("label_audit", {}),
            "reviews": ar.get("reviews", []),
            "alignment_stats": ar.get("stats", {}),
            "split": split,
        })
    write_jsonl(args.out, rows)
    _table(rows, args.table_prefix)
    print(f"[final-atomic] rows={len(rows)} -> {args.out}")


def _table(rows: list[dict], prefix: str) -> None:
    split_c = Counter(r["split"] for r in rows)
    split_pos = Counter(r["split"] for r in rows if r["y"] == 1)
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r.get("category", "")][0] += 1
        by_cat[r.get("category", "")][1] += int(r["y"])
    stats = {
        "n_atomic_claims": len(rows),
        "n_pairs": len({r["pair_id"] for r in rows}),
        "n_products": len({r["product_id"] for r in rows}),
        "n_pos": sum(int(r["y"]) for r in rows),
        "pos_rate": round(sum(int(r["y"]) for r in rows) / max(1, len(rows)), 4),
        "split": {
            s: {
                "n": split_c[s],
                "pos": split_pos[s],
                "pos_rate": round(split_pos[s] / max(1, split_c[s]), 4),
            }
            for s in ("train", "val", "test")
        },
        "coverage_dist": dict(sorted(Counter(r["coverage"] for r in rows).items())),
        "by_category": {
            k: {"n": v[0], "pos": v[1], "pos_rate": round(v[1] / max(1, v[0]), 4)}
            for k, v in sorted(by_cat.items())
        },
    }
    write_json(f"{prefix}.json", stats)
    lines = [
        "# Atomic Claim Dataset Stats",
        f"- atomic claims: **{stats['n_atomic_claims']}**",
        f"- pairs: **{stats['n_pairs']}**",
        f"- products: **{stats['n_products']}**",
        f"- positives: **{stats['n_pos']}** ({stats['pos_rate']:.1%})",
        "",
        "| split | n | y=1 | pos rate |",
        "|---|---:|---:|---:|",
    ]
    for s in ("train", "val", "test"):
        d = stats["split"][s]
        lines.append(f"| {s} | {d['n']} | {d['pos']} | {d['pos_rate']:.1%} |")
    lines += ["", "| category | n | y=1 | pos rate |", "|---|---:|---:|---:|"]
    for k, v in stats["by_category"].items():
        lines.append(f"| {k} | {v['n']} | {v['pos']} | {v['pos_rate']:.1%} |")
    from pathlib import Path
    Path(f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
