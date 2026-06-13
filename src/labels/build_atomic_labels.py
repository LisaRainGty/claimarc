"""Weak labels for atomic claim-level records.

Hard label: an atomic claim is risky (`y=1`) if at least one aligned consumer
comment refutes it.  Confidence follows the original Methodology_Data shape:
more aligned comments raise confidence; support comments discount but do not
erase refuting evidence.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import config
from common.io_utils import read_jsonl, write_jsonl


def _review_score(c: dict) -> float:
    s = config.STRENGTH_MULT.get(c.get("mention_strength", "weak"), 0.7)
    gamma = config.GAMMA if c.get("explicit_fact_hit") else 0.0
    rel_bonus = 1.15 if c.get("relation") == "refute" else 1.0
    return (1.0 + gamma) * s * rel_bonus


def compute_label(rec: dict) -> dict:
    reviews = rec.get("reviews", []) or []
    aligned = [c for c in reviews if int(c.get("y_supportability", 0) or 0) == 1]
    refutes = [c for c in aligned if c.get("relation") == "refute"]
    supports = [c for c in aligned if c.get("relation") == "support"]
    mixed = [c for c in aligned if c.get("relation") == "mixed"]
    n_total = int((rec.get("stats") or {}).get("N_total", len(reviews)) or len(reviews))
    n_aligned = len(aligned)

    y = 1 if refutes else 0
    s_ref = sum(_review_score(c) for c in refutes) + 0.5 * sum(_review_score(c) for c in mixed)
    s_sup = sum(_review_score(c) for c in supports)
    f_sat = 1.0 - math.exp(-n_aligned / config.K_SAT)
    f_cov = (n_aligned / (n_total + 1.0)) ** config.BETA_COV
    c_base = f_sat * f_cov
    audit = {
        "n_aligned": n_aligned,
        "n_total": n_total,
        "n_refute_aligned": len(refutes),
        "n_support_aligned": len(supports),
        "n_mixed_aligned": len(mixed),
        "S_refute": round(s_ref, 4),
        "S_support": round(s_sup, 4),
        "f_sat": round(f_sat, 4),
        "f_cov": round(f_cov, 4),
        "c_base": round(c_base, 4),
    }
    if y == 1:
        denom = s_ref + config.LAMBDA_POS * s_sup
        f_asym = s_ref / denom if denom > 0 else 1.0
        has_strong_ref = any(c.get("mention_strength") == "strong" or c.get("explicit_fact_hit") for c in refutes)
        phi = config.PHI_BONUS if has_strong_ref else 1.0
        c = min(1.0, c_base * f_asym * phi)
        audit.update({"f_asym": round(f_asym, 4), "phi_bonus": phi})
    else:
        # Atomic negatives without any aligned support are weak: keep them as
        # trainable counterexamples but do not let them dominate.
        f_support = 1.0 if supports else 0.55
        c = c_base * f_support
        audit.update({"f_support": f_support})
    c = max(float(config.C_FLOOR), c)
    return {"y": y, "c": round(c, 4), "label_audit": audit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_records", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for rec in read_jsonl(args.atomic_records):
        lab = compute_label(rec)
        rows.append({
            "atomic_id": rec["atomic_id"],
            "pair_id": rec["pair_id"],
            "product_id": rec["product_id"],
            "attribute_id": rec["attribute_id"],
            **lab,
        })
    write_jsonl(args.out, rows)
    pos = sum(1 for r in rows if r["y"] == 1)
    conf = Counter("floor" if r["c"] <= config.C_FLOOR else "above_floor" for r in rows)
    print(f"[atomic-labels] rows={len(rows)} y=1={pos} ({pos / max(1, len(rows)):.1%}) conf={dict(conf)} -> {args.out}")


if __name__ == "__main__":
    main()
