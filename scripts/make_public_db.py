#!/usr/bin/env python
"""Build the public reaction database: strip NameRXN-derived columns.

NameRXN is proprietary, so its labels (`NAME`, `CLASS`) are removed before public
release. Everything else — structures, our LLM labels, and templates — is kept.

Usage:
    python make_public_db.py \
        --in  /home/dparm/reaction_classification/data/reaction_db.parquet \
        --out reaction_db_public.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

# NameRXN-derived columns to drop.
DROP_COLS = ["NAME", "CLASS"]

# Columns kept in the public release (order preserved where present).
KEEP_COLS = [
    "REACTION", "SANITIZED_REACTION", "MAPPED_REACTION",
    "template_hash", "retro_template", "TEMPLATE_rr0rp0_ring0", "TEMPLATE_rr0rp1_ring0",
    "template_frequency", "source_library",
    "llm_class_number", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default="reaction_db_public.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.inp)
    n0, c0 = len(df), list(df.columns)
    keep = [c for c in KEEP_COLS if c in df.columns]
    out = df[keep].copy()

    dropped = [c for c in c0 if c not in keep]
    assert not (set(DROP_COLS) & set(out.columns)), "NameRXN columns still present!"
    assert len(out) == n0, "row count changed!"

    out.to_parquet(args.out, index=False)
    print(f"rows: {n0:,} (unchanged)")
    print(f"dropped columns: {dropped}")
    print(f"kept columns:    {keep}")
    print(f"wrote -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
