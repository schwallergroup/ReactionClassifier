from __future__ import annotations

import argparse
from typing import Optional

from rxn_classifier.classification import api as pipeline_api


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reaction classification pipeline from a config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON or YAML config file. See examples/config.example.json",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=5,
        help="Print first N rows of the result (default: 5)",
    )
    args = parser.parse_args(argv)

    # Load and run pipeline
    df = pipeline_api.run_from_config(args.config)

    # Best-effort fetch of output path for messaging
    try:
        data = pipeline_api._load_config(args.config)  # type: ignore[attr-defined]
        out_path = data.get("output_csv")
    except Exception:
        out_path = None

    print("Classification complete.")
    if out_path:
        print(f"Saved CSV: {out_path}")
    print(f"Rows: {len(df)}; Columns: {list(df.columns)}")
    if args.print_head > 0 and not df.empty:
        print("--- Head ---")
        print(df.head(args.print_head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

