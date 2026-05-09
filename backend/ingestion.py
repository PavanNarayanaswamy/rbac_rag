"""
Standalone ingestion script.

Usage::

    python ingestion.py            # incremental ingest from ./data/
    python ingestion.py --reset    # wipe the collection and re-index everything
    python ingestion.py --data-dir /path/to/other/data

The script writes the parent folder name into each chunk's ``access_label``
metadata - this is what the RBAC retrieval filter uses to enforce isolation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vector_service import DATA_DIR, ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest ./data/ into ChromaDB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Root folder containing role-named subfolders (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing chunks before ingesting.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Ingesting from: {args.data_dir}")
    if args.reset:
        print("Reset mode: wiping the existing collection first.")

    stats = ingest(args.data_dir, reset=args.reset)
    print("\nIngestion complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
