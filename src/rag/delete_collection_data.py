"""Compatibility entry point for Qdrant collection cleanup."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.qdrant_cleanup import delete_collection_data, main

__all__ = ["delete_collection_data", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
