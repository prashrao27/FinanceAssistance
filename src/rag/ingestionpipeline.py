"""Compatibility entry point for the document-ingestion workflow."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.ingestion import create_points, ingest_documents, load_embedded_datasets, main

__all__ = ["create_points", "ingest_documents", "load_embedded_datasets", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
