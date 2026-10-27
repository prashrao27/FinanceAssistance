"""Build and upload the project PDF knowledge base to Qdrant."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from core.settings import get_settings
from rag.embedding import make_dense_embedding, make_sparse_embedding
from rag.pdf_ingestion import read_news_synthesizer_chunks, read_pdf_chunks


logger = logging.getLogger(__name__)
_VECTOR_COLUMNS = {"dense_embedding", "sparse_embedding", "dense_text", "sparse_text"}
_DATASETS = (
    ("QAndA", "financeQnA.pdf", read_pdf_chunks),
    ("NewsSynthesiser", "News_Systhesizer.pdf", read_news_synthesizer_chunks),
    ("TaxEducation", "tax_education.pdf", read_news_synthesizer_chunks),
)


def load_embedded_datasets(data_directory: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read each configured PDF and compute its dense and sparse embeddings."""
    directory = data_directory or get_settings().data_directory
    datasets: dict[str, pd.DataFrame] = {}

    for document_type, filename, reader in _DATASETS:
        pdf_path = directory / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Knowledge source PDF does not exist: {pdf_path}")
        dataframe = pd.DataFrame(reader(pdf_path))
        if dataframe.empty:
            raise ValueError(f"Knowledge source PDF contained no chunks: {pdf_path}")
        texts = dataframe["combined_text"].astype(str).tolist()
        dataframe["sparse_embedding"] = make_sparse_embedding(texts)
        dataframe["dense_embedding"] = make_dense_embedding(texts)
        datasets[document_type] = dataframe
        logger.info("Embedded %d chunks from %s", len(dataframe), filename)
    return datasets


def _normalize_payload_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_payload_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload_value(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def create_points(dataframe: pd.DataFrame, document_type: str) -> list[models.PointStruct]:
    """Convert an embedded chunk DataFrame into Qdrant point objects."""
    required_columns = {"combined_text", "dense_embedding", "sparse_embedding"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{document_type} dataframe is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    points: list[models.PointStruct] = []
    for row_number, row in dataframe.reset_index(drop=True).iterrows():
        sparse_embedding = row["sparse_embedding"]
        if not hasattr(sparse_embedding, "indices") or not hasattr(
            sparse_embedding, "values"
        ):
            raise ValueError(
                f"{document_type} row {row_number} has an invalid sparse_embedding"
            )
        if len(sparse_embedding.indices) != len(sparse_embedding.values):
            raise ValueError(
                f"{document_type} row {row_number} sparse embedding indices and "
                "values have different lengths"
            )

        dense_embedding = row["dense_embedding"]
        if dense_embedding is None or len(dense_embedding) == 0:
            raise ValueError(
                f"{document_type} row {row_number} has an empty dense_embedding"
            )
        payload = {
            column: _normalize_payload_value(row[column])
            for column in dataframe.columns
            if column not in _VECTOR_COLUMNS
        }
        identity = "|".join(
            str(value)
            for value in (
                document_type,
                payload.get("source_file", ""),
                payload.get("document_title", ""),
                payload.get("section_path", ""),
                payload.get("page_start", ""),
                payload.get("page_end", ""),
                row["combined_text"],
            )
        )
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                vector={
                    "text-dense": [float(value) for value in dense_embedding],
                    "text-sparse": models.SparseVector(
                        indices=[int(index) for index in sparse_embedding.indices],
                        values=[float(value) for value in sparse_embedding.values],
                    ),
                },
                payload={"DocType": document_type, **payload},
            )
        )
    return points


def _prepare_collection(client: QdrantClient, collection_name: str) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Created Qdrant collection %s", collection_name)

    client.create_payload_index(
        collection_name=collection_name,
        field_name="DocType",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


def ingest_documents(
    *,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    data_directory: Path | None = None,
    batch_size: int = 64,
) -> int:
    """Embed project PDFs and upsert their chunks; return the uploaded count."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    settings = get_settings()
    collection = (collection_name or settings.qdrant_collection).strip()
    if not collection:
        raise ValueError("collection_name cannot be empty")

    owns_client = client is None
    qdrant = client
    if qdrant is None:
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL is not set in config or the environment")
        qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    try:
        _prepare_collection(qdrant, collection)
        uploaded_count = 0
        datasets = load_embedded_datasets(data_directory)
        for document_type, dataframe in datasets.items():
            points = create_points(dataframe, document_type)
            for start in range(0, len(points), batch_size):
                qdrant.upsert(
                    collection_name=collection,
                    points=points[start : start + batch_size],
                    wait=True,
                )
            uploaded_count += len(points)
            logger.info("Uploaded %d chunks for %s", len(points), document_type)
        return uploaded_count
    finally:
        if owns_client:
            qdrant.close()


def main() -> int:
    """Run the knowledge ingestion workflow from the command line."""
    logging.basicConfig(level=get_settings().log_level.upper())
    uploaded_count = ingest_documents()
    logger.info(
        "Uploaded %d knowledge chunks to collection %s",
        uploaded_count,
        get_settings().qdrant_collection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
