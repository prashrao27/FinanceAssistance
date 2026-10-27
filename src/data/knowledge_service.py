"""Hybrid Qdrant retrieval for the Finance Assistant knowledge base."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from qdrant_client import QdrantClient, models

from core.errors import FinanceAssistantError
from core.settings import get_settings
from rag.embedding import make_dense_embedding, make_sparse_embedding


_DENSE_VECTOR_NAME = "text-dense"
_SPARSE_VECTOR_NAME = "text-sparse"
_CANDIDATE_MULTIPLIER = 3
_MIN_CANDIDATES = 10
_MAX_CANDIDATES = 100

_client: QdrantClient | None = None
_client_lock = threading.Lock()
_search_cache: OrderedDict[
    tuple[str, str, tuple[str, ...], int],
    tuple[float, dict[str, Any]],
] = OrderedDict()
_cache_lock = threading.Lock()


class KnowledgeServiceError(FinanceAssistantError):
    """Raised when hybrid knowledge retrieval cannot be completed."""


def _get_client() -> QdrantClient:
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            settings = get_settings()
            qdrant_url = settings.qdrant_url or ""
            if not qdrant_url:
                raise KnowledgeServiceError(
                    "QDRANT_URL is not set. Configure it in the project .env "
                    "or the process environment."
                )
            _client = QdrantClient(
                url=qdrant_url,
                api_key=settings.qdrant_api_key,
            )
    return _client


def _cache_key(
    collection_name: str,
    query: str,
    doc_types: list[str] | None,
    limit: int,
) -> tuple[str, str, tuple[str, ...], int]:
    return (
        collection_name,
        query,
        tuple(sorted(doc_types or [])),
        limit,
    )


def _cached_result(
    key: tuple[str, str, tuple[str, ...], int],
) -> dict[str, Any] | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _search_cache.get(key)
        if entry is None:
            return None
        expires_at, result = entry
        if expires_at <= now:
            del _search_cache[key]
            return None
        _search_cache.move_to_end(key)
        return deepcopy(result)


def _store_result(
    key: tuple[str, str, tuple[str, ...], int],
    result: dict[str, Any],
) -> None:
    settings = get_settings()
    max_entries = settings.search_cache_max_entries
    with _cache_lock:
        _search_cache[key] = (
            time.monotonic() + settings.search_cache_ttl_seconds,
            deepcopy(result),
        )
        _search_cache.move_to_end(key)
        while len(_search_cache) > max_entries:
            _search_cache.popitem(last=False)


def search_knowledge(
    query: str,
    doc_types: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Retrieve knowledge chunks with dense+sparse vector search and RRF fusion."""
    normalized_query = query.strip()
    settings = get_settings()
    if not normalized_query:
        raise ValueError("query cannot be empty")
    if not 1 <= limit <= settings.search_max_results:
        raise ValueError(f"limit must be between 1 and {settings.search_max_results}")

    normalized_doc_types: list[str] | None = None
    if doc_types is not None:
        normalized_doc_types = sorted({doc_type.strip() for doc_type in doc_types})
        if not normalized_doc_types or any(not doc_type for doc_type in normalized_doc_types):
            raise ValueError("doc_types must contain non-empty document types")

    collection_name = settings.qdrant_collection.strip()
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION cannot be empty")

    key = _cache_key(collection_name, normalized_query, normalized_doc_types, limit)
    cached = _cached_result(key)
    if cached is not None:
        return cached

    dense_embeddings = make_dense_embedding([normalized_query])
    sparse_embeddings = make_sparse_embedding([normalized_query])
    if not dense_embeddings or not sparse_embeddings:
        raise KnowledgeServiceError("The embedding model returned no query vectors.")

    dense_vector = [float(value) for value in dense_embeddings[0]]
    sparse_embedding = sparse_embeddings[0]
    sparse_vector = models.SparseVector(
        indices=[int(index) for index in sparse_embedding.indices],
        values=[float(value) for value in sparse_embedding.values],
    )

    candidate_limit = min(
        max(limit * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES),
        _MAX_CANDIDATES,
    )
    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using=_DENSE_VECTOR_NAME,
            limit=candidate_limit,
        ),
        models.Prefetch(
            query=sparse_vector,
            using=_SPARSE_VECTOR_NAME,
            limit=candidate_limit,
        ),
    ]
    query_filter = None
    if normalized_doc_types:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="DocType",
                    match=models.MatchAny(any=normalized_doc_types),
                )
            ]
        )

    collection = _get_client().query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    results: list[dict[str, Any]] = []
    for point in collection.points:
        payload = point.payload or {}
        metadata = {
            name: value
            for name, value in payload.items()
            if name not in {"combined_text", "dense_text", "sparse_text"}
        }
        results.append(
            {
                "score": float(point.score),
                "text": payload.get("combined_text", ""),
                "metadata": metadata,
            }
        )

    result = {"query": normalized_query, "results": results}
    _store_result(key, result)
    return result
