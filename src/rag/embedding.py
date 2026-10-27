from functools import cache
from typing import Any

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from numpy.typing import NDArray

from core.settings import get_settings


@cache
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name=get_settings().sparse_embedding_model,
    )


@cache
def _dense_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=get_settings().dense_embedding_model,
    )


def make_sparse_embedding(texts: list[str]) -> list[SparseEmbedding]:
    return list(_sparse_model().embed(texts, batch_size=32))


def make_dense_embedding(texts: list[str]) -> list[NDArray[Any]]:
    return list(_dense_model().embed(texts, batch_size=32))
