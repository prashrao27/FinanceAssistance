"""Compatibility exports for Qdrant knowledge retrieval."""

from data.knowledge_service import KnowledgeServiceError, search_knowledge

__all__ = ["KnowledgeServiceError", "search_knowledge"]
