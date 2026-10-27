from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data import knowledge_service


class KnowledgeServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        knowledge_service._search_cache.clear()
        self.previous_client = knowledge_service._client
        self.client = Mock()
        self.client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.42,
                    payload={
                        "DocType": "QAndA",
                        "combined_text": "A bond is a debt instrument.",
                        "section_path": "1.2",
                        "dense_text": "embedding-only content",
                        "sparse_text": "keyword-only content",
                    },
                )
            ]
        )
        knowledge_service._client = self.client

    def tearDown(self) -> None:
        knowledge_service._search_cache.clear()
        knowledge_service._client = self.previous_client

    def test_hybrid_search_uses_rrf_and_caches_copied_results(self) -> None:
        with (
            patch.object(
                knowledge_service,
                "make_dense_embedding",
                return_value=[[0.1, 0.2]],
            ) as dense_embedding,
            patch.object(
                knowledge_service,
                "make_sparse_embedding",
                return_value=[
                    SimpleNamespace(indices=[1, 7], values=[0.4, 0.9])
                ],
            ) as sparse_embedding,
        ):
            result = knowledge_service.search_knowledge(
                " bond ",
                doc_types=["QAndA"],
                limit=3,
            )
            result["results"][0]["text"] = "modified by caller"
            cached_result = knowledge_service.search_knowledge(
                "bond",
                doc_types=["QAndA"],
                limit=3,
            )

        self.assertEqual(
            cached_result["results"][0]["text"],
            "A bond is a debt instrument.",
        )
        self.assertEqual(dense_embedding.call_count, 1)
        self.assertEqual(sparse_embedding.call_count, 1)
        self.assertEqual(self.client.query_points.call_count, 1)

        request = self.client.query_points.call_args.kwargs
        self.assertEqual(len(request["prefetch"]), 2)
        self.assertEqual(request["query"].fusion, knowledge_service.models.Fusion.RRF)
        self.assertEqual(
            request["query_filter"].must[0].match.any,
            ["QAndA"],
        )

    def test_empty_query_is_rejected_before_embedding(self) -> None:
        with patch.object(knowledge_service, "make_dense_embedding") as embed:
            with self.assertRaisesRegex(ValueError, "query cannot be empty"):
                knowledge_service.search_knowledge("  ")
        embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
