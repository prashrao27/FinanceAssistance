from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.settings import PROJECT_ROOT, get_settings
from workflow.ingestion import create_points


class SettingsAndIngestionTests(unittest.TestCase):
    def test_langsmith_tracing_uses_config_default(self) -> None:
        get_settings.cache_clear()
        try:
            with (
                patch(
                    "core.settings._read_config",
                    return_value={"langsmith": {"tracing": True}},
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                settings = get_settings()
            self.assertTrue(settings.langsmith_tracing)
        finally:
            get_settings.cache_clear()

    def test_environment_overrides_yaml_defaults(self) -> None:
        get_settings.cache_clear()
        try:
            with patch.dict(
                os.environ,
                {
                    "QDRANT_COLLECTION": "TestCollection",
                    "SEARCH_CACHE_TTL_SECONDS": "12",
                    "LANGSMITH_TRACING": "false",
                    "LANGSMITH_API_KEY": "test-key",
                    "LANGSMITH_PROJECT": "test-project",
                    "LANGSMITH_ENDPOINT": "https://example.test",
                    "CONVERSATION_DB_PATH": "tmp/test.sqlite3",
                    "CONVERSATION_HISTORY_TURNS": "5",
                },
            ):
                settings = get_settings()
            self.assertEqual(settings.qdrant_collection, "TestCollection")
            self.assertEqual(settings.search_cache_ttl_seconds, 12)
            self.assertFalse(settings.langsmith_tracing)
            self.assertEqual(settings.langsmith_api_key, "test-key")
            self.assertEqual(settings.langsmith_project, "test-project")
            self.assertEqual(settings.langsmith_endpoint, "https://example.test")
            self.assertEqual(
                settings.conversation_db_path,
                PROJECT_ROOT / "tmp/test.sqlite3",
            )
            self.assertEqual(settings.conversation_history_turns, 5)
        finally:
            get_settings.cache_clear()

    def test_invalid_langsmith_tracing_value_is_rejected(self) -> None:
        get_settings.cache_clear()
        try:
            with patch.dict(os.environ, {"LANGSMITH_TRACING": "sometimes"}):
                with self.assertRaisesRegex(
                    ValueError, "LANGSMITH_TRACING must be a boolean"
                ):
                    get_settings()
        finally:
            get_settings.cache_clear()

    def test_create_points_keeps_chunk_metadata_and_named_vectors(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "combined_text": "Bond fundamentals",
                    "dense_embedding": [0.1, 0.2],
                    "sparse_embedding": SimpleNamespace(
                        indices=[4, 9],
                        values=[0.3, 0.8],
                    ),
                    "section_path": "1.1",
                    "page_start": 2,
                    "dense_text": "semantic text",
                    "sparse_text": "keyword text",
                }
            ]
        )

        points = create_points(dataframe, "QAndA")

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload["DocType"], "QAndA")
        self.assertEqual(points[0].payload["section_path"], "1.1")
        self.assertNotIn("dense_text", points[0].payload)
        self.assertNotIn("sparse_text", points[0].payload)
        self.assertEqual(points[0].vector["text-dense"], [0.1, 0.2])
        self.assertEqual(
            points[0].vector["text-sparse"].indices,
            [4, 9],
        )


if __name__ == "__main__":
    unittest.main()
