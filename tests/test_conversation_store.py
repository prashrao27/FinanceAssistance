from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.conversation_store import ConversationStore, UserProfile


class ConversationStoreTests(unittest.TestCase):
    def test_turns_persist_and_are_scoped_by_user_and_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "assistant.sqlite3"
            first_store = ConversationStore(database_path)
            first_store.append_turn("user-a", "thread-a", "Question 1", "Answer 1")
            first_store.append_turn("user-a", "thread-a", "Question 2", "Answer 2")
            first_store.append_turn("user-a", "thread-b", "Other thread", "Other answer")
            first_store.append_turn("user-b", "thread-a", "Other user", "Other answer")

            reopened_store = ConversationStore(database_path)
            recent = reopened_store.get_history("user-a", "thread-a", limit=1)

            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["query"], "Question 2")
            self.assertEqual(recent[0]["answer"], "Answer 2")
            self.assertEqual(
                len(reopened_store.get_history("user-a", "thread-a")),
                2,
            )
            self.assertEqual(
                reopened_store.get_history("user-a", "thread-b")[0]["query"],
                "Other thread",
            )

    def test_profile_is_explicit_persistent_and_updatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "assistant.sqlite3"
            store = ConversationStore(database_path)
            self.assertEqual(store.get_profile("user-a"), UserProfile())

            saved = store.update_profile(
                "user-a",
                learning_level="beginner",
                explanation_preferences="Use examples",
                financial_goals=["Save for a home", "Build emergency savings"],
            )
            reopened_store = ConversationStore(database_path)
            self.assertEqual(reopened_store.get_profile("user-a"), saved)

            updated = reopened_store.update_profile(
                "user-a",
                learning_level="intermediate",
                financial_goals=[],
            )
            self.assertEqual(updated.learning_level, "intermediate")
            self.assertEqual(updated.explanation_preferences, "Use examples")
            self.assertEqual(updated.financial_goals, ())
            self.assertTrue(reopened_store.delete_profile("user-a"))
            self.assertEqual(reopened_store.get_profile("user-a"), UserProfile())

    def test_history_can_be_cleared_per_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "assistant.sqlite3")
            store.append_turn("user-a", "thread-a", "Question", "Answer")
            store.append_turn("user-a", "thread-b", "Keep this", "Answer")

            self.assertEqual(store.clear_history("user-a", "thread-a"), 1)
            self.assertEqual(store.get_history("user-a", "thread-a"), [])
            self.assertEqual(
                store.get_history("user-a", "thread-b")[0]["query"],
                "Keep this",
            )


if __name__ == "__main__":
    unittest.main()
