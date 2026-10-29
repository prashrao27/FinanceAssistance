"""SQLite persistence for local conversation turns and explicit user profiles."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


@dataclass(frozen=True)
class UserProfile:
    learning_level: str | None = None
    explanation_preferences: str | None = None
    financial_goals: tuple[str, ...] = ()


class ConversationStore:
    """Persist chat turns and opt-in learning profile fields in a local database."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        os.chmod(self.database_path, 0o600)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS conversation_turns_by_thread
                ON conversation_turns(user_id, thread_id, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    learning_level TEXT,
                    explanation_preferences TEXT,
                    financial_goals TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _validate_identity(user_id: str, thread_id: str | None = None) -> None:
        if not user_id.strip() or len(user_id) > 128:
            raise ValueError("user_id must contain between 1 and 128 characters")
        if thread_id is not None and (
            not thread_id.strip() or len(thread_id) > 128
        ):
            raise ValueError("thread_id must contain between 1 and 128 characters")

    def append_turn(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        answer: str,
    ) -> None:
        self._validate_identity(user_id, thread_id)
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not answer.strip():
            raise ValueError("answer cannot be empty")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns
                    (user_id, thread_id, query, answer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    thread_id,
                    query,
                    answer,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_history(
        self,
        user_id: str,
        thread_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        self._validate_identity(user_id, thread_id)
        if limit is not None and limit < 1:
            raise ValueError("history limit must be greater than zero")
        with self._connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT query, answer, created_at
                    FROM conversation_turns
                    WHERE user_id = ? AND thread_id = ?
                    ORDER BY id
                    """,
                    (user_id, thread_id),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT query, answer, created_at
                    FROM (
                        SELECT id, query, answer, created_at
                        FROM conversation_turns
                        WHERE user_id = ? AND thread_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id
                    """,
                    (user_id, thread_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def clear_history(self, user_id: str, thread_id: str) -> int:
        self._validate_identity(user_id, thread_id)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_turns WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )
        return cursor.rowcount

    def get_profile(self, user_id: str) -> UserProfile:
        self._validate_identity(user_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT learning_level, explanation_preferences, financial_goals
                FROM user_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return UserProfile()
        goals = json.loads(row["financial_goals"])
        if not isinstance(goals, list) or not all(
            isinstance(goal, str) for goal in goals
        ):
            raise ValueError(f"Stored financial goals for {user_id!r} are invalid")
        return UserProfile(
            learning_level=row["learning_level"],
            explanation_preferences=row["explanation_preferences"],
            financial_goals=tuple(goals),
        )

    def update_profile(
        self,
        user_id: str,
        *,
        learning_level: str | None = None,
        explanation_preferences: str | None = None,
        financial_goals: list[str] | None = None,
    ) -> UserProfile:
        self._validate_identity(user_id)
        existing = self.get_profile(user_id)
        updated = UserProfile(
            learning_level=(
                learning_level.strip()
                if learning_level is not None and learning_level.strip()
                else (None if learning_level == "" else existing.learning_level)
            ),
            explanation_preferences=(
                explanation_preferences.strip()
                if explanation_preferences is not None
                and explanation_preferences.strip()
                else (
                    None
                    if explanation_preferences == ""
                    else existing.explanation_preferences
                )
            ),
            financial_goals=(
                tuple(goal.strip() for goal in financial_goals if goal.strip())
                if financial_goals is not None
                else existing.financial_goals
            ),
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles
                    (user_id, learning_level, explanation_preferences,
                     financial_goals, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    learning_level = excluded.learning_level,
                    explanation_preferences = excluded.explanation_preferences,
                    financial_goals = excluded.financial_goals,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    updated.learning_level,
                    updated.explanation_preferences,
                    json.dumps(updated.financial_goals, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return updated

    def delete_profile(self, user_id: str) -> bool:
        self._validate_identity(user_id)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
            )
        return cursor.rowcount > 0
