from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .models import ChunkRecord, ConversationTurn, DocumentRecord


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS turn_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    artifact_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_state (
                    session_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value_json FROM app_state WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["value_json"])

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO app_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), now),
            )

    def replace_knowledge(
        self,
        documents: list[DocumentRecord],
        chunk_map: dict[str, list[ChunkRecord]],
    ) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            for document in documents:
                cursor = conn.execute(
                    """
                    INSERT INTO documents (
                        source_key, title, source_type, checksum, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.source_key,
                        document.title,
                        document.source_type,
                        document.checksum,
                        json.dumps(document.metadata),
                        now,
                        now,
                    ),
                )
                document_id = cursor.lastrowid
                for chunk in chunk_map.get(document.source_key, []):
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            document_id, chunk_index, content, content_hash, embedding_json, metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.metadata["content_hash"],
                            json.dumps(chunk.embedding),
                            json.dumps(chunk.metadata),
                            now,
                        ),
                    )

    def load_chunks(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_index,
                    c.content,
                    c.embedding_json,
                    c.metadata_json,
                    d.source_key,
                    d.title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY d.source_key, c.chunk_index
                """
            ).fetchall()
        return [
            {
                "source_key": row["source_key"],
                "source_title": row["title"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "embedding": json.loads(row["embedding_json"]),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def get_query_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT result_json, expires_at FROM query_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            expires_at = row["expires_at"]
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now(UTC):
                conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
                return None
            conn.execute(
                "UPDATE query_cache SET last_used_at = ? WHERE cache_key = ?",
                (utc_now(), cache_key),
            )
        return json.loads(row["result_json"])

    def put_query_cache(
        self,
        cache_key: str,
        query_hash: str,
        query_text: str,
        top_k: int,
        model_name: str,
        result: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        now = utc_now()
        expires_at = datetime.now(UTC).timestamp() + ttl_seconds
        expires_text = datetime.fromtimestamp(expires_at, tz=UTC).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO query_cache (
                    cache_key, query_hash, query_text, top_k, model_name, result_json,
                    created_at, last_used_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    last_used_at = excluded.last_used_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    query_hash,
                    query_text,
                    top_k,
                    model_name,
                    json.dumps(result),
                    now,
                    now,
                    expires_text,
                ),
            )

    def add_turn(
        self,
        user_id: str,
        chat_id: str,
        mode: str,
        user_text: str,
        assistant_text: str,
        artifact: dict[str, Any],
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO turn_history (
                    user_id, chat_id, mode, user_text, assistant_text, artifact_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    chat_id,
                    mode,
                    user_text,
                    assistant_text,
                    json.dumps(artifact),
                    utc_now(),
                ),
            )

    def get_recent_turns(
        self,
        user_id: str,
        chat_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT user_id, chat_id, mode, user_text, assistant_text, artifact_json, created_at
                FROM turn_history
                WHERE user_id = ? AND chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, chat_id, limit),
            ).fetchall()
        turns = [
            ConversationTurn(
                user_id=row["user_id"],
                chat_id=row["chat_id"],
                mode=row["mode"],
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                artifact=json.loads(row["artifact_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
        turns.reverse()
        return turns

    def get_last_turn(self, user_id: str, chat_id: str) -> ConversationTurn | None:
        turns = self.get_recent_turns(user_id=user_id, chat_id=chat_id, limit=1)
        return turns[0] if turns else None

    def get_session_state(self, session_key: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM session_state WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        if row is None:
            return {}
        return json.loads(row["state_json"])

    def set_session_state(self, session_key: str, state: dict[str, Any]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO session_state (session_key, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_key, json.dumps(state), now),
            )
