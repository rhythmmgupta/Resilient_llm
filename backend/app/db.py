from datetime import datetime, timezone
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from app.config import sqlite_path_from_url


def utc_now_iso() -> str:
    """Return current UTC server timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite persistence layer for sessions and multi-turn messages."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.db_path = sqlite_path_from_url(database_url)
        self._lock = threading.RLock()
        self._memory_conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn

        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_if_file(self, conn: sqlite3.Connection) -> None:
        if self.db_path != ":memory:":
            conn.close()

    def init_schema(self) -> None:
        """Create required tables and indexes if they do not already exist."""
        with self._lock:
            conn = self._connect()
            try:
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id  TEXT NOT NULL,
                        role        TEXT NOT NULL,
                        content     TEXT NOT NULL,
                        metadata    TEXT,
                        created_at  TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, id);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id    TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                self._close_if_file(conn)

    def ping(self) -> bool:
        """Verify that the SQLite database is reachable."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("SELECT 1;").fetchone()
                return True
            finally:
                self._close_if_file(conn)

    def get_session_system_prompt(self, session_id: str) -> Optional[str]:
        """Return stored system_prompt override for a session, if any."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                return row["system_prompt"]
            finally:
                self._close_if_file(conn)

    def get_llm_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return stored user/assistant messages for prompt construction, oldest first."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC;",
                    (session_id,),
                ).fetchall()
                return [{"role": row["role"], "content": row["content"]} for row in rows]
            finally:
                self._close_if_file(conn)

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Return full public session history for GET /sessions/{session_id}/messages."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC;",
                    (session_id,),
                ).fetchall()
                result: List[Dict[str, Any]] = []
                for row in rows:
                    item: Dict[str, Any] = {
                        "role": row["role"],
                        "content": row["content"],
                    }
                    if row["role"] == "assistant" and row["metadata"]:
                        try:
                            item["metadata"] = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            item["metadata"] = {}
                    result.append(item)
                return result
            finally:
                self._close_if_file(conn)

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        metadata: Dict[str, Any],
        system_prompt: Optional[str],
    ) -> None:
        """Atomically persist user message, assistant message, and session metadata."""
        now = utc_now_iso()
        metadata_json = json.dumps(metadata)
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt),
                            updated_at = excluded.updated_at;
                        """,
                        (session_id, system_prompt, now, now),
                    )
                    conn.execute(
                        """
                        INSERT INTO messages (session_id, role, content, metadata, created_at)
                        VALUES (?, 'user', ?, NULL, ?);
                        """,
                        (session_id, user_message, now),
                    )
                    conn.execute(
                        """
                        INSERT INTO messages (session_id, role, content, metadata, created_at)
                        VALUES (?, 'assistant', ?, ?, ?);
                        """,
                        (session_id, assistant_reply, metadata_json, now),
                    )
            finally:
                self._close_if_file(conn)
