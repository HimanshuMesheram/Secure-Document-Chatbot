import json
import sqlite3
from pathlib import Path

from core.config import HISTORY_DB


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection with a short busy timeout."""
    Path(HISTORY_DB).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(HISTORY_DB), timeout=10)
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            owner_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            messages TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (owner_id, thread_id)
        )
        """
    )
    return connection


def load_history(owner_id: str, thread_id: str) -> list[dict]:
    """Load only the conversation belonging to this owner."""
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT messages FROM chat_history
            WHERE owner_id = ? AND thread_id = ?
            """,
            (owner_id, thread_id),
        ).fetchone()

    if row is None:
        return []

    try:
        messages = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(messages, list):
        return []

    # Bound the amount of history sent to the LLM.
    return messages[-12:]


def save_history(
    owner_id: str,
    thread_id: str,
    messages: list[dict],
) -> None:
    """Save a conversation using parameterized SQL."""
    serialized = json.dumps(messages[-12:], ensure_ascii=False)

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_history
                (owner_id, thread_id, messages, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(owner_id, thread_id)
            DO UPDATE SET
                messages = excluded.messages,
                updated_at = CURRENT_TIMESTAMP
            """,
            (owner_id, thread_id, serialized),
        )


def delete_history(owner_id: str, thread_id: str) -> None:
    """Delete one conversation belonging to its owner."""
    with _connect() as connection:
        connection.execute(
            """
            DELETE FROM chat_history
            WHERE owner_id = ? AND thread_id = ?
            """,
            (owner_id, thread_id),
        )