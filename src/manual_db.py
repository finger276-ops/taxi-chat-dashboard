"""
SQLite layer for manual moderation actions.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime
import uuid
import pandas as pd


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_overrides (
            event_id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            status TEXT,
            priority TEXT,
            hidden INTEGER DEFAULT 0,
            note TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_merges (
            source_event_id TEXT PRIMARY KEY,
            target_event_id TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS message_overrides (
            message_id TEXT PRIMARY KEY,
            target_event_id TEXT,
            hidden INTEGER DEFAULT 0,
            note TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_message_exclusions (
            event_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (event_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS manual_events (
            event_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            status TEXT,
            main_tags TEXT,
            hidden INTEGER DEFAULT 0,
            note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            payload TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def log(conn: sqlite3.Connection, action: str, entity_type: str, entity_id: str, payload: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log(action, entity_type, entity_id, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (action, entity_type, entity_id, payload, now()),
    )
    conn.commit()


def get_event_overrides(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM event_overrides", conn)


def get_event_merges(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM event_merges", conn)


def get_message_overrides(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM message_overrides", conn)


def get_message_exclusions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM event_message_exclusions", conn)


def get_manual_events(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM manual_events", conn)


def create_manual_event(
    conn: sqlite3.Connection,
    title: str,
    summary: str = "",
    status: str = "новый",
    main_tags: str = "",
    hidden: bool = False,
    note: str = "",
) -> str:
    """Create a user-defined information event and return its id."""
    title = str(title or "").strip()
    if not title:
        raise ValueError("Укажите название инфоповода.")

    event_id = f"manual_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    created = now()
    conn.execute(
        """
        INSERT INTO manual_events(event_id, title, summary, status, main_tags, hidden, note, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, title, summary, status, main_tags, int(hidden), note, created, created),
    )
    conn.commit()
    log(conn, "create_manual_event", "event", event_id, f"title={title}; note={note}")
    return event_id


def save_event_override(
    conn: sqlite3.Connection,
    event_id: str,
    title: str | None = None,
    summary: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    hidden: bool | None = None,
    note: str | None = None,
) -> None:
    existing = conn.execute("SELECT * FROM event_overrides WHERE event_id = ?", (event_id,)).fetchone()
    values = {
        "title": title if title is not None else (existing["title"] if existing else None),
        "summary": summary if summary is not None else (existing["summary"] if existing else None),
        "status": status if status is not None else (existing["status"] if existing else None),
        "priority": priority if priority is not None else (existing["priority"] if existing else None),
        "hidden": int(hidden) if hidden is not None else (existing["hidden"] if existing else 0),
        "note": note if note is not None else (existing["note"] if existing else None),
    }
    conn.execute(
        """
        INSERT INTO event_overrides(event_id, title, summary, status, priority, hidden, note, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            title=excluded.title,
            summary=excluded.summary,
            status=excluded.status,
            priority=excluded.priority,
            hidden=excluded.hidden,
            note=excluded.note,
            updated_at=excluded.updated_at
        """,
        (
            event_id,
            values["title"],
            values["summary"],
            values["status"],
            values["priority"],
            values["hidden"],
            values["note"],
            now(),
        ),
    )
    conn.commit()
    log(conn, "save_event_override", "event", event_id)


def merge_events(conn: sqlite3.Connection, source_event_id: str, target_event_id: str, reason: str = "") -> None:
    if source_event_id == target_event_id:
        raise ValueError("Нельзя объединить инфоповод сам с собой.")
    conn.execute(
        """
        INSERT INTO event_merges(source_event_id, target_event_id, reason, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_event_id) DO UPDATE SET
            target_event_id=excluded.target_event_id,
            reason=excluded.reason,
            updated_at=excluded.updated_at
        """,
        (source_event_id, target_event_id, reason, now()),
    )
    conn.commit()
    log(conn, "merge_events", "event", source_event_id, f"target={target_event_id}; reason={reason}")


def move_message(conn: sqlite3.Connection, message_id: str, target_event_id: str, note: str = "") -> None:
    conn.execute(
        """
        INSERT INTO message_overrides(message_id, target_event_id, hidden, note, updated_at)
        VALUES (?, ?, 0, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            target_event_id=excluded.target_event_id,
            hidden=0,
            note=excluded.note,
            updated_at=excluded.updated_at
        """,
        (message_id, target_event_id, note, now()),
    )
    conn.commit()
    log(conn, "move_message", "message", message_id, f"target={target_event_id}; note={note}")


def hide_message(conn: sqlite3.Connection, message_id: str, hidden: bool = True, note: str = "") -> None:
    existing = conn.execute("SELECT * FROM message_overrides WHERE message_id = ?", (message_id,)).fetchone()
    target_event_id = existing["target_event_id"] if existing else None
    conn.execute(
        """
        INSERT INTO message_overrides(message_id, target_event_id, hidden, note, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            hidden=excluded.hidden,
            note=excluded.note,
            updated_at=excluded.updated_at
        """,
        (message_id, target_event_id, int(hidden), note, now()),
    )
    conn.commit()
    log(conn, "hide_message" if hidden else "unhide_message", "message", message_id, note)

def mark_message_irrelevant(conn: sqlite3.Connection, event_id: str, message_id: str, reason: str = "") -> None:
    """Exclude a message only from the selected information event.

    This does not delete or globally hide the message. It only removes the
    message-event relationship from the dashboard aggregation.
    """
    conn.execute(
        """
        INSERT INTO event_message_exclusions(event_id, message_id, reason, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(event_id, message_id) DO UPDATE SET
            reason=excluded.reason,
            updated_at=excluded.updated_at
        """,
        (event_id, message_id, reason, now()),
    )
    conn.commit()
    log(conn, "mark_message_irrelevant", "message", message_id, f"event={event_id}; reason={reason}")


def restore_message_relevance(conn: sqlite3.Connection, event_id: str, message_id: str) -> None:
    """Return a previously excluded message back to the selected event."""
    conn.execute(
        "DELETE FROM event_message_exclusions WHERE event_id = ? AND message_id = ?",
        (event_id, message_id),
    )
    conn.commit()
    log(conn, "restore_message_relevance", "message", message_id, f"event={event_id}")
