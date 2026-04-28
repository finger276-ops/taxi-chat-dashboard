"""Manual moderation storage.

Default mode: local SQLite.
Supabase mode: if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY are set,
manual actions are persisted to the dashboard_manual_rows table in Supabase.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import uuid
import json
import pandas as pd

from persistent_store import supabase_configured, get_supabase_client, _fetch_all


class SupabaseManualStore:
    is_supabase = True

    def __init__(self):
        self.client = get_supabase_client()


def connect(db_path: str | Path):
    """Return Supabase-backed store when configured; otherwise SQLite connection."""
    if supabase_configured():
        return SupabaseManualStore()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def is_supabase_conn(conn) -> bool:
    return bool(getattr(conn, "is_supabase", False))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload_df(conn: SupabaseManualStore, table_name: str) -> pd.DataFrame:
    rows = _fetch_all(conn.client, "dashboard_manual_rows", filters={"table_name": table_name})
    payloads = []
    for row in rows:
        payload = row.get("payload") or {}
        if isinstance(payload, dict):
            payloads.append(payload)
    return pd.DataFrame(payloads)


def _get_payload(conn: SupabaseManualStore, table_name: str, row_key: str) -> dict:
    response = conn.client.table("dashboard_manual_rows").select("payload").eq("row_key", row_key).limit(1).execute()
    data = response.data or []
    if not data:
        return {}
    payload = data[0].get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _upsert_payload(conn: SupabaseManualStore, table_name: str, row_key: str, payload: dict) -> None:
    payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
    conn.client.table("dashboard_manual_rows").upsert({
        "row_key": row_key,
        "table_name": table_name,
        "payload": payload,
        "updated_at": now(),
    }, on_conflict="row_key").execute()


def _delete_payload(conn: SupabaseManualStore, row_key: str) -> None:
    conn.client.table("dashboard_manual_rows").delete().eq("row_key", row_key).execute()


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

        CREATE TABLE IF NOT EXISTS event_key_messages (
            event_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            note TEXT,
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

        CREATE TABLE IF NOT EXISTS period_summaries (
            summary_key TEXT PRIMARY KEY,
            summary TEXT,
            note TEXT,
            period_ids TEXT,
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


def log(conn, action: str, entity_type: str, entity_id: str, payload: str = "") -> None:
    if is_supabase_conn(conn):
        row_key = f"audit_log:{now()}:{uuid.uuid4().hex[:8]}"
        _upsert_payload(conn, "audit_log", row_key, {
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": payload,
            "created_at": now(),
        })
        return
    conn.execute(
        "INSERT INTO audit_log(action, entity_type, entity_id, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (action, entity_type, entity_id, payload, now()),
    )
    conn.commit()


def get_event_overrides(conn) -> pd.DataFrame:
    if is_supabase_conn(conn):
        return _payload_df(conn, "event_overrides")
    return pd.read_sql_query("SELECT * FROM event_overrides", conn)


def get_event_merges(conn) -> pd.DataFrame:
    if is_supabase_conn(conn):
        return _payload_df(conn, "event_merges")
    return pd.read_sql_query("SELECT * FROM event_merges", conn)


def get_message_overrides(conn) -> pd.DataFrame:
    if is_supabase_conn(conn):
        return _payload_df(conn, "message_overrides")
    return pd.read_sql_query("SELECT * FROM message_overrides", conn)


def get_message_exclusions(conn) -> pd.DataFrame:
    if is_supabase_conn(conn):
        return _payload_df(conn, "event_message_exclusions")
    return pd.read_sql_query("SELECT * FROM event_message_exclusions", conn)


def get_key_message_pins(conn) -> pd.DataFrame:
    """Return messages manually pinned as key for particular information events."""
    if is_supabase_conn(conn):
        return _payload_df(conn, "event_key_messages")
    init_db(conn)
    return pd.read_sql_query("SELECT * FROM event_key_messages", conn)


def get_manual_events(conn) -> pd.DataFrame:
    if is_supabase_conn(conn):
        return _payload_df(conn, "manual_events")
    return pd.read_sql_query("SELECT * FROM manual_events", conn)


def get_dashboard_summary(conn, summary_key: str) -> dict:
    """Return editable dashboard/period summary by key."""
    summary_key = str(summary_key or "local:current").strip() or "local:current"
    if is_supabase_conn(conn):
        return _get_payload(conn, "period_summaries", f"period_summaries:{summary_key}")
    init_db(conn)
    row = conn.execute("SELECT * FROM period_summaries WHERE summary_key = ?", (summary_key,)).fetchone()
    if not row:
        return {}
    return dict(row)


def save_dashboard_summary(conn, summary_key: str, summary: str, note: str = "", period_ids: list[str] | None = None) -> None:
    """Save editable dashboard/period summary. Empty summary means fallback to auto summary."""
    summary_key = str(summary_key or "local:current").strip() or "local:current"
    payload = {
        "summary_key": summary_key,
        "summary": str(summary or ""),
        "note": str(note or ""),
        "period_ids": "|".join(str(x) for x in (period_ids or []) if str(x).strip()),
        "updated_at": now(),
    }
    if is_supabase_conn(conn):
        _upsert_payload(conn, "period_summaries", f"period_summaries:{summary_key}", payload)
        log(conn, "save_dashboard_summary", "period_summary", summary_key)
        return
    init_db(conn)
    conn.execute(
        """
        INSERT INTO period_summaries(summary_key, summary, note, period_ids, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(summary_key) DO UPDATE SET
            summary=excluded.summary,
            note=excluded.note,
            period_ids=excluded.period_ids,
            updated_at=excluded.updated_at
        """,
        (payload["summary_key"], payload["summary"], payload["note"], payload["period_ids"], payload["updated_at"]),
    )
    conn.commit()
    log(conn, "save_dashboard_summary", "period_summary", summary_key)


def create_manual_event(
    conn,
    title: str,
    summary: str = "",
    status: str = "новый",
    main_tags: str = "",
    hidden: bool = False,
    note: str = "",
) -> str:
    title = str(title or "").strip()
    if not title:
        raise ValueError("Укажите название инфоповода.")
    event_id = f"manual_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    created = now()
    payload = {
        "event_id": event_id,
        "title": title,
        "summary": summary,
        "status": status,
        "main_tags": main_tags,
        "hidden": int(hidden),
        "note": note,
        "created_at": created,
        "updated_at": created,
    }
    if is_supabase_conn(conn):
        _upsert_payload(conn, "manual_events", f"manual_events:{event_id}", payload)
        log(conn, "create_manual_event", "event", event_id, f"title={title}; note={note}")
        return event_id
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
    conn,
    event_id: str,
    title: str | None = None,
    summary: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    hidden: bool | None = None,
    note: str | None = None,
) -> None:
    if is_supabase_conn(conn):
        key = f"event_overrides:{event_id}"
        existing = _get_payload(conn, "event_overrides", key)
        payload = {
            "event_id": event_id,
            "title": title if title is not None else existing.get("title"),
            "summary": summary if summary is not None else existing.get("summary"),
            "status": status if status is not None else existing.get("status"),
            "priority": priority if priority is not None else existing.get("priority"),
            "hidden": int(hidden) if hidden is not None else int(existing.get("hidden", 0) or 0),
            "note": note if note is not None else existing.get("note"),
            "updated_at": now(),
        }
        _upsert_payload(conn, "event_overrides", key, payload)
        log(conn, "save_event_override", "event", event_id)
        return

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
        (event_id, values["title"], values["summary"], values["status"], values["priority"], values["hidden"], values["note"], now()),
    )
    conn.commit()
    log(conn, "save_event_override", "event", event_id)


def merge_events(conn, source_event_id: str, target_event_id: str, reason: str = "") -> None:
    if source_event_id == target_event_id:
        raise ValueError("Нельзя объединить инфоповод сам с собой.")
    payload = {
        "source_event_id": source_event_id,
        "target_event_id": target_event_id,
        "reason": reason,
        "updated_at": now(),
    }
    if is_supabase_conn(conn):
        _upsert_payload(conn, "event_merges", f"event_merges:{source_event_id}", payload)
        log(conn, "merge_events", "event", source_event_id, f"target={target_event_id}; reason={reason}")
        return
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


def move_message(conn, message_id: str, target_event_id: str, note: str = "") -> None:
    if is_supabase_conn(conn):
        key = f"message_overrides:{message_id}"
        existing = _get_payload(conn, "message_overrides", key)
        payload = {
            "message_id": message_id,
            "target_event_id": target_event_id,
            "hidden": 0,
            "note": note,
            "updated_at": now(),
        }
        _upsert_payload(conn, "message_overrides", key, payload)
        log(conn, "move_message", "message", message_id, f"target={target_event_id}; note={note}")
        return
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


def hide_message(conn, message_id: str, hidden: bool = True, note: str = "") -> None:
    if is_supabase_conn(conn):
        key = f"message_overrides:{message_id}"
        existing = _get_payload(conn, "message_overrides", key)
        payload = {
            "message_id": message_id,
            "target_event_id": existing.get("target_event_id"),
            "hidden": int(hidden),
            "note": note,
            "updated_at": now(),
        }
        _upsert_payload(conn, "message_overrides", key, payload)
        log(conn, "hide_message" if hidden else "unhide_message", "message", message_id, note)
        return
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


def mark_message_irrelevant(conn, event_id: str, message_id: str, reason: str = "") -> None:
    payload = {
        "event_id": event_id,
        "message_id": message_id,
        "reason": reason,
        "updated_at": now(),
    }
    if is_supabase_conn(conn):
        _upsert_payload(conn, "event_message_exclusions", f"event_message_exclusions:{event_id}:{message_id}", payload)
        log(conn, "mark_message_irrelevant", "message", message_id, f"event={event_id}; reason={reason}")
        return
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


def restore_message_relevance(conn, event_id: str, message_id: str) -> None:
    if is_supabase_conn(conn):
        _delete_payload(conn, f"event_message_exclusions:{event_id}:{message_id}")
        log(conn, "restore_message_relevance", "message", message_id, f"event={event_id}")
        return
    conn.execute("DELETE FROM event_message_exclusions WHERE event_id = ? AND message_id = ?", (event_id, message_id))
    conn.commit()
    log(conn, "restore_message_relevance", "message", message_id, f"event={event_id}")


def pin_key_message(conn, event_id: str, message_id: str, note: str = "") -> None:
    """Pin a message as manually selected key message inside an information event."""
    event_id = str(event_id or "").strip()
    message_id = str(message_id or "").strip()
    if not event_id or not message_id:
        raise ValueError("Не удалось определить инфоповод или сообщение.")
    payload = {
        "event_id": event_id,
        "message_id": message_id,
        "note": str(note or ""),
        "updated_at": now(),
    }
    if is_supabase_conn(conn):
        _upsert_payload(conn, "event_key_messages", f"event_key_messages:{event_id}:{message_id}", payload)
        log(conn, "pin_key_message", "message", message_id, f"event={event_id}; note={note}")
        return
    init_db(conn)
    conn.execute(
        """
        INSERT INTO event_key_messages(event_id, message_id, note, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(event_id, message_id) DO UPDATE SET
            note=excluded.note,
            updated_at=excluded.updated_at
        """,
        (event_id, message_id, str(note or ""), now()),
    )
    conn.commit()
    log(conn, "pin_key_message", "message", message_id, f"event={event_id}; note={note}")


def unpin_key_message(conn, event_id: str, message_id: str) -> None:
    """Remove a manual key-message pin from an information event."""
    event_id = str(event_id or "").strip()
    message_id = str(message_id or "").strip()
    if is_supabase_conn(conn):
        _delete_payload(conn, f"event_key_messages:{event_id}:{message_id}")
        log(conn, "unpin_key_message", "message", message_id, f"event={event_id}")
        return
    init_db(conn)
    conn.execute("DELETE FROM event_key_messages WHERE event_id = ? AND message_id = ?", (event_id, message_id))
    conn.commit()
    log(conn, "unpin_key_message", "message", message_id, f"event={event_id}")
