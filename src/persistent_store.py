"""Persistent Supabase storage for dashboard periods and generated tables.

The app keeps the local file-based mode as a fallback. If Supabase credentials
are present in Streamlit secrets or environment variables, generated tables can
be saved to and loaded from Supabase so uploaded periods survive Streamlit Cloud
restarts/redeploys.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = Any  # type: ignore
    create_client = None  # type: ignore

TABLES = ["events", "discussions", "messages", "discussion_messages", "event_discussions"]
ROW_KEY_COLUMNS = {
    "events": ["event_id"],
    "discussions": ["discussion_id"],
    "messages": ["message_id"],
    "discussion_messages": ["discussion_id", "message_id"],
    "event_discussions": ["event_id", "discussion_id"],
}
PAGE_SIZE = 1000
CHUNK_SIZE = 400


def _secret_value(*names: str) -> str:
    """Read a value from st.secrets first, then environment variables."""
    for name in names:
        if st is not None:
            try:
                if name in st.secrets:
                    return str(st.secrets[name])
            except Exception:
                pass
            # Also support [supabase] url/key sections.
            try:
                if "supabase" in st.secrets and name.lower().replace("supabase_", "") in st.secrets["supabase"]:
                    return str(st.secrets["supabase"][name.lower().replace("supabase_", "")])
            except Exception:
                pass
        value = os.getenv(name)
        if value:
            return str(value)
    return ""


def supabase_configured() -> bool:
    return bool(_secret_value("SUPABASE_URL") and _secret_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY", "SUPABASE_ANON_KEY"))


def get_supabase_client() -> Client:
    if create_client is None:
        raise RuntimeError("Пакет supabase не установлен. Добавьте supabase>=2 в requirements.txt")
    url = _secret_value("SUPABASE_URL")
    key = _secret_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY", "SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Не заданы SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY в secrets.")
    return create_client(url, key)


def safe_slug(value: str, fallback: str = "period") -> str:
    value = str(value or "").strip().lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def ascii_storage_component(value: str, fallback: str = "file") -> str:
    """Return a Supabase Storage-safe ASCII path component."""
    value = str(value or "").strip()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or fallback


def safe_storage_filename(filename: str) -> str:
    source = Path(filename or "upload.csv")
    stem = ascii_storage_component(source.stem, "upload")[:120]
    suffix = ascii_storage_component(source.suffix.lower().lstrip("."), "csv")
    if suffix not in {"csv", "txt", "xlsx", "xls", "xlsm"}:
        suffix = "csv"
    return f"{stem}.{suffix}"


def content_type_for_filename(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suffix == ".xls":
        return "application/vnd.ms-excel"
    return "application/octet-stream"


def make_period_id(period_name: str, source_filename: str = "") -> str:
    base = safe_slug(period_name, "period")[:70]
    digest = hashlib.md5(f"{period_name}|{source_filename}".encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{base}_{digest}"


def normalize_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def dataframe_to_payload_records(df: pd.DataFrame, table_name: str, period_id: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records: list[dict[str, Any]] = []
    key_cols = ROW_KEY_COLUMNS.get(table_name, [])
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

    for idx, row in work.iterrows():
        payload = {str(k): normalize_json_value(v) for k, v in row.to_dict().items()}
        if key_cols and all(col in payload and str(payload.get(col) or "").strip() for col in key_cols):
            row_id = "::".join(str(payload[col]) for col in key_cols)
        else:
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            row_id = f"row_{idx}_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:10]}"
        records.append({
            "period_id": period_id,
            "table_name": table_name,
            "row_id": str(row_id),
            "payload": payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


def chunked(items: list[Any], size: int = CHUNK_SIZE) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _fetch_all(client: Client, table: str, *, filters: dict[str, Any] | None = None, order: str | None = None) -> list[dict[str, Any]]:
    """Paginated select. Supabase defaults to 1000 rows per request."""
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        query = client.table(table).select("*")
        if filters:
            for key, value in filters.items():
                if isinstance(value, (list, tuple, set)):
                    query = query.in_(key, list(value))
                else:
                    query = query.eq(key, value)
        if order:
            query = query.order(order)
        response = query.range(start, start + PAGE_SIZE - 1).execute()
        data = response.data or []
        rows.extend(data)
        if len(data) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def list_periods(include_inactive: bool = False) -> pd.DataFrame:
    """Return uploaded periods from Supabase.

    By default only active periods are returned for dashboard filters.
    The history manager passes include_inactive=True to show hidden/archived
    periods as well. Periods with status=deleted are treated as removed.
    """
    client = get_supabase_client()
    rows = _fetch_all(client, "dashboard_periods", order="uploaded_at")
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["date_from", "date_to", "uploaded_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "status" not in df.columns:
            df["status"] = "active"
        df["status"] = df["status"].fillna("active").astype(str)
        if not include_inactive:
            df = df[df["status"].str.lower().isin(["active", "активный", ""])]
        else:
            df = df[~df["status"].str.lower().isin(["deleted", "удален", "удалён"])]
        df = df.sort_values("uploaded_at", ascending=False)
    return df


def load_table_from_supabase(period_ids: list[str], table_name: str) -> pd.DataFrame:
    client = get_supabase_client()
    all_payloads: list[dict[str, Any]] = []
    for period_id in period_ids:
        rows = _fetch_all(client, "dashboard_table_rows", filters={"period_id": period_id, "table_name": table_name})
        for row in rows:
            payload = row.get("payload") or {}
            if isinstance(payload, dict):
                payload.setdefault("period_id", period_id)
                all_payloads.append(payload)
    df = pd.DataFrame(all_payloads)
    return df


def _prefix_generated_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prefix generated event/discussion IDs by period to avoid collisions across periods."""
    if df is None or df.empty or "period_id" not in df.columns:
        return df
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df.apply(lambda r: f"{r['period_id']}__{r[col]}" if pd.notna(r[col]) and str(r[col]).strip() else r[col], axis=1)
    return df


def load_generated_tables_from_supabase(period_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not period_ids:
        return tuple(pd.DataFrame() for _ in TABLES)  # type: ignore[return-value]
    loaded = [load_table_from_supabase(period_ids, table_name) for table_name in TABLES]

    events, discussions, messages, discussion_messages, event_discussions = loaded
    events = _prefix_generated_ids(events, ["event_id"])
    discussions = _prefix_generated_ids(discussions, ["discussion_id"])
    discussion_messages = _prefix_generated_ids(discussion_messages, ["discussion_id"])
    event_discussions = _prefix_generated_ids(event_discussions, ["event_id", "discussion_id"])

    for df, cols in [(events, ["start_date", "end_date"]), (discussions, ["start_date", "end_date"]), (messages, ["datetime"]),]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return events, discussions, messages, discussion_messages, event_discussions


def read_generated_tables_from_dir(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    from io_utils import read_table
    return {table: read_table(str(data_dir), table) for table in TABLES}


def detect_period_dates(messages: pd.DataFrame) -> tuple[str | None, str | None]:
    if messages is None or messages.empty or "datetime" not in messages.columns:
        return None, None
    dt = pd.to_datetime(messages["datetime"], errors="coerce").dropna()
    if dt.empty:
        return None, None
    return dt.min().date().isoformat(), dt.max().date().isoformat()


def save_processed_tables(
    *,
    period_id: str,
    period_name: str,
    source_filename: str,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any] | None = None,
    replace: bool = True,
) -> None:
    client = get_supabase_client()
    messages = tables.get("messages", pd.DataFrame())
    date_from, date_to = detect_period_dates(messages)
    period_payload = {
        "period_id": period_id,
        "period_name": period_name,
        "date_from": date_from,
        "date_to": date_to,
        "source_filename": source_filename,
        "status": "active",
        "manifest": manifest or {},
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("dashboard_periods").upsert(period_payload, on_conflict="period_id").execute()

    if replace:
        client.table("dashboard_table_rows").delete().eq("period_id", period_id).execute()

    for table_name in TABLES:
        records = dataframe_to_payload_records(tables.get(table_name, pd.DataFrame()), table_name, period_id)
        for batch in chunked(records):
            client.table("dashboard_table_rows").upsert(batch, on_conflict="period_id,table_name,row_id").execute()


def save_processed_tables_from_dir(
    data_dir: str | Path,
    *,
    period_id: str,
    period_name: str,
    source_filename: str,
    manifest: dict[str, Any] | None = None,
    replace: bool = True,
) -> None:
    tables = read_generated_tables_from_dir(data_dir)
    save_processed_tables(
        period_id=period_id,
        period_name=period_name,
        source_filename=source_filename,
        tables=tables,
        manifest=manifest,
        replace=replace,
    )


def _normalize_date_for_db(value: Any) -> str | None:
    """Normalize edited period dates to ISO YYYY-MM-DD without MM/DD ambiguity."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value or "").strip()
    if not text or text.lower() in {"nat", "nan", "none"}:
        return None

    iso_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if iso_match:
        try:
            return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))).isoformat()
        except ValueError:
            return None

    ru_match = re.match(r"^(\d{1,2})[.](\d{1,2})[.](\d{2,4})$", text)
    if ru_match:
        day, month, year = [int(x) for x in ru_match.groups()]
        if year < 100:
            year += 2000
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def get_period(period_id: str) -> dict[str, Any] | None:
    client = get_supabase_client()
    response = client.table("dashboard_periods").select("*").eq("period_id", period_id).limit(1).execute()
    data = response.data or []
    return data[0] if data else None


def update_period_metadata(
    period_id: str,
    *,
    period_name: str | None = None,
    date_from: Any = None,
    date_to: Any = None,
    source_filename: str | None = None,
    status: str | None = None,
    manifest_updates: dict[str, Any] | None = None,
) -> None:
    """Update editable period metadata without touching generated rows."""
    payload: dict[str, Any] = {}
    if period_name is not None:
        payload["period_name"] = str(period_name).strip() or period_id
    if date_from is not None:
        payload["date_from"] = _normalize_date_for_db(date_from)
    if date_to is not None:
        payload["date_to"] = _normalize_date_for_db(date_to)
    if source_filename is not None:
        payload["source_filename"] = str(source_filename).strip()
    if status is not None:
        payload["status"] = str(status).strip() or "active"

    if manifest_updates:
        current = get_period(period_id) or {}
        manifest = current.get("manifest") or {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.update(manifest_updates)
        payload["manifest"] = manifest

    if not payload:
        return
    client = get_supabase_client()
    client.table("dashboard_periods").update(payload).eq("period_id", period_id).execute()


def set_period_status(period_id: str, status: str) -> None:
    update_period_metadata(period_id, status=status)


def delete_period(period_id: str, *, hard: bool = False) -> None:
    """Soft-hide or permanently delete a period.

    Hard delete removes dashboard_periods and generated table rows via cascade.
    Manual rows are intentionally left intact, because they can include global
    moderator notes; orphaned records are ignored by the dashboard.
    """
    client = get_supabase_client()
    if hard:
        client.table("dashboard_periods").delete().eq("period_id", period_id).execute()
    else:
        set_period_status(period_id, "hidden")


def save_uploaded_csv_to_storage(period_id: str, filename: str, file_bytes: bytes) -> str:
    """Optionally save raw CSV to Supabase Storage.

    Requires a bucket named by SUPABASE_STORAGE_BUCKET, default dashboard-csv.
    The dashboard does not depend on this file to render periods; processed
    tables are stored in Postgres. If the bucket is not configured correctly,
    the caller may catch and ignore the exception.
    """
    client = get_supabase_client()
    bucket = _secret_value("SUPABASE_STORAGE_BUCKET") or "dashboard-csv"
    safe_period = ascii_storage_component(period_id, "period")[:100]
    safe_name = safe_storage_filename(filename)
    digest = hashlib.md5(file_bytes or b"").hexdigest()[:8]
    path = f"{safe_period}/{digest}_{safe_name}"
    try:
        client.storage.from_(bucket).upload(path, file_bytes, {"content-type": content_type_for_filename(filename), "upsert": "true"})
    except TypeError:
        client.storage.from_(bucket).upload(path, file_bytes)
    return path
