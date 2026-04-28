"""
Streamlit dashboard for taxi chat information events.

Run locally:
    python -m streamlit run src/app.py -- --data-dir data/processed --db-path data/manual_actions.sqlite
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from io_utils import read_table, read_source_csv
from manual_db import (
    connect,
    get_event_overrides,
    get_event_merges,
    get_message_overrides,
    get_message_exclusions,
    get_manual_events,
    create_manual_event,
    save_event_override,
    merge_events,
    move_message,
    hide_message,
    mark_message_irrelevant,
    restore_message_relevance,
)
from settings import STATUS_OPTIONS
from preprocess import run_preprocess, run_preprocess_from_dataframe
from persistent_store import (
    supabase_configured,
    list_periods,
    load_generated_tables_from_supabase,
    save_processed_tables_from_dir,
    make_period_id,
    save_uploaded_csv_to_storage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default=os.getenv("DASHBOARD_DATA_DIR", "data/processed"))
    parser.add_argument("--db-path", default=os.getenv("DASHBOARD_DB_PATH", "data/manual_actions.sqlite"))
    parser.add_argument("--upload-dir", default=os.getenv("DASHBOARD_UPLOAD_DIR", "data/uploads"))
    args, _ = parser.parse_known_args()
    return args


def get_secret_value(name: str, default: str = "") -> str:
    """Read Streamlit Secret first, then environment variable. Never expose values in UI."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value or "").strip()


def render_admin_mode() -> bool:
    """Return True when user is allowed to upload files and edit/moderate data."""
    admin_password = get_secret_value("ADMIN_PASSWORD", "")

    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False

    if not admin_password:
        st.sidebar.warning(
            "ADMIN_PASSWORD не настроен. Режим загрузки и правок пока доступен всем пользователям со ссылкой."
        )
        return True

    if st.session_state.get("is_admin"):
        st.sidebar.success("Режим: администратор")
        if st.sidebar.button("Выйти из режима правок"):
            st.session_state["is_admin"] = False
            st.rerun()
        return True

    st.sidebar.info("Режим: просмотр")
    with st.sidebar.expander("Вход администратора", expanded=False):
        entered = st.text_input("Пароль администратора", type="password", key="admin_password_input")
        if st.button("Войти", key="admin_login_button"):
            if entered == admin_password:
                st.session_state["is_admin"] = True
                st.success("Режим администратора включен.")
                st.rerun()
            else:
                st.error("Неверный пароль.")

    return False


@st.cache_data(show_spinner=False)
def load_generated_tables(data_dir: str):
    events = read_table(data_dir, "events")
    discussions = read_table(data_dir, "discussions")
    messages = read_table(data_dir, "messages")
    discussion_messages = read_table(data_dir, "discussion_messages")
    event_discussions = read_table(data_dir, "event_discussions")

    for df, cols in [
        (events, ["start_date", "end_date"]),
        (discussions, ["start_date", "end_date"]),
        (messages, ["datetime"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["is_hidden"]:
        if col in events.columns:
            events[col] = events[col].astype(str).str.lower().isin(["true", "1", "yes", "да"])

    for col in ["message_count", "chat_count", "author_count", "negative_count", "toxic_count", "discussion_count"]:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)

    for col in ["negative_share", "toxic_share", "importance_score"]:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0.0)

    return events, discussions, messages, discussion_messages, event_discussions


def resolve_merge_map(merges: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    if merges is None or merges.empty:
        return mapping

    raw = dict(zip(merges["source_event_id"], merges["target_event_id"]))

    def resolve(x: str) -> str:
        seen = set()
        while x in raw and x not in seen:
            seen.add(x)
            x = raw[x]
        return x

    for source in raw:
        mapping[source] = resolve(source)
    return mapping


def normalize_title_for_auto_merge(title: str) -> str:
    """Normalize event title so visually identical topics are merged in the dashboard."""
    value = str(title or "").strip().lower().replace("ё", "е")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*[—–-]\s*волна\s*\d+\s*$", "", value)
    value = value.strip(" .,:;!?'\"«»()[]{}")
    return value

def normalize_manual_tags(tags: str) -> str:
    """Return tags in the same pipe-separated format as generated events."""
    raw = str(tags or "").replace(";", ",").replace("|", ",")
    items = []
    seen = set()
    for item in raw.split(","):
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            continue
        key = item.lower().replace("ё", "е")
        if key not in seen:
            seen.add(key)
            items.append(item)
    return "|".join(items)


def append_manual_events(events: pd.DataFrame, manual_events: pd.DataFrame) -> pd.DataFrame:
    """Append user-created information events to the generated event table."""
    if manual_events is None or manual_events.empty:
        return events

    events = events.copy()
    rows = []
    existing_ids = set(events.get("event_id", pd.Series(dtype=str)).astype(str))
    for _, row in manual_events.iterrows():
        event_id = str(row.get("event_id", "")).strip()
        if not event_id or event_id in existing_ids:
            continue
        rows.append({
            "event_id": event_id,
            "event_title": str(row.get("title", "")).strip() or "Новый инфоповод",
            "event_summary": str(row.get("summary", "")).strip(),
            "main_tag": "ручной",
            "main_tags": normalize_manual_tags(row.get("main_tags", "")),
            "keywords": "",
            "key_phrases": "",
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "discussion_count": 0,
            "message_count": 0,
            "chat_count": 0,
            "author_count": 0,
            "negative_count": 0,
            "toxic_count": 0,
            "negative_share": 0.0,
            "toxic_share": 0.0,
            "importance_score": 0.0,
            "status": str(row.get("status", "")).strip() or "новый",
            "is_hidden": str(row.get("hidden", "0")).lower() in ["1", "true", "yes", "да"],
            "is_manual": True,
        })

    if not rows:
        return events

    manual_df = pd.DataFrame(rows)
    for col in events.columns:
        if col not in manual_df.columns:
            manual_df[col] = np.nan
    for col in manual_df.columns:
        if col not in events.columns:
            events[col] = np.nan

    return pd.concat([events, manual_df[events.columns]], ignore_index=True)





SUMMARY_STOPWORDS = {
    "это", "как", "что", "или", "для", "при", "про", "без", "есть", "нет", "все", "уже", "еще", "ещё",
    "там", "тут", "они", "она", "его", "мне", "нам", "вам", "тебя", "себя", "так", "такой", "такая",
    "если", "когда", "куда", "где", "кто", "чем", "надо", "нужно", "можно", "нельзя", "будет", "были",
    "был", "была", "будут", "этот", "эта", "эти", "эту", "того", "тоже", "очень", "просто",
    "сейчас", "сегодня", "вчера", "завтра", "потом", "теперь", "больше", "меньше", "через", "после",
    "водитель", "водители", "такси", "чат", "чате", "сообщение", "сказал", "говорят", "пишут",
    "яндекс", "yandex", "про", "таксометр", "машина", "заказ", "заказы",
}

SUMMARY_RULES = [
    ("законопроект и новые требования к работе такси", [r"законопроект\w*", r"\bзакон\w*", r"регулирован\w*", r"требован\w*", r"минтранс", r"госдум", r"реестр", r"разрешени\w*"]),
    ("налоги, патенты и самозанятость водителей", [r"налог\w*", r"патент\w*", r"самозанят\w*", r"нпд", r"деклараци\w*", r"фнс", r"штраф\w*"]),
    ("повышение или снижение коэффициентов", [r"коэфф\w*", r"кэф\w*", r"повыш\w*.{0,40}коэфф\w*", r"сниз\w*.{0,40}коэфф\w*", r"вырос\w*.{0,40}коэфф\w*"]),
    ("приоритет, тарифы и распределение заказов", [r"приоритет\w*", r"тариф\w*", r"эконом", r"комфорт", r"комисси\w*", r"раздач\w* заказ\w*", r"распределен\w* заказ\w*"]),
    ("невозможность загрузить или установить обновление", [r"обновлен\w*", r"обновить", r"скачать", r"загруз\w*", r"установ\w*", r"верси\w*", r"апдейт"]),
    ("сбои, ошибки и зависания приложения", [r"сбой\w*", r"ошибк\w*", r"не работает", r"завис\w*", r"вылета\w*", r"лага\w*", r"глюч\w*", r"баг\w*"]),
    ("проблемы с заказами, отменами и назначением поездок", [r"отмен\w*", r"назнач\w*", r"принять заказ", r"заказ\w*.{0,40}не приход", r"не дает заказ", r"цепочк\w*", r"подач\w*"]),
    ("блокировки, доступ к аккаунту и проверки", [r"блокиров\w*", r"заблок\w*", r"аккаунт\w*", r"доступ\w*", r"провер\w*", r"верификац\w*", r"фотоконтроль"]),
    ("оплата, выплаты и удержания", [r"оплат\w*", r"выплат\w*", r"деньг\w*", r"баланс\w*", r"удерж\w*", r"перевод\w*", r"компенсац\w*"]),
    ("забастовка, бойкот и коллективные действия", [r"забастов\w*", r"бойкот\w*", r"стачк\w*", r"не выход\w* на лини\w*", r"акци\w* протест\w*"]),
    ("WB Такси, условия запуска и сравнение с агрегаторами", [r"wb\s*такси", r"wildberries", r"вайлдбер\w*", r"вб\s*такси", r"wb"]),
    ("Фастен и альтернативные сервисы для водителей", [r"фастен", r"fasten", r"альтернатив\w* агрегатор\w*", r"нов\w* сервис\w*"]),
    ("карты, адреса, геолокация и навигация", [r"карт\w*", r"адрес\w*", r"геолокац\w*", r"gps", r"навигатор", r"точк\w* подач\w*", r"маршрут\w*"]),
    ("аэропорты, очереди и заказы из аэропорта", [r"аэропорт\w*", r"шереметьево", r"домодедово", r"внуково", r"пулково", r"очеред\w*"]),
    ("детские кресла и требования к заказам с детьми", [r"детск\w* кресл\w*", r"кресл\w*", r"ребен\w*", r"ребён\w*", r"дет\w* тариф\w*"]),
]


def normalize_text_for_summary(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|t\.me/\S+", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def summary_tokens(text: str) -> list[str]:
    text = normalize_text_for_summary(text)
    tokens = re.findall(r"[а-яa-z0-9]{3,}", text)
    return [t for t in tokens if t not in SUMMARY_STOPWORDS and not t.isdigit()]


def top_readable_phrases(texts, top_n: int = 5) -> list[str]:
    """Return short readable frequent phrases as fallback details for summaries."""
    counter: dict[str, int] = {}
    for text in texts:
        tokens = summary_tokens(str(text)[:2500])
        for a, b in zip(tokens, tokens[1:]):
            if a == b:
                continue
            phrase = f"{a} {b}"
            counter[phrase] = counter.get(phrase, 0) + 1
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [p for p, c in ranked[:top_n] if c >= 2]


def extract_summary_mentions(event_title: str, main_tags: str, event_messages: pd.DataFrame, keywords: str = "", phrases: str = "") -> list[str]:
    """Extract user-facing thesis bullets from the most frequent signals in event messages."""
    text_series = event_messages.get("text_clean", pd.Series(dtype=str)).dropna().astype(str)
    if text_series.empty:
        base = [p.strip() for p in str(phrases or "").split("|") if p.strip()]
        base += [k.strip() for k in str(keywords or "").split("|") if k.strip()]
        return base[:5]

    full_text = "\n".join(text_series.head(350).tolist())
    normalized = normalize_text_for_summary(full_text)

    scored: list[tuple[int, str]] = []
    for label, patterns in SUMMARY_RULES:
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, normalized, flags=re.IGNORECASE))
        if count > 0:
            scored.append((count, label))

    title_tags = normalize_text_for_summary(f"{event_title} {main_tags}")
    boosted = []
    for count, label in scored:
        boost = 2 if any(token in title_tags for token in summary_tokens(label)[:2]) else 0
        boosted.append((count + boost, label))

    mentions = [label for _, label in sorted(boosted, key=lambda x: x[0], reverse=True)]

    for phrase in top_readable_phrases(text_series, top_n=5):
        phrase = phrase.strip()
        if phrase and all(phrase not in m for m in mentions):
            mentions.append(phrase)
        if len(mentions) >= 6:
            break

    clean: list[str] = []
    seen: set[str] = set()
    for item in mentions:
        key = normalize_text_for_summary(item)
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append(item)
        if len(clean) >= 6:
            break

    return clean


def build_event_description(event_title: str, main_tags: str, event_messages: pd.DataFrame, keywords: str = "", phrases: str = "") -> str:
    """Build concise thesis-style summary for table/card display."""
    mentions = extract_summary_mentions(event_title, main_tags, event_messages, keywords=keywords, phrases=phrases)
    if not mentions:
        return "В теме обсуждались связанные сообщения по выбранному инфоповоду."
    return "В теме обсуждались: " + "; ".join(mentions[:6]) + "."

def build_auto_title_merge_map(events: pd.DataFrame) -> dict[str, str]:
    """Map duplicate event titles to one representative event id.

    This is intentionally done at the dashboard layer: source data stays detailed,
    while the user sees one consolidated topic for identical event titles.
    """
    if events.empty or "final_event_id" not in events.columns or "event_title" not in events.columns:
        return {}

    work_events = events.copy()
    if "is_manual" in work_events.columns:
        work_events = work_events[~work_events["is_manual"].astype(str).str.lower().isin(["true", "1", "yes", "да"])]

    representatives = (
        work_events.groupby("final_event_id", as_index=False)
        .agg(
            event_title=("event_title", "first"),
            message_count=("message_count", "sum"),
            importance_score=("importance_score", "max"),
        )
    )
    representatives["title_key"] = representatives["event_title"].apply(normalize_title_for_auto_merge)
    representatives = representatives[representatives["title_key"].astype(str).str.len() > 0]

    title_merge_map: dict[str, str] = {}
    for _, group in representatives.groupby("title_key", sort=False):
        if len(group) <= 1:
            continue
        ordered = group.sort_values(["message_count", "importance_score"], ascending=False)
        target_id = str(ordered.iloc[0]["final_event_id"])
        for source_id in ordered["final_event_id"].astype(str):
            title_merge_map[source_id] = target_id

    return title_merge_map


def apply_manual_edits(
    events: pd.DataFrame,
    messages: pd.DataFrame,
    discussion_messages: pd.DataFrame,
    event_discussions: pd.DataFrame,
    overrides: pd.DataFrame,
    merges: pd.DataFrame,
    message_overrides: pd.DataFrame,
    message_exclusions: pd.DataFrame | None = None,
    manual_events: pd.DataFrame | None = None,
):
    events = append_manual_events(events.copy(), manual_events)
    links = event_discussions.copy()
    msg_links = discussion_messages.merge(links, on="discussion_id", how="left")

    merge_map = resolve_merge_map(merges)

    def apply_manual_event_merge(event_id):
        return merge_map.get(event_id, event_id)

    links["event_id"] = links["event_id"].apply(apply_manual_event_merge)
    msg_links["event_id"] = msg_links["event_id"].apply(apply_manual_event_merge)
    events["final_event_id"] = events["event_id"].apply(apply_manual_event_merge)

    auto_title_merge_map = build_auto_title_merge_map(events)

    def canonical_event_id(event_id):
        event_id = merge_map.get(event_id, event_id)
        return auto_title_merge_map.get(str(event_id), event_id)

    if auto_title_merge_map:
        links["event_id"] = links["event_id"].apply(canonical_event_id)
        msg_links["event_id"] = msg_links["event_id"].apply(canonical_event_id)
        events["final_event_id"] = events["final_event_id"].apply(canonical_event_id)

    if message_overrides is not None and not message_overrides.empty:
        move_map = {
            r["message_id"]: canonical_event_id(r["target_event_id"])
            for _, r in message_overrides.iterrows()
            if str(r.get("target_event_id", "")).strip()
        }
        if move_map:
            msg_links["event_id"] = msg_links.apply(
                lambda r: move_map.get(r["message_id"], r["event_id"]),
                axis=1,
            )

    if message_exclusions is not None and not message_exclusions.empty and len(msg_links):
        exclusions = message_exclusions.copy()
        exclusions["event_id"] = exclusions["event_id"].apply(canonical_event_id)
        excluded_pairs = set(
            zip(
                exclusions["message_id"].astype(str),
                exclusions["event_id"].astype(str),
            )
        )
        msg_links = msg_links[
            ~msg_links.apply(
                lambda r: (str(r.get("message_id", "")), str(r.get("event_id", ""))) in excluded_pairs,
                axis=1,
            )
        ]

    msg_event = (
        msg_links[["message_id", "event_id"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"event_id": "final_event_id"})
    )

    enriched_messages = messages.merge(msg_event, on="message_id", how="left")

    if message_overrides is not None and not message_overrides.empty:
        hidden_msg = set(
            message_overrides.loc[
                message_overrides["hidden"].astype(str).isin(["1", "true", "True"]),
                "message_id",
            ]
        )
        enriched_messages["message_hidden"] = enriched_messages["message_id"].isin(hidden_msg)
    else:
        enriched_messages["message_hidden"] = False

    rows = []
    for final_id, group in events.groupby("final_event_id", sort=False):
        target = events[events["event_id"] == final_id]
        base = target.iloc[0] if len(target) else group.iloc[0]

        group_messages = enriched_messages[
            (enriched_messages["final_event_id"] == final_id) & (~enriched_messages["message_hidden"].astype(bool))
        ]

        all_tags = sorted(set(t for tags in group.get("main_tags", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
        if len(group_messages) and "tags" in group_messages.columns:
            message_tags = sorted(set(t for tags in group_messages["tags"].fillna("") for t in str(tags).split("|") if t.strip()))
            all_tags = sorted(set(all_tags) | set(message_tags))
        keywords = sorted(set(t for tags in group.get("keywords", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
        phrases = sorted(set(t for tags in group.get("key_phrases", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))

        start_date = group_messages["datetime"].min() if "datetime" in group_messages and len(group_messages) else group["start_date"].min()
        end_date = group_messages["datetime"].max() if "datetime" in group_messages and len(group_messages) else group["end_date"].max()

        msg_count = int(group_messages["message_id"].nunique()) if len(group_messages) else int(group["message_count"].sum())
        chat_count = int(group_messages["chat_id"].nunique()) if "chat_id" in group_messages and len(group_messages) else int(group["chat_count"].sum())
        author_count = int(group_messages["author_id"].nunique()) if "author_id" in group_messages and len(group_messages) else int(group["author_count"].sum())
        negative_count = int(group_messages["is_negative"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_negative" in group_messages and len(group_messages) else int(group["negative_count"].sum())
        toxic_count = int(group_messages["is_toxic"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_toxic" in group_messages and len(group_messages) else int(group["toxic_count"].sum())

        event_title = str(base.get("event_title", ""))
        manual_summary = str(base.get("event_summary", "") or "").strip() if str(base.get("is_manual", "")).lower() in ["true", "1", "yes", "да"] else ""
        event_summary = manual_summary or build_event_description(
            event_title,
            "|".join(all_tags),
            group_messages,
            keywords="|".join(keywords),
            phrases="|".join(phrases),
        )

        source_event_ids = sorted({str(x) for x in group.get("event_id", pd.Series(dtype=str)).dropna().astype(str).tolist()})

        rows.append({
            "event_id": final_id,
            "source_event_ids": "|".join(source_event_ids),
            "source_event_count": len(source_event_ids),
            "event_title": event_title,
            "event_summary": event_summary,
            "main_tag": base.get("main_tag", ""),
            "main_tags": "|".join(all_tags),
            "keywords": "|".join(keywords),
            "key_phrases": "|".join(phrases),
            "start_date": start_date,
            "end_date": end_date,
            "discussion_count": int(links[links["event_id"] == final_id]["discussion_id"].nunique()) if len(links) else 0,
            "message_count": msg_count,
            "chat_count": chat_count,
            "author_count": author_count,
            "negative_count": negative_count,
            "toxic_count": toxic_count,
            "importance_score": float(group["importance_score"].max()),
            "status": base.get("status", "новый"),
            "is_hidden": bool(base.get("is_hidden", False)) if str(base.get("is_manual", "")).lower() in ["true", "1", "yes", "да"] else False,
            "is_manual": str(base.get("is_manual", "")).lower() in ["true", "1", "yes", "да"],
        })

    visible_events = pd.DataFrame(rows)

    if overrides is not None and not overrides.empty and len(visible_events):
        ov = overrides.set_index("event_id")
        for idx, row in visible_events.iterrows():
            event_id = row["event_id"]
            if event_id not in ov.index:
                continue
            o = ov.loc[event_id]
            if str(o.get("title", "")).strip():
                visible_events.at[idx, "event_title"] = o["title"]
            if str(o.get("summary", "")).strip():
                visible_events.at[idx, "event_summary"] = o["summary"]
            if str(o.get("status", "")).strip():
                visible_events.at[idx, "status"] = o["status"]
            if str(o.get("priority", "")).strip():
                visible_events.at[idx, "priority"] = o["priority"]
            hidden_val = str(o.get("hidden", "0")).lower()
            visible_events.at[idx, "is_hidden"] = hidden_val in ["1", "true", "yes", "да"]

    if len(visible_events):
        visible_events["negative_share"] = np.where(
            visible_events["message_count"] > 0,
            visible_events["negative_count"] / visible_events["message_count"],
            0,
        )
        visible_events["toxic_share"] = np.where(
            visible_events["message_count"] > 0,
            visible_events["toxic_count"] / visible_events["message_count"],
            0,
        )
        visible_events = visible_events.sort_values(["importance_score", "message_count"], ascending=False)

    return visible_events, enriched_messages


def format_pct(value) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "0%"


def format_date(value) -> str:
    """Return a user-facing date without time: DD.MM.YYYY."""
    dt = pd.to_datetime(value, errors="coerce")
    return dt.strftime("%d.%m.%Y") if pd.notna(dt) else ""


def format_date_series(series: pd.Series) -> pd.Series:
    """Format pandas datetime/string series as DD.MM.YYYY strings for display tables."""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%d.%m.%Y").fillna("")


def format_period(row: pd.Series) -> str:
    start = pd.to_datetime(row.get("start_date"), errors="coerce")
    end = pd.to_datetime(row.get("end_date"), errors="coerce")
    start_s = format_date(start)
    end_s = format_date(end)

    if not start_s:
        return end_s
    if not end_s or start_s == end_s:
        return start_s
    return f"{start_s} — {end_s}"


def get_selected_rows(event) -> list[int]:
    try:
        return list(event.selection.rows)
    except Exception:
        try:
            return list(event.get("selection", {}).get("rows", []))
        except Exception:
            return []


def normalize_search_query(value: str) -> str:
    """Normalize user-entered text search query for Russian text."""
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def filter_messages_by_word(messages: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return visible messages whose text contains the entered word or phrase."""
    q = normalize_search_query(query)
    if not q or "text_clean" not in messages.columns:
        return messages.iloc[0:0].copy()

    work = messages.copy()
    if "message_hidden" in work.columns:
        work = work[~work["message_hidden"].astype(bool)]

    text = (
        work["text_clean"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("ё", "е", regex=False)
    )
    return work[text.str.contains(q, regex=False, na=False)].copy()


def apply_filters(events: pd.DataFrame, messages: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    st.sidebar.header("Фильтры")

    filtered = events.copy()
    empty_messages = messages.iloc[0:0].copy() if isinstance(messages, pd.DataFrame) else pd.DataFrame()
    if filtered.empty:
        return filtered, "", empty_messages

    word_query = st.sidebar.text_input(
        "Слово в тексте сообщений",
        placeholder="например: налог, коэффициент, обновление",
        help="Фильтр ищет введенное слово или фразу именно в текстах сообщений, а не в названии темы.",
    )
    word_matches = filter_messages_by_word(messages, word_query) if word_query else empty_messages

    all_tags = sorted(set(t for tags in filtered.get("main_tags", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
    selected_tags = st.sidebar.multiselect("Теги", all_tags)

    statuses = [s for s in STATUS_OPTIONS if s in set(filtered.get("status", pd.Series(dtype=str)).fillna("").astype(str))]
    selected_statuses = st.sidebar.multiselect("Статус", statuses)

    only_attention = st.sidebar.checkbox("Только требующие внимания", value=False)

    with st.sidebar.expander("Дополнительно", expanded=False):
        show_hidden = st.checkbox("Показывать скрытые", value=False)
        negative_only = st.checkbox("Только с негативом", value=False)
        min_importance = st.slider(
            "Минимальная важность",
            0.0,
            float(max(filtered["importance_score"].max(), 1.0)),
            0.0,
        )

    if not show_hidden and "is_hidden" in filtered.columns:
        filtered = filtered[~filtered["is_hidden"].astype(bool)]

    if word_query:
        matched_event_ids = set(word_matches.get("final_event_id", pd.Series(dtype=str)).dropna().astype(str))
        filtered = filtered[filtered["event_id"].astype(str).isin(matched_event_ids)]

    if selected_tags:
        filtered = filtered[
            filtered["main_tags"].fillna("").apply(
                lambda x: bool(set(selected_tags) & {t.strip() for t in str(x).split("|") if t.strip()})
            )
        ]

    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]

    filtered = filtered[filtered["importance_score"] >= min_importance]

    if negative_only:
        filtered = filtered[filtered["negative_count"] > 0]

    if only_attention:
        filtered = filtered[
            (filtered["importance_score"] >= filtered["importance_score"].quantile(0.70))
            | (filtered["negative_share"] >= 0.25)
            | (filtered["toxic_share"] >= 0.10)
            | (filtered["main_tags"].fillna("").str.contains("Забастовка|Законы", regex=True))
        ]

    return filtered.sort_values(["importance_score", "message_count"], ascending=False), word_query, word_matches

def create_manual_event_form(conn, key_prefix: str = "manual_event", *, compact: bool = False) -> str | None:
    """Render a form for creating a user-defined information event."""
    with st.form(f"{key_prefix}_form"):
        title = st.text_input("Название инфоповода", placeholder="Например: Проблемы с детскими креслами")
        summary = st.text_area(
            "Описание",
            placeholder="Например: В теме обсуждались: требования к детским креслам; отказы от заказов с детьми; штрафы.",
            height=90 if compact else 130,
        )
        tags = st.text_input("Теги", placeholder="через запятую: Яндекс, детские кресла, тарифы")
        status = st.selectbox(
            "Статус",
            STATUS_OPTIONS,
            index=STATUS_OPTIONS.index("новый") if "новый" in STATUS_OPTIONS else 0,
            key=f"{key_prefix}_status",
        )
        note = st.text_input("Комментарий модератора", value="", key=f"{key_prefix}_note")
        submitted = st.form_submit_button("Создать инфоповод")
        if submitted:
            try:
                new_id = create_manual_event(
                    conn,
                    title=title,
                    summary=summary,
                    status=status,
                    main_tags=normalize_manual_tags(tags),
                    note=note,
                )
                st.success("Инфоповод создан. Теперь в него можно переносить сообщения.")
                st.cache_data.clear()
                return new_id
            except Exception as e:
                st.error(str(e))
    return None


def show_manual_event_creator(conn):
    with st.sidebar.expander("Создать инфоповод", expanded=False):
        st.caption("Используйте, если нужной темы нет в списке. Новый инфоповод появится в таблице и в списке для переноса сообщений.")
        new_id = create_manual_event_form(conn, key_prefix="sidebar_create_event", compact=True)
        if new_id:
            st.rerun()


def show_kpis(events: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Инфоповодов", f"{len(events):,}".replace(",", " "))
    c2.metric("Сообщений", f"{int(events['message_count'].sum()) if len(events) else 0:,}".replace(",", " "))
    c3.metric("Чатов", int(events["chat_count"].max()) if len(events) else 0)
    c4.metric("Негатив", format_pct(events["negative_count"].sum() / events["message_count"].sum()) if len(events) and events["message_count"].sum() else "0%")
    c5.metric("Высокая важность", int((events["importance_score"] >= events["importance_score"].quantile(0.75)).sum()) if len(events) else 0)



def _safe_int_delta(value) -> str:
    try:
        return f"{int(value):+d}"
    except Exception:
        return "0"



def _safe_pct_delta(current: float, previous: float) -> str:
    try:
        if previous in [0, None] or pd.isna(previous):
            return "н/д"
        return f"{((current - previous) / previous) * 100:+.0f}%"
    except Exception:
        return "н/д"



def build_period_comparison_df(messages: pd.DataFrame, period_ids: list[str], periods_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build period-level comparison metrics from visible messages."""
    if messages is None or messages.empty or "period_id" not in messages.columns:
        return pd.DataFrame()

    work = messages.copy()
    if period_ids:
        work = work[work["period_id"].astype(str).isin([str(x) for x in period_ids])]
    if work.empty:
        return pd.DataFrame()

    if "message_hidden" in work.columns:
        work = work[~work["message_hidden"].astype(bool)]
    if work.empty:
        return pd.DataFrame()

    if "datetime" in work.columns:
        work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")

    if "is_negative" in work.columns:
        work["__is_negative"] = work["is_negative"].astype(str).str.lower().isin(["true", "1", "yes", "да"]).astype(int)
    elif "sentiment" in work.columns:
        work["__is_negative"] = work["sentiment"].astype(str).str.lower().str.contains("neg").astype(int)
    else:
        work["__is_negative"] = 0

    agg_dict = {
        "message_count": ("period_id", "size"),
        "negative_count": ("__is_negative", "sum"),
    }
    if "datetime" in work.columns:
        agg_dict["date_from"] = ("datetime", "min")
        agg_dict["date_to"] = ("datetime", "max")
    if "chat_title" in work.columns:
        agg_dict["chat_count"] = ("chat_title", pd.Series.nunique)

    summary = work.groupby("period_id", dropna=False).agg(**agg_dict).reset_index()
    summary["negative_share"] = np.where(
        summary["message_count"] > 0,
        summary["negative_count"] / summary["message_count"],
        0.0,
    )

    if periods_meta is not None and not periods_meta.empty and "period_id" in periods_meta.columns:
        meta_cols = [c for c in ["period_id", "period_name", "date_from", "date_to", "uploaded_at"] if c in periods_meta.columns]
        meta = periods_meta[meta_cols].copy().drop_duplicates(subset=["period_id"])
        summary = summary.merge(meta, on="period_id", how="left", suffixes=("", "_meta"))
        if "date_from_meta" in summary.columns:
            summary["date_from"] = summary["date_from_meta"].combine_first(summary.get("date_from"))
        if "date_to_meta" in summary.columns:
            summary["date_to"] = summary["date_to_meta"].combine_first(summary.get("date_to"))
        drop_cols = [c for c in ["date_from_meta", "date_to_meta"] if c in summary.columns]
        if drop_cols:
            summary = summary.drop(columns=drop_cols)
    else:
        summary["period_name"] = summary["period_id"]

    # Normalize period names to plain strings. Supabase/JSON values can occasionally
    # come back as mixed objects; sorting mixed Python objects may crash pandas.
    if "period_name" not in summary.columns:
        summary["period_name"] = summary["period_id"]
    summary["period_name"] = summary["period_name"].where(
        summary["period_name"].notna(),
        summary["period_id"],
    ).map(lambda value: "" if pd.isna(value) else str(value))

    # Build a stable string sort key instead of sorting raw datetimes.
    # This avoids pandas TypeError when selected periods contain a mix of
    # timezone-aware, timezone-naive, empty, or malformed dates.
    date_from_series = summary["date_from"] if "date_from" in summary.columns else pd.Series([pd.NaT] * len(summary), index=summary.index)
    sort_date = pd.to_datetime(date_from_series, errors="coerce", utc=True)
    if "uploaded_at" in summary.columns:
        uploaded_at = pd.to_datetime(summary["uploaded_at"], errors="coerce", utc=True)
        sort_date = sort_date.combine_first(uploaded_at)

    summary["sort_date"] = sort_date.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("9999-12-31 23:59:59")
    summary["period_name_sort"] = summary["period_name"].fillna("").astype(str)
    summary = summary.sort_values(
        ["sort_date", "period_name_sort"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["negative_share_pct"] = (summary["negative_share"] * 100).round(1)
    summary["label"] = summary["period_name"]
    return summary



def render_period_comparison(messages: pd.DataFrame, period_ids: list[str]):
    """Render visual comparison across two or more selected periods."""
    if not period_ids or len(period_ids) < 2:
        return

    try:
        periods_meta = list_periods()
    except Exception:
        periods_meta = pd.DataFrame()

    summary = build_period_comparison_df(messages, period_ids, periods_meta)
    if summary.empty or len(summary) < 2:
        return

    latest = summary.iloc[-1]
    previous = summary.iloc[-2]
    msg_delta = int(latest["message_count"] - previous["message_count"])
    neg_count_delta = int(latest["negative_count"] - previous["negative_count"])
    neg_share_delta_pp = float((latest["negative_share"] - previous["negative_share"]) * 100)

    st.subheader("Сравнение периодов")
    st.caption(
        f"Сейчас сравниваются {len(summary)} период(а/ов). Последний период: {latest['period_name']}. "
        f"Сравнение ведется с предыдущим периодом: {previous['period_name']}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Сообщения · {latest['period_name']}",
        f"{int(latest['message_count']):,}".replace(",", " "),
        delta=_safe_int_delta(msg_delta),
        help="Разница по общему количеству сообщений относительно предыдущего выбранного периода.",
    )
    c2.metric(
        "Изменение сообщений",
        _safe_pct_delta(float(latest["message_count"]), float(previous["message_count"])),
        help="Процентное изменение количества сообщений относительно предыдущего периода.",
    )
    c3.metric(
        f"Негатив · {latest['period_name']}",
        format_pct(latest["negative_share"]),
        delta=f"{neg_share_delta_pp:+.1f} п.п.",
        help="Доля негативных сообщений и ее изменение в процентных пунктах относительно предыдущего периода.",
    )
    c4.metric(
        "Негативных сообщений",
        f"{int(latest['negative_count']):,}".replace(",", " "),
        delta=_safe_int_delta(neg_count_delta),
        help="Изменение числа негативных сообщений относительно предыдущего периода.",
    )

    chart_source = summary[["label", "message_count", "negative_share_pct"]].copy().rename(columns={
        "label": "Период",
        "message_count": "Сообщений",
        "negative_share_pct": "Негатив, %",
    })

    left, right = st.columns(2)
    with left:
        st.caption("Общее количество сообщений по периодам")
        st.bar_chart(chart_source.set_index("Период")[["Сообщений"]], use_container_width=True)
    with right:
        st.caption("Доля негатива по периодам")
        st.line_chart(chart_source.set_index("Период")[["Негатив, %"]], use_container_width=True)

    table = summary.copy()
    table["Период"] = table["period_name"]
    table["Начало"] = format_date_series(table.get("date_from", pd.Series(dtype=str)))
    table["Конец"] = format_date_series(table.get("date_to", pd.Series(dtype=str)))
    table["Сообщений"] = table["message_count"].astype(int)
    table["Негативных"] = table["negative_count"].astype(int)
    table["Доля негатива"] = table["negative_share"].apply(format_pct)
    display_cols = [c for c in ["Период", "Начало", "Конец", "Сообщений", "Негативных", "Доля негатива"] if c in table.columns]
    st.dataframe(table[display_cols], use_container_width=True, hide_index=True)



def word_message_results_table(messages: pd.DataFrame, events: pd.DataFrame, query: str):
    """Show all messages that match the sidebar word filter."""
    if not str(query or "").strip():
        return

    st.subheader(f"Сообщения со словом: «{query}»")

    if messages.empty:
        st.info("Сообщений с таким словом в тексте не найдено.")
        return

    table = messages.copy().sort_values("datetime", ascending=False)
    table["Дата"] = format_date_series(table.get("datetime", pd.Series(dtype=str)))
    table["Чат"] = table.get("chat_title", "")
    table["Автор"] = table.get("author", "")
    table["Текст"] = table.get("text_clean", "").fillna("").astype(str).str.slice(0, 500)
    event_title_map = events.set_index("event_id")["event_title"].to_dict() if len(events) else {}
    table["Инфоповод"] = table.get("final_event_id", "").map(event_title_map).fillna("")
    table["Ссылка"] = table.get("message_link", "")

    st.caption(f"Найдено сообщений: {len(table):,}".replace(",", " "))
    cols = ["Дата", "Чат", "Автор", "Инфоповод", "Текст", "Ссылка"]
    cols = [c for c in cols if c in table.columns]

    st.dataframe(
        table[cols],
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Текст": st.column_config.TextColumn(width="large"),
            "Инфоповод": st.column_config.TextColumn(width="medium"),
            "Ссылка": st.column_config.LinkColumn("Ссылка"),
        },
    )


def event_table(events: pd.DataFrame) -> str | None:
    if events.empty:
        st.info("Нет инфоповодов по выбранным фильтрам.")
        return None

    table = events.copy()
    table["Период"] = table.apply(format_period, axis=1)
    table["Негатив"] = table["negative_share"].apply(format_pct)
    table["Токсичность"] = table["toxic_share"].apply(format_pct)
    table["Теги"] = table["main_tags"].fillna("").astype(str).str.replace("|", ", ", regex=False)
    table["Название"] = table["event_title"]
    table["Описание"] = table["event_summary"]
    table["Сообщений"] = table["message_count"]
    table["Чатов"] = table["chat_count"]
    table["Авторов"] = table["author_count"]
    table["Важность"] = table["importance_score"].round(1)
    table["Статус"] = table["status"]

    display_cols = [
        "Название",
        "Описание",
        "Теги",
        "Период",
        "Сообщений",
        "Чатов",
        "Негатив",
        "Важность",
        "Статус",
    ]

    st.subheader("Инфоповоды")
    selected = st.dataframe(
        table[display_cols],
        use_container_width=True,
        height=520,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Описание": st.column_config.TextColumn(width="large"),
            "Название": st.column_config.TextColumn(width="medium"),
            "Теги": st.column_config.TextColumn(width="medium"),
            "Сообщений": st.column_config.NumberColumn(format="%d"),
            "Чатов": st.column_config.NumberColumn(format="%d"),
            "Важность": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    rows = get_selected_rows(selected)
    if rows:
        return table.iloc[rows[0]]["event_id"]
    return table.iloc[0]["event_id"]


def message_preview_cards(event_messages: pd.DataFrame, limit: int = 8):
    if event_messages.empty:
        st.info("Сообщения не найдены.")
        return

    work = event_messages.copy()
    work["text_len"] = work["text_clean"].fillna("").astype(str).str.len()
    if "is_negative" in work.columns:
        work["_rank_negative"] = work["is_negative"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    else:
        work["_rank_negative"] = 0

    sample = (
        work.sort_values(["_rank_negative", "text_len"], ascending=False)
        .head(limit)
        .sort_values("datetime")
    )

    for _, row in sample.iterrows():
        when = format_date(row.get("datetime"))
        chat = row.get("chat_title", "")
        author = row.get("author", "")
        text = str(row.get("text_clean", "")).strip()
        sentiment = str(row.get("sentiment", "")).strip()
        link = str(row.get("message_link", "")).strip()

        st.markdown(
            f"""
<div style="padding: 0.75rem 0; border-bottom: 1px solid rgba(128,128,128,.25);">
  <div style="font-size: 0.88rem; opacity: .75;">{when} · {chat} · {author} · {sentiment}</div>
  <div style="margin-top: .25rem; white-space: pre-wrap;">{text[:1200]}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if link.startswith("http"):
            st.markdown(f"[Открыть сообщение]({link})")


def show_event_card(event_id: str, events: pd.DataFrame, messages: pd.DataFrame, conn, can_edit: bool = True):
    selected = events[events["event_id"] == event_id]
    if selected.empty:
        st.warning("Инфоповод не найден.")
        return

    ev = selected.iloc[0]
    event_messages = messages[
        (messages["final_event_id"] == event_id)
        & (~messages.get("message_hidden", pd.Series([False] * len(messages))).astype(bool))
    ].copy()
    event_messages = event_messages.sort_values("datetime")

    st.markdown("---")
    st.header(ev["event_title"])
    st.info(str(ev.get("event_summary", "")))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сообщений", int(ev.get("message_count", 0)))
    c2.metric("Чатов", int(ev.get("chat_count", 0)))
    c3.metric("Авторов", int(ev.get("author_count", 0)))
    c4.metric("Негатив", format_pct(ev.get("negative_share", 0)))
    c5.metric("Важность", round(float(ev.get("importance_score", 0)), 1))

    tags = str(ev.get("main_tags", "")).replace("|", " · ")

    st.caption(f"Теги: {tags}")

    tab_names = ["Ключевые сообщения", "Вся лента"] + (["Правки"] if can_edit else [])
    tabs = st.tabs(tab_names)
    tab_messages = tabs[0]
    tab_all = tabs[1]
    tab_edit = tabs[2] if can_edit else None

    with tab_messages:
        message_preview_cards(event_messages, limit=10)

    with tab_all:
        if event_messages.empty:
            st.info("Сообщения не найдены.")
        else:
            table = event_messages.copy()
            table["Текст"] = table["text_clean"].fillna("").astype(str).str.slice(0, 300)
            table["Дата"] = format_date_series(table["datetime"])
            table["Чат"] = table.get("chat_title", "")
            table["Автор"] = table.get("author", "")
            table["Тональность"] = table.get("sentiment", "")
            table["Теги"] = table.get("tags", "")

            table["Ссылка"] = table.get("message_link", "")
            cols = ["Дата", "Чат", "Автор", "Тональность", "Теги", "Текст", "Ссылка"]
            msg_select = st.dataframe(
                table[cols],
                use_container_width=True,
                height=420,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Ссылка": st.column_config.LinkColumn("Ссылка"),
                    "Текст": st.column_config.TextColumn(width="large"),
                },
            )

            rows = get_selected_rows(msg_select)
            if rows:
                row = table.iloc[rows[0]]
                st.markdown("#### Полный текст")
                st.write(row.get("text_clean", ""))
                if can_edit:
                    with st.expander("Перенести, исключить или скрыть сообщение", expanded=False):
                        st.caption(
                            "«Нерелевант» убирает сообщение только из текущего инфоповода. "
                            "Сообщение остается в базе и поиске."
                        )
                        target_options = events[["event_id", "event_title", "message_count"]].copy()
                        target_options["label"] = target_options.apply(
                            lambda r: f"{str(r['event_title'])[:110]} · {int(r.get('message_count', 0))} сообщ.",
                            axis=1,
                        )
                        current_matches = target_options.index[target_options["event_id"] == event_id].tolist()
                        current_idx = int(current_matches[0]) if current_matches else 0
                        target_label = st.selectbox("Перенести в инфоповод", target_options["label"].tolist(), index=current_idx)
                        msg_note = st.text_input("Комментарий", value="", key=f"msg_note_{event_id}_{row['message_id']}")
                        col_a, col_b, col_c = st.columns(3)
                        if col_a.button("Перенести", key=f"move_{event_id}_{row['message_id']}"):
                            target_id = target_options.loc[target_options["label"] == target_label, "event_id"].iloc[0]
                            move_message(conn, row["message_id"], target_id, note=msg_note)
                            st.success("Сообщение перенесено.")
                            st.cache_data.clear()
                            st.rerun()
                        if col_b.button("Нерелевант", key=f"irrelevant_{event_id}_{row['message_id']}"):
                            mark_message_irrelevant(conn, event_id, row["message_id"], reason=msg_note)
                            st.success("Сообщение исключено из текущего инфоповода как нерелевантное.")
                            st.cache_data.clear()
                            st.rerun()
                        if col_c.button("Скрыть везде", key=f"hide_{event_id}_{row['message_id']}"):
                            hide_message(conn, row["message_id"], hidden=True, note=msg_note)
                            st.success("Сообщение скрыто во всех разделах дашборда.")
                            st.cache_data.clear()
                            st.rerun()

                        st.markdown("##### Создать новый инфоповод для этого сообщения")
                        st.caption("Если подходящей темы нет в списке, создайте новую — выбранное сообщение сразу будет перенесено туда.")
                        with st.form(f"create_event_for_message_{event_id}_{row['message_id']}"):
                            new_title = st.text_input("Название новой темы", key=f"new_event_title_{event_id}_{row['message_id']}")
                            new_summary = st.text_area(
                                "Описание новой темы",
                                value="",
                                height=90,
                                key=f"new_event_summary_{event_id}_{row['message_id']}",
                            )
                            new_tags = st.text_input(
                                "Теги новой темы",
                                value=str(row.get("tags", "")).replace("|", ", "),
                                key=f"new_event_tags_{event_id}_{row['message_id']}",
                            )
                            create_and_move = st.form_submit_button("Создать и перенести сообщение")
                            if create_and_move:
                                try:
                                    new_event_id = create_manual_event(
                                        conn,
                                        title=new_title,
                                        summary=new_summary,
                                        status="новый",
                                        main_tags=normalize_manual_tags(new_tags),
                                        note=f"Создано из сообщения {row['message_id']}",
                                    )
                                    move_message(conn, row["message_id"], new_event_id, note="Перенесено в новый инфоповод")
                                    st.success("Новый инфоповод создан, сообщение перенесено.")
                                    st.cache_data.clear()
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

    if can_edit and tab_edit is not None:
        with tab_edit:
            st.markdown("#### Ручная правка инфоповода")
            st.caption("Описание можно скорректировать вручную. После сохранения оно будет показываться в таблице и карточке вместо автоматического описания.")
            with st.form(f"edit_event_{event_id}"):
                title = st.text_input("Название", value=str(ev.get("event_title", "")))
                summary = st.text_area("Описание", value=str(ev.get("event_summary", "")), height=150, help="Например: В теме обсуждались: законопроект; повышение коэффициентов; невозможность загрузить обновление.")
                status = st.selectbox(
                    "Статус",
                    STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(ev.get("status")) if ev.get("status") in STATUS_OPTIONS else 0,
                )
                hidden = st.checkbox("Скрыть инфоповод", value=bool(ev.get("is_hidden", False)))
                note = st.text_area("Комментарий модератора", value="", height=80)
                submitted = st.form_submit_button("Сохранить")
                if submitted:
                    save_event_override(conn, event_id, title=title, summary=summary, status=status, hidden=hidden, note=note)
                    st.success("Правки сохранены.")
                    st.cache_data.clear()
                    st.rerun()

            st.markdown("#### Нерелевантные сообщения")
            exclusions = get_message_exclusions(conn)
            event_exclusions = exclusions[exclusions["event_id"] == event_id] if not exclusions.empty else exclusions
            if event_exclusions.empty:
                st.info("Для этого инфоповода пока нет сообщений, помеченных как нерелевантные.")
            else:
                excluded_ids = event_exclusions["message_id"].astype(str).tolist()
                excluded_messages = messages[messages["message_id"].astype(str).isin(excluded_ids)].copy()
                if excluded_messages.empty:
                    st.info("Есть записи об исключениях, но сообщения не найдены в текущей выгрузке.")
                else:
                    reason_map = event_exclusions.set_index("message_id")["reason"].to_dict()
                    excluded_messages["Дата"] = format_date_series(excluded_messages.get("datetime", pd.Series(dtype=str)))
                    excluded_messages["Чат"] = excluded_messages.get("chat_title", "")
                    excluded_messages["Автор"] = excluded_messages.get("author", "")
                    excluded_messages["Причина"] = excluded_messages["message_id"].map(reason_map).fillna("")
                    excluded_messages["Текст"] = excluded_messages["text_clean"].fillna("").astype(str).str.slice(0, 350)
                    excluded_cols = ["Дата", "Чат", "Автор", "Причина", "Текст"]
                    excluded_select = st.dataframe(
                        excluded_messages[excluded_cols],
                        use_container_width=True,
                        height=220,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "Текст": st.column_config.TextColumn(width="large"),
                        },
                    )
                    excluded_rows = get_selected_rows(excluded_select)
                    if excluded_rows:
                        restored_row = excluded_messages.iloc[excluded_rows[0]]
                        st.write(restored_row.get("text_clean", ""))
                        if st.button("Вернуть сообщение в инфоповод", key=f"restore_{event_id}_{restored_row['message_id']}"):
                            restore_message_relevance(conn, event_id, restored_row["message_id"])
                            st.success("Сообщение возвращено в инфоповод.")
                            st.cache_data.clear()
                            st.rerun()

            st.markdown("#### Объединить с другим инфоповодом")
            st.caption("Объединение переносит всю видимую тему целиком: все исходные события/волны обсуждения, которые скрыты под выбранной строкой.")
            candidates = events[events["event_id"] != event_id][["event_id", "event_title", "message_count"]].copy()
            candidates["label"] = candidates.apply(
                lambda r: f"{str(r['event_title'])[:120]} · {int(r.get('message_count', 0))} сообщ.",
                axis=1,
            )
            target_label = st.selectbox("Целевой инфоповод", candidates["label"].tolist() if len(candidates) else [])
            reason = st.text_input("Причина объединения", value="")
            if st.button("Объединить", disabled=not bool(target_label)):
                target_id = candidates.loc[candidates["label"] == target_label, "event_id"].iloc[0]
                try:
                    source_ids_raw = str(ev.get("source_event_ids", "") or "")
                    source_ids = [x.strip() for x in source_ids_raw.split("|") if x.strip()] or [str(event_id)]
                    merged_count = 0
                    for source_id in source_ids:
                        if source_id == str(target_id):
                            continue
                        merge_events(conn, source_event_id=source_id, target_event_id=target_id, reason=reason)
                        merged_count += 1
                    st.success(f"Инфоповоды объединены: перенесено внутренних событий/волн: {merged_count}.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


def show_message_search(messages: pd.DataFrame, events: pd.DataFrame, conn):
    st.subheader("Поиск по сообщениям")

    col1, col2, col3 = st.columns(3)
    q = col1.text_input("Текст")
    tag = col2.text_input("Тег")
    chat = col3.text_input("Чат")

    filtered = messages[~messages.get("message_hidden", pd.Series([False] * len(messages))).astype(bool)].copy()

    if q:
        filtered = filtered[filtered["text_clean"].fillna("").str.lower().str.contains(q.lower(), regex=False)]
    if tag and "tags" in filtered.columns:
        filtered = filtered[filtered["tags"].fillna("").str.lower().str.contains(tag.lower(), regex=False)]
    if chat and "chat_title" in filtered.columns:
        filtered = filtered[filtered["chat_title"].fillna("").str.lower().str.contains(chat.lower(), regex=False)]

    filtered = filtered.sort_values("datetime", ascending=False).head(500)
    filtered["Текст"] = filtered["text_clean"].fillna("").astype(str).str.slice(0, 350)
    filtered["Дата"] = format_date_series(filtered["datetime"])
    filtered["Чат"] = filtered.get("chat_title", "")
    filtered["Автор"] = filtered.get("author", "")
    event_title_map = events.set_index("event_id")["event_title"].to_dict() if len(events) else {}
    filtered["Инфоповод"] = filtered.get("final_event_id", "").map(event_title_map).fillna("")
    filtered["Теги"] = filtered.get("tags", "")
    filtered["Тональность"] = filtered.get("sentiment", "")
    filtered["Ссылка"] = filtered.get("message_link", "")
    cols = ["Дата", "Чат", "Автор", "Теги", "Тональность", "Инфоповод", "Текст", "Ссылка"]
    cols = [c for c in cols if c in filtered.columns]

    st.dataframe(
        filtered[cols],
        use_container_width=True,
        height=640,
        hide_index=True,
        column_config={
            "Ссылка": st.column_config.LinkColumn("Ссылка"),
            "Текст": st.column_config.TextColumn(width="large"),
        },
    )



def safe_upload_name(filename: str) -> str:
    source = Path(filename or "uploaded.csv")
    name = re.sub(r"[^0-9A-Za-zА-Яа-я_. -]+", "_", source.stem).strip("._-") or "uploaded"
    suffix = source.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".xlsm"}:
        suffix = ".csv"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{name}{suffix}"


def read_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}



@st.cache_data(show_spinner=False, ttl=60)
def load_generated_tables_remote(period_ids: tuple[str, ...]):
    return load_generated_tables_from_supabase(list(period_ids))


def render_period_selector() -> list[str]:
    """Render Supabase period selector and return selected period IDs."""
    periods = list_periods()
    if periods.empty:
        st.sidebar.info("В Supabase пока нет сохраненных периодов. Загрузите файл в разделе «Загрузка файла».")
        return []

    def period_label(row) -> str:
        name = str(row.get("period_name", "") or row.get("period_id", ""))
        date_from = format_date(row.get("date_from")) if "date_from" in row else ""
        date_to = format_date(row.get("date_to")) if "date_to" in row else ""
        dates = f" · {date_from}–{date_to}" if date_from or date_to else ""
        return f"{name}{dates}"

    labels = {str(row["period_id"]): period_label(row) for _, row in periods.iterrows()}
    options = list(labels.keys())
    default = options[:1]
    selected = st.sidebar.multiselect(
        "Периоды",
        options=options,
        default=default,
        format_func=lambda x: labels.get(x, x),
        help="Можно выбрать один период или несколько. При выборе нескольких периодов дашборд объединит данные.",
    )
    if not selected:
        st.sidebar.warning("Выберите хотя бы один период.")
    return selected
def show_upload_page(data_dir: Path, upload_dir: Path):
    st.subheader("Загрузка файла нового периода")
    st.write(
        "Загрузите новый CSV или Excel-файл из Brand Analytics, Медиалогии или другой системы. "
        "Дашборд приведет данные к единому формату и пересоберет сообщения, обсуждения и инфоповоды."
    )

    persistent_enabled = supabase_configured()
    selected_period_ids: list[str] = []
    if persistent_enabled:
        st.success("Постоянное хранение включено: исходные файлы и обработанные периоды будут сохраняться в Supabase.")
    else:
        st.warning(
            "Supabase не настроен. Загруженные файлы и пересчитанные таблицы сохранятся только в текущем runtime Streamlit Cloud "
            "и после redeploy могут сброситься к версии из GitHub."
        )

    manifest = read_manifest(data_dir)
    if manifest:
        c1, c2, c3 = st.columns(3)
        c1.metric("Текущих сообщений", f"{int(manifest.get('rows_messages', 0)):,}".replace(",", " "))
        c2.metric("Текущих обсуждений", f"{int(manifest.get('rows_discussions', 0)):,}".replace(",", " "))
        c3.metric("Текущих инфоповодов", f"{int(manifest.get('rows_events', 0)):,}".replace(",", " "))

    default_period_name = datetime.now().strftime("%d.%m.%Y")
    period_name = st.text_input(
        "Название периода",
        value=default_period_name,
        help="Например: 24.04.2026–30.04.2026. Так период будет отображаться в фильтре.",
    )

    mode = st.radio(
        "Режим загрузки",
        [
            "Заменить текущую выборку новым файлом",
            "Добавить файл в историю загрузок и пересобрать все загруженные периоды",
        ],
        help=(
            "В режиме добавления дашборд объединит все CSV/Excel-файлы из папки data/uploads. "
            "Если исходный файл старого периода не был загружен в историю, он не попадет в пересборку."
        ),
    )

    uploaded = st.file_uploader("Файл нового периода", type=["csv", "xlsx", "xls", "xlsm"])

    with st.expander("Параметры алгоритма", expanded=False):
        col1, col2 = st.columns(2)
        window_minutes = col1.number_input("Окно обсуждения, минут", min_value=10, max_value=240, value=60, step=5)
        similarity_threshold = col2.slider("Порог похожести", min_value=0.10, max_value=0.60, value=0.28, step=0.01)
        event_gap_hours = col1.number_input("Разрыв между волнами, часов", min_value=0.5, max_value=24.0, value=3.0, step=0.5)
        event_window_hours = col2.number_input("Максимальная длина волны, часов", min_value=2.0, max_value=72.0, value=16.0, step=2.0)
        cluster_method = st.selectbox("Метод кластеризации", ["tfidf", "none"], index=0)

    col_a, col_b = st.columns([1, 2])
    run = col_a.button("Загрузить и пересобрать", type="primary", disabled=uploaded is None)

    if run and uploaded is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        saved_path = upload_dir / safe_upload_name(uploaded.name)
        saved_path.write_bytes(uploaded.getbuffer())

        try:
            with st.spinner("Читаю файл и пересобираю инфоповоды…"):
                if persistent_enabled or mode.startswith("Заменить"):
                    manifest = run_preprocess(
                        input_path=saved_path,
                        output=data_dir,
                        window_minutes=int(window_minutes),
                        cluster_method=cluster_method,
                        similarity_threshold=float(similarity_threshold),
                        event_gap_hours=float(event_gap_hours),
                        event_window_hours=float(event_window_hours),
                    )
                else:
                    csv_paths = sorted([p for p in upload_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"}])
                    raw_parts = []
                    used_files = []
                    for path in csv_paths:
                        try:
                            raw_parts.append(read_source_csv(path))
                            used_files.append(path.name)
                        except Exception as e:
                            st.warning(f"Файл {path.name} пропущен: {e}")
                    if not raw_parts:
                        st.error("Не удалось прочитать ни один файл.")
                        return
                    combined = pd.concat(raw_parts, ignore_index=True)
                    manifest = run_preprocess_from_dataframe(
                        raw=combined,
                        output=data_dir,
                        source_file="; ".join(used_files),
                        window_minutes=int(window_minutes),
                        cluster_method=cluster_method,
                        similarity_threshold=float(similarity_threshold),
                        event_gap_hours=float(event_gap_hours),
                        event_window_hours=float(event_window_hours),
                    )


            if persistent_enabled:
                period_title = str(period_name or uploaded.name).strip() or uploaded.name
                period_id = make_period_id(period_title, uploaded.name)
                with st.spinner("Сохраняю период в Supabase…"):
                    save_processed_tables_from_dir(
                        data_dir,
                        period_id=period_id,
                        period_name=period_title,
                        source_filename=uploaded.name,
                        manifest=manifest,
                        replace=True,
                    )
                    try:
                        save_uploaded_csv_to_storage(period_id, uploaded.name, bytes(uploaded.getbuffer()))
                    except Exception as storage_error:
                        st.warning(f"Обработанные данные сохранены в Supabase, но исходный файл не удалось сохранить в Storage: {storage_error}")
                st.success(f"Период сохранен в Supabase: {period_title}")

            st.cache_data.clear()
            st.success(
                "Данные пересобраны: "
                f"{manifest.get('rows_messages', 0)} сообщений, "
                f"{manifest.get('rows_discussions', 0)} обсуждений, "
                f"{manifest.get('rows_events', 0)} инфоповодов."
            )
            st.info("Перейдите в раздел «Инфоповоды» или обновите страницу, чтобы увидеть новую выборку.")
            if st.button("Открыть обновленные инфоповоды"):
                st.rerun()
        except Exception as e:
            st.error("Не удалось обработать файл.")
            st.exception(e)

    with st.expander("История загруженных файлов", expanded=False):
        if persistent_enabled:
            try:
                periods = list_periods()
                if periods.empty:
                    st.write("Пока нет сохраненных периодов в Supabase.")
                else:
                    view = periods.copy()
                    for col in ["date_from", "date_to", "uploaded_at"]:
                        if col in view.columns:
                            view[col] = pd.to_datetime(view[col], errors="coerce").dt.strftime("%d.%m.%Y").fillna("")
                    cols = [c for c in ["period_name", "date_from", "date_to", "source_filename", "uploaded_at", "status"] if c in view.columns]
                    st.dataframe(view[cols].rename(columns={
                        "period_name": "Период",
                        "date_from": "Начало",
                        "date_to": "Конец",
                        "source_filename": "Файл",
                        "uploaded_at": "Загружено",
                        "status": "Статус",
                    }), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Не удалось получить историю периодов из Supabase: {e}")
        else:
            upload_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(upload_dir.glob("*.csv"), reverse=True)
            if not files:
                st.write("Пока нет загруженных файлов.")
            else:
                history = pd.DataFrame({
                    "Файл": [f.name for f in files],
                    "Размер, КБ": [round(f.stat().st_size / 1024, 1) for f in files],
                    "Дата загрузки": [datetime.fromtimestamp(f.stat().st_mtime).strftime("%d.%m.%Y") for f in files],
                })
                st.dataframe(history, use_container_width=True, hide_index=True)
                if st.button("Очистить историю загруженных файлов"):
                    for f in files:
                        f.unlink(missing_ok=True)
                    st.success("История загрузок очищена.")
                    st.rerun()


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Дайджест водительских чатов",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Дайджест водительских чатов")
    st.caption("Версия 2.2: добавлен универсальный импорт CSV/Excel из Медиалогии, Brand Analytics и других систем")

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)
    upload_dir = Path(args.upload_dir)
    persistent_enabled = supabase_configured()
    selected_period_ids: list[str] = []

    if persistent_enabled:
        st.sidebar.success("Хранилище: Supabase")
    else:
        st.sidebar.info("Хранилище: локальные файлы")

    can_edit = render_admin_mode()
    pages = ["Инфоповоды", "Поиск сообщений"] + (["Загрузка файла"] if can_edit else [])
    page = st.sidebar.radio("Раздел", pages, label_visibility="collapsed")

    if page == "Загрузка файла":
        show_upload_page(data_dir, upload_dir)
        return

    conn = connect(db_path)

    if persistent_enabled:
        try:
            selected_period_ids = render_period_selector()
            if not selected_period_ids:
                st.info("Пока нет выбранных периодов. Откройте «Загрузка файла» и сохраните первый период в Supabase.")
                st.stop()
            events_raw, discussions, messages, discussion_messages, event_discussions = load_generated_tables_remote(tuple(selected_period_ids))
        except Exception as e:
            st.error("Не удалось загрузить данные из Supabase.")
            st.exception(e)
            st.stop()
    else:
        if not data_dir.exists():
            st.error(f"Папка с обработанными данными не найдена: {data_dir}")
            st.info("Откройте раздел «Загрузка файла» и загрузите исходный файл для первой сборки дашборда.")
            st.stop()
        events_raw, discussions, messages, discussion_messages, event_discussions = load_generated_tables(str(data_dir))

    overrides = get_event_overrides(conn)
    merges = get_event_merges(conn)
    msg_overrides = get_message_overrides(conn)
    msg_exclusions = get_message_exclusions(conn)
    manual_events = get_manual_events(conn)

    events, enriched_messages = apply_manual_edits(
        events_raw,
        messages,
        discussion_messages,
        event_discussions,
        overrides,
        merges,
        msg_overrides,
        msg_exclusions,
        manual_events,
    )

    if page == "Поиск сообщений":
        show_message_search(enriched_messages, events, conn)
        return

    if can_edit:
        show_manual_event_creator(conn)
    else:
        st.sidebar.caption("Загрузка файлов и ручная модерация доступны после входа администратора.")

    if persistent_enabled and len(selected_period_ids) >= 2:
        render_period_comparison(enriched_messages, selected_period_ids)

    filtered_events, word_query, word_matches = apply_filters(events, enriched_messages)
    show_kpis(filtered_events)
    if word_query:
        word_message_results_table(word_matches, events, word_query)
    selected_event_id = event_table(filtered_events)
    if selected_event_id:
        show_event_card(selected_event_id, events, enriched_messages, conn, can_edit=can_edit)


if __name__ == "__main__":
    main()
