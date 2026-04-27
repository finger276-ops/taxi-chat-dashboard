"""
Streamlit dashboard for taxi chat information events.

Run:
    streamlit run src/app.py -- --data-dir data/processed --db-path data/manual_actions.sqlite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import json
import textwrap

import numpy as np
import pandas as pd
import streamlit as st

from io_utils import read_table
from manual_db import (
    connect,
    get_event_overrides,
    get_event_merges,
    get_message_overrides,
    save_event_override,
    merge_events,
    move_message,
    hide_message,
)
from settings import STATUS_OPTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default=os.getenv("DASHBOARD_DATA_DIR", "data/processed"))
    parser.add_argument("--db-path", default=os.getenv("DASHBOARD_DB_PATH", "data/manual_actions.sqlite"))
    args, _ = parser.parse_known_args()
    return args


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

    if "is_hidden" in events.columns:
        events["is_hidden"] = events["is_hidden"].astype(str).str.lower().isin(["true", "1", "yes", "да"])

    for col in ["message_count", "chat_count", "author_count", "negative_count", "toxic_count"]:
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


def apply_manual_edits(
    events: pd.DataFrame,
    discussions: pd.DataFrame,
    messages: pd.DataFrame,
    discussion_messages: pd.DataFrame,
    event_discussions: pd.DataFrame,
    overrides: pd.DataFrame,
    merges: pd.DataFrame,
    message_overrides: pd.DataFrame,
):
    events = events.copy()
    links = event_discussions.copy()
    msg_links = discussion_messages.merge(links, on="discussion_id", how="left")

    merge_map = resolve_merge_map(merges)
    if merge_map:
        links["event_id"] = links["event_id"].apply(lambda x: merge_map.get(x, x))
        msg_links["event_id"] = msg_links["event_id"].apply(lambda x: merge_map.get(x, x))
        events["final_event_id"] = events["event_id"].apply(lambda x: merge_map.get(x, x))
    else:
        events["final_event_id"] = events["event_id"]

    # Apply message-level moves.
    if message_overrides is not None and not message_overrides.empty:
        move_map = {
            r["message_id"]: r["target_event_id"]
            for _, r in message_overrides.iterrows()
            if str(r.get("target_event_id", "")).strip()
        }
        if move_map:
            msg_links["event_id"] = msg_links.apply(
                lambda r: move_map.get(r["message_id"], r["event_id"]),
                axis=1,
            )

    msg_event = (
        msg_links[["message_id", "event_id"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"event_id": "final_event_id"})
    )

    enriched_messages = messages.merge(msg_event, on="message_id", how="left")

    if message_overrides is not None and not message_overrides.empty:
        hidden_msg = set(message_overrides.loc[message_overrides["hidden"].astype(str).isin(["1", "true", "True"]), "message_id"])
        enriched_messages["message_hidden"] = enriched_messages["message_id"].isin(hidden_msg)
    else:
        enriched_messages["message_hidden"] = False

    # Aggregate generated events after merges.
    rows = []
    for final_id, group in events.groupby("final_event_id", sort=False):
        target = events[events["event_id"] == final_id]
        base = target.iloc[0] if len(target) else group.iloc[0]

        group_messages = enriched_messages[
            (enriched_messages["final_event_id"] == final_id) & (~enriched_messages["message_hidden"])
        ]

        all_tags = sorted(set(t for tags in group["main_tags"].fillna("") for t in str(tags).split("|") if t.strip()))
        start_date = group_messages["datetime"].min() if "datetime" in group_messages else group["start_date"].min()
        end_date = group_messages["datetime"].max() if "datetime" in group_messages else group["end_date"].max()

        rows.append({
            "event_id": final_id,
            "event_title": base.get("event_title", ""),
            "event_summary": base.get("event_summary", ""),
            "main_tag": base.get("main_tag", ""),
            "main_tags": "|".join(all_tags),
            "keywords": base.get("keywords", ""),
            "start_date": start_date,
            "end_date": end_date,
            "discussion_count": int(links[links["event_id"] == final_id]["discussion_id"].nunique()) if len(links) else 0,
            "message_count": int(group_messages["message_id"].nunique()) if len(group_messages) else int(group["message_count"].sum()),
            "chat_count": int(group_messages["chat_id"].nunique()) if "chat_id" in group_messages else int(group["chat_count"].sum()),
            "author_count": int(group_messages["author_id"].nunique()) if "author_id" in group_messages else int(group["author_count"].sum()),
            "negative_count": int(group_messages["is_negative"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_negative" in group_messages else int(group["negative_count"].sum()),
            "toxic_count": int(group_messages["is_toxic"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_toxic" in group_messages else int(group["toxic_count"].sum()),
            "importance_score": float(group["importance_score"].max()),
            "status": base.get("status", "новый"),
            "is_hidden": False,
        })

    visible_events = pd.DataFrame(rows)

    # Apply event overrides.
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

    return visible_events, enriched_messages, links


def format_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def event_filters(events: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Фильтры инфоповодов")

    show_hidden = st.sidebar.checkbox("Показывать скрытые", value=False)
    if not show_hidden and "is_hidden" in events.columns:
        events = events[~events["is_hidden"].astype(bool)]

    all_tags = sorted(set(t for tags in events.get("main_tags", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
    selected_tags = st.sidebar.multiselect("Теги", all_tags)

    statuses = sorted(set(events.get("status", pd.Series(dtype=str)).fillna("").astype(str))) if len(events) else []
    selected_statuses = st.sidebar.multiselect("Статус", statuses)

    q = st.sidebar.text_input("Поиск по названию / описанию")
    min_importance = st.sidebar.slider("Минимальная важность", 0.0, float(max(events["importance_score"].max(), 1.0)) if len(events) else 1.0, 0.0)

    filtered = events.copy()
    if selected_tags:
        filtered = filtered[
            filtered["main_tags"].fillna("").apply(
                lambda x: any(tag in str(x).split("|") for tag in selected_tags)
            )
        ]
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if q:
        qq = q.lower()
        filtered = filtered[
            filtered["event_title"].fillna("").str.lower().str.contains(qq, regex=False)
            | filtered["event_summary"].fillna("").str.lower().str.contains(qq, regex=False)
            | filtered["keywords"].fillna("").str.lower().str.contains(qq, regex=False)
        ]
    filtered = filtered[filtered["importance_score"] >= min_importance]
    return filtered


def show_overview(events: pd.DataFrame, messages: pd.DataFrame):
    st.subheader("Сводка")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Инфоповодов", len(events))
    col2.metric("Сообщений", int(events["message_count"].sum()) if len(events) else 0)
    col3.metric("Чатов", int(messages["chat_id"].nunique()) if "chat_id" in messages else 0)
    col4.metric("Авторов", int(messages["author_id"].nunique()) if "author_id" in messages else 0)

    col5, col6, col7 = st.columns(3)
    neg = int(events["negative_count"].sum()) if "negative_count" in events else 0
    tox = int(events["toxic_count"].sum()) if "toxic_count" in events else 0
    col5.metric("Негативных сообщений", neg)
    col6.metric("Токсичных сообщений", tox)
    col7.metric("Средняя важность", round(float(events["importance_score"].mean()), 2) if len(events) else 0)

    st.subheader("Топ тегов по инфоповодам")
    tag_counter = {}
    for tags in events.get("main_tags", pd.Series(dtype=str)).fillna(""):
        for tag in str(tags).split("|"):
            tag = tag.strip()
            if tag:
                tag_counter[tag] = tag_counter.get(tag, 0) + 1
    if tag_counter:
        tag_df = pd.DataFrame(
            [{"tag": k, "events": v} for k, v in tag_counter.items()]
        ).sort_values("events", ascending=False)
        st.bar_chart(tag_df.set_index("tag"))
    else:
        st.info("Теги не найдены.")


def show_events_table(events: pd.DataFrame) -> str | None:
    st.subheader("Инфоповоды")

    table = events.copy()
    if len(table) == 0:
        st.info("Нет инфоповодов по выбранным фильтрам.")
        return None

    table["period"] = (
        table["start_date"].dt.strftime("%d.%m %H:%M").fillna("")
        + " — "
        + table["end_date"].dt.strftime("%d.%m %H:%M").fillna("")
    )
    table["negative_share"] = table["negative_share"].apply(format_pct)
    table["toxic_share"] = table["toxic_share"].apply(format_pct)

    show_cols = [
        "event_id",
        "event_title",
        "main_tags",
        "period",
        "message_count",
        "chat_count",
        "author_count",
        "negative_share",
        "toxic_share",
        "importance_score",
        "status",
    ]
    show_cols = [c for c in show_cols if c in table.columns]

    event = st.dataframe(
        table[show_cols],
        use_container_width=True,
        height=430,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    rows = event.selection.rows if hasattr(event, "selection") else []
    if rows:
        return table.iloc[rows[0]]["event_id"]

    return table.iloc[0]["event_id"]


def show_event_card(event_id: str, events: pd.DataFrame, messages: pd.DataFrame, conn):
    selected = events[events["event_id"] == event_id]
    if selected.empty:
        st.warning("Инфоповод не найден.")
        return

    ev = selected.iloc[0]

    st.markdown("---")
    st.subheader(ev["event_title"])
    st.write(ev.get("event_summary", ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сообщений", int(ev.get("message_count", 0)))
    c2.metric("Чатов", int(ev.get("chat_count", 0)))
    c3.metric("Авторов", int(ev.get("author_count", 0)))
    c4.metric("Негатив", format_pct(ev.get("negative_share", 0)))
    c5.metric("Важность", ev.get("importance_score", 0))

    with st.expander("Ручная правка инфоповода", expanded=False):
        with st.form(f"edit_event_{event_id}"):
            title = st.text_input("Название", value=str(ev.get("event_title", "")))
            summary = st.text_area("Описание", value=str(ev.get("event_summary", "")), height=120)
            status = st.selectbox(
                "Статус",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(ev.get("status")) if ev.get("status") in STATUS_OPTIONS else 0,
            )
            hidden = st.checkbox("Скрыть инфоповод", value=bool(ev.get("is_hidden", False)))
            note = st.text_area("Комментарий модератора", value="", height=80)
            submitted = st.form_submit_button("Сохранить правки")
            if submitted:
                save_event_override(conn, event_id, title=title, summary=summary, status=status, hidden=hidden, note=note)
                st.success("Правки сохранены.")
                st.cache_data.clear()
                st.rerun()

    with st.expander("Объединить с другим инфоповодом", expanded=False):
        candidates = events[events["event_id"] != event_id][["event_id", "event_title"]].copy()
        candidates["label"] = candidates["event_id"] + " — " + candidates["event_title"].astype(str).str.slice(0, 100)
        target_label = st.selectbox("Целевой инфоповод", candidates["label"].tolist() if len(candidates) else [])
        reason = st.text_input("Причина объединения", value="")
        if st.button("Объединить выбранный инфоповод с целевым", disabled=not bool(target_label)):
            target_id = target_label.split(" — ", 1)[0]
            try:
                merge_events(conn, source_event_id=event_id, target_event_id=target_id, reason=reason)
                st.success(f"Инфоповод {event_id} объединен с {target_id}.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.subheader("Сообщения внутри инфоповода")
    event_messages = messages[
        (messages["final_event_id"] == event_id)
        & (~messages.get("message_hidden", pd.Series([False] * len(messages))).astype(bool))
    ].copy()

    if event_messages.empty:
        st.info("Сообщения не найдены.")
        return

    event_messages = event_messages.sort_values("datetime")
    event_messages["preview"] = event_messages["text_clean"].fillna("").astype(str).str.slice(0, 250)
    msg_cols = ["message_id", "datetime", "chat_title", "author", "sentiment", "tags", "preview", "message_link"]
    msg_cols = [c for c in msg_cols if c in event_messages.columns]

    msg_event = st.dataframe(
        event_messages[msg_cols],
        use_container_width=True,
        height=360,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    rows = msg_event.selection.rows if hasattr(msg_event, "selection") else []

    if rows:
        row = event_messages.iloc[rows[0]]
        st.markdown("#### Полный текст выбранного сообщения")
        st.write(row.get("text_clean", ""))

        link = row.get("message_link", "")
        if isinstance(link, str) and link.startswith("http"):
            st.markdown(f"[Открыть в Telegram]({link})")

        with st.expander("Ручная правка выбранного сообщения", expanded=False):
            target_options = events[["event_id", "event_title"]].copy()
            target_options["label"] = target_options["event_id"] + " — " + target_options["event_title"].astype(str).str.slice(0, 100)
            current_idx = int(target_options[target_options["event_id"] == event_id].index[0]) if event_id in target_options["event_id"].values else 0
            target_label = st.selectbox("Перенести в инфоповод", target_options["label"].tolist(), index=current_idx)
            msg_note = st.text_input("Комментарий к переносу / скрытию", value="")
            col_a, col_b = st.columns(2)
            if col_a.button("Перенести сообщение"):
                target_id = target_label.split(" — ", 1)[0]
                move_message(conn, row["message_id"], target_id, note=msg_note)
                st.success("Сообщение перенесено.")
                st.cache_data.clear()
                st.rerun()
            if col_b.button("Скрыть сообщение"):
                hide_message(conn, row["message_id"], hidden=True, note=msg_note)
                st.success("Сообщение скрыто.")
                st.cache_data.clear()
                st.rerun()


def show_message_search(messages: pd.DataFrame, events: pd.DataFrame, conn):
    st.subheader("Поиск по сообщениям")

    col1, col2, col3 = st.columns(3)
    q = col1.text_input("Текст")
    tag = col2.text_input("Тег содержит")
    chat = col3.text_input("Чат содержит")

    filtered = messages[~messages.get("message_hidden", pd.Series([False] * len(messages))).astype(bool)].copy()

    if q:
        filtered = filtered[filtered["text_clean"].fillna("").str.lower().str.contains(q.lower(), regex=False)]
    if tag and "tags" in filtered.columns:
        filtered = filtered[filtered["tags"].fillna("").str.lower().str.contains(tag.lower(), regex=False)]
    if chat and "chat_title" in filtered.columns:
        filtered = filtered[filtered["chat_title"].fillna("").str.lower().str.contains(chat.lower(), regex=False)]

    filtered = filtered.sort_values("datetime", ascending=False).head(500)
    filtered["preview"] = filtered["text_clean"].fillna("").astype(str).str.slice(0, 300)
    cols = ["message_id", "datetime", "chat_title", "author", "tags", "sentiment", "final_event_id", "preview", "message_link"]
    cols = [c for c in cols if c in filtered.columns]

    st.dataframe(filtered[cols], use_container_width=True, height=600, hide_index=True)


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Инфоповоды в чатах такси",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Инфоповоды в Telegram-чатах такси")
    st.caption("MVP: сообщения → обсуждения → инфоповоды + ручная модерация")

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)

    if not data_dir.exists():
        st.error(f"Папка с обработанными данными не найдена: {data_dir}")
        st.stop()

    conn = connect(db_path)

    events_raw, discussions, messages, discussion_messages, event_discussions = load_generated_tables(str(data_dir))
    overrides = get_event_overrides(conn)
    merges = get_event_merges(conn)
    msg_overrides = get_message_overrides(conn)

    events, enriched_messages, links = apply_manual_edits(
        events_raw,
        discussions,
        messages,
        discussion_messages,
        event_discussions,
        overrides,
        merges,
        msg_overrides,
    )

    filtered_events = event_filters(events)

    tab1, tab2, tab3 = st.tabs(["Обзор", "Инфоповоды", "Поиск сообщений"])

    with tab1:
        show_overview(filtered_events, enriched_messages)

    with tab2:
        selected_event_id = show_events_table(filtered_events)
        if selected_event_id:
            show_event_card(selected_event_id, filtered_events, enriched_messages, conn)

    with tab3:
        show_message_search(enriched_messages, filtered_events, conn)


if __name__ == "__main__":
    main()
