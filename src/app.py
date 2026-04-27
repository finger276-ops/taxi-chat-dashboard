"""
Streamlit dashboard for taxi chat information events.

Run locally:
    python -m streamlit run src/app.py -- --data-dir data/processed --db-path data/manual_actions.sqlite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from io_utils import read_table
from manual_db import (
    connect,
    get_event_overrides,
    get_event_merges,
    get_message_overrides,
    get_message_exclusions,
    save_event_override,
    merge_events,
    move_message,
    hide_message,
    mark_message_irrelevant,
    restore_message_relevance,
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


def apply_manual_edits(
    events: pd.DataFrame,
    messages: pd.DataFrame,
    discussion_messages: pd.DataFrame,
    event_discussions: pd.DataFrame,
    overrides: pd.DataFrame,
    merges: pd.DataFrame,
    message_overrides: pd.DataFrame,
    message_exclusions: pd.DataFrame | None = None,
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

    if message_exclusions is not None and not message_exclusions.empty and len(msg_links):
        exclusions = message_exclusions.copy()
        exclusions["event_id"] = exclusions["event_id"].apply(lambda x: merge_map.get(x, x))
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
        keywords = sorted(set(t for tags in group.get("keywords", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
        phrases = sorted(set(t for tags in group.get("key_phrases", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))

        start_date = group_messages["datetime"].min() if "datetime" in group_messages and len(group_messages) else group["start_date"].min()
        end_date = group_messages["datetime"].max() if "datetime" in group_messages and len(group_messages) else group["end_date"].max()

        msg_count = int(group_messages["message_id"].nunique()) if len(group_messages) else int(group["message_count"].sum())
        chat_count = int(group_messages["chat_id"].nunique()) if "chat_id" in group_messages and len(group_messages) else int(group["chat_count"].sum())
        author_count = int(group_messages["author_id"].nunique()) if "author_id" in group_messages and len(group_messages) else int(group["author_count"].sum())
        negative_count = int(group_messages["is_negative"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_negative" in group_messages and len(group_messages) else int(group["negative_count"].sum())
        toxic_count = int(group_messages["is_toxic"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_toxic" in group_messages and len(group_messages) else int(group["toxic_count"].sum())

        rows.append({
            "event_id": final_id,
            "event_title": base.get("event_title", ""),
            "event_summary": base.get("event_summary", ""),
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
            "is_hidden": False,
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


def apply_filters(events: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Фильтры")

    filtered = events.copy()
    if filtered.empty:
        return filtered

    q = st.sidebar.text_input("Поиск по теме", placeholder="например: налог, забастовка, WB")
    all_tags = sorted(set(t for tags in filtered.get("main_tags", pd.Series(dtype=str)).fillna("") for t in str(tags).split("|") if t.strip()))
    selected_tags = st.sidebar.multiselect("Теги", all_tags)

    statuses = [s for s in STATUS_OPTIONS if s in set(filtered.get("status", pd.Series(dtype=str)).fillna("").astype(str))]
    selected_statuses = st.sidebar.multiselect("Статус", statuses)

    min_messages = st.sidebar.slider(
        "Минимум сообщений",
        min_value=1,
        max_value=int(max(filtered["message_count"].max(), 1)),
        value=min(3, int(max(filtered["message_count"].max(), 1))),
    )

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

    if q:
        q_low = q.lower()
        haystack = (
            filtered["event_title"].fillna("").astype(str)
            + " "
            + filtered["event_summary"].fillna("").astype(str)
            + " "
            + filtered.get("keywords", pd.Series([""] * len(filtered))).fillna("").astype(str)
            + " "
            + filtered.get("key_phrases", pd.Series([""] * len(filtered))).fillna("").astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(q_low, regex=False, na=False)]

    if selected_tags:
        filtered = filtered[
            filtered["main_tags"].fillna("").apply(
                lambda x: bool(set(selected_tags) & {t.strip() for t in str(x).split("|") if t.strip()})
            )
        ]

    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]

    filtered = filtered[filtered["message_count"] >= min_messages]
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

    return filtered.sort_values(["importance_score", "message_count"], ascending=False)


def show_kpis(events: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Инфоповодов", f"{len(events):,}".replace(",", " "))
    c2.metric("Сообщений", f"{int(events['message_count'].sum()) if len(events) else 0:,}".replace(",", " "))
    c3.metric("Чатов", int(events["chat_count"].max()) if len(events) else 0)
    c4.metric("Негатив", format_pct(events["negative_count"].sum() / events["message_count"].sum()) if len(events) and events["message_count"].sum() else "0%")
    c5.metric("Высокая важность", int((events["importance_score"] >= events["importance_score"].quantile(0.75)).sum()) if len(events) else 0)


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


def show_event_card(event_id: str, events: pd.DataFrame, messages: pd.DataFrame, conn):
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
    st.write(ev.get("event_summary", ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сообщений", int(ev.get("message_count", 0)))
    c2.metric("Чатов", int(ev.get("chat_count", 0)))
    c3.metric("Авторов", int(ev.get("author_count", 0)))
    c4.metric("Негатив", format_pct(ev.get("negative_share", 0)))
    c5.metric("Важность", round(float(ev.get("importance_score", 0)), 1))

    tags = str(ev.get("main_tags", "")).replace("|", " · ")
    keywords = str(ev.get("keywords", "")).replace("|", ", ")
    phrases = str(ev.get("key_phrases", "")).replace("|", ", ")

    st.caption(f"Теги: {tags}")
    if phrases or keywords:
        st.caption(f"Ключевые сигналы: {phrases or keywords}")

    tab_messages, tab_all, tab_edit = st.tabs(["Ключевые сообщения", "Вся лента", "Правки"])

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

    with tab_edit:
        st.markdown("#### Ручная правка инфоповода")
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
                merge_events(conn, source_event_id=event_id, target_event_id=target_id, reason=reason)
                st.success(f"Инфоповод объединен с {target_id}.")
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


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Инфоповоды в чатах такси",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Инфоповоды в Telegram-чатах такси")
    st.caption("Версия 0.6: даты в интерфейсе отображаются без времени в формате ДД.ММ.ГГГГ")

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
    msg_exclusions = get_message_exclusions(conn)

    events, enriched_messages = apply_manual_edits(
        events_raw,
        messages,
        discussion_messages,
        event_discussions,
        overrides,
        merges,
        msg_overrides,
        msg_exclusions,
    )

    page = st.sidebar.radio("Раздел", ["Инфоповоды", "Поиск сообщений"], label_visibility="collapsed")

    if page == "Поиск сообщений":
        show_message_search(enriched_messages, events, conn)
        return

    filtered_events = apply_filters(events)
    show_kpis(filtered_events)
    selected_event_id = event_table(filtered_events)
    if selected_event_id:
        show_event_card(selected_event_id, filtered_events, enriched_messages, conn)


if __name__ == "__main__":
    main()
