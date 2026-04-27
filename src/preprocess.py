"""
Preprocess Telegram chat CSV into normalized analytical tables.

Usage:
    python src/preprocess.py --input data/chats.csv --output data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from settings import (
    TAG_COLUMNS_DEFAULT,
    TEXT_PREFIXES_TO_REMOVE,
    RUSSIAN_STOPWORDS,
    KEYWORD_STOPWORDS,
    CRITICAL_TAG_WEIGHTS,
    TITLE_RULES,
    MICROTOPIC_TITLES,
)
from io_utils import read_source_csv, write_table, write_manifest


def stable_hash(value: str, prefix: str = "") -> str:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}" if prefix else digest


def normalize_spaces(value: str) -> str:
    value = "" if value is None else str(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clean_text(message: str, recognized: str) -> tuple[str, str]:
    message = normalize_spaces(message)
    recognized = normalize_spaces(recognized)

    if message:
        text = message
        source = "message"
    else:
        text = recognized
        source = "recognized"

    for prefix in TEXT_PREFIXES_TO_REMOVE:
        text = text.replace(prefix, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_spaces(text)

    if recognized.startswith("Тексты с изображений"):
        source = "image_ocr" if not message else "message_plus_image_ocr"
    elif recognized.startswith("Расшифровки"):
        source = "transcript" if not message else "message_plus_transcript"

    return text, source


def parse_datetime(series: pd.Series) -> pd.Series:
    prepared = (
        series.fillna("")
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    return pd.to_datetime(prepared, errors="coerce", dayfirst=True)


def detect_tag_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in TAG_COLUMNS_DEFAULT if col in df.columns]


def row_tags(row: pd.Series, tag_cols: list[str]) -> list[str]:
    tags = []
    for tag in tag_cols:
        val = str(row.get(tag, "")).strip().lower()
        if val in {"да", "yes", "true", "1", "+"}:
            tags.append(tag)
    return tags


def normalize_messages(raw: pd.DataFrame, tag_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)

    text_pairs = df.apply(
        lambda r: clean_text(r.get("Сообщение", ""), r.get("Автораспознанный текст", "")),
        axis=1,
    )
    df["text_clean"] = [x[0] for x in text_pairs]
    df["text_source"] = [x[1] for x in text_pairs]

    df["datetime"] = parse_datetime(df.get("Дата", pd.Series([""] * len(df))))
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    # Robust message id: prefer Id сообщения, then link, then row number.
    if "Id сообщения" in df.columns:
        source_id = df["Id сообщения"].where(df["Id сообщения"].str.strip() != "", df.get("Ссылка", ""))
    else:
        source_id = df.get("Ссылка", pd.Series([""] * len(df)))
    df["message_id"] = [
        stable_hash(v if str(v).strip() else f"row_{i}", prefix="m_")
        for i, v in enumerate(source_id.tolist())
    ]

    df["chat_id"] = [
        stable_hash(v or b or "unknown", prefix="c_")
        for v, b in zip(df.get("Профиль блога", ""), df.get("Блог", ""))
    ]

    df["author_id"] = [
        stable_hash(v or a or "unknown", prefix="a_")
        for v, a in zip(df.get("Профиль автора", ""), df.get("Автор", ""))
    ]

    tag_lists = df.apply(lambda r: row_tags(r, tag_cols), axis=1)
    df["tags"] = ["|".join(tags) for tags in tag_lists]
    df["tag_count"] = [len(tags) for tags in tag_lists]

    # Narrow rule-based topic used before clustering. For very short replies,
    # include a small parent-post context, but do not let parent text dominate.
    parent_context = df.get("Текст родительского поста", "").fillna("").astype(str).str.slice(0, 300)
    df["microtopic"] = [
        classify_microtopic((text if len(str(text)) > 45 else f"{text} {parent}"), tags)
        for text, parent, tags in zip(df["text_clean"].astype(str), parent_context, df["tags"].astype(str))
    ]

    def as_int(col_name: str) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        return pd.to_numeric(df[col_name].str.replace(",", ".", regex=False), errors="coerce").fillna(0).astype(int)

    df["duplicate_count"] = as_int("Количество дублей")
    df["views"] = as_int("Просмотры")
    df["engagement"] = as_int("Вовлечённость")

    df["is_negative"] = df.get("Тональность", "").astype(str).str.lower().str.contains("негатив", na=False)
    df["is_toxic"] = df.get("Токсичность", "").astype(str).str.strip().ne("")

    keep_cols = {
        "message_id": "message_id",
        "№": "source_row_no",
        "date": "date",
        "datetime": "datetime",
        "Тип": "message_type",
        "Ссылка": "message_link",
        "Сообщение": "message_raw",
        "Автораспознанный текст": "recognized_raw",
        "text_clean": "text_clean",
        "text_source": "text_source",
        "Текст родительского поста": "parent_text",
        "Ссылка на родительский пост": "parent_link",
        "Дата публикации родительского поста": "parent_date",
        "Площадка": "platform",
        "Тип площадки": "platform_type",
        "Автор": "author",
        "Профиль автора": "author_profile",
        "Тип автора": "author_type",
        "author_id": "author_id",
        "Блог": "chat_title",
        "Профиль блога": "chat_profile",
        "Тип блога": "chat_type",
        "chat_id": "chat_id",
        "Тональность": "sentiment",
        "Токсичность": "toxicity",
        "WOM": "wom",
        "Страна": "country",
        "Регион": "region",
        "Город": "city",
        "Количество дублей": "duplicate_count_raw",
        "duplicate_count": "duplicate_count",
        "views": "views",
        "engagement": "engagement",
        "tags": "tags",
        "tag_count": "tag_count",
        "microtopic": "microtopic",
        "is_negative": "is_negative",
        "is_toxic": "is_toxic",
    }

    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    messages = df[list(available.keys())].rename(columns=available)
    messages["date"] = messages["date"].fillna("")
    messages["datetime"] = pd.to_datetime(messages["datetime"], errors="coerce")

    tag_rows = []
    for message_id, tags in zip(messages["message_id"], messages["tags"]):
        for tag in str(tags).split("|"):
            tag = tag.strip()
            if tag:
                tag_rows.append({"message_id": message_id, "tag": tag})
    message_tags = pd.DataFrame(tag_rows, columns=["message_id", "tag"])

    return messages, message_tags


def tag_set(value: str) -> set[str]:
    return {x.strip() for x in str(value).split("|") if x.strip()}


def tag_signature(value: str) -> str:
    return "|".join(sorted(tag_set(value)))


def regex_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_microtopic(text: str, tags: str | Iterable[str] = "") -> str:
    """
    Rule-based microtopic layer.

    Why it exists: broad tags like «яндекс» or «Коэффициент» are too coarse.
    Before semantic/lexical clustering, we first assign every message/discussion
    to a narrower microtopic and only then allow clustering inside that bucket.
    This reduces unrelated messages inside one information event.
    """
    t = normalize_spaces(text).lower().replace("ё", "е")
    if isinstance(tags, str):
        tag_values = tag_set(tags)
    else:
        tag_values = {str(x).strip() for x in tags if str(x).strip()}

    # Critical and highly specific topics first.
    if "Забастовка" in tag_values or regex_any(t, [r"\bзабастов\w*", r"\bбойкот\w*", r"\bстачк\w*", r"\bмитинг\w*", r"коллективн\w+\s+акци"]):
        return "strike"
    if "WB Такси" in tag_values or regex_any(t, [r"\bwb\b", r"wildberries", r"\bвб\b", r"валбер", r"вайлдбер"]):
        return "wb_launch"
    if "Фастен" in tag_values or regex_any(t, [r"fasten", r"фаст[еэо]н", r"фастон"]):
        return "fasten_service"
    if "Законы и налоги" in tag_values or regex_any(t, [r"налог\w*", r"патент\w*", r"самозанят\w*", r"минтранс", r"реестр\w*", r"закон\w*", r"разрешени\w*", r"лиценз\w*", r"штраф\w*", r"провер\w*"]):
        return "tax_law"

    # Product/app operational issues.
    if regex_any(t, [r"не\s+работа\w*", r"не\s+открыва\w*", r"не\s+груз\w*", r"не\s+заход\w*", r"завис\w*", r"висит", r"сбой\w*", r"ошибк\w*", r"глюк\w*", r"вылета\w*", r"приложени\w*", r"яндекс\s+про"]):
        if regex_any(t, [r"нет\s+заказ\w*", r"заказ\w*\s+не\s+приход", r"пропал\w*\s+заказ", r"заказ\w*\s+пропал", r"распределени\w*\s+заказ"]):
            return "app_orders"
        return "app_bug"

    # Money and account issues.
    if regex_any(t, [r"оплат\w*", r"выплат\w*", r"деньг\w*", r"перевод\w*", r"задолж\w*", r"баланс\w*", r"комисс\w*"]):
        return "payments"
    if regex_any(t, [r"блокир\w*", r"заблок\w*", r"\bбан\b", r"аккаунт\w*", r"доступ\w*", r"деактив\w*", r"профил\w*", r"самозанят\w*\s+не\s+подтверж"]):
        return "account_block"

    # Operational subtopics.
    if regex_any(t, [r"кресл\w*", r"детск\w+", r"ребен\w*", r"ребенк\w*", r"бустер\w*"]):
        return "child_seat"
    if "Коэффициент" in tag_values or regex_any(t, [r"коэф\w*", r"коэффициент\w*", r"\bкэф\w*", r"приоритет\w*", r"тариф\w*", r"ценник\w*", r"стоимост\w*", r"подач\w*", r"комфорт", r"эконом"]):
        return "coeff_priority"
    if regex_any(t, [r"аэропорт\w*", r"пулково", r"шереметьево", r"внуково", r"домодедово"]):
        return "airport"
    if regex_any(t, [r"карт\w*", r"навигатор\w*", r"адрес\w*", r"геолокац\w*", r"локац\w*", r"подъезд\w*", r"maps", r"улиц\w*", r"маршрут\w*"]):
        return "gps_map"
    if regex_any(t, [r"поддержк\w*", r"диспетчер\w*", r"парк\w*", r"таксопарк\w*", r"оператор\w*"]):
        return "support"

    if "яндекс" in tag_values or regex_any(t, [r"яндекс", r"\bяши\b", r"\bяше\b", r"\bяшу\b", r"yandex"]):
        return "general_yandex"
    return "other"


def topic_bucket_for(row: pd.Series) -> str:
    tag = main_tag([row.get("main_tags", row.get("tags", ""))])
    microtopic = str(row.get("microtopic", "other") or "other")
    return f"{tag}::{microtopic}"


def should_start_new_discussion(prev_row: pd.Series, row: pd.Series, window_minutes: int) -> bool:
    if pd.isna(prev_row["datetime"]) or pd.isna(row["datetime"]):
        return False

    gap = row["datetime"] - prev_row["datetime"]
    if gap > pd.Timedelta(minutes=window_minutes):
        return True

    prev_micro = str(prev_row.get("microtopic", "other") or "other")
    curr_micro = str(row.get("microtopic", "other") or "other")
    if prev_micro != curr_micro and gap > pd.Timedelta(minutes=12):
        return True

    prev_tags = tag_set(prev_row.get("tags", ""))
    curr_tags = tag_set(row.get("tags", ""))
    if prev_tags and curr_tags and not (prev_tags & curr_tags) and gap > pd.Timedelta(minutes=8):
        return True

    return False


def make_discussions(messages: pd.DataFrame, window_minutes: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    messages = messages.copy()
    messages["parent_link"] = messages.get("parent_link", "").fillna("").astype(str).str.strip()
    messages["datetime"] = pd.to_datetime(messages["datetime"], errors="coerce")
    messages["sort_date"] = messages["datetime"].fillna(pd.Timestamp("1970-01-01"))

    discussion_links = []

    # 1) Parent post is a useful anchor, but comments under one post often drift
    # into several sub-discussions. Split large parent threads by time, tags and microtopic.
    has_parent = messages["parent_link"].ne("")
    for parent_link, group in messages[has_parent].sort_values(["parent_link", "sort_date", "message_id"]).groupby("parent_link", sort=False):
        current_no = 0
        prev = None
        for _, row in group.iterrows():
            if prev is None or should_start_new_discussion(prev, row, max(25, window_minutes // 2)):
                current_no += 1
            did = stable_hash(f"{parent_link}::{current_no:04d}", prefix="d_parent_")
            discussion_links.append({"discussion_id": did, "message_id": row["message_id"], "discussion_source": "parent_link_segment"})
            prev = row

    # 2) For messages without parent, segment by chat, time, tag overlap and microtopic.
    no_parent = messages[~has_parent].sort_values(["chat_id", "sort_date", "message_id"]).copy()
    for chat_id, group in no_parent.groupby("chat_id", sort=False):
        current_no = 0
        prev = None
        for _, row in group.iterrows():
            if prev is None:
                current_no += 1
            else:
                if should_start_new_discussion(prev, row, window_minutes):
                    current_no += 1
            did = f"d_time_{chat_id}_{current_no:05d}"
            discussion_links.append({"discussion_id": did, "message_id": row["message_id"], "discussion_source": "time_window"})
            prev = row

    discussion_messages = pd.DataFrame(discussion_links)
    enriched = discussion_messages.merge(messages, on="message_id", how="left")

    rows = []
    for did, group in enriched.groupby("discussion_id", sort=False):
        group = group.sort_values(["sort_date", "message_id"])

        tags = sorted(set(t for tags in group["tags"].fillna("") for t in str(tags).split("|") if t.strip()))
        parent_texts = [normalize_spaces(x) for x in group.get("parent_text", pd.Series([], dtype=str)).fillna("").astype(str).unique() if normalize_spaces(x)]
        message_texts = [normalize_spaces(x) for x in group["text_clean"].fillna("").astype(str).tolist() if normalize_spaces(x)]

        # Message text should dominate. Parent text is only a compact context; otherwise
        # comments under the same parent post become artificially too similar.
        parent_block = "\n".join(parent_texts[:1])[:350]
        messages_block = "\n".join(message_texts[:80])
        if len(messages_block) < 120 and parent_block:
            discussion_text = normalize_spaces((messages_block + "\n" + parent_block).strip())
        else:
            discussion_text = normalize_spaces((messages_block + "\n" + parent_block[:180]).strip())

        microtopic_counts = Counter(group.get("microtopic", pd.Series(["other"])).fillna("other").astype(str))
        microtopic = microtopic_counts.most_common(1)[0][0] if microtopic_counts else "other"

        rep_messages = []
        for _, r in group.head(5).iterrows():
            text = normalize_spaces(r.get("text_clean", ""))
            if text:
                rep_messages.append(text[:300])

        rows.append({
            "discussion_id": did,
            "discussion_source": group["discussion_source"].iloc[0],
            "start_date": group["datetime"].min(),
            "end_date": group["datetime"].max(),
            "chat_id": group["chat_id"].iloc[0] if group["chat_id"].nunique() == 1 else "",
            "chat_title": group["chat_title"].iloc[0] if "chat_title" in group else "",
            "parent_link": group["parent_link"].iloc[0] if "parent_link" in group else "",
            "main_tags": "|".join(tags),
            "microtopic": microtopic,
            "topic_bucket": main_tag(["|".join(tags)]) + "::" + microtopic,
            "message_count": int(group["message_id"].nunique()),
            "author_count": int(group["author_id"].nunique()) if "author_id" in group else 0,
            "negative_count": int(group["is_negative"].astype(bool).sum()) if "is_negative" in group else 0,
            "toxic_count": int(group["is_toxic"].astype(bool).sum()) if "is_toxic" in group else 0,
            "discussion_text": discussion_text,
            "representative_messages": "\n---\n".join(rep_messages),
        })

    discussions = pd.DataFrame(rows)
    for col in ["start_date", "end_date"]:
        discussions[col] = pd.to_datetime(discussions[col], errors="coerce")

    return discussions, discussion_messages


def tokenize_ru(text: str, *, for_keywords: bool = False) -> list[str]:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|t\.me/\S+", " ", text)
    tokens = re.findall(r"[а-яa-z0-9]{3,}", text)
    stop = KEYWORD_STOPWORDS if for_keywords else RUSSIAN_STOPWORDS
    return [t for t in tokens if t not in stop and not t.isdigit()]


def top_keywords(texts: Iterable[str], top_n: int = 7) -> list[str]:
    """
    Возвращает чистые ключевые слова для карточки инфоповода.
    В отличие от TF-IDF токенизации, здесь жестче режем мат, бренды и слишком общие слова.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_ru(str(text)[:3500], for_keywords=True))
    return [w for w, _ in counter.most_common(top_n)]


def top_phrases(texts: Iterable[str], top_n: int = 5) -> list[str]:
    """
    Простая вытяжка устойчивых 2-словных фраз.
    Нужна не для ML, а для более понятного названия/описания.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize_ru(str(text)[:3500], for_keywords=True)
        for a, b in zip(tokens, tokens[1:]):
            if a != b:
                counter[f"{a} {b}"] += 1
    return [p for p, c in counter.most_common(top_n) if c >= 2]


def main_tag(tags_series: Iterable[str]) -> str:
    counter: Counter[str] = Counter()
    for tags in tags_series:
        for tag in str(tags).split("|"):
            tag = tag.strip()
            if tag:
                # "яндекс" слишком широкий тег, по возможности уступает более предметным тегам.
                weight = 0.45 if tag == "яндекс" else 1.0
                counter[tag] += weight
    return counter.most_common(1)[0][0] if counter else "Без тега"


def tag_set_from_series(tags_series: Iterable[str]) -> set[str]:
    result = set()
    for tags in tags_series:
        for tag in str(tags).split("|"):
            tag = tag.strip()
            if tag:
                result.add(tag)
    return result


def build_title(
    tag: str,
    keywords: list[str],
    all_tags: Iterable[str] | None = None,
    microtopic: str = "other",
    phrases: list[str] | None = None,
) -> str:
    all_tags = set(all_tags or [])
    kw = set(keywords)
    phrases = phrases or []

    # Microtopic is more precise than a broad tag. Prefer it when available.
    if microtopic in MICROTOPIC_TITLES and microtopic != "other":
        # Keep the title stable and clean. Specific clues are shown in the
        # summary/keywords, not injected into the title where noisy OCR/chat words
        # can make the table look unreliable.
        return MICROTOPIC_TITLES[microtopic]

    for rule in TITLE_RULES:
        if all_tags & set(rule.get("tags", set())) and (not rule.get("keywords") or kw & set(rule.get("keywords", set()))):
            return rule["title"]

    if tag == "Забастовка":
        return "Призывы к забастовке или бойкоту"
    if tag == "Приложение и сбои":
        return "Сбои и проблемы в приложении"
    if tag == "Яндекс Про":
        return "Проблемы с Яндекс Про"
    if tag == "Законы и налоги":
        return "Законы, налоги и регулирование такси"
    if tag == "Коэффициент":
        return "Коэффициенты, приоритет и тарифы"
    if tag == "WB Такси":
        return "Запуск и обсуждение WB Такси"
    if tag == "Фастен":
        return "Обсуждение сервиса Фастен"

    if tag == "яндекс":
        if any(k in kw for k in ["дивиденды", "акции", "акционеры"]):
            return "Финансовые новости и обсуждение Яндекса"
        if any(k in kw for k in ["кресло", "кресла", "детским"]):
            return "Детские кресла и требования к заказам"
        return "Общее обсуждение Яндекса"

    if keywords:
        return f"{tag}: {', '.join(keywords[:3])}"
    return f"Обсуждение: {tag}"


def summarize_event(group: pd.DataFrame, tag: str, keywords: list[str], phrases: list[str]) -> str:
    start = pd.to_datetime(group["start_date"], errors="coerce").min()
    end = pd.to_datetime(group["end_date"], errors="coerce").max()
    start_s = start.strftime("%d.%m.%Y %H:%M") if pd.notna(start) else "неизвестно"
    end_s = end.strftime("%d.%m.%Y %H:%M") if pd.notna(end) else "неизвестно"
    msg_count = int(group["message_count"].sum())
    chat_count = int(group["chat_id"].replace("", np.nan).nunique()) if "chat_id" in group else 0
    author_count = int(group["author_count"].sum()) if "author_count" in group else 0

    details = phrases[:3] or keywords[:5]
    detail_text = ", ".join(details) if details else "без устойчивых ключевых слов"

    return (
        f"Тема «{tag}» обсуждалась с {start_s} по {end_s}. "
        f"Внутри: {msg_count} сообщений, {chat_count} чатов, {author_count} участников. "
        f"Ключевые сигналы: {detail_text}."
    )


def split_labels_by_fixed_time_window(
    labels: pd.Series,
    discussions: pd.DataFrame,
    window_hours: float = 12.0,
) -> pd.Series:
    """
    Дополнительное дробление широких кластеров на временные волны.
    Это защищает от ситуации, когда большой тег вроде «яндекс» или «Коэффициент»
    склеивает обсуждения за несколько дней в один инфоповод.
    """
    if window_hours <= 0 or len(discussions) == 0:
        return labels.astype(int)

    d = discussions.copy()
    d["_label"] = labels.loc[d.index].astype(int)
    d["_start"] = pd.to_datetime(d["start_date"], errors="coerce")
    fallback = pd.Timestamp("1970-01-01")
    d["_bucket"] = (
        d["_start"].fillna(fallback).astype("int64")
        // int(pd.Timedelta(hours=window_hours).value)
    )

    mapping = {}
    next_label = 0
    result = pd.Series(index=d.index, dtype=int)
    for idx, row in d.iterrows():
        key = (int(row["_label"]), int(row["_bucket"]))
        if key not in mapping:
            mapping[key] = next_label
            next_label += 1
        result.loc[idx] = mapping[key]
    return result.astype(int)

def dynamic_threshold(bucket: str, base: float) -> float:
    """Stricter threshold for broad/noisy buckets."""
    if "general_yandex" in bucket or bucket.endswith("::other"):
        return min(0.72, base + 0.12)
    if "coeff_priority" in bucket:
        return min(0.68, base + 0.07)
    if "app_bug" in bucket or "app_orders" in bucket:
        return min(0.66, base + 0.05)
    return base


def cluster_sparse_greedy(
    group: pd.DataFrame,
    threshold: float,
    max_gap_hours: float,
    max_event_span_hours: float,
    max_features: int,
) -> pd.Series:
    """
    Fast clustering inside a narrow topic+time bucket.

    We still use connected components, but only after splitting by microtopic and
    fixed time bucket. This removes the worst source of false merges: long chains
    across broad tags over several days.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(group) == 0:
        return pd.Series([], dtype=int)
    if len(group) == 1:
        return pd.Series([0], index=group.index, dtype=int)

    texts = group["discussion_text"].fillna("").astype(str).tolist()
    min_df = 1 if len(group) < 12 else 2
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_ru,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.72,
        max_features=max_features,
        sublinear_tf=True,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.Series(range(len(group)), index=group.index, dtype=int)

    nonzero_mask = np.asarray(X.getnnz(axis=1) > 0).ravel()
    result = pd.Series(index=group.index, dtype=int)
    if int(nonzero_mask.sum()) < 2:
        result.loc[:] = range(len(group))
        return result.astype(int)

    Xn = X[nonzero_mask]
    sim = Xn @ Xn.T
    sim.setdiag(0)
    sim.data[sim.data < threshold] = 0
    sim.eliminate_zeros()

    _, labels = connected_components(sim, directed=False, return_labels=True)
    result.iloc[np.where(nonzero_mask)[0]] = labels

    next_label = int(labels.max()) + 1 if len(labels) else 0
    for pos in np.where(~nonzero_mask)[0]:
        result.iloc[pos] = next_label
        next_label += 1

    return result.astype(int)


def cluster_discussions_tfidf(
    discussions: pd.DataFrame,
    similarity_threshold: float = 0.38,
    max_features: int = 6000,
    max_gap_hours: float = 0.75,
    max_event_span_hours: float = 8.0,
) -> pd.Series:
    texts = discussions["discussion_text"].fillna("").astype(str)
    if len(texts) == 0:
        return pd.Series([], dtype=int)
    if len(texts) == 1:
        return pd.Series([0], index=discussions.index)

    d = discussions.copy()
    if "topic_bucket" not in d.columns:
        d["topic_bucket"] = d.apply(topic_bucket_for, axis=1)

    # Narrow time bucket before lexical clustering. This is the main protection
    # against putting several independent waves into one information event.
    d["_start"] = pd.to_datetime(d["start_date"], errors="coerce")
    fallback = pd.Timestamp("1970-01-01")
    bucket_hours = max(1.0, float(max_event_span_hours))
    d["_time_bucket"] = (
        d["_start"].fillna(fallback).astype("int64")
        // int(pd.Timedelta(hours=bucket_hours).value)
    )
    d["_cluster_bucket"] = d["topic_bucket"].astype(str) + "::t" + d["_time_bucket"].astype(str)

    result = pd.Series(index=d.index, dtype=int)
    next_label = 0

    # Cluster independently inside narrow topic + time buckets.
    for bucket, group in d.groupby("_cluster_bucket", sort=False):
        topic_part = str(group["topic_bucket"].iloc[0]) if len(group) else str(bucket)
        bucket_threshold = dynamic_threshold(topic_part, similarity_threshold)
        local_labels = cluster_sparse_greedy(
            group,
            threshold=bucket_threshold,
            max_gap_hours=max_gap_hours,
            max_event_span_hours=max_event_span_hours,
            max_features=max_features,
        )
        unique_local = {int(v): i + next_label for i, v in enumerate(sorted(local_labels.unique()))}
        result.loc[group.index] = local_labels.map(unique_local)
        next_label += len(unique_local)

    return result.astype(int)


def cluster_discussions_embeddings(
    discussions: pd.DataFrame,
    similarity_threshold: float = 0.68,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> pd.Series:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    texts = discussions["discussion_text"].fillna("").astype(str).tolist()
    if len(texts) == 0:
        return pd.Series([], dtype=int)
    if len(texts) == 1:
        return pd.Series([0], index=discussions.index)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = normalize(embeddings)

    distance_threshold = max(0.01, min(0.99, 1.0 - similarity_threshold))
    kwargs = {
        "n_clusters": None,
        "distance_threshold": distance_threshold,
        "linkage": "average",
        "compute_full_tree": True,
    }
    try:
        clustering = AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        clustering = AgglomerativeClustering(affinity="cosine", **kwargs)
    labels = clustering.fit_predict(embeddings)
    return pd.Series(labels, index=discussions.index)


def refine_labels_by_tag(labels: pd.Series, discussions: pd.DataFrame) -> pd.Series:
    """
    Prevent obviously unrelated broad tags from being merged too aggressively.
    We split every cluster by its main tag.
    """
    new_labels = []
    label_map = {}
    next_id = 0
    for i, row in discussions.iterrows():
        raw_label = int(labels.loc[i])
        mt = main_tag([row.get("main_tags", "")])
        key = (raw_label, mt)
        if key not in label_map:
            label_map[key] = next_id
            next_id += 1
        new_labels.append(label_map[key])
    return pd.Series(new_labels, index=discussions.index)


def split_labels_by_time_gap(
    labels: pd.Series,
    discussions: pd.DataFrame,
    max_gap_hours: float = 12.0,
) -> pd.Series:
    """
    Split broad clusters into separate information events when discussions are far apart in time.
    This is important for live chats: one broad topic can recur for several days, but each wave may be a separate info event.
    """
    d = discussions.copy()
    d["_label"] = labels.loc[d.index].astype(int)
    d["_start"] = pd.to_datetime(d["start_date"], errors="coerce")
    d["_end"] = pd.to_datetime(d["end_date"], errors="coerce")

    new_labels = pd.Series(index=d.index, dtype=int)
    next_label = 0

    for _, group in d.sort_values(["_label", "_start"]).groupby("_label", sort=False):
        current_label = next_label
        next_label += 1
        prev_end = None

        for idx, row in group.iterrows():
            start = row["_start"]
            end = row["_end"]

            if prev_end is not None and pd.notna(start) and pd.notna(prev_end):
                if start - prev_end > pd.Timedelta(hours=max_gap_hours):
                    current_label = next_label
                    next_label += 1

            new_labels.loc[idx] = current_label

            if pd.notna(end):
                prev_end = max(prev_end, end) if prev_end is not None and pd.notna(prev_end) else end
            elif pd.notna(start):
                prev_end = max(prev_end, start) if prev_end is not None and pd.notna(prev_end) else start

    return new_labels.astype(int)


def make_events(
    discussions: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = discussions.copy()
    d["cluster_label"] = labels.values
    d["event_id"] = d["cluster_label"].apply(lambda x: f"e_{int(x):05d}")

    event_discussions = d[["event_id", "discussion_id"]].copy()

    rows = []
    for event_id, group in d.groupby("event_id", sort=True):
        tag = main_tag(group["main_tags"])
        keywords = top_keywords(group["discussion_text"].fillna("").astype(str), top_n=7)
        phrases = top_phrases(group["discussion_text"].fillna("").astype(str), top_n=5)
        microtopic_counter = Counter(group.get("microtopic", pd.Series(["other"])).fillna("other").astype(str))
        microtopic = microtopic_counter.most_common(1)[0][0] if microtopic_counter else "other"

        start = pd.to_datetime(group["start_date"], errors="coerce").min()
        end = pd.to_datetime(group["end_date"], errors="coerce").max()
        msg_count = int(group["message_count"].sum())
        discussion_count = int(group["discussion_id"].nunique())
        chat_count = int(group["chat_id"].replace("", np.nan).nunique()) if "chat_id" in group else 0
        author_count = int(group["author_count"].sum()) if "author_count" in group else 0
        negative_count = int(group["negative_count"].sum()) if "negative_count" in group else 0
        toxic_count = int(group["toxic_count"].sum()) if "toxic_count" in group else 0

        all_tags = sorted(tag_set_from_series(group["main_tags"]))
        tag_weight = max([CRITICAL_TAG_WEIGHTS.get(t, 1.0) for t in all_tags] or [1.0])
        negative_share = negative_count / msg_count if msg_count else 0.0
        toxic_share = toxic_count / msg_count if msg_count else 0.0

        importance_score = (
            math.log1p(msg_count) * 2
            + math.log1p(max(chat_count, 1)) * 2.5
            + math.log1p(max(author_count, 1)) * 0.8
            + negative_share * 4
            + toxic_share * 2
            + tag_weight
        )

        rows.append({
            "event_id": event_id,
            "event_title": build_title(tag, keywords, all_tags, microtopic=microtopic, phrases=phrases),
            "event_summary": summarize_event(group, MICROTOPIC_TITLES.get(microtopic, tag), keywords, phrases),
            "main_tag": tag,
            "microtopic": microtopic,
            "main_tags": "|".join(all_tags),
            "keywords": "|".join(keywords),
            "key_phrases": "|".join(phrases),
            "start_date": start,
            "end_date": end,
            "discussion_count": discussion_count,
            "message_count": msg_count,
            "chat_count": chat_count,
            "author_count": author_count,
            "negative_count": negative_count,
            "toxic_count": toxic_count,
            "negative_share": round(negative_share, 4),
            "toxic_share": round(toxic_share, 4),
            "importance_score": round(float(importance_score), 2),
            "status": "новый",
            "is_hidden": False,
        })

    events = pd.DataFrame(rows).sort_values(["importance_score", "message_count"], ascending=False)
    return events, event_discussions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source CSV")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--cluster-method", choices=["tfidf", "embeddings", "none"], default="tfidf")
    parser.add_argument("--similarity-threshold", type=float, default=0.28)
    parser.add_argument("--event-gap-hours", type=float, default=3.0, help="Split clusters into separate events when the time gap is larger than this value")
    parser.add_argument("--event-window-hours", type=float, default=16.0, help="Additionally limit one event to a fixed time span")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw = read_source_csv(args.input)
    tag_cols = detect_tag_columns(raw)

    messages, message_tags = normalize_messages(raw, tag_cols)
    discussions, discussion_messages = make_discussions(messages, window_minutes=args.window_minutes)

    # Filter empty discussions from clustering, but preserve them as singleton events.
    clusterable = discussions[discussions["discussion_text"].fillna("").str.len() > 10].copy()
    non_clusterable = discussions.drop(clusterable.index).copy()

    if args.cluster_method == "none":
        labels = pd.Series(range(len(clusterable)), index=clusterable.index)
    elif args.cluster_method == "embeddings":
        labels = cluster_discussions_embeddings(
            clusterable,
            similarity_threshold=args.similarity_threshold,
            model_name=args.embedding_model,
        )
    else:
        labels = cluster_discussions_tfidf(
            clusterable,
            similarity_threshold=args.similarity_threshold,
            max_gap_hours=args.event_gap_hours,
            max_event_span_hours=args.event_window_hours,
        )

    if len(clusterable):
        labels = refine_labels_by_tag(labels, clusterable)
        labels = split_labels_by_time_gap(labels, clusterable, max_gap_hours=args.event_gap_hours)
        labels = split_labels_by_fixed_time_window(labels, clusterable, window_hours=args.event_window_hours)

    if len(non_clusterable):
        start_label = int(labels.max()) + 1 if len(labels) else 0
        singleton_labels = pd.Series(range(start_label, start_label + len(non_clusterable)), index=non_clusterable.index)
        all_discussions = pd.concat([clusterable, non_clusterable], axis=0).sort_index()
        all_labels = pd.concat([labels, singleton_labels]).loc[all_discussions.index]
    else:
        all_discussions = clusterable
        all_labels = labels

    events, event_discussions = make_events(all_discussions, all_labels)

    paths = {
        "messages": str(write_table(messages, output, "messages")),
        "message_tags": str(write_table(message_tags, output, "message_tags")),
        "discussions": str(write_table(discussions, output, "discussions")),
        "discussion_messages": str(write_table(discussion_messages, output, "discussion_messages")),
        "events": str(write_table(events, output, "events")),
        "event_discussions": str(write_table(event_discussions, output, "event_discussions")),
    }

    manifest = {
        "source_file": str(Path(args.input).resolve()),
        "rows_source": int(len(raw)),
        "rows_messages": int(len(messages)),
        "rows_discussions": int(len(discussions)),
        "rows_events": int(len(events)),
        "tag_columns": tag_cols,
        "window_minutes": args.window_minutes,
        "cluster_method": args.cluster_method,
        "similarity_threshold": args.similarity_threshold,
        "event_gap_hours": args.event_gap_hours,
        "event_window_hours": args.event_window_hours,
        "paths": paths,
    }
    write_manifest(output, manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
