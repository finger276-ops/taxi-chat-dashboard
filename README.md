# Taxi Chat Dashboard

MVP-дашборд для анализа Telegram-чатов такси: сообщения → обсуждения → инфоповоды.

## Что делает проект

1. Загружает исходный CSV с разделителем `;` и кодировкой UTF-8 BOM.
2. Нормализует поля: дата, чат, автор, текст, ссылки, тональность, токсичность.
3. Превращает бинарные колонки тегов в нормальный список тегов.
4. Собирает первичные обсуждения:
   - комментарии группируются по `Ссылка на родительский пост`;
   - остальные сообщения группируются внутри чата по временному окну и пересечению тегов.
5. Кластеризует обсуждения в инфоповоды.
6. Создает таблицы:
   - `messages`
   - `message_tags`
   - `discussions`
   - `discussion_messages`
   - `events`
   - `event_discussions`
7. Запускает Streamlit-дашборд.
8. Хранит ручные правки, скрытия, статусы, объединения и перенос сообщений в SQLite.

## Быстрый старт

```bash
cd taxi_chat_dashboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-preprocess.txt
```

Положите исходный CSV в папку `data/`, например:

```text
data/chats.csv
```

Предобработка:

```bash
python src/preprocess.py --input data/chats.csv --output data/processed --window-minutes 60 --cluster-method tfidf --event-gap-hours 1
```

Запуск дашборда:

```bash
streamlit run src/app.py -- --data-dir data/processed --db-path data/manual_actions.sqlite
```

## Рекомендуемый первый режим

Для старта используйте:

```bash
python src/preprocess.py --input data/chats.csv --output data/processed --window-minutes 60 --cluster-method tfidf --similarity-threshold 0.25 --event-gap-hours 1
```

Если инфоповоды получаются слишком крупными — увеличьте `--similarity-threshold`, например до `0.35`.
Если слишком дробными — уменьшите до `0.20` или увеличьте `--event-gap-hours`, например до `6` или `12`.

## Ручные действия в дашборде

В интерфейсе можно:

- переименовать инфоповод;
- изменить краткое описание;
- поставить статус;
- скрыть инфоповод;
- объединить один инфоповод с другим;
- перенести выбранное сообщение в другой инфоповод;
- скрыть отдельное сообщение.

Все правки пишутся в `manual_actions.sqlite`, исходный CSV не меняется.
