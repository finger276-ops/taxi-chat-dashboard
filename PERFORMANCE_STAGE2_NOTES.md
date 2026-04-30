# Performance optimization — stage 2

This patch keeps the existing dashboard behavior but reduces repeated work on large tables.

## What changed

1. Prepared reusable message helper columns:
   - `_message_hidden_bool`
   - `_final_event_id_str`
   - `_source_final_event_id_str`
   - `_search_text`

   These avoid repeated `astype(str)`, `lower()`, `replace("ё", "е")` and hidden-message conversions inside filters, search, KPIs and message feeds.

2. Made event cards more lazy:
   - selected event messages are sliced and sorted only when the user opens “Ключевые сообщения” or “Вся лента”;
   - the “Правки” section no longer pays for building the full message feed.

3. Removed repeated DB reads inside event cards:
   - pinned key messages and irrelevant-message exclusions are now taken from the already loaded `manual_tables` bundle.

4. Cached consolidation layer:
   - `cached_build_consolidated_events()` avoids rebuilding macro-events on every UI click when data, edits and consolidation level have not changed.

5. Optimized Supabase generated-table loads:
   - `load_table_from_supabase()` selects only `period_id,payload` instead of `*`;
   - generated IDs are prefixed with vectorized string operations instead of row-wise `DataFrame.apply()`.

6. Added safe SQL indexes for existing Supabase projects:
   - `(table_name, period_id)` on `dashboard_table_rows`;
   - `(status, uploaded_at desc)` on `dashboard_periods`;
   - `(table_name, updated_at desc)` on `dashboard_manual_rows`.

## Files changed

- `src/app.py`
- `src/persistent_store.py`
- `sql/supabase_schema.sql`

No new Python dependencies are required.

## Deployment note

For an existing Supabase project, run the new index statements from `sql/supabase_schema.sql` once in SQL Editor. They use `create index if not exists`, so they are safe to run repeatedly.
