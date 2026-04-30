-- Supabase schema for taxi chat dashboard persistence.
-- Run this once in Supabase SQL Editor.

create table if not exists public.dashboard_periods (
    period_id text primary key,
    period_name text not null,
    date_from date,
    date_to date,
    source_filename text,
    uploaded_at timestamptz not null default now(),
    status text not null default 'active',
    manifest jsonb not null default '{}'::jsonb
);

create table if not exists public.dashboard_table_rows (
    period_id text not null references public.dashboard_periods(period_id) on delete cascade,
    table_name text not null,
    row_id text not null,
    payload jsonb not null,
    updated_at timestamptz not null default now(),
    primary key (period_id, table_name, row_id)
);

create index if not exists idx_dashboard_table_rows_period_table
    on public.dashboard_table_rows(period_id, table_name);

create table if not exists public.dashboard_manual_rows (
    row_key text primary key,
    table_name text not null,
    payload jsonb not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_dashboard_manual_rows_table
    on public.dashboard_manual_rows(table_name);

-- Optional: create a private Storage bucket named dashboard-csv in the Supabase UI.
-- The app can work without Storage because processed rows are stored in Postgres.

-- Extra indexes for faster multi-period dashboard loads.
-- These are safe to run on an existing Supabase project.
create index if not exists idx_dashboard_table_rows_table_period
    on public.dashboard_table_rows(table_name, period_id);

create index if not exists idx_dashboard_periods_status_uploaded
    on public.dashboard_periods(status, uploaded_at desc);

create index if not exists idx_dashboard_manual_rows_table_updated
    on public.dashboard_manual_rows(table_name, updated_at desc);
