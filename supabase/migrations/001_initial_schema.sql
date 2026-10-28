create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.pigs (
  id uuid primary key default gen_random_uuid(),
  ai_identity text,
  pen_tag integer not null unique check (pen_tag between 1 and 2147483647),
  growth_stage text not null check (length(trim(growth_stage)) > 0),
  registered_weight numeric(8, 2) not null default 0 check (registered_weight >= 0),
  status text not null default 'Active' check (status in ('Active', 'Quarantined', 'Inactive')),
  notes text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.observations (
  id uuid primary key default gen_random_uuid(),
  pig_id uuid references public.pigs(id) on delete set null,
  ai_identity text,
  pen_tag integer check (pen_tag between 1 and 2147483647),
  weight_kg numeric(8, 2) check (weight_kg is null or weight_kg >= 0),
  behavior text,
  confidence numeric(4, 3) check (confidence is null or confidence between 0 and 1),
  bounding_box jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.alerts (
  id uuid primary key default gen_random_uuid(),
  pig_id uuid references public.pigs(id) on delete set null,
  ai_identity text,
  pen_tag integer check (pen_tag between 1 and 2147483647),
  issue text not null check (length(trim(issue)) > 0),
  severity text not null default 'Medium' check (severity in ('Low', 'Medium', 'High', 'Critical')),
  status text not null default 'Pending Review'
    check (status in ('Pending Review', 'Referred to Vet', 'Resolved')),
  details text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.feed_logs (
  id uuid primary key default gen_random_uuid(),
  pig_id uuid references public.pigs(id) on delete set null,
  pen_tag integer not null check (pen_tag between 1 and 2147483647),
  feed_type text not null check (length(trim(feed_type)) > 0),
  quantity_kg numeric(10, 2) not null check (quantity_kg > 0),
  cost_per_kg numeric(10, 2) not null default 0 check (cost_per_kg >= 0),
  fed_on date not null default current_date,
  created_at timestamptz not null default now()
);

create index if not exists observations_created_at_idx
  on public.observations (created_at desc);
create index if not exists observations_pig_id_created_at_idx
  on public.observations (pig_id, created_at desc);
create index if not exists alerts_created_at_idx
  on public.alerts (created_at desc);
create index if not exists alerts_status_created_at_idx
  on public.alerts (status, created_at desc);
create index if not exists feed_logs_fed_on_idx
  on public.feed_logs (fed_on desc);
create index if not exists feed_logs_pig_id_fed_on_idx
  on public.feed_logs (pig_id, fed_on desc);

drop trigger if exists pigs_set_updated_at on public.pigs;
create trigger pigs_set_updated_at
  before update on public.pigs
  for each row execute function public.set_updated_at();

drop trigger if exists alerts_set_updated_at on public.alerts;
create trigger alerts_set_updated_at
  before update on public.alerts
  for each row execute function public.set_updated_at();

alter table public.pigs enable row level security;
alter table public.observations enable row level security;
alter table public.alerts enable row level security;
alter table public.feed_logs enable row level security;

revoke all on table
  public.pigs,
  public.observations,
  public.alerts,
  public.feed_logs
from anon, authenticated;

grant select, insert, update, delete on table
  public.pigs,
  public.observations,
  public.alerts,
  public.feed_logs
to service_role;

-- The application uses the Supabase service-role key only in the FastAPI backend.
-- No anon/authenticated policies are added, so browser clients cannot access farm data directly.
