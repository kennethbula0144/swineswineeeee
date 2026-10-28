alter table public.pigs
  drop constraint if exists pigs_pen_tag_key;

create index if not exists pigs_pen_tag_idx
  on public.pigs (pen_tag);

create or replace function public.enforce_pig_pen_capacity()
returns trigger
language plpgsql
as $$
begin
  perform pg_advisory_xact_lock(73104, new.pen_tag);

  if (
    select count(*)
    from public.pigs
    where pen_tag = new.pen_tag
      and id is distinct from new.id
  ) >= 20 then
    raise exception 'Pen % has reached its limit of 20 pigs.', new.pen_tag
      using errcode = 'P0001';
  end if;

  return new;
end;
$$;

drop trigger if exists pigs_enforce_pen_capacity on public.pigs;
create trigger pigs_enforce_pen_capacity
  before insert or update of pen_tag on public.pigs
  for each row execute function public.enforce_pig_pen_capacity();
