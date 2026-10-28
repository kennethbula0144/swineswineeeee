create or replace function public.enforce_unique_pig_identity_per_pen()
returns trigger
language plpgsql
as $$
declare
  identity_key text;
begin
  identity_key := regexp_replace(
    lower(coalesce(new.ai_identity, '')),
    '[^a-z0-9]',
    '',
    'g'
  );

  if identity_key = '' or identity_key = 'unknown' then
    return new;
  end if;

  perform pg_advisory_xact_lock(
    73105,
    hashtext(new.pen_tag::text || ':' || identity_key)
  );

  if exists (
    select 1
    from public.pigs
    where pen_tag = new.pen_tag
      and regexp_replace(
        lower(coalesce(ai_identity, '')),
        '[^a-z0-9]',
        '',
        'g'
      ) = identity_key
      and id is distinct from new.id
  ) then
    raise exception 'AI identity % is already assigned to pen %.',
      new.ai_identity, new.pen_tag
      using errcode = 'P0001';
  end if;

  return new;
end;
$$;

drop trigger if exists pigs_enforce_unique_identity_per_pen on public.pigs;
create trigger pigs_enforce_unique_identity_per_pen
  before insert or update of ai_identity, pen_tag on public.pigs
  for each row execute function public.enforce_unique_pig_identity_per_pen();
