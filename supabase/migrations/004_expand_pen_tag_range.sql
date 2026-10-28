  alter table public.pigs
    drop constraint if exists pigs_pen_tag_check,
    alter column pen_tag type integer using pen_tag::integer,
    add constraint pigs_pen_tag_check
      check (pen_tag between 1 and 2147483647);

  alter table public.observations
    drop constraint if exists observations_pen_tag_check,
    alter column pen_tag type integer using pen_tag::integer,
    add constraint observations_pen_tag_check
      check (pen_tag is null or pen_tag between 1 and 2147483647);

  alter table public.alerts
    drop constraint if exists alerts_pen_tag_check,
    alter column pen_tag type integer using pen_tag::integer,
    add constraint alerts_pen_tag_check
      check (pen_tag is null or pen_tag between 1 and 2147483647);

  alter table public.feed_logs
    drop constraint if exists feed_logs_pen_tag_check,
    alter column pen_tag type integer using pen_tag::integer,
    add constraint feed_logs_pen_tag_check
      check (pen_tag between 1 and 2147483647);
