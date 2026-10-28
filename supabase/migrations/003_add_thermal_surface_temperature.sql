alter table public.observations
  add column if not exists thermal_surface_c numeric(5, 2)
  check (thermal_surface_c is null or thermal_surface_c between -40 and 300);
