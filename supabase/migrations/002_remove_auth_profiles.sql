drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.create_profile_for_auth_user();
drop table if exists public.profiles;
