import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status
from supabase import Client, create_client

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    pass


class SupabaseOperationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY"
    )
    if not url or not secret_key:
        raise ConfigurationError(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY in the project .env file."
        )
    try:
        return create_client(url, secret_key)
    except Exception as exc:
        logger.exception("Unable to initialize the Supabase client.")
        raise ConfigurationError("Supabase client initialization failed.") from exc


def list_records(
    table: str,
    *,
    order_by: str,
    descending: bool = False,
    limit: int = 1000,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        query = get_client().table(table).select("*")
        for column, value in (filters or {}).items():
            query = query.eq(column, value)
        result = query.order(order_by, desc=descending).limit(limit).execute()
        return result.data
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Supabase list operation failed for table %s.", table)
        raise SupabaseOperationError(f"Could not list {table}.") from exc


def count_records(table: str, **filters: Any) -> int:
    try:
        query = get_client().table(table).select("id", count="exact")
        for column, value in filters.items():
            query = query.eq(column, value)
        result = query.execute()
        if result.count is None:
            raise SupabaseOperationError(f"Could not count {table} records.")
        return result.count
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Supabase count operation failed for table %s.", table)
        raise SupabaseOperationError(f"Could not count {table} records.") from exc


def create_record(table: str, values: dict[str, Any]) -> dict[str, Any]:
    try:
        query = get_client().table(table)
        if values.get("id"):
            result = query.upsert(values, on_conflict="id").execute()
        else:
            result = query.insert(values).execute()
        return result.data[0]
    except ConfigurationError:
        raise
    except Exception as exc:
        if getattr(exc, "code", None) == "P0001":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=getattr(exc, "message", str(exc)),
            ) from exc
        logger.exception("Supabase insert operation failed for table %s.", table)
        raise SupabaseOperationError(f"Could not create a {table} record.") from exc


def update_record(
    table: str, record_id: str, values: dict[str, Any]
) -> dict[str, Any]:
    try:
        result = (
            get_client()
            .table(table)
            .update(values)
            .eq("id", record_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested record does not exist.",
            )
        return result.data[0]
    except HTTPException:
        raise
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Supabase update operation failed for table %s.", table)
        raise SupabaseOperationError(f"Could not update {table}.") from exc


def delete_record(table: str, record_id: str) -> None:
    try:
        result = get_client().table(table).delete().eq("id", record_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested record does not exist.",
            )
    except HTTPException:
        raise
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Supabase delete operation failed for table %s.", table)
        raise SupabaseOperationError(f"Could not delete {table}.") from exc
