"""Supabase PostgreSQL client — direct httpx REST pattern (reused from NeuralQuant).

Avoids supabase-py SDK to prevent uvicorn RemoteProtocolError issues.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Shared httpx client (connection pooling)
_client: Optional[httpx.Client] = None


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required")
        _client = httpx.Client(
            base_url=f"{SUPABASE_URL.rstrip('/')}/rest/v1",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            timeout=30.0,
        )
    return _client


def rest_get(
    table: str,
    select: str = "*",
    filters: Optional[dict] = None,
    order: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """GET rows from a Supabase table."""
    client = get_client()
    params: dict = {"select": select, "limit": limit, "offset": offset}
    if order:
        params["order"] = order
    if filters:
        for k, v in filters.items():
            params[k] = f"eq.{v}"

    resp = client.get(f"/{table}", params=params)
    resp.raise_for_status()
    return resp.json()


def rest_insert(table: str, rows: list[dict]) -> list[dict]:
    """INSERT rows into a Supabase table. Returns inserted rows."""
    client = get_client()
    resp = client.post(
        f"/{table}",
        json=rows,
        headers={"Prefer": "return=representation"},
    )
    resp.raise_for_status()
    return resp.json() if resp.text else []


def rest_upsert(table: str, rows: list[dict], on_conflict: str = "ticker,date") -> list[dict]:
    """UPSERT rows. Returns upserted rows."""
    client = get_client()
    resp = client.post(
        f"/{table}",
        json=rows,
        headers={
            "Prefer": "resolution=merge-duplicates,return=representation",
        },
        params={"on_conflict": on_conflict},
    )
    resp.raise_for_status()
    return resp.json() if resp.text else []


def rest_delete(table: str, filters: dict) -> list[dict]:
    """DELETE rows matching filters."""
    client = get_client()
    params = {}
    for k, v in filters.items():
        params[k] = f"eq.{v}"
    resp = client.delete(f"/{table}", params=params, headers={"Prefer": "return=representation"})
    resp.raise_for_status()
    return resp.json() if resp.text else []


def table_exists(table: str) -> bool:
    """Check if a table exists (by trying to read 1 row)."""
    try:
        rest_get(table, select="*", limit=1)
        return True
    except httpx.HTTPStatusError:
        return False
