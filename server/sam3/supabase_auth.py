"""Supabase-based per-user API key validation for the SAM3 server."""
import hashlib
import os

import httpx
from fastapi import HTTPException

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def validate_api_key(raw_key: str) -> None:
    """Validate key against Supabase api_keys table.

    Raises HTTPException 401 if invalid/revoked, 503 if Supabase unreachable.
    If SUPABASE_URL is not configured, validation is skipped (dev mode).
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return  # dev mode: no Supabase configured

    if not raw_key:
        raise HTTPException(status_code=401, detail="API key required")

    key_hash = hash_api_key(raw_key)

    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/api_keys",
            params={
                "key_hash": f"eq.{key_hash}",
                "active": "eq.true",
                "select": "id",
                "limit": "1",
            },
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            timeout=5.0,
        )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    if resp.status_code != 200 or not resp.json():
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
