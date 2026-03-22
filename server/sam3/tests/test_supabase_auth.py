import hashlib
from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException

import os
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-key"

from supabase_auth import hash_api_key, validate_api_key  # noqa: E402


def test_hash_api_key():
    result = hash_api_key("tl_pro_abc123")
    assert len(result) == 64
    assert result == hashlib.sha256("tl_pro_abc123".encode()).hexdigest()


def test_validate_valid_key():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": "some-uuid"}]
    with patch("supabase_auth.httpx.get", return_value=mock_resp):
        validate_api_key("tl_pro_valid")  # must not raise


def test_validate_invalid_key():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = []
    with patch("supabase_auth.httpx.get", return_value=mock_resp):
        with pytest.raises(HTTPException) as exc_info:
            validate_api_key("tl_pro_bad")
        assert exc_info.value.status_code == 401


def test_validate_empty_key():
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key("")
    assert exc_info.value.status_code == 401


def test_validate_supabase_unreachable():
    import httpx
    with patch("supabase_auth.httpx.get", side_effect=httpx.RequestError("timeout")):
        with pytest.raises(HTTPException) as exc_info:
            validate_api_key("tl_pro_any")
        assert exc_info.value.status_code == 503


def test_validate_supabase_error_status():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = []
    with patch("supabase_auth.httpx.get", return_value=mock_resp):
        with pytest.raises(HTTPException) as exc_info:
            validate_api_key("tl_pro_any")
        assert exc_info.value.status_code == 401
