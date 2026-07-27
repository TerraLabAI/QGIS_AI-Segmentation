

































from __future__ import annotations

import re
from typing import Iterable, Sequence

from .server_dials import (
    dial_in_range,
    dial_list,
    dial_str,
    dial_url,
    parse_version,
    read_value,
    safe_web_url,
)



_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")



_VERSION_CLAUSE = r"(?:===|[<>!~=]=|[<>])\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,39}"
_VERSION_SPEC_RE = re.compile(
    rf"^\s*{_VERSION_CLAUSE}(?:\s*,\s*{_VERSION_CLAUSE})*\s*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")

_ASSET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")




_MAX_EXTRA_PACKAGES = 8
_MAX_MIRRORS = 4
_MAX_DIGESTS = 256
_MAX_VERSION_ENTRIES = 32





def valid_package_name(value) -> bool:

    return isinstance(value, str) and bool(_PACKAGE_NAME_RE.match(value))


def valid_version_spec(value) -> bool:







    if not isinstance(value, str) or not value.strip():
        return False
    return bool(_VERSION_SPEC_RE.match(value))


def _clean_sha256(value):

    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if _SHA256_RE.match(text) else None


def _served_map(path: str) -> dict:

    value = read_value(path)
    return value if isinstance(value, dict) else {}





def package_specs(
    shipped: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:












    served = _served_map("install.packages")
    if not served:
        return list(shipped)

    shipped_names = {name for name, _spec in shipped}
    out: list[tuple[str, str]] = []
    for name, spec in shipped:
        candidate = served.get(name)
        if valid_version_spec(candidate) and isinstance(candidate, str):
            out.append((name, candidate.strip()))
        else:
            out.append((name, spec))

    extras = []
    for name in served:
        keep = name not in shipped_names and valid_package_name(name)
        keep = keep and valid_version_spec(served.get(name)) and isinstance(served.get(name), str)
        if keep:
            extras.append(name)
    extras.sort()
    for name in extras[:_MAX_EXTRA_PACKAGES]:
        out.append((name, str(served[name]).strip()))
    return out


def torch_index_url(shipped: str) -> str:

    return dial_url("install.torch_index_url", shipped)


def version_pin(path: str, shipped):






    served = read_value(path)
    if isinstance(served, str) and served.strip() and valid_version_spec(served):
        return served.strip()
    return shipped





def pip_retries(shipped: int) -> int:

    return int(dial_in_range("install.pip.retries", shipped, 1, 20))


def pip_timeout_s(shipped: int) -> int:

    return int(dial_in_range("install.pip.timeout_s", shipped, 5, 300))


def package_timeout_s(package_name: str, shipped: int) -> int:






    served = _served_map("install.timeouts").get(package_name)
    if served is None:
        return shipped
    return int(dial_in_range(
        f"install.timeouts.{package_name}", shipped, 60, 21_600))


def verify_timeout_s(package_name: str, shipped: int) -> int:








    served = _served_map("install.verify_timeouts").get(package_name)
    if served is None:
        return shipped
    return int(dial_in_range(
        f"install.verify_timeouts.{package_name}", shipped, 10, 1800))


def network_retry_attempts(shipped: int) -> int:

    return int(dial_in_range("install.retry.network_attempts", shipped, 1, 10))


def network_retry_backoff_s(attempt: int, shipped_base: int) -> int:





    base = int(dial_in_range("install.retry.backoff_base_s", shipped_base, 1, 120))
    return min(300, base * (2 ** max(0, attempt - 1)))


def install_logic_version(shipped: str) -> str:








    served = dial_str("install.logic_version", "")
    if not served:
        return shipped
    served_parts = parse_version(served)
    shipped_parts = parse_version(shipped)
    if served_parts is None or shipped_parts is None:
        return shipped
    width = max(len(served_parts), len(shipped_parts))
    served_parts += (0,) * (width - len(served_parts))
    shipped_parts += (0,) * (width - len(shipped_parts))
    return served if served_parts > shipped_parts else shipped





def checkpoint_source(
    filename: str, shipped_url: str, shipped_sha256: str,
) -> tuple[str, str, tuple[str, ...]]:























    entry = _served_map("install.checkpoint").get(filename)
    if not isinstance(entry, dict):
        return shipped_url, shipped_sha256, ()

    url = safe_web_url(entry.get("url"), "")
    digest = _clean_sha256(entry.get("sha256"))
    if url and digest:
        primary, expected = url, digest
    else:
        primary, expected = shipped_url, shipped_sha256

    mirrors: list[str] = []
    raw = entry.get("mirrors")
    if isinstance(raw, (list, tuple)):
        for item in raw[:_MAX_MIRRORS]:
            candidate = safe_web_url(item, "")
            if candidate and candidate != primary and candidate not in mirrors:
                mirrors.append(candidate)



    if not expected:
        return shipped_url, shipped_sha256, ()
    return primary, expected, tuple(mirrors)






def checkpoint_max_retries(shipped: int) -> int:

    return int(dial_in_range("install.download.max_retries", shipped, 1, 20))


def checkpoint_idle_timeout_ms(shipped: int) -> int:

    return int(dial_in_range(
        "install.download.idle_timeout_ms", shipped, 10_000, 1_800_000))


def checkpoint_hard_timeout_ms(shipped: int) -> int:

    return int(dial_in_range(
        "install.download.hard_timeout_ms", shipped, 60_000, 21_600_000))


def replace_attempts(shipped: int) -> int:

    return int(dial_in_range("install.replace.attempts", shipped, 1, 20))


def replace_delay_s(shipped: float) -> float:

    return float(dial_in_range("install.replace.delay_s", shipped, 0.1, 30.0))


def min_free_gb_full(shipped: float) -> float:

    return float(dial_in_range("install.disk.min_free_gb_full", shipped, 1.0, 40.0))


def min_free_gb_automatic(shipped: float) -> float:

    return float(dial_in_range("install.disk.min_free_gb_automatic", shipped, 0.2, 40.0))





def _digest_table(
    path: str,
    shipped: dict[str, str],
    shipped_tag: str,
    served_tag: str,
) -> dict[str, str]:









    served_raw = _served_map(path)
    served: dict[str, str] = {}
    for name, value in list(served_raw.items())[:_MAX_DIGESTS]:
        if not isinstance(name, str) or not _ASSET_RE.match(name):
            continue
        digest = _clean_sha256(value)
        if digest:
            served[name] = digest
    if served_tag != shipped_tag:
        return served
    merged = dict(served)
    merged.update(shipped)
    return merged


def uv_version(shipped: str) -> str:

    served = dial_str("install.uv.version", "")
    return served if _TAG_RE.match(served or "") else shipped


def uv_digests(shipped: dict[str, str], shipped_version: str) -> dict[str, str]:

    return _digest_table(
        "install.uv.sha256", shipped, shipped_version, uv_version(shipped_version))


def uv_download_timeout_ms(shipped: int) -> int:

    return int(dial_in_range(
        "install.uv.download_timeout_ms", shipped, 10_000, 1_800_000))


def uv_http_timeout_s(shipped: int) -> int:





    return int(dial_in_range("install.uv.http_timeout_s", shipped, 30, 1800))


def uv_http_retries(shipped: int) -> int:

    return int(dial_in_range("install.uv.http_retries", shipped, 1, 20))


def python_release_tag(shipped: str) -> str:

    served = dial_str("install.python.release_tag", "")
    return served if _TAG_RE.match(served or "") else shipped


def python_versions(shipped: dict[tuple[int, int], str]) -> dict[tuple[int, int], str]:







    served = _served_map("install.python.versions")
    if not served:
        return dict(shipped)
    out = dict(shipped)
    for key, value in list(served.items())[:_MAX_VERSION_ENTRIES]:
        parsed_key = parse_version(key) if isinstance(key, str) else None
        if parsed_key is None or len(parsed_key) != 2:
            continue
        minor = (parsed_key[0], parsed_key[1])
        if minor not in shipped:
            continue
        if isinstance(value, str) and parse_version(value) is not None:
            out[minor] = value.strip()
    return out


def python_digests(shipped: dict[str, str], shipped_tag: str) -> dict[str, str]:

    return _digest_table(
        "install.python.sha256", shipped, shipped_tag, python_release_tag(shipped_tag))

















def classifier_markers(name: str, shipped: Iterable[str]) -> tuple[str, ...]:









    return tuple(dial_list(
        f"install.classifier.{name}",
        tuple(item.lower() for item in shipped),
        normalize=str.lower,
    ))
