"""Server-tunable pieces of the environment bootstrap.

The install path is the most release-blocked surface in the plugin. Everything
it needs is pinned in source: which packages, which version ranges, which
download addresses, which digests, how long to wait and how often to retry.
When an upstream package or host goes away, NEW installs break and stay broken
until a plugin release clears the plugin repository, which takes days. The
readers here let those pins be corrected from the product configuration in the
meantime.

Cache-only, and that is not an optimisation. This code runs before any sign-in
and before any successful call, so it reads the configuration a previous
session mirrored to disk (``config_cache.prime_from_disk``) and NEVER networks
itself. With an empty cache every reader returns the shipped constant, so a
first install on a fresh machine behaves exactly as it does today.

Validation is stricter here than anywhere else in the plugin, because the
failure mode is a broken install on a machine nobody can reach:

  - a package spec is a bare name plus a version range, so a URL, a local
    path, an extra, an environment marker or a smuggled ``--flag`` is refused
    before it can reach a command line;
  - every address must be https with a host;
  - every digest must be 64 hex characters, and it is verified exactly as
    today: a download whose digest is unknown or wrong still fails closed;
  - every number must be positive and inside a range we would accept from
    ourselves.

Anything that does not pass leaves the shipped constant in place, per value.
One bad entry never costs the others.

Pure Python with no Qt at import time, so the install path, the managers and
the tests can all import it.
"""
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

# A distribution name as PyPI allows it, and nothing else: no extras, no
# environment marker, no path separator, no leading dash.
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# One or more comma-separated PEP 440 clauses. Each clause is a comparison
# operator followed by a version. A URL ("://"), a path ("/"), an option
# ("--only-binary") and an extra ("[dev]") all fail to match.
_VERSION_CLAUSE = r"(?:===|[<>!~=]=|[<>])\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,39}"
_VERSION_SPEC_RE = re.compile(
    rf"^\s*{_VERSION_CLAUSE}(?:\s*,\s*{_VERSION_CLAUSE})*\s*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# A release tag or a tool version: digits, dots and dashes, nothing that could
# travel out of a path segment.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
# A standalone-Python or uv asset filename, used as a digest-table key.
_ASSET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")

# Bounds on how much a served table may hold. An install reads these a handful
# of times, so the caps are about keeping a damaged configuration cheap rather
# than about the happy path.
_MAX_EXTRA_PACKAGES = 8
_MAX_MIRRORS = 4
_MAX_DIGESTS = 256
_MAX_VERSION_ENTRIES = 32


# -- small validators ------------------------------------------------------


def valid_package_name(value) -> bool:
    """Whether ``value`` is a bare distribution name we would install."""
    return isinstance(value, str) and bool(_PACKAGE_NAME_RE.match(value))


def valid_version_spec(value) -> bool:
    """Whether ``value`` is a version range and nothing else.

    Anything carrying a scheme, a path, an option, an extra or an environment
    marker is refused. So is a blank one: a blank retune of a shipped package
    would silently widen its range to unbounded, which is a mistake far more
    often than it is an intention.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    return bool(_VERSION_SPEC_RE.match(value))


def _clean_sha256(value):
    """The digest in the form the verifiers compare against, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if _SHA256_RE.match(text) else None


def _served_map(path: str) -> dict:
    """A served ``{key: value}`` map, or ``{}``. Never raises."""
    value = read_value(path)
    return value if isinstance(value, dict) else {}


# -- packages --------------------------------------------------------------


def package_specs(
    shipped: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The packages to install, with any server correction applied.

    ``install.packages`` is a ``{name: version_spec}`` map. A key naming a
    shipped package RETUNES its version range in place, which is what unbricks
    an install after an upstream yank. A key naming anything else is APPENDED,
    bounded, and in sorted order so the result is the same on every machine.

    Order matters to the caller (torch is installed before the packages that
    build against it), so shipped order is never disturbed and additions go
    last. Every name and every range is validated on its own; one bad entry
    leaves that package exactly as it shipped.
    """
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
    """The CPU wheel index used where the default index is too heavy."""
    return dial_url("install.torch_index_url", shipped)


def version_pin(path: str, shipped):
    """A served version range for a one-off pin, or the shipped one.

    For the pins that are not part of the install list: the exact versions a
    repair path falls back to when the normal range produced a broken build.
    Validated like any other spec, so nothing but a version range gets through.
    """
    served = read_value(path)
    if isinstance(served, str) and served.strip() and valid_version_spec(served):
        return served.strip()
    return shipped


# -- pip and the retry ladder ----------------------------------------------


def pip_retries(shipped: int) -> int:
    """``--retries`` handed to the installer for one package."""
    return int(dial_in_range("install.pip.retries", shipped, 1, 20))


def pip_timeout_s(shipped: int) -> int:
    """``--timeout`` handed to the installer, per connection attempt."""
    return int(dial_in_range("install.pip.timeout_s", shipped, 5, 300))


def package_timeout_s(package_name: str, shipped: int) -> int:
    """How long one package's install may run before it is killed.

    ``install.timeouts`` is a ``{package_name: seconds}`` map. The floor is a
    minute, because a value below that would turn a slow link into a failed
    install for everyone on it.
    """
    served = _served_map("install.timeouts").get(package_name)
    if served is None:
        return shipped
    return int(dial_in_range(
        f"install.timeouts.{package_name}", shipped, 60, 21_600))


def verify_timeout_s(package_name: str, shipped: int) -> int:
    """How long one package's post-install check may run before it is killed.

    ``install.verify_timeouts`` is a ``{package_name: seconds}`` map, keyed the
    same way as ``install.timeouts``. A check that runs out of time is read as
    a broken package, and that verdict escalates to a full reinstall, so the
    floor exists to stop a merely slow machine from repeating that reinstall
    for ever.
    """
    served = _served_map("install.verify_timeouts").get(package_name)
    if served is None:
        return shipped
    return int(dial_in_range(
        f"install.verify_timeouts.{package_name}", shipped, 10, 1800))


def network_retry_attempts(shipped: int) -> int:
    """How many times a package install is retried after a network error."""
    return int(dial_in_range("install.retry.network_attempts", shipped, 1, 10))


def network_retry_backoff_s(attempt: int, shipped_base: int) -> int:
    """Seconds to wait before retry ``attempt`` (1-based), doubling each time.

    Only the first wait is tunable; the doubling and the ceiling stay in code,
    so no configuration can make a user sit through a wait we never intended.
    """
    base = int(dial_in_range("install.retry.backoff_base_s", shipped_base, 1, 120))
    return min(300, base * (2 ** max(0, attempt - 1)))


def install_logic_version(shipped: str) -> str:
    """The install-logic version stamped into the dependency hash.

    Raising it makes every existing environment re-run the install, so this is
    a fleet-wide "reinstall now" lever: powerful, and expensive for every user
    who gets it. A served value is therefore accepted ONLY when it parses as a
    version and is STRICTLY GREATER than the shipped one. Never lower, never
    equal, never anything that does not parse.
    """
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


# -- downloaded artefacts --------------------------------------------------


def checkpoint_source(
    filename: str, shipped_url: str, shipped_sha256: str,
) -> tuple[str, str, tuple[str, ...]]:
    """Where to fetch the model weights, and what they must hash to.

    ``install.checkpoint`` is a map keyed by the weight file's name, each entry
    ``{url, sha256, mirrors: [url, ...]}``. Keying by filename lets one
    configuration describe every variant the fleet installs, and lets a client
    pick its own without the server knowing which one it runs.

    A dead third-party host is then survivable without a release: the mirrors
    are extra addresses to try for the SAME file. The digest is not optional
    and is checked exactly as today, so a mirror can only ever serve the bytes
    the entry already claims.

    An entry is taken whole or not at all. A served address with no valid
    digest beside it would be an unverifiable download, so both fall back
    together to what shipped.

    Note the trust boundary this sets: an entry carries the address AND the
    digest, so the check passes by construction and whoever controls the
    response controls the bytes, which the model loader then unpickles. That
    is a deliberate trade for being able to move a dead host without a
    release. It rests on the backend not being compromised, unlike the uv and
    Python digests below, where the shipped value wins on a collision.
    """
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
    # A mirror serves the same file, so it is only usable when the digest that
    # file must match is known. It always is (the shipped one at worst), but
    # say it rather than assume it.
    if not expected:
        return shipped_url, shipped_sha256, ()
    return primary, expected, tuple(mirrors)


# The download tuning lives under its own key, not under install.checkpoint:
# that one is a map keyed by weight filename, and mixing scalars into it would
# invite an author to put these inside a per-file entry, where nothing reads
# them.
def checkpoint_max_retries(shipped: int) -> int:
    """How many times the weight download is attempted before giving up."""
    return int(dial_in_range("install.download.max_retries", shipped, 1, 20))


def checkpoint_idle_timeout_ms(shipped: int) -> int:
    """How long a silent connection is tolerated before the download aborts."""
    return int(dial_in_range(
        "install.download.idle_timeout_ms", shipped, 10_000, 1_800_000))


def checkpoint_hard_timeout_ms(shipped: int) -> int:
    """The ceiling on one download attempt, however live the connection is."""
    return int(dial_in_range(
        "install.download.hard_timeout_ms", shipped, 60_000, 21_600_000))


def replace_attempts(shipped: int) -> int:
    """How many times a locked file is re-tried before the move fails."""
    return int(dial_in_range("install.replace.attempts", shipped, 1, 20))


def replace_delay_s(shipped: float) -> float:
    """Seconds between two attempts at moving a locked file into place."""
    return float(dial_in_range("install.replace.delay_s", shipped, 0.1, 30.0))


# -- the two bundled tool downloads ----------------------------------------


def _digest_table(
    path: str,
    shipped: dict[str, str],
    shipped_tag: str,
    served_tag: str,
) -> dict[str, str]:
    """Merge a served digest table with the shipped one.

    A digest table describes ONE release. While the served release matches the
    shipped one the merge is additive, so no shipped digest can be replaced by
    a served value and a wrong entry can only ever add an unused key. When the
    server moves the release, the shipped digests describe a different artefact
    and are dropped: keeping them would either verify the wrong bytes or fail
    closed on every machine.
    """
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
    merged.update(shipped)  # shipped wins on a collision
    return merged


def uv_version(shipped: str) -> str:
    """The version of the bundled installer tool to fetch and to expect."""
    served = dial_str("install.uv.version", "")
    return served if _TAG_RE.match(served or "") else shipped


def uv_digests(shipped: dict[str, str], shipped_version: str) -> dict[str, str]:
    """Asset name to digest for the tool version in force."""
    return _digest_table(
        "install.uv.sha256", shipped, shipped_version, uv_version(shipped_version))


def uv_download_timeout_ms(shipped: int) -> int:
    """Inactivity timeout on the tool download."""
    return int(dial_in_range(
        "install.uv.download_timeout_ms", shipped, 10_000, 1_800_000))


def uv_http_timeout_s(shipped: int) -> int:
    """Seconds the installer waits on one stalled transfer before it gives up.

    The wheels it fetches are large, so the tool's own default is short enough
    to fail a slow link that would have finished.
    """
    return int(dial_in_range("install.uv.http_timeout_s", shipped, 30, 1800))


def uv_http_retries(shipped: int) -> int:
    """How many times the installer re-sends a transfer that failed."""
    return int(dial_in_range("install.uv.http_retries", shipped, 1, 20))


def python_release_tag(shipped: str) -> str:
    """The standalone-Python release the interpreter is fetched from."""
    served = dial_str("install.python.release_tag", "")
    return served if _TAG_RE.match(served or "") else shipped


def python_versions(shipped: dict[tuple[int, int], str]) -> dict[tuple[int, int], str]:
    """Minor version to patch version, with any server correction applied.

    ``install.python.versions`` is a ``{"3.12": "3.12.12"}`` map. Only a
    dotted-integer patch version is accepted, and only for a minor version this
    build already knows how to fetch: a new key would name an interpreter the
    rest of the plugin has no wheels for.
    """
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
    """Asset name to digest for the interpreter release in force."""
    return _digest_table(
        "install.python.sha256", shipped, shipped_tag, python_release_tag(shipped_tag))


# -- installer-error classifier vocabularies -------------------------------
# Which remedy a failed install offers is decided by matching the installer's
# own output against a keyword table. The installer is written in Rust and does
# its own TLS, so its wording is neither pip's nor OpenSSL's, and a tool version
# or a system language we have not seen can print a phrase the shipped table
# does not carry. A phrase we do not carry costs the user the right remedy and
# costs us the class the failure is reported under, since that class is the
# branch that matched.
#
# Union-only, like every other served list: a deploy may ADD a phrase and can
# never drop a shipped one, so no configuration can take a remedy away. The
# ORDER the classifiers run in stays in code, because that order carries meaning
# the server has no way to express.


def classifier_markers(name: str, shipped: Iterable[str]) -> tuple[str, ...]:
    """The phrases one installer-error classifier matches on, served extras included.

    ``install.classifier.<name>`` is a list of EXTRA phrases for the classifier
    called ``name``. Both sides come back lower-cased, because every caller
    matches against lower-cased installer output.

    Resolved per call rather than at import, so a configuration a later session
    mirrors to disk reaches a classifier that was imported before it existed.
    """
    return tuple(dial_list(
        f"install.classifier.{name}",
        tuple(item.lower() for item in shipped),
        normalize=str.lower,
    ))
