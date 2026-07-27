"""The cached product configuration: memory, disk, and the local override.

The configuration is fetched once per session by a hidden background task.
Kept in memory only, a user who is offline at startup would have no
configuration at all, and the install path (which runs before any successful
call) could never benefit from it. So the last good fetch is mirrored to a
small JSON file next to the plugin's other cached state, written atomically.

Every read is a memory read. The file is touched on the background task's
thread only, by ``prime_from_disk`` before the fetch and by ``set_config``
after it, and each of them publishes a resolved snapshot. ``get_config`` then
hands that snapshot back with no disk access at all, which is what makes it
safe to call while the dock is being built.

Precedence, resolved once at publish time: this session's fetch, then the copy
left on disk by an earlier session, with the local override merged on top of
whichever won. The override file is absent from a released plugin.

The disk copy is not authenticated. Anything with write access to the user's
home directory can put a syntactically valid file there, so a value read back
from disk MAY retune the plugin and may NEVER switch a part of it off: kill
switches are stripped on the way in, and only a live fetch can disable a
feature. That keeps the offline value of the copy without its risk, and no
signature or key is needed for it.

The copy holds only values the server already serves to any client that asks,
so it adds no secret to disk. No expiry either: a stale configuration is far
better than none, and the timestamp is stored so a caller can weigh its age.

Pure Python with no Qt at import time, so the controller, the dock, the
headless path and the dial readers can all reach it. Every function is
best-effort and never raises: a missing directory, a read-only volume or a
truncated file all mean "no cached configuration".
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import NamedTuple

from .cache_paths import PLUGIN_CACHE_DIR

CONFIG_FILENAME = "server_config.json"

# A configuration blob is a few kilobytes. Refuse anything wildly larger rather
# than pulling a damaged or foreign file into memory.
_MAX_BYTES = 2 * 1024 * 1024

# Stamped on every file this module writes. A payload without it was not
# written here as the mirror of a live fetch, so it is not read back at all.
_FILE_SOURCE = "live_fetch"

# Where the configuration in force came from. Only LIVE is allowed to disable a
# feature; see the module docstring.
SOURCE_NONE = "none"
SOURCE_DISK = "disk"
SOURCE_LIVE = "live"


class _Snapshot(NamedTuple):
    """One published configuration, replaced whole and never edited in place."""

    config: dict
    fetched_at: float | None
    source: str


_EMPTY = _Snapshot({}, None, SOURCE_NONE)

# The one published state. A reader takes a single reference to it, so it can
# never see a publish half applied and neither side needs a lock.
_state: _Snapshot = _EMPTY

_override: dict | None = None
_override_loaded = False


# -- the file --------------------------------------------------------------


def config_cache_path() -> str:
    """Absolute path of the cached configuration file (may not exist)."""
    return os.path.join(PLUGIN_CACHE_DIR, CONFIG_FILENAME)


# Windows refuses to rename onto a file another process holds open, and a
# scanner or a second QGIS reads this one at every start. Retrying costs under
# a second on a background thread; giving up costs the offline mirror for the
# whole session, and with it the server dials on the next offline start.
_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY_S = 0.2


def _move_config_into_place(tmp_path: str, path: str) -> None:
    """Move the finished temporary file onto the config, past a transient lock.

    Raises the last PermissionError when every attempt lost, so the caller
    reports the write as failed exactly as before.
    """
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_DELAY_S)


def save_config(config: dict) -> bool:
    """Mirror a freshly fetched configuration to disk. True when written.

    Writes a file, so it belongs on a background thread. Atomic: the payload
    lands in a temporary file in the same directory and is moved into place
    with os.replace, so a crash mid-write can never leave a half-written file
    behind for the next start to read.
    """
    if not isinstance(config, dict) or not config:
        return False
    path = config_cache_path()
    tmp_path = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = json.dumps(
            {"source": _FILE_SOURCE, "fetched_at": time.time(), "config": config}
        )
        handle, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix=".server_config-", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)
        _move_config_into_place(tmp_path, path)
    except Exception:  # noqa: BLE001 -- cache write is best-effort  # nosec B110
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # nosec B110
        return False
    return True


def _without_kill_switches(config: dict) -> dict:
    """A copy of ``config`` that cannot turn any part of the plugin off.

    Applied to everything read back from disk. The file is not authenticated,
    so it may retune but never disable: every switch that would say no is
    dropped, and the shipped default (on) stands until a live fetch says
    otherwise.
    """
    out = dict(config)
    if out.get("automatic_mode_enabled") is not True:
        out.pop("automatic_mode_enabled", None)
    features = out.get("features")
    if isinstance(features, dict):
        kept = {name: on for name, on in features.items() if on is True}
        if len(kept) != len(features):
            out["features"] = kept
    elif features is not None:
        # A "features" that is not a map disables nothing today, but it helps
        # nothing either. Drop it so no later reader has to think about it.
        out.pop("features", None)
    return out


# Install dials that choose WHAT gets downloaded or run, as opposed to how long
# to wait for it. Anything here turns a write to one unauthenticated JSON file
# into code execution at the next dependency repair, so the disk copy does not
# carry them and the shipped constants stand until a live fetch replaces them.
_INSTALL_KEYS_THAT_RUN_CODE = ("packages", "torch_index_url")
_INSTALL_SUBKEYS_THAT_RUN_CODE = {
    # Moving the release tag makes the digest merge drop the shipped digests
    # (they describe a different artefact), leaving a served table authoritative.
    "python": ("release_tag",),
    "uv": ("version",),
}


def _without_code_execution_dials(config: dict) -> dict:
    """A copy of ``config`` whose install section can only retune, never
    choose a different package, index or release to fetch and run."""
    install = config.get("install")
    if not isinstance(install, dict):
        return config
    out = dict(config)
    safe = {k: v for k, v in install.items() if k not in _INSTALL_KEYS_THAT_RUN_CODE}
    for parent, subkeys in _INSTALL_SUBKEYS_THAT_RUN_CODE.items():
        child = safe.get(parent)
        if isinstance(child, dict):
            safe[parent] = {k: v for k, v in child.items() if k not in subkeys}
    out["install"] = safe
    return out


def load_config() -> tuple[dict, float | None]:
    """The cached configuration and the time it was fetched.

    Reads the file, so it belongs on a background thread. Kill switches are
    stripped on the way out (see ``_without_kill_switches``), so what the
    caller gets can only retune.

    Returns ``({}, None)`` when there is nothing usable on disk: absent file,
    unreadable directory, corrupt JSON, a file this module did not write, or an
    unexpected shape.
    """
    path = config_cache_path()
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MAX_BYTES:
            return {}, None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 -- cache read is best-effort  # nosec B110
        return {}, None
    if not isinstance(data, dict) or data.get("source") != _FILE_SOURCE:
        return {}, None
    config = data.get("config")
    if not isinstance(config, dict) or not config:
        return {}, None
    fetched_at = data.get("fetched_at")
    if not isinstance(fetched_at, (int, float)) or isinstance(fetched_at, bool):
        fetched_at = None
    return _without_code_execution_dials(_without_kill_switches(config)), fetched_at


def clear_config() -> None:
    """Forget the configuration in force and delete the copy on disk.

    Back to the state of a cold start: nothing published, nothing on disk, and
    the override file to be read again. This is the one reset point, so a
    caller never has to reach into the module's internals to get a clean store.
    """
    global _state, _override, _override_loaded
    _state = _EMPTY
    _override = None
    _override_loaded = False
    try:
        os.unlink(config_cache_path())
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        pass


# -- the local override ----------------------------------------------------


def override_path() -> str:
    """Absolute path of the local override file (may not exist)."""
    plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(plugin_root, ".debug", "dev_policy.json")


def get_override() -> dict:
    """The local override dict, or {} when absent or unreadable. Read once.

    Reads a file on the first call, so it belongs on a background thread, like
    the two publishers that call it. Deep-merged over the cached configuration
    so a value can be exercised in QGIS without a server change. The file is
    absent from a released plugin, so this is a no-op there.
    """
    global _override, _override_loaded
    if _override_loaded:
        return _override or {}
    _override_loaded = True
    try:
        path = override_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _override = data
    except Exception:  # noqa: BLE001 -- optional file, best-effort  # nosec B110
        _override = None
    return _override or {}


def deep_merge(base: dict, override: dict) -> dict:
    """A copy of base with override merged in, recursing into nested dicts."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


# -- the store -------------------------------------------------------------


def _resolve_with_override(config: dict) -> dict:
    """``config`` with the local override merged on top."""
    override = get_override()
    return deep_merge(config, override) if override else config


def prime_from_disk() -> bool:
    """Publish the copy an earlier session left on disk. True when one was.

    Reads the file and the override, resolves both into one snapshot and
    publishes it. This and ``set_config`` are the only places the configuration
    touches disk, and both belong on the background task's thread.

    Does nothing once anything is in force, so the file is read at most once per
    session and a late call can never put an older configuration back. Never
    raises.
    """
    global _state
    if _state.source != SOURCE_NONE:
        return False
    try:
        config, fetched_at = load_config()
        resolved = _resolve_with_override(config)
    except Exception:  # noqa: BLE001 -- the disk copy is best-effort  # nosec B110
        return False
    if not resolved:
        return False
    _state = _Snapshot(resolved, fetched_at, SOURCE_DISK)
    return True


def set_config(config: dict) -> None:
    """Publish a freshly fetched configuration and mirror it to disk.

    Belongs on the thread that did the fetch, never on the GUI thread: it
    writes a file. The write is best-effort, since the published snapshot
    serves this session either way. The override is merged into what is
    published but not into what is written, so a dev file never lands in the
    copy an offline start will read.
    """
    global _state
    if not isinstance(config, dict) or not config:
        return
    try:
        resolved = _resolve_with_override(config)
    except Exception:  # noqa: BLE001 -- the override is best-effort  # nosec B110
        resolved = config
    _state = _Snapshot(resolved, time.time(), SOURCE_LIVE)
    save_config(config)


def get_config() -> dict:
    """The configuration in force, or ``{}`` when there is none.

    A pure memory read: no disk, no network, no lock, nothing to block on. Safe
    on the GUI thread and from any worker. Callers all fail open on an empty
    dict.

    The result is the published snapshot itself, so read it and never edit it.
    A publish replaces the whole object rather than changing this one, which is
    what lets a caller use ``is`` to tell whether its configuration still
    holds.
    """
    return _state.config


def config_source() -> str:
    """Where the configuration in force came from.

    ``live`` for this session's fetch, ``disk`` for the copy an earlier session
    left, ``none`` when there is no configuration. Only a live one is trusted
    to disable a feature.
    """
    return _state.source


def config_age_s() -> float | None:
    """Seconds since the configuration in force was fetched, or None.

    None means no configuration, or one whose fetch time is unknown. Nothing
    expires on age; this only lets a caller weigh how old the values are.
    """
    fetched_at = _state.fetched_at
    if fetched_at is None:
        return None
    return max(0.0, time.time() - fetched_at)
