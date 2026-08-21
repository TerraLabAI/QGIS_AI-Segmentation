"""Server-tunable dials read from the cached product configuration.

A dial is a value the plugin ships with a working default for, and that the
server may override so the fleet can be retuned without a plugin release.

Every read is fail-open: the shipped constant wins whenever the cache is empty
(startup, offline, an older server), the key is absent, or the served value is
the wrong shape. No accessor here raises, networks, or touches disk, so it is
safe on the GUI thread and safe from a worker thread.

List dials are UNION-only on purpose. A deploy can ADD an entry but can never
remove a shipped one, so a bad config can never silence a shipped error code or
drop a shipped safety value, and a wrong addition is undone by the next deploy.
They are bounded too, in entry count and entry length, so no served list can
make a membership test expensive.

Served COPY is the one exception: it replaces the shipped sentence rather than
adding to it, because a line that reads badly has to be removable the same day.
Union-only guards the sets and lists that carry policy, where dropping a
shipped entry could relax something the plugin promises; a sentence promises
nothing on its own.

Pure Python with no Qt at import time, so the controller, the dock and the
headless path can all import it.
"""
from __future__ import annotations

import html
import math
import re
from itertools import islice
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

# One shared empty mapping for "no configuration". Handing back the same object
# every time keeps the change token below (object identity) meaningful when
# nothing has been published yet.
_NO_CONFIG: dict = {}

# A served list is a handful of short codes. These caps are what keeps a
# damaged or hostile configuration from turning a membership test into a lot of
# work, and they are applied before anything is stripped or copied.
_MAX_LIST_ENTRIES = 64
_MAX_ENTRY_CHARS = 128

# A served address is opened in the user's browser and pasted into rich text.
_MAX_URL_CHARS = 2048
# Characters a URL may not carry unencoded. Rejecting them here means the
# string cannot break out of an href attribute even before it is escaped.
_URL_FORBIDDEN = set('"<>\\^`{|}')


def _server_config() -> dict:
    """The cached product configuration, or ``{}``. Never raises.

    The returned object doubles as a change token: the store publishes a new
    dict on every change and never mutates one in place, so ``is`` on the
    result tells a caller whether the configuration it worked from still holds.
    """
    try:
        from .config_cache import get_config

        config = get_config()
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return _NO_CONFIG
    return config if isinstance(config, dict) else _NO_CONFIG


def read_value(path: str) -> Any:
    """Dotted-path lookup into the cached configuration, or None."""
    try:
        value: Any = _server_config()
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return None
    return value


def _is_finite_dial_value(value: Any) -> bool:
    """Whether a served value is a finite number and not a bool."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )


def dial(path: str, fallback):
    """Strictly positive numeric dial, coerced to the fallback's kind.

    An int fallback yields an int. Zero, a negative, a bool, a string or an
    infinity all return the shipped constant.
    """
    try:
        value = read_value(path)
        if not _is_finite_dial_value(value) or value <= 0:
            return fallback
        return type(fallback)(value)
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return fallback


def dial_in_range(path: str, fallback, low: float, high: float):
    """Numeric dial bounded by ``low`` and ``high``, both inclusive.

    Use it where zero is a legal value (a ratio, or a "0 means off" setting)
    instead of ``dial``, which rejects anything not strictly positive.
    """
    try:
        value = read_value(path)
        if not _is_finite_dial_value(value) or not low <= value <= high:
            return fallback
        return type(fallback)(value)
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return fallback


def dial_pair(path: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """Two-element ``(low, high)`` dial, for example a clamp range.

    Valid only as two finite numbers with ``0 < low <= high``; anything else
    returns the shipped constant.
    """
    try:
        value = read_value(path)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            if _is_finite_dial_value(low) and _is_finite_dial_value(high) and 0 < low <= high:
                return (float(low), float(high))
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        pass
    return fallback


def dial_list(
    path: str,
    base: Iterable[str] = (),
    normalize: Callable[[str], str] | None = None,
) -> frozenset:
    """Additive string-set dial: the served list holds EXTRA entries.

    The result is always a superset of the shipped ``base``. Blank and
    non-string entries are dropped; an absent or malformed list leaves the base
    unchanged. ``normalize`` (for example ``str.lower``) applies to served
    entries only, so casing drift in the configuration cannot defeat a match.

    Bounded on both axes: entries past the cap are never looked at, and an
    over-long entry is skipped before it is stripped or copied.
    """
    merged = set(base)
    try:
        value = read_value(path)
        if isinstance(value, (list, tuple)):
            for item in islice(value, _MAX_LIST_ENTRIES):
                if not isinstance(item, str) or len(item) > _MAX_ENTRY_CHARS:
                    continue
                entry = item.strip()
                if entry:
                    merged.add(normalize(entry) if normalize else entry)
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        pass
    return frozenset(merged)


def dial_str(path: str, fallback: str, allowed: Iterable[str] | None = None) -> str:
    """String dial. A non-blank served string wins, optionally restricted to
    ``allowed``; anything else returns the shipped constant."""
    try:
        value = read_value(path)
        if isinstance(value, str):
            value = value.strip()
            if value and (allowed is None or value in allowed):
                return value
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        pass
    return fallback


def safe_web_url(candidate: Any, fallback: str) -> str:
    """``candidate`` when it is a usable https web address, else ``fallback``.

    The single gate between an address we do not control and the two places one
    ends up: the desktop URL handler, and the href of a rich-text label. Both
    are dangerous with the wrong string, so the rules are strict and the same
    for both. Rejected: anything that is not a string, an over-long one, any
    control character or inner whitespace, any character a URL may not carry
    unencoded, any scheme other than https, and anything with no host.

    https only. An http address would be a downgrade we chose ourselves, and
    any other scheme hands the string to a local application instead of the
    browser. The accepted string comes back as written, minus outer whitespace.
    """
    try:
        if not isinstance(candidate, str):
            return fallback
        text = candidate.strip()
        if not text or len(text) > _MAX_URL_CHARS:
            return fallback
        # Checked on the raw string: the parser below silently drops tabs and
        # newlines, so checking the parsed parts would look past them.
        for ch in text:
            if ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F or ch in _URL_FORBIDDEN:
                return fallback
        parts = urlsplit(text)
        if parts.scheme.lower() != "https" or not parts.hostname:
            return fallback
        return text
    except Exception:  # noqa: BLE001 -- a bad address is not an error  # nosec B110
        return fallback


def dial_url(path: str, fallback: str) -> str:
    """Web-address dial: the served address only when it is a usable https one."""
    return safe_web_url(read_value(path), fallback)


def dial_bool(path: str, fallback: bool) -> bool:
    """Boolean dial. Only a real ``true`` or ``false`` wins; 0, 1, "yes" and
    every other shape return the shipped constant."""
    try:
        value = read_value(path)
        if isinstance(value, bool):
            return value
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        pass
    return fallback


# -- served copy -----------------------------------------------------------
# Text the user reads. The server sends it already written in the language the
# plugin asked for (see request_context), so it carries no tr() and needs none.
# Same contract and same key names as the sibling plugin, so one blob can
# describe both.
#
# Three properties make it safe to render:
#   - it replaces ONE shipped string at a time, so a bad entry costs that
#     string and never the screen around it;
#   - it is capped, stripped of control characters, and stripped of the angle
#     brackets that could open a tag, here, before any widget sees it;
#   - the call sites that build rich text escape it on top of that.
#
# The angle brackets are REMOVED rather than escaped because these labels are
# AutoText: Qt decides per string whether to render markup, so an escaped
# entity would show as itself ("&#x27;") in a label that decided the text was
# plain. Removing the character is the only answer that reads correctly under
# both decisions. A served string cannot contain "<" or ">".
#
# Absent config, absent entry, wrong type or an empty string all mean the
# shipped string, which is what a user sees today.
_MAX_COPY_CHARS = 400
# Tab and newline stay: a sentence may legitimately wrap. Everything else in
# the control ranges goes, including the C1 block.
_COPY_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f<>]")


def clean_served_text(value: Any, max_chars: int = _MAX_COPY_CHARS) -> str | None:
    """One served string, cleaned, or None when nothing usable is left.

    The cap applies BEFORE the scrub, so an over-long value is never fully
    copied.
    """
    if not isinstance(value, str):
        return None
    text = _COPY_CTRL_RE.sub("", value[:max_chars]).strip()
    return text or None


def dial_text(container_path: str, key: str, max_chars: int = _MAX_COPY_CHARS) -> str | None:
    """One entry of a served ``{key: text}`` map, or None when unusable.

    Validation is per entry, so one bad entry never costs the others.
    """
    try:
        container = read_value(container_path)
        if isinstance(container, dict):
            return clean_served_text(container.get(key), max_chars)
    except Exception:  # noqa: BLE001 -- copy is best-effort  # nosec B110
        pass
    return None


# The configuration request asks for a language, and the server can only
# answer in the few it is authored in. Every other UI language gets a served
# sentence in a language the reader did not choose, while the plugin ships that
# same sentence translated. So a served sentence is only allowed to replace the
# shipped one when the request could name the reader's language.
_served_copy_allowed: bool | None = None


def served_copy_allowed() -> bool:
    """True when a served sentence may replace the shipped, translated one.

    The QGIS UI language is fixed for the session, so this resolves once.
    """
    global _served_copy_allowed
    if _served_copy_allowed is None:
        try:
            from .request_context import ui_language

            _served_copy_allowed = ui_language() is not None
        except Exception:  # noqa: BLE001 -- copy is best-effort
            _served_copy_allowed = False
    return _served_copy_allowed


def dial_copy(
    string_id: str,
    fallback: str,
    max_chars: int = _MAX_COPY_CHARS,
    escape: bool = False,
) -> str:
    """Served replacement for one shipped user-facing string, or that string.

    ``string_id`` is a FLAT key under ``copy``, so it may carry dots without
    the server having to nest anything.

    ``escape=True`` for a label the plugin builds rich text into (a link, a
    span): the only character left to deal with there is "&", and it has to
    become an entity or the parser eats it. The shipped fallback comes back
    untouched either way, so an absent entry changes nothing at all.

    Placeholders in a served sentence must be filled with ``str.replace``,
    never ``str.format``: a stray brace in served text would raise on a paint
    or click path.
    """
    if not served_copy_allowed():
        return fallback
    served = dial_text("copy", string_id, max_chars)
    if served is None:
        return fallback
    return html.escape(served, quote=False) if escape else served


class ServerDialSet(frozenset):
    """Read-through frozenset: ``in`` also matches served extras at ``path``.

    The union is additive (see the module docstring), so a deploy can only add
    members. Iteration and set algebra see the shipped entries only, which
    keeps the shipped set the one that is displayed or exported.
    """

    def __new__(cls, path: str, base: Iterable[str], normalize=None):
        obj = super().__new__(cls, base)
        obj._path = path
        obj._normalize = normalize
        obj._memo = None
        return obj

    def _served(self) -> frozenset:
        """The served extras for the configuration in force.

        Membership is a hot path, so the resolved extras are kept until the
        configuration is replaced by a new one. The memo is a single tuple,
        written and read in one step, so a worker thread reading it while
        another publishes sees either the old pair or the new one and never a
        mixture. That is why there is no lock here.
        """
        config = _server_config()
        memo = self._memo
        if memo is not None and memo[0] is config:
            return memo[1]
        served = dial_list(self._path, (), normalize=self._normalize)
        self._memo = (config, served)
        return served

    def __contains__(self, item) -> bool:
        if frozenset.__contains__(self, item):
            return True
        return item in self._served()


class ServerDialMap(dict):
    """Read-through mapping: ``[]`` and ``get`` prefer the served entry.

    Each entry is validated on its own by ``dial``, so one malformed served
    value leaves the shipped value in place instead of poisoning the map. Keys
    stay the shipped set, so the server can retune an entry but never add or
    remove one.
    """

    def __init__(self, path: str, defaults: dict):
        super().__init__(defaults)
        self._path = path

    def __getitem__(self, key):
        return dial(f"{self._path}.{key}", dict.__getitem__(self, key))

    def get(self, key, default=None):
        if key in self:
            return self[key]  # routes through the dial-aware __getitem__
        return default


# -- feature switches ------------------------------------------------------


def feature_enabled(name: str) -> bool:
    """Server kill switch for one named feature, under the ``features`` key.

    Fail-open: an absent map, an absent key or garbage all mean enabled. Only
    an explicit ``false`` disables, so a feature can be turned off fleet-wide
    without a release, and never turns itself off by accident.

    A configuration in force always decides. When it has no opinion, which is
    every cold start until the fetch lands, the last live answer stands
    instead of the shipped default: a feature withdrawn because it is broken
    used to be back on at every restart, on the machines the withdrawal was
    written for. See ``kill_switch_memory``, which can only say no.
    """
    return feature_switch("features." + name, True)


def feature_switch(path: str, shipped: bool) -> bool:
    """One kill switch, with the last live answer standing in for a cold cache.

    The configuration in force decides whenever it has an opinion, exactly
    like ``dial_bool``. When it has none, and the switch ships ON, the last
    live answer is consulted before the shipped default: a feature withdrawn
    because it is broken used to come back at every restart, because the disk
    copy of the configuration deliberately carries no kill switch.

    A switch that ships OFF needs no memory. It is already off, and the
    memory, by design, can only say off.
    """
    served = read_value(path)
    if served is False:
        return False
    if served is True:
        return True
    if not shipped:
        return False
    name = path.rsplit(".", 1)[-1]
    try:
        from .kill_switch_memory import is_remembered_off

        return not is_remembered_off(name)
    except Exception:  # noqa: BLE001 -- a memory is best-effort  # nosec B110
        return True


def correct_ai_cloud_enabled() -> bool:
    """Whether the Automatic review's AI fix is answered off the machine.

    Default ON. The AI lane of the Correct step is a paid, server-answered
    lane: it is the same model the run itself used, it costs a credit per fix,
    and the on-device model is no longer an answer for it. The free way to fix
    a polygon in Correct is the Manual lane beside it, which needs no network,
    no account and no install.

    An explicit false still withdraws the lane fleet-wide, and the step then
    falls back to Manual, which is where a refused click already lands.

    Semi-Auto never reads this. Its own engine cards decide, and its on-device
    side stays offline by product rule.
    """
    return feature_switch("features.correct_ai_cloud", True)


def crop_webp_enabled() -> bool:
    """Whether the click path may pack its crop as lossless WebP.

    Default OFF, like correct_ai_cloud above and for the same reason: the
    format is a wire contract, and every server has to accept webp before any
    client sends it. Off keeps today's PNG, so a plugin that never hears from
    the server changes nothing. Deliberately NOT feature_enabled(), which is
    fail-open and would turn this on for a fleet the server never reached.
    """
    return dial_bool("features.crop_webp", False)


def gzip_request_bodies_enabled() -> bool:
    """Whether a large request body may travel gzipped.

    Default ON: every route that takes a large body reads the encoding, so a
    plugin that never hears from the server still gets the shorter upload, and
    a client whose dial read fails on a slow link is exactly the one that
    needs it. A served false is what takes it back from the whole fleet.
    """
    return feature_switch("features.gzip_request_bodies", True)


def automatic_mode_enabled() -> bool:
    """Whether Automatic mode is available.

    Honours the original top-level ``automatic_mode_enabled`` flag and the
    generic ``features.automatic_mode`` switch. Either one saying no is enough.

    Only an explicit ``false`` says no, exactly like ``feature_enabled``. A
    served 0, "", [] or {} is garbage, and garbage means "use what shipped",
    the same as an absent value: nothing the server can get wrong may take
    Automatic mode away from the whole fleet.

    The two keys share one entry in the memory of what was last turned off, so
    the memory is read here rather than through ``feature_switch``: a live
    ``automatic_mode_enabled: false`` beside a ``features.automatic_mode:
    true`` is a no, and the copy on disk keeps only the true half of it. Read
    through the generic switch, that true would answer for the pair at the
    next start and hand back a mode the server had withdrawn.
    """
    if read_value("automatic_mode_enabled") is False:
        return False
    if read_value("features.automatic_mode") is False:
        return False
    try:
        from .kill_switch_memory import AUTOMATIC_MODE_NAME, is_remembered_off

        return not is_remembered_off(AUTOMATIC_MODE_NAME)
    except Exception:  # noqa: BLE001 -- a memory is best-effort  # nosec B110
        return True


# -- version comparison ----------------------------------------------------


def parse_version(text) -> tuple[int, ...] | None:
    """``"1.6.9"`` -> ``(1, 6, 9)``. None on anything that is not dotted ints."""
    if not isinstance(text, str):
        return None
    parts = text.strip().lstrip("vV").split(".")
    out = []
    for part in parts:
        part = part.strip()
        if not part.isdigit():
            return None
        out.append(int(part))
    return tuple(out) if out else None


def is_served_update_recommended(installed_version: str) -> bool:
    """True when the served ``min_recommended_version`` parses strictly higher
    than the installed version. Garbage or missing on either side means no."""
    installed = parse_version(installed_version)
    minimum = parse_version(read_value("min_recommended_version"))
    if installed is None or minimum is None:
        return False
    width = max(len(installed), len(minimum))
    installed += (0,) * (width - len(installed))
    minimum += (0,) * (width - len(minimum))
    return minimum > installed
