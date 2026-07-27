"""Pure decision logic for when an Automatic detection may run.

Reference-image (exemplar) detection is materially better with two or more
positive examples than with a single one, so the pure example-driven path
(no text prompt) requires at least two positives before a run is allowed. A
text prompt stands on its own and lifts that requirement (examples are then a
bonus on top of the prompt). The exclude (negative) example is a refinement
offered only once the positive set is strong enough.

The two minimums below are client fallbacks for the server's exemplar block,
read through the resolvers here so the gate and the example store always agree.

No Qt, no QGIS, no I/O: this is the single source of truth for the gate, shared
by the dock button, the run guard, and the headless/MCP path, and it is unit
tested directly.
"""
from __future__ import annotations

# Two references detect far better than one: this threshold drives the
# second-example NUDGE and the exclude-button unlock. It no longer blocks a
# run (the floor below is any non-empty query; the single-example path runs
# through the explicit escape link). Product decision, not a tuned value, and
# the client fallback for `exemplar.min_example_positives`.
MIN_EXAMPLE_POSITIVES = 2

# The recommended default combination is a text prompt PLUS at least this many
# positive examples: the model grounds the word and the look together, which is
# its most accurate mode. Product decision, not a tuned value, and the client
# fallback for `exemplar.min_meta_positives`.
MIN_META_POSITIVES = 1


def _policy_min(getter_name: str, fallback: int) -> int:
    """One gate minimum resolved from the cached server policy, else the client
    constant. The reader is cache-only (no network, no disk), so this stays
    I/O-free and safe on any thread; any failure keeps the shipped value, and
    the exemplar ceilings resolve from the same block, so the gate and the
    example store cannot drift apart."""
    try:
        from . import detection_policy
        return int(getattr(detection_policy, getter_name)(fallback))
    except Exception:  # noqa: BLE001 - policy is optional, never break the gate
        return fallback


def min_example_positives() -> int:
    """Positives needed before the exclude example is offered."""
    return _policy_min("exemplar_min_example_positives", MIN_EXAMPLE_POSITIVES)


def min_meta_positives() -> int:
    """Positives the recommended prompt-plus-example combination asks for."""
    return _policy_min("exemplar_min_meta_positives", MIN_META_POSITIVES)


def meta_satisfied(has_text: bool, positives: int) -> bool:
    """True when the run uses the recommended prompt-plus-example combination.

    This is the DEFAULT path the UI steers toward (text AND at least
    min_meta_positives() positive examples). It is deliberately stricter than
    ``can_detect``: a run that passes the floor but not this check is allowed
    only through the explicit "detect anyway" escape, never as the default.
    """
    return has_text and positives >= min_meta_positives()


def can_detect(has_text: bool, positives: int, excludes: int = 0) -> bool:
    """True when a detection may start: the query is NON-EMPTY.

    The floor is deliberately permissive: a text prompt, or at least one
    positive example, each form a complete query the model can run. Quality
    steering lives ABOVE this floor (meta_satisfied gates the default green
    button; the single-example and text-only paths run only through the
    explicit escape link, and the UI nudges toward a second example).

    ``excludes`` never affects the decision (an exclude is a refinement, never a
    query on its own); it is accepted for a complete, self-documenting signature.
    """
    return has_text or positives >= 1


def exclude_available(positives: int) -> bool:
    """True when the exclude (negative) example may be offered.

    The exclude is a bonus refinement on an already-strong positive set, so it
    unlocks only once at least min_example_positives() positives exist. Below
    that it stays hidden.
    """
    return positives >= min_example_positives()
