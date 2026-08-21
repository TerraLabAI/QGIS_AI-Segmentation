"""Pure decision logic for when an Automatic detection may run.

A run needs a typed prompt. Examples are optional and they sharpen it; they
never replace it. An example on its own says "find things that look like
this", which the model grounds far more loosely than a word does, and the
runs that came back empty or full of noise were the ones with no word in
them. The exclude (negative) example is a refinement offered only once the
positive set is strong enough.

The two minimums below are client fallbacks for the server's exemplar block,
read through the resolvers here so the gate and the example store always agree.

No Qt, no QGIS, no I/O: this is the single source of truth for the gate, shared
by the dock button, the run guard, and the headless/MCP path, and it is unit
tested directly.
"""
from __future__ import annotations

# Two references detect better than one: this threshold drives the
# second-example NUDGE and the exclude-button unlock. It does not block a run
# (the floor below is any non-empty query). Product decision, not a tuned
# value, and the client fallback for `exemplar.min_example_positives`.
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


# UNREACHABLE (2026-07-30): nothing in src/ calls this any more. It gated the
# green Detect button until the button moved to the plain can_detect floor, and
# it is kept because the same combination is still the model's most accurate
# mode, so a future nudge (never a gate) would ask exactly this question.
def meta_satisfied(has_text: bool, positives: int) -> bool:
    """True when the run uses the most accurate prompt-plus-example
    combination: text AND at least min_meta_positives() positive examples.

    Stricter than ``can_detect``, which is what decides whether a run may
    start. This one only describes quality, and gates nothing.
    """
    return has_text and positives >= min_meta_positives()


def can_detect(has_text: bool, positives: int, excludes: int = 0) -> bool:
    """True when a detection may start: there is a TYPED PROMPT.

    This is the whole gate on the green Detect button. Examples are optional
    on top of the word, never instead of it: the model grounds a word far
    tighter than a picture, and an example-only run is the one that comes back
    empty or full of look-alikes. Quality steering above this floor is advice
    only (the UI nudges toward a second example), never a block.

    ``positives`` and ``excludes`` never affect the decision; they are taken
    for a complete, self-documenting signature and because every caller
    already has them to hand.
    """
    return has_text


def exclude_available(positives: int) -> bool:
    """True when the exclude (negative) example may be offered.

    The exclude is a bonus refinement on an already-strong positive set, so it
    unlocks only once at least min_example_positives() positives exist. Below
    that it stays hidden.
    """
    return positives >= min_example_positives()
