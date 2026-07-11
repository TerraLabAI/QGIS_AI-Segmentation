"""Pure decision logic for when an Automatic detection may run.

Reference-image (exemplar) detection is materially better with two or more
positive examples than with a single one, so the pure example-driven path
(no text prompt) requires at least two positives before a run is allowed. A
text prompt stands on its own and lifts that requirement (examples are then a
bonus on top of the prompt). The exclude (negative) example is a refinement
offered only once the positive set is strong enough.

No Qt, no QGIS, no I/O: this is the single source of truth for the gate, shared
by the dock button, the run guard, and the headless/MCP path, and it is unit
tested directly.
"""
from __future__ import annotations

# Two references detect far better than one, so the example-only path needs at
# least this many positives. Product decision, not a tuned value.
MIN_EXAMPLE_POSITIVES = 2


def can_detect(has_text: bool, positives: int, excludes: int = 0) -> bool:
    """True when a detection may start.

    - A valid text prompt is allowed with any number of examples (text stands
      on its own; examples are a bonus on top).
    - With no text, the pure example path needs at least MIN_EXAMPLE_POSITIVES
      positives: one positive alone must not run.

    ``excludes`` never affects the decision (an exclude is a refinement, never a
    query on its own); it is accepted for a complete, self-documenting signature.
    """
    if has_text:
        return True
    return positives >= MIN_EXAMPLE_POSITIVES


def exclude_available(positives: int) -> bool:
    """True when the exclude (negative) example may be offered.

    The exclude is a bonus refinement on an already-strong positive set, so it
    unlocks only once at least MIN_EXAMPLE_POSITIVES positives exist. Below that
    it stays hidden.
    """
    return positives >= MIN_EXAMPLE_POSITIVES
