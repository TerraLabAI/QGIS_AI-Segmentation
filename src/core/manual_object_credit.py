"""What a saved object costs in Semi-Auto, and what has already been paid for.

Pure Python: no Qt, no QGIS, no network, so the rule the user is billed by sits
in one readable place instead of being re-derived inside a Save handler.

Three decisions live here, and only these three:

- **The billed unit is the object, not the click.** A user places as many points
  as an outline needs, and a click still being adjusted has produced nothing
  they keep. So the credit lands at Save, once per object.
- **Only what the network answered is billed.** A session whose clicks were all
  answered on the user's own machine costs nothing, which is what keeps the
  offline mode free and unlimited exactly as it shipped.
- **An object is charged once for its lifetime.** Re-open it, correct it, save
  it again: the ledger already holds its identity, so nothing more is spent.

The affordability boundary is NOT re-derived here. It is
``credit_gate.run_affordable``, the same primitive Automatic's Detect gate uses,
called with the cost of one object, so the two modes can never disagree about
what an empty balance means.
"""
from __future__ import annotations

import uuid

# What one saved object costs when the server names no rate. The shipped
# figure, and the fallback ``credits_per_object`` falls to.
CREDITS_PER_OBJECT = 1


def credits_per_object() -> float:
    """How many credits one saved object costs. One, unless the server says
    otherwise.

    The single place the object-to-credit rate is decided, so no Save handler
    re-derives it. Same shape and same cap as ``credit_gate.credits_per_tile``,
    which does the job for Automatic.

    Capped at one, and that cap is the point. The panel promises one credit per
    object saved before the user saves anything, so a rate ABOVE one would bill
    more than that sentence said. A rate BELOW one only ever bills less than
    was shown, so it is safe from a screen that cannot express it: that is the
    discount lever, and it is the one worth having during an outage. Zero is
    refused too, since a free save still consumes the service.
    """
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "gate.credits_per_object", float(CREDITS_PER_OBJECT), 0.01, 1.0))
    except Exception:  # noqa: BLE001 -- billing math must never break on config
        return float(CREDITS_PER_OBJECT)


def save_affordable(balance) -> bool:
    """True when one more object may be saved against ``balance`` credits.

    An unknown balance (``None``) is always affordable: the caller has no number
    to refuse on, and a plugin that has not read the account yet must not be the
    thing that stops a user working.
    """
    try:
        from .credit_gate import run_affordable

        return bool(run_affordable(credits_per_object(), balance))
    except Exception:  # noqa: BLE001 -- a gate must never break on config
        return True


class ManualObjectLedger:
    """One Semi-Auto session's billing memory.

    Minted when a session starts with its clicks going to TerraLab's servers,
    dropped when that session ends. It carries no account and no balance: it
    only answers what this session has already paid for.

    ``remote_answered`` is written from the thread that answers a click and read
    on the GUI thread at Save. A plain bool assignment is what makes that safe,
    so keep it one: nothing here may grow into a structure a reader could catch
    half-written.
    """

    def __init__(self, session_id: str | None = None) -> None:
        # The idempotency key the server dedupes on. A retry after a dropped
        # answer must reach the same key, so it is minted once per session and
        # never per request.
        self.session_id: str = session_id or str(uuid.uuid4())
        self.remote_answered: bool = False
        self._charged: set[int] = set()
        self._wire_index: dict[int, int] = {}
        self._next_index: int = 0

    def note_remote_answer(self) -> None:
        """A click on the object being traced came back from the network."""
        self.remote_answered = True

    def start_next_object(self) -> None:
        """The object just saved is behind us. What the next one costs depends
        on where ITS clicks are answered, so the flag starts clean."""
        self.remote_answered = False

    def object_is_billable(self, det_id) -> bool:
        """Whether saving ``det_id`` right now should spend a credit.

        No network answer means no charge, whatever the option says: a click the
        machine answered because the link was down is not something to bill.
        """
        if not self.remote_answered:
            return False
        return not self.already_charged(det_id)

    def already_charged(self, det_id) -> bool:
        """Whether this object has been paid for in this session."""
        try:
            return int(det_id) in self._charged
        except (TypeError, ValueError):
            return False

    def wire_index(self, det_id) -> int:
        """The number this object travels under, stable for the whole session.

        The server keys its charge on (session, index), so an object that is
        sent twice has to carry the same index both times or the second send
        would look like a new object and cost a second credit.
        """
        try:
            key = int(det_id)
        except (TypeError, ValueError):
            key = -1
        known = self._wire_index.get(key)
        if known is not None:
            return known
        assigned = self._next_index
        self._next_index += 1
        self._wire_index[key] = assigned
        return assigned

    def mark_charged(self, det_id) -> None:
        """Record that this object is paid for. Called on the server's answer,
        never on the send: an object whose charge never landed must stay
        billable, or the user would get it free by losing their connection."""
        try:
            self._charged.add(int(det_id))
        except (TypeError, ValueError):
            pass

    def charged_count(self) -> int:
        """How many objects this session has paid for."""
        return len(self._charged)
