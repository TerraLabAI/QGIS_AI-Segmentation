




















from __future__ import annotations

import math
import uuid



CREDITS_PER_OBJECT = 1


def credits_per_object() -> float:














    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "gate.credits_per_object", float(CREDITS_PER_OBJECT), 0.01, 1.0))
    except Exception:  # noqa: BLE001
        return float(CREDITS_PER_OBJECT)


def object_cost() -> int:







    return int(math.ceil(credits_per_object()))


def save_affordable(balance) -> bool:






    try:
        from .credit_gate import run_affordable

        return bool(run_affordable(object_cost(), balance))
    except Exception:  # noqa: BLE001
        return True


class ManualObjectLedger:












    def __init__(self, session_id: str | None = None) -> None:



        self.session_id: str = session_id or str(uuid.uuid4())
        self.remote_answered: bool = False
        self._charged: set[str] = set()
        self._wire_index: dict[str, int] = {}
        self._next_index: int = 0

    def _object_key(self, det_id) -> str:








        try:
            return str(int(det_id))
        except (TypeError, ValueError):
            return f"id:{det_id}"

    def note_remote_answer(self) -> None:

        self.remote_answered = True

    def start_next_object(self) -> None:


        self.remote_answered = False

    def object_is_billable(self, det_id) -> bool:





        if not self.remote_answered:
            return False
        return not self.already_charged(det_id)

    def already_charged(self, det_id) -> bool:

        return self._object_key(det_id) in self._charged

    def wire_index(self, det_id) -> int:








        key = self._object_key(det_id)
        known = self._wire_index.get(key)
        if known is not None:
            return known
        assigned = self._next_index
        self._next_index += 1
        self._wire_index[key] = assigned
        return assigned

    def mark_charged(self, det_id) -> None:



        self._charged.add(self._object_key(det_id))

    def charged_count(self) -> int:

        return len(self._charged)
