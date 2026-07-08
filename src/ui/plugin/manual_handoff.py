






from __future__ import annotations

from .manual_handoff_edit import (
    ManualHandoffEditMixin,
)
from .manual_handoff_fold import (
    ManualHandoffFoldMixin,
)
from .manual_handoff_select import (
    ManualHandoffSelectMixin,
)
from .manual_handoff_session import (
    ManualHandoffSessionMixin,
)


class ManualHandoffMixin(
    ManualHandoffSessionMixin,
    ManualHandoffFoldMixin,
    ManualHandoffSelectMixin,
    ManualHandoffEditMixin,
):
    pass


__all__ = [
    "ManualHandoffMixin",
]
