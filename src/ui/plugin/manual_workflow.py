






from __future__ import annotations

from .manual_workflow_canvas import (
    ManualWorkflowCanvasMixin,
)
from .manual_workflow_export import (
    ManualWorkflowExportMixin,
)
from .manual_workflow_refine import (
    ManualWorkflowRefineMixin,
)
from .manual_workflow_save import (
    ManualWorkflowSaveMixin,
)
from .manual_workflow_session import (
    ManualWorkflowSessionMixin,
)
from .manual_workflow_start import (
    ManualWorkflowStartMixin,
)


class ManualWorkflowMixin(
    ManualWorkflowStartMixin,
    ManualWorkflowCanvasMixin,
    ManualWorkflowSaveMixin,
    ManualWorkflowExportMixin,
    ManualWorkflowSessionMixin,
    ManualWorkflowRefineMixin,
):

    pass


__all__ = [
    "ManualWorkflowMixin",
]
