

from __future__ import annotations

import threading
from contextlib import contextmanager

_request_scope = threading.local()


@contextmanager
def request_feedback(feedback):

    previous = getattr(_request_scope, "feedback", None)
    _request_scope.feedback = feedback
    try:
        yield feedback
    finally:
        _request_scope.feedback = previous


def current_request_feedback():

    return getattr(_request_scope, "feedback", None)


__all__ = ["current_request_feedback", "request_feedback"]
