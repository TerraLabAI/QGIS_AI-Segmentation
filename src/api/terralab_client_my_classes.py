

from __future__ import annotations

import json



_TIMEOUT_MY_CLASSES_MS = 200_000


class TerraLabMyClassesMixin:


    def post_my_classes(self, payload: dict, auth: dict | None = None) -> dict:




        from ..core.error_policy import LINK_OR_TIMEOUT_CODES

        body = json.dumps(payload).encode("utf-8")
        answer = None
        for _attempt in range(2):
            answer = self._request(
                "POST", "/api/ai-segmentation/my-classes", auth=auth, body=body,
                timeout_ms=_TIMEOUT_MY_CLASSES_MS, require_body=True, wall_clock=True)
            if not (isinstance(answer, dict) and answer.get("code") in LINK_OR_TIMEOUT_CODES):
                break
        return answer if isinstance(answer, dict) else {
            "error": "Unreadable answer", "code": "BAD_RESPONSE"}
