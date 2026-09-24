







from __future__ import annotations

import json
import math

from ..core import transport_dials as _td
from ..core.server_dials import dial_in_range
from .terralab_client_primitives import (
    _TIMEOUT_API,
    _TIMEOUT_CHECKOUT_LINK,
    _TIMEOUT_INTERACTIVE,
    _TIMEOUT_TRANSLATE,
)
from .terralab_client_retry import (
    _note_skipped_tuning,
)






_account_shape: dict = {"bundles_usage": None}


class TerraLabAccountMixin:







    def charge_saved_object(self, auth: dict, session_id: str,
                            polygon_index: int,
                            area_m2: float | None = None,
                            polygon_wkt: str | None = None) -> dict:




















        fields: dict = {
            "session_id": str(session_id), "polygon_index": int(polygon_index)}
        try:
            if area_m2 is not None and not isinstance(area_m2, bool):
                area = float(area_m2)
                if math.isfinite(area) and area >= 0:
                    fields["area_m2"] = area
        except (TypeError, ValueError, OverflowError):
            pass  # nosec B110
        if polygon_wkt:
            fields["polygon_wkt"] = str(polygon_wkt)
        try:
            body = json.dumps(fields, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            body = json.dumps(
                {"session_id": str(session_id),
                 "polygon_index": int(polygon_index)},
                allow_nan=False,
            ).encode("utf-8")
        headers = dict(auth or {})
        try:
            from ..core.request_context import plugin_version

            version = plugin_version()
            if version:
                headers["x-plugin-version"] = version
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return self._request(
            "POST", "/api/ai-segmentation/save-polygon", auth=headers, body=body,
            timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE), require_body=True,
        )

    def get_usage(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/usage", auth=auth, timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
            require_body=True,
        )

    def get_account(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/account", auth=auth, timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
            require_body=True,
        )

    def delete_account(self, auth: dict, confirm: str) -> dict:















        body = json.dumps({"confirm": confirm}).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/account/delete",
            auth=auth, body=body,
            timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
        )

    def get_account_and_usage(self, auth: dict) -> tuple[dict, dict]:




















        if _account_shape["bundles_usage"] is not False:
            bundled = self._request(
                "GET", "/api/plugin/account?include=usage", auth=auth,
                timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
                require_body=True,
            )
            if not isinstance(bundled, dict) or "error" in bundled:
                failure = bundled if isinstance(bundled, dict) else {
                    "error": "Invalid server response", "code": "SERVER_ERROR"}
                return failure, failure
            usage = bundled.get("usage")
            if isinstance(usage, dict):
                _account_shape["bundles_usage"] = True
                return {k: v for k, v in bundled.items() if k != "usage"}, usage
            _account_shape["bundles_usage"] = False
        account, usage = self.request_many([
            {"method": "GET", "path": "/api/plugin/account",
             "auth": auth, "timeout_ms": _td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
             "require_body": True},
            {"method": "GET", "path": "/api/plugin/usage",
             "auth": auth, "timeout_ms": _td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
             "require_body": True},
        ])
        return account, usage

    def get_usage_with_account(self, auth: dict) -> dict:















        try:
            account, usage = self.get_account_and_usage(auth=auth)
        finally:





            try:
                if not self.retain_thread_nam():
                    self.release_thread_nam()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        if not isinstance(usage, dict) or "error" in usage:
            return usage if isinstance(usage, dict) else {
                "error": "Invalid server response", "code": "SERVER_ERROR"}
        account_ok: dict | None = (
            account if isinstance(account, dict) and "error" not in account
            else None)
        return {"usage": usage, "account": account_ok}

    def get_config(self, product: str, auth: dict | None = None) -> dict:






















        from ..core import config_cache
        from ..core.request_context import config_query

        path = config_query(product)
        etag = config_cache.config_etag()
        auth = auth or None
        answer = self._request(
            "GET", path, auth=auth,
            extra_headers={"If-None-Match": etag} if etag else None,
            timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
            require_body=True,
        )
        if isinstance(answer, dict) and answer.get("not_modified") is True:
            cached = config_cache.get_config()
            if cached:
                return cached



            return self._request(
                "GET", path, auth=auth,
                timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
                require_body=True,
            )
        if isinstance(answer, dict):
            new_etag = answer.pop("etag", None)
            if new_etag:
                config_cache.remember_etag(new_etag)
        return answer

    def get_segment_catalog(self, timeout_ms: int | None = None) -> dict:

        if timeout_ms is None:
            timeout_ms = _td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE)
        return self._request(
            "GET", "/api/ai-segmentation/presets", timeout_ms=timeout_ms, require_body=True)

    def translate_prompt(self, text: str, auth: dict | None = None) -> dict:







        body = json.dumps({"text": text}).encode("utf-8")
        answer = self._request(
            "POST", "/api/plugin/translate-prompt", auth=auth, body=body,
            timeout_ms=_td.translate_timeout_ms(_TIMEOUT_TRANSLATE), require_body=True,
        )
        _note_skipped_tuning("prompt translation", answer)
        return answer

    def get_seg_run_plan(
        self,
        prompt: str,
        zone_area_m2: float | None,
        native_mupp: float | None,
        auth: dict | None = None,
        exemplar_size_m: float | None = None,
        rewritten_from: str | None = None,
    ) -> dict:







        from ..core.request_context import plugin_version

        payload: dict = {
            "prompt": prompt,
            "zone_area_m2": zone_area_m2,
            "native_mupp": native_mupp,
            "plugin_version": plugin_version() or "unknown",
        }




        if exemplar_size_m is not None and exemplar_size_m > 0:
            payload["exemplar_size_m"] = float(exemplar_size_m)



        if isinstance(rewritten_from, str) and rewritten_from.strip():
            payload["rewritten_from"] = rewritten_from.strip()
        body = json.dumps(payload).encode("utf-8")
        answer = self._request(
            "POST", "/api/plugin/seg-run-plan", auth=auth, body=body,
            timeout_ms=_td.translate_timeout_ms(_TIMEOUT_TRANSLATE), require_body=True,
        )
        _note_skipped_tuning("run plan", answer)
        return answer

    def get_plugin_login_link(
        self,
        target: str,
        cta_source: str,
        auth: dict | None = None,
        locale: str | None = None,
    ) -> dict:













        from ..core.request_context import plugin_version

        payload: dict = {
            "target": target,
            "cta_source": cta_source,
            "plugin_version": plugin_version() or "unknown",
        }
        if locale:
            payload["locale"] = locale
        body = json.dumps(payload).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/login-link", auth=auth, body=body,
            timeout_ms=dial_in_range(
                "tuning.network.checkout_link_timeout_ms",
                _TIMEOUT_CHECKOUT_LINK, 1000, 30000),
            require_body=True,
        )







    def get_seg_history(
        self,
        auth: dict,
        limit: int = 12,
        before: str | None = None,
        favorites_only: bool = False,
        deleted: bool = False,
    ) -> dict:





        from urllib.parse import quote

        params = [f"limit={int(limit)}"]
        if before:
            params.append("before={}".format(quote(str(before), safe="")))
        if favorites_only:
            params.append("favorites_only=true")
        if deleted:
            params.append("deleted=true")
        path = "/api/ai-segmentation/history?" + "&".join(params)
        return self._request(
            "GET", path, auth=auth, timeout_ms=_td.api_timeout_ms(_TIMEOUT_API), require_body=True)

    def get_seg_run_detail(
        self,
        auth: dict,
        run_id: str | None = None,
        group_key: str | None = None,
    ) -> dict:





        from urllib.parse import quote

        if run_id:
            path = "/api/ai-segmentation/history/run?run_id={}".format(
                quote(str(run_id), safe=""))
        elif group_key:
            path = "/api/ai-segmentation/history/run?group_key={}".format(
                quote(str(group_key), safe=""))
        else:
            return {"error": "missing run identifier", "code": "CLIENT_ERROR"}
        return self._request(
            "GET", path, auth=auth, timeout_ms=_td.api_timeout_ms(_TIMEOUT_API), require_body=True)

    def set_seg_run_favorite(self, auth: dict, run_id: str, is_favorite: bool) -> dict:

        body = json.dumps({"run_id": run_id, "is_favorite": bool(is_favorite)}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/favorite",
            auth=auth, body=body, timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
        )

    def delete_seg_run(self, auth: dict, run_id: str) -> dict:

        body = json.dumps({"run_id": run_id}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/delete",
            auth=auth, body=body, timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
        )

    def undelete_seg_run(self, auth: dict, run_id: str) -> dict:

        body = json.dumps({"run_id": run_id}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/undelete",
            auth=auth, body=body, timeout_ms=_td.interactive_timeout_ms(_TIMEOUT_INTERACTIVE),
        )

    def fetch_run_masks(self, auth: dict, request_id: str) -> dict | list:






        from urllib.parse import quote

        path = "/api/ai-segmentation/image/{}?type=masks&stream=1".format(
            quote(str(request_id), safe=""))
        return self._request(
            "GET", path, auth=auth, timeout_ms=_td.api_timeout_ms(_TIMEOUT_API), allow_list=True,
            require_body=True)

    def poll_pairing(self, code: str, timeout_ms: int | None = None) -> dict:








        from urllib.parse import quote
        if timeout_ms is None:
            timeout_ms = dial_in_range("tuning.pairing.poll_timeout_ms", 10_000, 2000, 30000)
        return self._request(
            "GET",
            f"/api/plugin/pair/poll?code={quote(code, safe='')}",
            timeout_ms=timeout_ms,
            require_body=True,
        )

    def cancel_pairing(self, code: str) -> dict:





        body = json.dumps({"code": code, "product": "ai-segmentation"}).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/pair/cancel", body=body,
            timeout_ms=dial_in_range(
                "tuning.pairing.cancel_timeout_ms", 5_000, 1000, 30000),
        )
