













from __future__ import annotations

import time

from ...core.self_exemplar import (
    SELF_EXEMPLAR_INDEX_OFFSET,
    self_exemplar_boxes_for_image,
    self_exemplar_wire_index,
)
from .retry_policy import _CachedReply


class AutoSelfExemplarMixin:


    def _self_ex_log(self, message: str) -> None:

        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                f"Auto detection: {message}", "AI Segmentation",
                level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _self_ex_tile_eligible(self, reply, tile_idx: int, png_bytes) -> bool:


        if not self._self_ex_settings or self._stop_requested:
            return False
        if isinstance(reply, _CachedReply) or not png_bytes:
            return False
        if tile_idx in self._self_ex_held or tile_idx in self._self_ex_tried:
            return False
        if not 0 <= tile_idx < SELF_EXEMPLAR_INDEX_OFFSET:
            return False

        if self._tile_depth.get(tile_idx, 0) != 0 or tile_idx in self._parent_of:
            return False
        if self._stamps or self._exemplar_stamps_in or self._tile_exemplars.get(tile_idx):
            return False
        return bool((self._prompt or "").strip())

    def _build_self_exemplar_submission(
        self, tile_idx: int, png_bytes: bytes, boxes: list,
    ) -> dict:



        from ...core.cloud_detection import mask_scale_field, tile_png_to_base64

        submission = {
            "run_id": self._run_id,
            "prompt": self._prompt,
            "image_b64": tile_png_to_base64(png_bytes),
            "tile_index": self_exemplar_wire_index(tile_idx),
            "crs_authid": self._crs_authid,
            "max_masks": self._max_masks,
            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": boxes,


            "parent_tile_index": tile_idx,
            "self_exemplar_of": tile_idx,
        }
        run_mask_scale = mask_scale_field(self._mask_scale)
        if run_mask_scale is not None:
            submission["mask_scale"] = run_mask_scale
        return submission

    def _self_ex_try_fire(
        self, reply, tile_idx: int, tile_spec, tile_transform, png_bytes,
        first_answer: dict, in_flight: dict,
    ) -> bool:



        if not self._self_ex_tile_eligible(reply, tile_idx, png_bytes):
            return False
        self._self_ex_tried.add(tile_idx)
        settings = self._self_ex_settings
        boxes = self_exemplar_boxes_for_image(
            first_answer, png_bytes, settings["top_k"], settings["min_score"])
        n_masks = len(first_answer.get("masks") or [])
        if not boxes:
            self.self_ex_no_confident += 1
            self._self_ex_log(
                f"tile {tile_idx} pass 1 answered ({n_masks} masks), "
                f"pass 2 skipped: no mask above {settings['min_score']:.2f}")
            return False
        submission = self._build_self_exemplar_submission(tile_idx, png_bytes, boxes)
        try:
            reply2 = self._client.post_detection_async(submission, self._auth)
        except Exception as exc:  # noqa: BLE001
            self._self_ex_note_fallback("POST_ERROR")
            self._self_ex_log(
                f"tile {tile_idx} pass 2 not sent ({type(exc).__name__}); "
                "keeping pass 1")
            return False
        now = time.monotonic()
        self._self_ex_held[tile_idx] = (first_answer, tile_transform)
        self.self_ex_sent += 1
        self.self_ex_upload_bytes += len(submission["image_b64"])
        in_flight[reply2] = (
            tile_idx, tile_spec, tile_transform, png_bytes,
            now + self._stream_reply_budget_s)


        self._submit_at[tile_idx] = now
        self._self_ex_sent_at[tile_idx] = now
        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        self._upload_last_sent.pop(tile_idx, None)
        self._reply_byte_at.pop(tile_idx, None)
        self._watch_upload(reply2, tile_idx)
        self._self_ex_log(
            f"tile {tile_idx} pass 1 answered ({n_masks} masks), pass 2 sent "
            f"as wire tile {submission['tile_index']} with {len(boxes)} box(es)")
        return True

    def _self_ex_take_answer(self, tile_idx: int, answer: dict) -> bool:



        held = self._self_ex_held.pop(tile_idx, None)
        if held is None:
            return False
        sent_at = self._self_ex_sent_at.pop(tile_idx, None)
        if sent_at is not None:
            self.self_ex_answer_s += time.monotonic() - sent_at
        self._submit_at.pop(tile_idx, None)
        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        self._reply_byte_at.pop(tile_idx, None)
        self._dead_reply_tiles.discard(tile_idx)
        self.self_ex_used += 1
        if answer.get("self_exemplar") is not True:


            self._self_ex_settings = None
        n1 = len(held[0].get("masks") or [])
        n2 = len(answer.get("masks") or [])
        self._self_ex_log(
            f"tile {tile_idx} pass 2 (wire tile "
            f"{self_exemplar_wire_index(tile_idx)}) completed: {n2} masks "
            f"(pass 1 had {n1}), server free path "
            f"{answer.get('self_exemplar') is True}")
        return True

    def _self_ex_note_fallback(self, code: str) -> None:
        key = str(code or "UNKNOWN")
        self.self_ex_fallback[key] = self.self_ex_fallback.get(key, 0) + 1

    def _self_ex_release_held(self, tile_idx: int, code: str):


        held = self._self_ex_held.pop(tile_idx, None)
        if held is None:
            return None
        self._self_ex_sent_at.pop(tile_idx, None)
        self._submit_at.pop(tile_idx, None)
        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        self._reply_byte_at.pop(tile_idx, None)
        self._dead_reply_tiles.discard(tile_idx)
        self._self_ex_note_fallback(code)
        self._self_ex_log(
            f"tile {tile_idx} pass 2 (wire tile "
            f"{self_exemplar_wire_index(tile_idx)}) ended {code or 'UNKNOWN'}: "
            "keeping pass 1")
        return held

    def _self_ex_settle_first(
        self, tile_idx: int, tile_spec, code: str, inline: bool = False,
    ) -> bool:





        held = self._self_ex_release_held(tile_idx, code)
        if held is None:
            return False
        first_answer, tile_transform = held
        _, _, tile_w, tile_h = tile_spec
        self._cache_answer(tile_idx, first_answer)
        if inline:
            if self._emit_completed(first_answer, tile_idx, tile_w, tile_h, tile_transform):
                self.tiles_succeeded += 1
                self._completed_idx.add(tile_idx)
            return True
        self._convert_pool.submit(
            self._plan_completed(first_answer, tile_idx, tile_w, tile_h, tile_transform))
        self.tiles_succeeded += 1
        self._completed_idx.add(tile_idx)
        return True

    def _self_ex_profile(self) -> dict:

        get = getattr
        return {
            "self_ex_sent": int(get(self, "self_ex_sent", 0)),
            "self_ex_used": int(get(self, "self_ex_used", 0)),
            "self_ex_no_confident": int(get(self, "self_ex_no_confident", 0)),

            **{
                f"self_ex_fallback_{code}": int(n)
                for code, n in (get(self, "self_ex_fallback", None) or {}).items()
            },
            "self_ex_upload_mb": float(get(self, "self_ex_upload_bytes", 0)) / 1048576.0,
            "self_ex_answer_s": float(get(self, "self_ex_answer_s", 0.0)),
        }
