



































from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsPointXY


_EDGE_BAND_PX = 2


class ManualWiderCropMixin:


    def _wider_crop_enabled(self) -> bool:

        from ...core.server_dials import dial_in_range

        return dial_in_range("tuning.click.wider_crop_on_edge", 0, 0, 1) >= 1

    def _first_answer_meets_crop_edge(self, mask) -> bool:





        import numpy as np

        if mask is None or mask.ndim != 2 or not mask.any():
            return False
        from ...core.server_dials import dial_in_range

        band = int(dial_in_range(
            "tuning.click.wider_crop_edge_px", _EDGE_BAND_PX, 1, 16))
        top = bool(np.any(mask[:band]))
        bottom = bool(np.any(mask[-band:]))
        left = bool(np.any(mask[:, :band]))
        right = bool(np.any(mask[:, -band:]))
        if not (top or bottom or left or right):
            return False
        if self._is_online_layer:
            return True
        try:
            minx, miny, maxx, maxy = self._current_crop_info["bounds"]
            height, width = mask.shape
            step_x = (maxx - minx) / float(width)
            step_y = (maxy - miny) / float(height)
            ext = self._current_layer.extent()
            beyond = {
                "left": minx > ext.xMinimum() + step_x,
                "right": maxx < ext.xMaximum() - step_x,
                "top": maxy < ext.yMaximum() - step_y,
                "bottom": miny > ext.yMinimum() + step_y,
            }
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError, KeyError):
            return False
        return ((top and beyond["top"]) or (bottom and beyond["bottom"])
                or (left and beyond["left"]) or (right and beyond["right"]))

    def _wider_crop_plan(self):






        from ...core.server_dials import dial_in_range

        info = self._current_crop_info
        if info is None:
            return None
        minx, miny, maxx, maxy = info["bounds"]
        centre = QgsPointXY((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        factor = dial_in_range("tuning.click.wider_crop_factor", 1.0, 1.25, 4.0)
        try:
            if self._is_online_layer:
                held = self._current_crop_actual_mupp
                if not held or held <= 0:
                    return None
                _canvas_mupp, wider = self._online_crop_mupp_now(held * factor)
            else:
                held = self._current_crop_scale_factor
                if not held or held <= 0:
                    return None
                ceiling = dial_in_range(
                    "tuning.manual.max_crop_scale_factor", 8.0, 2.0, 20.0)
                wider = min(held * factor, ceiling)
        except (RuntimeError, AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
        if not wider or wider <= held * 1.05:
            return None
        return centre, wider, (centre, held)

    def _wider_crop_read(self, centre, resolution, on_encoded=None) -> bool:

        return bool(self._extract_and_encode_crop(
            centre, mupp_override=resolution,
            on_encoded=on_encoded or self._invalidate_history_logits, quiet=True))



    def _wider_crop_awaiting(self) -> bool:

        state = getattr(self, "_wider_crop_state", None)
        return bool(state) and state.get("phase") in ("pending", "fallback")

    def _wider_crop_replay_in_hand(self) -> bool:

        state = getattr(self, "_wider_crop_state", None)
        return bool(state) and state.get("kind") == "later" and state.get("phase") == "pending"

    def _wider_crop_reset(self, keep_object: bool = False) -> None:


        self._wider_crop_state = None
        self._wider_crop_outcome = None
        self._wider_crop_trigger = None
        self._wider_crop_queued = []
        if not keep_object:
            self._wider_crop_object_used = False

    def _wider_crop_resolution_now(self):

        if self._is_online_layer:
            return self._current_crop_actual_mupp
        return self._current_crop_scale_factor

    def _wider_crop_cancel(self) -> None:




        if not self._wider_crop_awaiting():
            return
        state = self._wider_crop_state
        self._wider_crop_state = {"phase": "cancelled", "kind": state.get("kind"),
                                  "wider": state.get("wider")}
        self._wider_crop_outcome = None
        self._wider_crop_trigger = None
        self._wider_crop_queued = []

    def _wider_crop_replay_failed(self) -> None:

        if getattr(self, "_replaying_manual_click", False):
            self._wider_crop_cancel()

    def _wider_crop_fresh_click(self) -> None:


        if getattr(self, "_replaying_manual_click", False):
            return
        if (self._active_crop_points_positive or self._active_crop_points_negative
                or self._frozen_sessions):
            return
        if getattr(self, "_pending_manual_click", None) is not None:
            return
        state = getattr(self, "_wider_crop_state", None)
        if state and state.get("phase") not in ("answered", "cancelled"):
            self._wider_crop_state = None
        self._wider_crop_object_used = False
        self._wider_crop_queued = []

    def _wider_crop_answer_so_far(self, kind: str) -> None:


        history = self._mask_state_history
        info = self._current_crop_info
        minx, miny, maxx, maxy = info["bounds"]
        crs = (self.current_transform_info or {}).get("crs")
        if crs is None:
            try:
                layer_crs = self._current_layer.crs()
                crs = layer_crs.authid() or layer_crs.toWkt()
            except (RuntimeError, AttributeError):
                crs = None
        self._wider_crop_state["answer"] = {
            "pre": history[-1] if history else None,
            "post": {
                "mask": self.current_mask,
                "score": self.current_score,
                "transform_info": {"bbox": (minx, maxx, miny, maxy),
                                   "img_shape": tuple(self.current_mask.shape),
                                   "crs": crs},
                "low_res_mask": self.current_low_res_mask,
                "display_polygon": getattr(self, "_unfrozen_display_polygon", None),
            },
            "point": getattr(self, "_last_click_point", None),
            "kind": kind,
        }

    def _wider_crop_settle(self) -> None:



        if not self._wider_crop_awaiting():
            if getattr(self, "_wider_crop_queued", None):
                self._wider_crop_queued = []
            return
        self._pending_manual_click = None
        self._wider_crop_cancel()

    def _wider_crop_click_dropped(self) -> bool:



        if not self._wider_crop_awaiting():
            return False
        queued = list(getattr(self, "_wider_crop_queued", None) or [])
        saved = (self._wider_crop_state or {}).get("answer")
        pending = getattr(self, "_pending_manual_click", None)
        self._pending_manual_click = None
        self._wider_crop_reset(keep_object=True)
        if saved and saved.get("point") is not None:
            x, y = saved["point"]
            if saved.get("pre") is not None:
                self._mask_state_history.append(saved["pre"])
            self.prompts.add_positive_point(x, y)
            self._active_crop_points_positive.append((x, y))
            self._last_click_point = (x, y)
            self._last_click_polarity = "positive"
            self._restore_mask_state(saved["post"])
            if self.map_tool and pending is not None:
                self.map_tool.add_marker(pending["canvas_point"], is_positive=True)
            QgsMessageLog.logMessage(
                "Semi-Auto: the wider crop did not come; the click keeps its own answer",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            try:
                self._update_ui_after_prediction()
            except RuntimeError:
                pass  # nosec B110
        if queued:
            from qgis.PyQt.QtCore import QTimer

            self._wider_crop_queued = queued
            QTimer.singleShot(0, self._wider_crop_drain_queue)
        return True

    def _wider_crop_queue_click(self, entry: dict) -> bool:


        if getattr(self, "_pending_manual_click", None) is None:
            return False
        if not self._wider_crop_awaiting():
            return False
        queue = getattr(self, "_wider_crop_queued", None)
        if queue is None:
            queue = self._wider_crop_queued = []
        queue.append(entry)
        return True

    def _wider_crop_unqueue_last(self) -> bool:

        queue = getattr(self, "_wider_crop_queued", None)
        if not queue:
            return False
        queue.pop()
        return True

    def _wider_crop_drain_queue(self) -> None:


        queue = getattr(self, "_wider_crop_queued", None)
        if not queue or self._wider_crop_awaiting():
            return
        if self._encoding_in_progress or getattr(self, "_pending_manual_click", None):
            return
        self._pending_manual_click = queue.pop(0)
        self._replay_pending_manual_click()



    def _wider_crop_before_first_answer(self, mask, is_first_point: bool) -> bool:




        state = getattr(self, "_wider_crop_state", None)
        if state and (state.get("kind") == "later" or not is_first_point):

            self._wider_crop_state = state = None
        if state and state.get("phase") == "cancelled":

            self._wider_crop_state = state = None
        phase = state.get("phase") if state else None

        self._wider_crop_outcome = None
        self._wider_crop_trigger = None
        if phase == "pending":

            state["phase"] = "answered"
            state["bounds"] = tuple(self._current_crop_info["bounds"])
            from ...core.server_dials import dial_in_range

            flood = dial_in_range("tuning.click.wider_crop_max_share", 0.0, 0.3, 0.95)
            share = float(mask.mean()) if mask is not None and mask.size else 0.0
            if share <= flood:
                self._wider_crop_outcome = "wider"
                self._wider_crop_trigger = "first"
                QgsMessageLog.logMessage(
                    f"Semi-Auto: first answer taken on the wider crop (share {share:.2f})",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                return False
            centre, resolution = state["small"]
            state["phase"] = "fallback"
            QgsMessageLog.logMessage(
                f"Semi-Auto: the wider crop flooded (share {share:.2f}); "
                "asking the first window again",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            if self._wider_crop_read(centre, resolution):
                return True
            self._wider_crop_state = None
            return False
        if phase == "fallback":
            self._wider_crop_state = None
            self._wider_crop_outcome = "fallback"
            self._wider_crop_trigger = "first"
            return False
        if not is_first_point:
            return False

        self._wider_crop_object_used = False
        if self._headless or not self._wider_crop_enabled():
            return False
        if (self._frozen_sessions or self._is_refining_saved_object
                or getattr(self, "_refine_handoff_active", False)
                or getattr(self, "_unfrozen_display_polygon", None) is not None):
            return False
        if not getattr(getattr(self, "predictor", None), "last_answer_was_remote", False):
            return False
        if not self._first_answer_meets_crop_edge(mask):
            return False
        plan = self._wider_crop_plan()
        if plan is None:
            return False
        centre, wider, small = plan
        self._wider_crop_state = {"phase": "pending", "kind": "first", "small": small,
                                  "wider": wider}
        self._wider_crop_answer_so_far("first")
        self._wider_crop_object_used = True
        if not self._wider_crop_read(centre, wider):
            self._wider_crop_state = None
            return False
        QgsMessageLog.logMessage(
            "Semi-Auto: first answer meets the crop edge; reading a wider crop",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _wider_crop_later_allowed(self, n_points: int) -> bool:





        from ...core.server_dials import dial_in_range

        if dial_in_range("tuning.click.wider_crop_later", 0, 0, 1) < 1:
            return False
        last = dial_in_range("tuning.click.wider_crop_later_max_click", 0, 0, 50)
        if last < 2:
            return False
        return 2 <= n_points <= int(last)

    def _wider_crop_later_encoded(self) -> None:



        self._invalidate_history_logits()
        self.current_low_res_mask = None

    def _wider_crop_later_back(self) -> None:


        state = getattr(self, "_wider_crop_state", None) or {}
        logits = state.get("logits") or ()
        for entry, held in zip(self._mask_state_history, logits):
            entry["low_res_mask"] = held
        self.current_low_res_mask = state.get("live_logits")

    def _shape_on_wider_grid(self):




        import numpy as np

        mask = self.current_mask
        info = self.current_transform_info
        crop = self._current_crop_info
        if mask is None or info is None or crop is None:
            return None
        try:
            ominx, omaxx, ominy, omaxy = info["bbox"]
            oh, ow = mask.shape
            minx, miny, maxx, maxy = crop["bounds"]
            h, w = crop["img_shape"]


            finer = min((omaxx - ominx) / ow, (omaxy - ominy) / oh,
                        (maxx - minx) / w, (maxy - miny) / h)
            if (oh, ow) == (h, w) and np.allclose(
                    (ominx, omaxx, ominy, omaxy), (minx, maxx, miny, maxy),
                    rtol=0.0, atol=0.01 * abs(finer)):
                return mask
            out = np.zeros((h, w), dtype=bool)
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                return out
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1

            def weights(n_new, new_lo, new_step, old_lo, old_step, i0, i1):

                edges = (new_lo + np.arange(n_new + 1) * new_step - old_lo) / old_step
                a = edges[:-1, None]
                b = edges[1:, None]
                i = np.arange(i0, i1)[None, :]
                over = np.clip(np.minimum(b, i + 1) - np.maximum(a, i), 0, None)
                return (over / (b - a)).astype(np.float32)

            wc = weights(w, minx, (maxx - minx) / w, ominx, (omaxx - ominx) / ow, c0, c1)

            wr = weights(h, -maxy, (maxy - miny) / h, -omaxy, (omaxy - ominy) / oh, r0, r1)
            keep_r = np.flatnonzero(wr.any(axis=1))
            keep_c = np.flatnonzero(wc.any(axis=1))
            if keep_r.size == 0 or keep_c.size == 0:
                return out
            part = mask[r0:r1, c0:c1].astype(np.float32)
            cover = wr[keep_r] @ part @ wc[keep_c].T
            out[np.ix_(keep_r, keep_c)] = cover >= 0.5 - 1e-6
            return out
        except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError):
            return None

    def _wider_crop_before_later_answer(self, mask, raw_answer, n_points: int) -> bool:





        state = getattr(self, "_wider_crop_state", None)
        self._wider_crop_outcome = None
        self._wider_crop_trigger = None
        if state and state.get("kind") == "later":
            phase = state.get("phase")
            if phase == "pending":
                state["phase"] = "answered"
                state["bounds"] = tuple(self._current_crop_info["bounds"])
                from ...core.server_dials import dial_in_range

                flood = dial_in_range("tuning.click.wider_crop_max_share", 0.0, 0.3, 0.95)
                share = (float(raw_answer.mean())
                         if raw_answer is not None and raw_answer.size else 0.0)
                if share <= flood:
                    self._wider_crop_outcome = "wider"
                    self._wider_crop_trigger = "later"
                    QgsMessageLog.logMessage(
                        f"Semi-Auto: later click taken on the wider crop (share {share:.2f})",
                        "AI Segmentation", level=Qgis.MessageLevel.Info)
                    return False
                centre, resolution = state["small"]
                state["phase"] = "fallback"
                QgsMessageLog.logMessage(
                    f"Semi-Auto: the wider crop flooded (share {share:.2f}); "
                    "asking the click's own window again",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                if self._wider_crop_read(centre, resolution,
                                         on_encoded=self._wider_crop_later_back):
                    return True
                self._wider_crop_state = None
                return False
            if phase == "fallback":
                self._wider_crop_state = None
                self._wider_crop_outcome = "fallback"
                self._wider_crop_trigger = "later"
                return False
        if getattr(self, "_wider_crop_object_used", False):
            return False
        if self._headless or not self._wider_crop_enabled():
            return False
        if not self._wider_crop_later_allowed(n_points):
            return False
        if (self._frozen_sessions or self._is_refining_saved_object
                or getattr(self, "_refine_handoff_active", False)
                or getattr(self, "_unfrozen_display_polygon", None) is not None):
            return False
        if not getattr(getattr(self, "predictor", None), "last_answer_was_remote", False):
            return False
        if not self._first_answer_meets_crop_edge(mask):
            return False
        plan = self._wider_crop_plan()
        if plan is None:
            return False
        centre, wider, small = plan


        self._wider_crop_state = {
            "phase": "pending", "kind": "later", "small": small, "wider": wider,
            "logits": [e.get("low_res_mask") for e in self._mask_state_history[:-1]],
            "live_logits": (self._mask_state_history[-1].get("low_res_mask")
                            if self._mask_state_history else None),
        }
        self._wider_crop_answer_so_far("later")
        self._wider_crop_object_used = True
        if not self._wider_crop_read(centre, wider,
                                     on_encoded=self._wider_crop_later_encoded):
            self._wider_crop_state = None
            return False
        QgsMessageLog.logMessage(
            "Semi-Auto: later click's shape meets the crop edge; reading a wider crop",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _wider_crop_status(self):



        state = getattr(self, "_wider_crop_state", None)
        if not state or state.get("phase") not in ("answered", "cancelled"):
            return None
        if self._active_crop_points_positive or self._active_crop_points_negative:
            return None
        if state["phase"] == "cancelled":
            wider = state.get("wider")
            held = self._wider_crop_resolution_now()
            if not wider or not held or abs(held - wider) > 0.01 * wider:
                return None
            return "no_crop"
        info = self._current_crop_info
        if info is None or tuple(info["bounds"]) != state.get("bounds"):
            return None
        return "no_crop"
