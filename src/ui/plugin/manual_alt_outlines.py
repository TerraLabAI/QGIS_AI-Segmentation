




























from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QEvent, QObject, Qt, QTimer


_ALT_OUTLINE_SAME_IOU = 0.9



_ALT_OUTLINE_REMOTE_CANDIDATES = (0, 1, 2)

_ALT_OUTLINE_BUSY_RETRY_MS = 300

_ALT_OUTLINE_FETCH_RETRY_S = 2.5
_ALT_OUTLINE_FETCH_TRIES = 2



_ALT_OUTLINE_ARM_MS = 150


def _ordered_distinct_outlines(picked, others):


    import numpy as np

    kept = []
    areas = []
    ranked = [picked] + sorted(
        others, key=lambda c: int(np.count_nonzero(c[0])))
    for cand in ranked:
        mask = cand[0]
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        duplicate = False
        for prev, prev_area in zip((k[0] for k in kept), areas):
            if prev.shape != mask.shape:
                continue
            inter = int(np.count_nonzero(np.logical_and(prev, mask)))
            union = area + prev_area - inter
            if union and inter / union >= _ALT_OUTLINE_SAME_IOU:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
            areas.append(area)
    return kept


def _focus_takes_text(widget) -> bool:

    from qgis.PyQt.QtWidgets import (
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
    )
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return True
    if isinstance(widget, QComboBox) and widget.isEditable():
        return True
    node = widget
    while node is not None:

        if "Console" in type(node).__name__ or "Scintilla" in type(node).__name__ \
                or "console" in (node.objectName() or "").lower():
            return True
        node = node.parentWidget()
    return False


class _AltOutlineKeyFilter(QObject):


    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self._plugin = plugin

    def _step_for(self, event) -> int:
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return 0
        key = event.key()
        if key == Qt.Key.Key_Backtab:
            step = -1
        elif key == Qt.Key.Key_Tab:
            step = -1 if mods & Qt.KeyboardModifier.ShiftModifier else 1
        else:
            return 0
        plugin = self._plugin
        tool = getattr(plugin, "map_tool", None)
        if tool is None or not tool.isActive():
            return 0
        from qgis.PyQt.QtWidgets import QApplication
        focus = QApplication.focusWidget()
        if focus is not None:
            if _focus_takes_text(focus):
                return 0
            canvas = plugin.iface.mapCanvas()
            dock = getattr(plugin, "dock_widget", None)
            on_canvas = focus is canvas or canvas.isAncestorOf(focus)
            in_dock = dock is not None and (focus is dock or dock.isAncestorOf(focus))
            if not (on_canvas or in_dock):
                return 0
        return step

    def eventFilter(self, _obj, event):
        try:
            etype = event.type()
            if etype not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
                return False
            step = self._step_for(event)
            if not step:
                return False
            if etype == QEvent.Type.ShortcutOverride:
                event.accept()
                return True


            self._plugin._cycle_alt_outline(step)
            return True
        except Exception:  # noqa: BLE001
            return False


class ManualAltOutlinesMixin:


    def _keep_alt_outlines(self, masks, scores, low_res_masks, picked_idx,
                           img_height, img_width) -> None:






        self._forget_alt_outlines()
        try:
            import numpy as np

            picked = (self.current_mask, float(scores[picked_idx]),
                      self.current_low_res_mask)
            others = []
            for i in range(len(scores)):
                if i == picked_idx:
                    continue
                others.append((
                    np.array(masks[i][:img_height, :img_width], copy=True),
                    float(scores[i]),
                    np.array(low_res_masks[i:i + 1], copy=True),
                ))
            kept = _ordered_distinct_outlines(picked, others)
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Alternative outlines unavailable: {e}", "AI Segmentation",
                level=Qgis.MessageLevel.Info)
            return
        if len(kept) < 2 or kept[0][0] is not self.current_mask:
            return
        self._alt_outlines = kept
        self._alt_outline_index = 0
        self._alt_outline_info = self.current_transform_info
        self._alt_outline_armed = False
        self._arm_alt_outlines_later()
        self._install_alt_outline_keys()

    def _keep_remote_alt_outline_seed(self, col: int, row: int,
                                      img_height: int, img_width: int) -> None:


        self._forget_alt_outlines()
        if self.current_mask is None:
            return
        self._alt_remote_seed = {
            "mask": self.current_mask,
            "info": self.current_transform_info,
            "col": int(col),
            "row": int(row),
            "shape": (int(img_height), int(img_width)),
            "picked": (self.current_mask,
                       float(getattr(self, "current_score", 0.0) or 0.0),
                       self.current_low_res_mask),
        }
        self._alt_outline_armed = False
        self._arm_alt_outlines_later()
        self._install_alt_outline_keys()

    def _arm_alt_outlines_later(self) -> None:
        gen = getattr(self, "_alt_outline_gen", 0)

        def _arm():
            if getattr(self, "_alt_outline_gen", 0) == gen:
                self._alt_outline_armed = True
        try:
            QTimer.singleShot(_ALT_OUTLINE_ARM_MS, _arm)
        except Exception:  # noqa: BLE001
            self._alt_outline_armed = True

    def _install_alt_outline_keys(self) -> None:
        if getattr(self, "_alt_outline_key_filter", None) is not None:
            return
        try:
            from qgis.PyQt.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                return
            key_filter = _AltOutlineKeyFilter(self, self.dock_widget)
            app.installEventFilter(key_filter)
            self._alt_outline_key_filter = key_filter
        except Exception:  # noqa: BLE001  # nosec B110
            self._alt_outline_key_filter = None

    def _remove_alt_outline_keys(self) -> None:
        key_filter = getattr(self, "_alt_outline_key_filter", None)
        self._alt_outline_key_filter = None
        if key_filter is None:
            return
        try:
            from qgis.PyQt.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(key_filter)
            key_filter.deleteLater()
        except RuntimeError:
            pass

    def _forget_alt_outlines(self) -> None:

        self._alt_outline_gen = getattr(self, "_alt_outline_gen", 0) + 1
        self._alt_outlines = []
        self._alt_outline_index = 0
        self._alt_outline_info = None
        self._alt_outline_armed = False
        self._alt_remote_seed = None
        self._abandon_remote_alt_fetch()
        self._remove_alt_outline_keys()

    def _alt_outlines_available(self) -> bool:


        cands = getattr(self, "_alt_outlines", None)
        if not cands or len(cands) < 2:
            return False
        idx = getattr(self, "_alt_outline_index", 0)
        if self.current_mask is None or cands[idx][0] is not self.current_mask:
            return False
        if self.current_transform_info is not getattr(self, "_alt_outline_info", None):
            return False
        if getattr(self, "_encoding_in_progress", False):
            return False
        if getattr(self, "_pending_manual_click", None) is not None:
            return False
        if getattr(self, "_click_wait_started_here", False):
            return False
        try:
            if tuple(self.prompts.point_count) != (1, 0):
                return False
        except (AttributeError, TypeError):
            return False
        return True

    def _cycle_alt_outline(self, step: int = 1) -> bool:

        if not getattr(self, "_alt_outlines", None) \
                and getattr(self, "_alt_remote_seed", None) is not None:
            if not getattr(self, "_alt_outline_armed", True):
                return False
            self._start_remote_alt_fetch(step)
            return False
        if not self._alt_outlines_available():
            if getattr(self, "_alt_outlines", None) and self.current_mask is not \
                    self._alt_outlines[self._alt_outline_index][0]:
                self._forget_alt_outlines()
            return False
        if not getattr(self, "_alt_outline_armed", True):
            return False
        cands = self._alt_outlines
        self._alt_outline_index = (self._alt_outline_index + step) % len(cands)
        mask, score, low_res = cands[self._alt_outline_index]
        self.current_mask = mask
        self.current_score = score
        self.current_low_res_mask = low_res
        self._last_prediction_empty = False


        self._manual_tab_switches_session = getattr(
            self, "_manual_tab_switches_session", 0) + 1
        self._update_ui_after_prediction()
        return True

    def _remote_alt_seed_valid(self, seed) -> bool:


        if seed is None or getattr(self, "_alt_remote_seed", None) is not seed:
            return False
        if self.current_mask is None or self.current_mask is not seed["mask"]:
            return False
        if self.current_transform_info is not seed["info"]:
            return False
        if getattr(self, "_encoding_in_progress", False):
            return False
        if getattr(self, "_pending_manual_click", None) is not None:
            return False
        if getattr(self, "_click_wait_started_here", False):
            return False
        try:
            return tuple(self.prompts.point_count) == (1, 0)
        except (AttributeError, TypeError):
            return False

    def _abandon_remote_alt_fetch(self) -> None:
        calls = getattr(self, "_alt_fetch_calls", None) or []
        self._alt_fetch_calls = []
        self._alt_fetch_state = None
        for call in calls:
            try:
                call.abandon()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _start_remote_alt_fetch(self, step: int) -> None:



        import time

        seed = self._alt_remote_seed
        if getattr(self, "_alt_fetch_state", None) is not None:
            return
        if not self._remote_alt_seed_valid(seed):
            return
        if time.monotonic() < seed.get("retry_at", 0.0):
            return
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.hover_preview_client import preview_refine_url

            handle = self.predictor.hover_preview_handle()
            url = preview_refine_url()
            auth = get_auth_header()
        except Exception as e:  # noqa: BLE001
            self._end_failed_remote_alt_fetch(seed, f"no route ({type(e).__name__})")
            return
        if not handle or not url or not auth:
            self._end_failed_remote_alt_fetch(seed, "no route")
            return
        from ...core.hover_preview_client import preview_frame_holds_crop

        token, size = handle
        if not preview_frame_holds_crop(size, seed["shape"]):
            self._end_failed_remote_alt_fetch(seed, "crop changed")
            return
        state = {"seed": seed, "token": token,
                 "frame": (int(size[0]), int(size[1])), "step": int(step), "url": url,
                 "auth": auth, "next": 0, "busy_retried": False,
                 "others": [], "codes": [], "held": None}
        self._alt_fetch_state = state
        self._send_remote_alt_candidate(state)

    def _send_remote_alt_candidate(self, state) -> None:
        if getattr(self, "_alt_fetch_state", None) is not state:
            return
        from ...core.hover_preview_client import HoverPreviewCall, build_preview_body

        seed = state["seed"]
        k = _ALT_OUTLINE_REMOTE_CANDIDATES[state["next"]]

        def _answer(answer, k=k):
            self._on_remote_alt_answer(state, k, answer)
        call = HoverPreviewCall(
            state["url"], build_preview_body(state["token"], seed["col"],
                                             seed["row"], opening_candidate=k),
            state["auth"], _answer)
        self._alt_fetch_calls = [call]
        if not call.send() and getattr(self, "_alt_fetch_state", None) is state:
            self._on_remote_alt_answer(state, k, {"error": "not sent", "code": "NOT_SENT"})

    def _remote_alt_fetch_current(self, state) -> bool:


        if getattr(self, "_alt_fetch_state", None) is not state:
            return False
        if state["held"] is None:
            ok = self._remote_alt_seed_valid(state["seed"])
        else:
            ok = (getattr(self, "_alt_outlines", None) is state["held"]
                  and self._alt_outlines_available())
        if not ok:
            return False
        try:
            handle = self.predictor.hover_preview_handle()
        except Exception:  # noqa: BLE001
            return False
        return bool(handle) and handle[0] == state["token"]

    def _on_remote_alt_answer(self, state, k: int, answer) -> None:
        if getattr(self, "_alt_fetch_state", None) is not state:
            return
        self._alt_fetch_calls = []
        if not self._remote_alt_fetch_current(state):
            self._abandon_remote_alt_fetch()
            return
        code = ""
        if not isinstance(answer, dict):
            code = "unreadable"
        elif answer.get("error") or answer.get("code"):
            code = str(answer.get("code") or "refused")
        if code == "PREVIEW_BUSY" and not state["busy_retried"]:
            state["busy_retried"] = True
            QTimer.singleShot(_ALT_OUTLINE_BUSY_RETRY_MS,
                              lambda: self._send_remote_alt_candidate(state))
            return
        state["busy_retried"] = False
        if code:
            state["codes"].append(code)
        else:
            self._take_remote_alt_answer(state, answer)
        state["next"] += 1


        if code and code != "PREVIEW_BUSY":
            state["next"] = len(_ALT_OUTLINE_REMOTE_CANDIDATES)
        if state["next"] < len(_ALT_OUTLINE_REMOTE_CANDIDATES):
            self._send_remote_alt_candidate(state)
            return
        self._alt_fetch_state = None
        if state["held"] is None:
            codes = state["codes"]
            self._end_failed_remote_alt_fetch(
                state["seed"],
                "nothing distinct" + (f" ({', '.join(codes)})" if codes else ""))

    def _take_remote_alt_answer(self, state, answer) -> None:


        try:
            from ...core.hover_preview_client import read_preview_answer

            seed = state["seed"]

            height, width = state["frame"]
            read = read_preview_answer(answer, height, width,
                                       crop_shape=seed["shape"])
            if read is None:
                return
            state["others"].append(read)
            kept = _ordered_distinct_outlines(seed["picked"], state["others"])
        except Exception as e:  # noqa: BLE001
            self._note_remote_alt_fetch(f"unreadable ({type(e).__name__})")
            return
        if len(kept) < 2 or kept[0][0] is not seed["mask"]:
            return
        if state["held"] is None:
            self._alt_outlines = kept
            self._alt_outline_index = 0
            self._alt_outline_info = self.current_transform_info
            self._alt_outline_armed = True
            state["held"] = kept
            self._cycle_alt_outline(state["step"])
            return
        shown = self._alt_outlines[self._alt_outline_index][0]
        index = next((i for i, c in enumerate(kept) if c[0] is shown), None)
        if index is None:
            return
        self._alt_outlines = kept
        self._alt_outline_index = index
        state["held"] = kept

    def _end_failed_remote_alt_fetch(self, seed, reason: str) -> None:


        import time

        self._note_remote_alt_fetch(reason)
        if seed is None or getattr(self, "_alt_remote_seed", None) is not seed:
            return
        seed["failures"] = seed.get("failures", 0) + 1
        seed["retry_at"] = time.monotonic() + _ALT_OUTLINE_FETCH_RETRY_S
        if seed["failures"] >= _ALT_OUTLINE_FETCH_TRIES:
            self._forget_alt_outlines()

    @staticmethod
    def _note_remote_alt_fetch(reason: str) -> None:
        QgsMessageLog.logMessage(
            f"Other outlines from the server unavailable: {reason}",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
