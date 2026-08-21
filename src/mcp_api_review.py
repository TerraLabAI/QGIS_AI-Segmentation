









from __future__ import annotations

from .mcp_api_guard import gui_thread_only

_DISPLAY_MODES = ("normal", "outline", "confidence", "random")

_NO_OPEN_REVIEW = (
    "No open detection review. A run started through this API saves itself and "
    "leaves nothing to review, so pass confidence= and refine= to detect_auto "
    "instead. This call works on a run a person started in the panel and has "
    "not exported yet."
)


class SegmentationReviewMixin:


    def _review_mutation_error(self) -> dict | None:

        plugin = self._plugin
        if any(getattr(plugin, name, False) for name in (
                "_qgis_bridge_active", "_refine_handoff_active",
                "_refine_install_pending", "_ai_add_install_pending")):
            return {"_error": (
                "Finish the active correction or wait for its setup before changing the review.")}
        return None

    def review_status(self) -> dict:

















        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"open": False, "_error": _NO_OPEN_REVIEW}

        out: dict = {"open": True}
        out.update(self._count_review_kept())
        try:
            removed = plugin._review_removed_fids()
        except Exception:  # noqa: BLE001
            removed = set()
        out["removed"] = len(removed)
        out["confidence"] = float(getattr(plugin, "_auto_confidence", 0.0))
        out["display_mode"] = str(getattr(plugin, "_auto_display_mode", "") or "")
        journal = getattr(plugin, "_auto_correct_journal", None)
        out["corrections"] = int(getattr(journal, "count", 0) or 0)
        return out

    @gui_thread_only
    def review_filter(
        self,
        confidence: float | None = None,
        min_size_m2: float | None = None,
        max_size_m2: float | None = None,
    ) -> dict:























        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        if confidence is None and min_size_m2 is None and max_size_m2 is None:
            return {"_error": (
                "Pass confidence, min_size_m2 or max_size_m2. Nothing to do "
                "with all three left out.")}

        busy = self._review_mutation_error()
        if busy:
            return busy

        from .mcp_api_refine import _refine_number

        try:
            if min_size_m2 is not None:
                min_size_m2 = _refine_number("min_size_m2", min_size_m2)
            if max_size_m2 is not None:
                max_size_m2 = _refine_number("max_size_m2", max_size_m2)
        except (TypeError, ValueError, OverflowError):
            return {"_error": "Size limits must be finite numbers."}

        if confidence is not None:
            conf, conf_err = self._confidence_in_range(confidence)
            if conf_err:
                return conf_err
            try:
                plugin._auto_confidence = conf
            except (AttributeError, RuntimeError):
                return {"_error": "Cannot set the review confidence on this build."}
            dock = getattr(plugin, "dock_widget", None)
            for widget_name in ("auto_confidence_spin", "auto_review_confidence_spin",
                                "auto_review_confidence_slider"):
                widget = getattr(dock, widget_name, None) if dock is not None else None
                if widget is not None:



                    value = conf * 100.0 if "review" in widget_name else conf
                    self._write_review_widget(widget, value)




        size_result = None
        if min_size_m2 is not None or max_size_m2 is not None:
            size_result = self.apply_refine(
                min_size_m2=min_size_m2, max_size_m2=max_size_m2)
        if isinstance(size_result, dict) and "_error" in size_result:
            return size_result
        if isinstance(size_result, dict):
            out = {"confidence": float(plugin._auto_confidence)}
            out["kept_instances"] = size_result.get("kept_instances")
            out["total_found"] = size_result.get("total_found")
            return out



        out = {"confidence": float(getattr(plugin, "_auto_confidence", 0.0))}
        out.update(self._reslice_open_review({"confidence": out["confidence"]}))
        out.pop("applied", None)
        return out

    def _write_review_widget(self, widget, value) -> None:

        try:
            was_blocked = widget.blockSignals(True)
            try:
                widget.setValue(value)
            except TypeError:
                widget.setValue(int(value))
            finally:
                widget.blockSignals(was_blocked)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    @gui_thread_only
    def set_display_mode(self, mode: str) -> dict:














        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        wanted = (mode or "").strip().lower() if isinstance(mode, str) else ""
        if wanted not in _DISPLAY_MODES:
            from .mcp_api import not_found_error
            return not_found_error("display mode", wanted, list(_DISPLAY_MODES))

        dock = getattr(plugin, "dock_widget", None)
        setter = getattr(dock, "set_auto_display_mode", None) if dock is not None else None
        if callable(setter):
            try:
                setter(wanted)
            except (RuntimeError, AttributeError):
                pass


        applier = getattr(plugin, "_on_auto_display_mode_changed", None)
        if not callable(applier):
            return {"_error": "This build has no display-mode control."}
        try:
            applier(wanted)
        except Exception as err:  # noqa: BLE001
            return {"_error": f"Could not set the display mode: {err}"}
        return {"mode": wanted}

    @gui_thread_only
    def review_remove_object(self, index: int) -> dict:















        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        idx, err = self._review_index(index)
        if err:
            return err
        busy = self._review_mutation_error()
        if busy:
            return busy
        remove = getattr(plugin, "_remove_detection_index", None)
        if not callable(remove):
            return {"_error": "This build cannot remove a reviewed object."}
        try:
            remove(idx)
        except Exception as err_obj:  # noqa: BLE001
            return {"_error": f"Could not remove object {idx}: {err_obj}"}
        out = {"removed": idx}
        out.update(self._count_review_kept())
        return out

    @gui_thread_only
    def review_merge_objects(self, indices: list[int]) -> dict:
















        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        if not isinstance(indices, (list, tuple)):
            return {"_error": "indices must be a list of object indexes."}

        picked: list[int] = []
        for raw in indices:
            idx, err = self._review_index(raw)
            if err:
                return err
            if idx not in picked:
                picked.append(idx)
        if len(picked) < 2:
            return {"_error": "Pick at least two different objects to merge."}
        busy = self._review_mutation_error()
        if busy:
            return busy

        try:
            from .core.shape_edits import KIND_MERGE, apply_merge, merge_plan

            objects = plugin._auto_objects
            pairs = [(idx, float(objects[idx][1])) for idx in picked]
            plan = merge_plan(pairs, frozenset(plugin._review_removed_fids()))
            if plan is None:
                return {"_error": (
                    "Those objects cannot be merged. One of them may already "
                    "have been removed.")}

            from .core.geometry_ops import merge_geometries

            geoms = [objects[idx][0] for idx in (plan.target, *plan.absorbed)
                     if objects[idx][0] is not None]
            merged = merge_geometries(geoms)
            if merged is None or merged.isEmpty():
                return {"_error": "Those shapes could not be joined. Nothing changed."}





            from .core.geometry_ops import bridge_seam_gap, polygon_part_count

            if polygon_part_count(merged) > 1:
                seam_tol = 0.0
                tol_reader = getattr(plugin, "_merge_seam_tolerance", None)
                if callable(tol_reader):
                    seam_tol = float(tol_reader() or 0.0)
                bridged = bridge_seam_gap(merged, seam_tol)
                if bridged is None:
                    return {"_error": (
                        "Those objects do not touch, so joining them would "
                        "make one row out of two objects. Nothing changed.")}
                merged = bridged

            edit = apply_merge(objects, plan, plugin._object_row(merged, plan.score))
            plugin._auto_correction_removed.update(plan.absorbed)
            plugin._record_shape_edit(edit, fids=plan.absorbed)
            plugin._after_shape_edit(changed=(plan.target,))
            plugin._track_shape_edit(KIND_MERGE, "merged", len(plan.absorbed) + 1)
        except Exception as err:  # noqa: BLE001
            return {"_error": f"Merge failed: {err}"}

        out = {"merged": len(picked), "target_index": plan.target}
        out.update(self._count_review_kept())
        return out

    @gui_thread_only
    def review_undo_last(self) -> dict:









        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        undo = getattr(plugin, "_on_auto_correction_undo_requested", None)
        if not callable(undo):
            return {"_error": "This build has no review undo."}
        try:
            undo()
        except Exception as err:  # noqa: BLE001
            return {"_error": f"Undo failed: {err}"}
        journal = getattr(plugin, "_auto_correct_journal", None)
        out = {"corrections_left": int(getattr(journal, "count", 0) or 0)}
        out.update(self._count_review_kept())
        return out

    @gui_thread_only
    def review_clear_corrections(self) -> dict:









        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        clear = getattr(plugin, "_on_auto_correction_clear_requested", None)
        if not callable(clear):
            return {"_error": "This build has no way to clear review corrections."}
        busy = self._review_mutation_error()
        if busy:
            return busy
        try:
            clear()
        except Exception as err:  # noqa: BLE001
            return {"_error": f"Clear failed: {err}"}
        return self._count_review_kept()

    def _review_index(self, value):

        objects = getattr(self._plugin, "_auto_objects", None) or []
        try:
            idx = int(value)
            if isinstance(value, bool) or (not isinstance(value, str) and value != idx):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return None, {"_error": f"Object index must be a whole number, got {value!r}."}
        if idx < 0 or idx >= len(objects):


            from .mcp_api import not_found_error
            return None, not_found_error(
                "object index", str(idx), [],
                note="Read them from review_status().",
                valid_range=(0, len(objects) - 1),
            )
        return idx, None
