"""The post-run review for the public API: filter, recolour, correct, undo.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file.

Every call here needs an OPEN review, which is what the panel shows after a run
a person started and has not exported yet. A run started through this API saves
itself and leaves nothing open, so decide with ``detect_auto``'s own
``confidence`` and ``refine`` arguments instead. Nothing here costs anything.
"""
from __future__ import annotations

_DISPLAY_MODES = ("normal", "outline", "confidence", "random")

_NO_OPEN_REVIEW = (
    "No open detection review. A run started through this API saves itself and "
    "leaves nothing to review, so pass confidence= and refine= to detect_auto "
    "instead. This call works on a run a person started in the panel and has "
    "not exported yet."
)


class SegmentationReviewMixin:
    """Read and correct what a finished run left on screen."""

    def review_status(self) -> dict:
        """Report what the open review holds, so a caller can act on indexes.

        Returns
        -------
        dict with keys:
            "open"            -- bool, whether a review is open at all.
            "total_found"     -- int, objects the run produced.
            "kept_instances"  -- int, how many the current settings keep.
            "removed"         -- int, objects a correction has dropped.
            "confidence"      -- float, the cutoff now in force.
            "display_mode"    -- str, the colouring now in force.
            "corrections"     -- int, corrections that undo could take back.

        Object indexes run from 0 to ``total_found - 1`` and are what
        :meth:`review_remove_object` and :meth:`review_merge_objects` take.
        They stay stable for the life of one review. Costs nothing.
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"open": False, "_error": _NO_OPEN_REVIEW}

        out: dict = {"open": True}
        out.update(self._count_review_kept())
        removed = set()
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

    def review_filter(
        self,
        confidence: float | None = None,
        min_size_m2: float | None = None,
        max_size_m2: float | None = None,
    ) -> dict:
        """Re-filter an open review by confidence and object size.

        This is the panel's Keep step. Nothing is re-detected: the run already
        returned every plausible object, and this only decides which of them
        stay on the map.

        Parameters
        ----------
        confidence : float | None
            Cutoff in [0.05, 0.95]. Lower keeps more objects and more false
            positives. None leaves the cutoff alone.
        min_size_m2 : float | None
            Drop objects under this ground area. 0 keeps every size.
        max_size_m2 : float | None
            Drop objects over this ground area. 0 means no limit.

        Returns
        -------
        dict with "confidence", "kept_instances" and "total_found", or
        "_error". Costs nothing.
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        if confidence is None and min_size_m2 is None and max_size_m2 is None:
            return {"_error": (
                "Pass confidence, min_size_m2 or max_size_m2. Nothing to do "
                "with all three left out.")}

        if confidence is not None:
            conf, conf_err = self._confidence_in_range(confidence)
            if conf_err:
                return conf_err
            try:
                plugin._auto_confidence = conf
            except (AttributeError, RuntimeError):
                return {"_error": "Cannot set the review confidence on this build."}
            dock = getattr(plugin, "dock_widget", None)
            for widget_name in ("auto_confidence_spin", "auto_review_confidence_spin"):
                widget = getattr(dock, widget_name, None) if dock is not None else None
                if widget is not None:
                    # The spin carries a percentage on the review panel and a
                    # fraction on the run panel, so each is written in its own
                    # unit rather than one being converted into the other.
                    value = conf * 100.0 if "review" in widget_name else conf
                    self._write_review_widget(widget, value)

        size_result = self.apply_refine(
            min_size_m2=min_size_m2, max_size_m2=max_size_m2)
        if isinstance(size_result, dict) and "_error" not in size_result:
            out = {"confidence": float(plugin._auto_confidence)}
            out["kept_instances"] = size_result.get("kept_instances")
            out["total_found"] = size_result.get("total_found")
            return out

        # No size argument was given, so only the cutoff moved. Re-derive the
        # visible set on that alone.
        out = {"confidence": float(getattr(plugin, "_auto_confidence", 0.0))}
        out.update(self._reslice_open_review({"confidence": out["confidence"]}))
        out.pop("applied", None)
        return out

    def _write_review_widget(self, widget, value) -> None:
        """Set a review widget without waking its debounce."""
        try:
            widget.blockSignals(True)
            try:
                widget.setValue(value)
            except TypeError:
                widget.setValue(int(value))
            finally:
                widget.blockSignals(False)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def set_display_mode(self, mode: str) -> dict:
        """Recolour an open review's objects on the canvas.

        Parameters
        ----------
        mode : str
            One of "normal" (one fill for all), "outline" (no fill),
            "confidence" (coloured by score) or "random" (a different colour
            per object, which is the easiest way to see two objects that touch).

        Returns
        -------
        dict with "mode", or "_error". Costs nothing and changes no geometry.
        """
        plugin = self._plugin
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
        # The dock combo follows silently by design; the plugin is what stores
        # the mode and repaints, so it has to be called as well.
        applier = getattr(plugin, "_on_auto_display_mode_changed", None)
        if not callable(applier):
            return {"_error": "This build has no display-mode control."}
        try:
            applier(wanted)
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"Could not set the display mode: {err}"}
        return {"mode": wanted}

    def review_remove_object(self, index: int) -> dict:
        """Drop one object from an open review.

        Mirrors the panel's "Delete this polygon". The object is marked
        removed, not rewritten, so :meth:`review_undo_last` puts it back.

        Parameters
        ----------
        index : int
            Object index from :meth:`review_status`, 0-based.

        Returns
        -------
        dict with "removed" (the index), "kept_instances" and "total_found",
        or "_error". Costs nothing.
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        idx, err = self._review_index(index)
        if err:
            return err
        remove = getattr(plugin, "_remove_detection_index", None)
        if not callable(remove):
            return {"_error": "This build cannot remove a reviewed object."}
        try:
            remove(idx)
        except Exception as err_obj:  # noqa: BLE001 - the API never raises
            return {"_error": f"Could not remove object {idx}: {err_obj}"}
        out = {"removed": idx}
        out.update(self._count_review_kept())
        return out

    def review_merge_objects(self, indices: list[int]) -> dict:
        """Join several objects of an open review into one.

        Mirrors the panel's "Merge with neighbours". Use it when one real
        object came back cut in two, which happens where an object crosses the
        edge between two tiles. Pure geometry: no server call, no charge.

        Parameters
        ----------
        indices : list[int]
            At least two object indexes from :meth:`review_status`.

        Returns
        -------
        dict with "merged" (how many objects went in), "kept_instances" and
        "total_found", or "_error". Costs nothing.
        """
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

            # Same rule as the panel: a join stitches ONE object back together.
            # A tile seam is the only gap it may close, so bridge that one on
            # the review's own tolerance and refuse anything wider, instead of
            # writing two distinct objects into a single row.
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
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"Merge failed: {err}"}

        out = {"merged": len(picked), "target_index": plan.target}
        out.update(self._count_review_kept())
        return out

    def review_undo_last(self) -> dict:
        """Take back the last correction made to an open review.

        Mirrors the panel's "Undo last". It reaches removals, merges and shape
        edits, newest first.

        Returns
        -------
        dict with "corrections_left", or "_error". Costs nothing.
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        undo = getattr(plugin, "_on_auto_correction_undo_requested", None)
        if not callable(undo):
            return {"_error": "This build has no review undo."}
        try:
            undo()
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"Undo failed: {err}"}
        journal = getattr(plugin, "_auto_correct_journal", None)
        out = {"corrections_left": int(getattr(journal, "count", 0) or 0)}
        out.update(self._count_review_kept())
        return out

    def review_clear_corrections(self) -> dict:
        """Take back every correction made to an open review.

        Mirrors the panel's "Clear all". The detections go back to what the run
        produced. It does not re-run anything and costs nothing.

        Returns
        -------
        dict with "kept_instances" and "total_found", or "_error".
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_OPEN_REVIEW}
        clear = getattr(plugin, "_on_auto_correction_clear_requested", None)
        if not callable(clear):
            return {"_error": "This build has no way to clear review corrections."}
        try:
            clear()
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"Clear failed: {err}"}
        return self._count_review_kept()

    def _review_index(self, value):
        """Validate one object index, as (index, error_dict_or_None)."""
        objects = getattr(self._plugin, "_auto_objects", None) or []
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return None, {"_error": f"Object index must be a whole number, got {value!r}."}
        if idx < 0 or idx >= len(objects):
            # Imported here, not at the top: mcp_api assembles this mixin, so
            # the import can only run once that module is built.
            from .mcp_api import not_found_error
            return None, not_found_error(
                "object index", str(idx), [],
                note="Read them from review_status().",
                valid_range=(0, len(objects) - 1),
            )
        return idx, None
