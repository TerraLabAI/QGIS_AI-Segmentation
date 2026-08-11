"""First-time setup, install/download workers, activation, pairing, settings.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import sys

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ...core.qt_compat import safe_disconnect
from ...core.window_focus import bring_qgis_window_to_front
from ..background_workers import DepsInstallWorker, DownloadWorker, VerifyWorker
from ..error_report_dialog import show_error_report
from .shared import (
    SETTINGS_KEY_LAST_MANUAL_SESSION_TS,
    _get_change_path_instructions,
)


def _notify_ui(callback, *args) -> None:
    """Call a dialog callback, tolerating a dialog that is already gone.

    The removal worker outlives the window that started it (QGIS can be closing
    while the delete finishes), so a dead Qt receiver must not turn into an
    unhandled exception inside a signal handler."""
    if callback is None:
        return
    try:
        callback(*args)
    except RuntimeError:
        pass  # nosec B110 - the receiving widget was destroyed


_INSTALL_ATTEMPT_KEY = "TerraLab/ai_seg_install_attempts"

# How many 1.5s turns the install waits for the SAM pipe to go idle before it
# starts anyway. A crop encode round trip is seconds, so this covers a normal
# one without letting a wedged worker postpone an install for good.
_INSTALL_PIPE_WAIT_MAX = 8


def _bump_install_attempt() -> int:
    """Count this install attempt and return its 1-based index.

    Persisted rather than held in memory: a user who fails, restarts QGIS and
    tries again is on their second attempt, and an in-session counter would
    call it their first. Without it install_started/completed/failed cannot be
    read per user at all, because one person retrying four times is
    indistinguishable from four people failing once.
    """
    try:
        from qgis.PyQt.QtCore import QSettings
        s = QSettings()
        n = int(s.value(_INSTALL_ATTEMPT_KEY, 0, type=int)) + 1
        s.setValue(_INSTALL_ATTEMPT_KEY, n)
        return n
    except Exception:  # nosec B110 - a counter must never block an install
        return 0


def _clear_install_attempts() -> None:
    """Start the count over once an install has actually succeeded."""
    try:
        from qgis.PyQt.QtCore import QSettings
        QSettings().setValue(_INSTALL_ATTEMPT_KEY, 0)
    except Exception:  # nosec B110
        pass


def _drop_untagged_account_history() -> None:
    """Forget every history blob that is not tied to one account.

    Called whenever the signed-in account changes (sign-in, sign-out, forced
    sign-out). The per-account stores are already invisible to the other
    accounts; what has to go is the run cache and the machine-wide Recent list
    older versions wrote with no account attached to them. Best-effort: a
    settings failure must never block a sign-in or a sign-out.
    """
    try:
        from ...core.presets.run_history_cache import (
            clear_run_history_cache,
            reset_account_fingerprint_cache,
        )
        from ...core.presets.segment_history import clear_unscoped_recent_objects

        # First, so every store below resolves against the account that is
        # signed in now rather than the one that just left.
        reset_account_fingerprint_cache()
        clear_run_history_cache()
        clear_unscoped_recent_objects()
    except Exception:
        pass  # nosec B110 - the cache is a warm start, never the source of truth


class EnvSetupMixin:
    """First-time setup, install/download workers, activation, pairing, settings."""

    def toggle_dock_widget(self):
        just_created = not self._dock_created
        self._ensure_dock_widget()
        if self.dock_widget:
            if just_created or not self.dock_widget.isVisible():
                self.dock_widget.show()
                self.dock_widget.raise_()
                QgsMessageLog.logMessage(
                    f"Dock shown (first_create={just_created})",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            else:
                self.dock_widget.hide()
                QgsMessageLog.logMessage(
                    "Dock hidden", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # This runs only when Manual is on screen: the dock opened on it, or the
        # user switched to it (_ensure_interactive_setup). Both callers latch
        # their own guard BEFORE scheduling this, so it cannot fire twice in a
        # session, and the warm-up below starts no install and asks nothing.
        self._warm_local_ai_for_manual()

        # All environment checks (legacy cleanup, venv status, checkpoint
        # presence) run off the UI thread: get_venv_status walks the venv and
        # hashes dependency specs, which froze the dock for 1-2 s on open.
        if self._startup_check_worker is not None and self._startup_check_worker.isRunning():
            return
        from ..background_workers import StartupCheckWorker
        self._startup_check_worker = StartupCheckWorker()
        self._startup_check_worker.done.connect(self._on_startup_check_finished)
        self._startup_check_worker.start()

        # Check for plugin updates with retries - QGIS repo metadata may not
        # be ready after just a few seconds, especially on slower connections.
        from qgis.PyQt.QtCore import QTimer
        self._update_check_delays = [5000, 30000, 60000, 120000]
        self._update_check_index = 0
        QTimer.singleShot(
            self._update_check_delays[0], self._check_for_plugin_update)

    def _on_startup_check_finished(self, venv_ready: bool, message: str, checkpoint_ok: bool):
        try:
            self._apply_startup_check(venv_ready, message, checkpoint_ok)
        finally:
            # Unconditional: the key check and the balance read used to sit
            # inside the local-install branch, so a user on the cloud engine
            # never revalidated and never saw a balance all session.
            self._refresh_activation_async()

    def _apply_startup_check(self, venv_ready: bool, message: str, checkpoint_ok: bool):
        # Cache the venv-installed state for the Refine-in-Manual env gate.
        self._env_ready = bool(venv_ready)
        if not self.dock_widget:
            return

        if message.startswith("startup_error:"):
            detail = message[len("startup_error:"):].strip()
            QgsMessageLog.logMessage(
                f"Dependency check error: {detail}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            # A fixed sentence, not the first 50 characters of a Python
            # exception cut mid-word. The full detail is in the message log
            # two lines above, which is what a bug report carries.
            self.dock_widget.set_dependency_status(
                False, tr("Could not check the AI components. See the log for details."))
            # Do NOT leave the env-check latched done on an error: reset the guard
            # so a later Manual/Refine re-entry re-attempts the check instead of
            # stranding the error status until a manual Install click.
            self._interactive_setup_done = False
            return

        if venv_ready:
            self.dock_widget.set_dependency_status(True, "✓ " + tr("AI ready"))
            QgsMessageLog.logMessage(
                "✓ Virtual environment verified successfully",
                "AI Segmentation",
                level=Qgis.MessageLevel.Success
            )
            self._start_device_info_worker()
            if checkpoint_ok:
                self.dock_widget.set_checkpoint_status(True, tr("AI ready"))
                self._load_predictor()
            else:
                # An install asked for by Automatic leaves the on-device
                # packages out, so "deps ready, model not downloaded" would
                # offer a download of weights nothing here can load. Report it
                # as the setup it is: ok=False is what puts the Install button
                # back with its own label, and the string below never reaches
                # the user (set_dependency_status shows it for a DLL fault
                # only), same as every other status this call carries.
                model_ok = True
                try:
                    from ...core.venv_manager import local_model_ready
                    model_ok, _why = local_model_ready()
                except Exception:  # noqa: BLE001 -- never block on the probe
                    pass  # nosec B110
                if not model_ok:
                    self.dock_widget.set_dependency_status(
                        False, "Local model packages are not installed")
                    return
                # Model missing but deps ok.
                self.dock_widget.set_dependency_status(
                    True, tr("Almost ready: the AI file is still missing."))
                # A pending Refine handoff / background install is waiting on the
                # model: download it now so the deferred import can complete,
                # instead of stranding the user behind a manual Download button.
                if self._pending_refine_import or getattr(self, "_refine_install_pending", False):
                    self._auto_download_checkpoint()
                    return
                # Show the Download-Model button ONLY in Manual mode: without the
                # mode guard this install UI leaked onto the Automatic page when a
                # StartupCheckWorker finished after a switch back to Automatic.
                from ..ai_segmentation_dockwidget import Mode
                if self.dock_widget._mode == Mode.INTERACTIVE:
                    self.dock_widget.install_button.setVisible(True)
                    self.dock_widget.install_button.setEnabled(True)
                    self.dock_widget.install_button.setText(tr("Download AI model"))
                    self.dock_widget.setup_group.setVisible(True)
                    # Model download is pending: keep the setup section across a
                    # mode round trip (see dock _setup_section_wanted).
                    self.dock_widget._setup_section_wanted = True
        else:
            self.dock_widget.set_dependency_status(False, message)
            QgsMessageLog.logMessage(
                f"Virtual environment status: {message}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            # Auto-trigger install for upgrades
            if "need updating" in message:
                self._on_install_requested()

    def _check_for_plugin_update(self):
        """Trigger the update check on the dock widget, retrying if needed."""
        if not self.dock_widget:
            return
        self.dock_widget.check_for_updates()

        # If notification is still hidden and we have retries left, schedule next
        if not self.dock_widget.update_notification_widget.isVisible() and hasattr(self, "_update_check_delays"):
            self._update_check_index += 1
            if self._update_check_index < len(self._update_check_delays):
                from qgis.PyQt.QtCore import QTimer
                delay = self._update_check_delays[self._update_check_index]
                QTimer.singleShot(delay, self._check_for_plugin_update)

    def _load_predictor(self):
        """Kick off predictor initialization in the background (#34)."""
        # A session the network is already serving must not start an on-device
        # load. On a machine with no venv the load fails, and the failure path
        # tears the live fix session down with "the on-device AI could not
        # start", which is a sentence about something the user never asked for.
        if self._cloud_correct_predictor_active():
            return
        # Re-entry while a load is in flight must not drop the running thread's
        # last reference (GC of a live QThread hard-aborts QGIS).
        worker = getattr(self, "_predictor_worker", None)
        if worker is not None and worker.isRunning():
            return
        # The on-device packages must be on disk before a worker is spawned.
        # get_venv_status does not cover them by design: it answers "can the
        # plugin run", and Automatic is a cloud mode needing none of them. Since
        # MANUAL_ONLY_PACKAGES made a failed torch/SAM install non-fatal, such
        # an install writes the deps hash and reads ready forever, so every
        # start path here spawned a worker that died on the import ("No module
        # named 'segment_anything'"), once per QGIS start, with nothing the user
        # could do about it. local_model_ready is the gate the Manual click path
        # already uses; this is the funnel, so it belongs here too.
        # Fails CLOSED, not open. local_model_ready is filesystem-only, so it
        # raises when site-packages cannot be read at all, and a venv we cannot
        # read is exactly the one whose worker dies on import: opening the gate
        # there would recreate the crash loop this guard exists to stop. The one
        # exception is the import itself, which is what an older build without
        # the helper would hit, and that build has nothing to gate anyway.
        try:
            from ...core.venv_manager import local_model_ready
        except ImportError:
            local_model_ready = None
        if local_model_ready is not None:
            try:
                model_ok, why = local_model_ready()
            except Exception as exc:  # noqa: BLE001 - unreadable venv, treat as not ready
                model_ok, why = False, f"cannot read the environment: {exc}"
        else:
            model_ok, why = True, ""
        if not model_ok:
            # Report it through the SAME handler a real load failure uses, never
            # a hand-rolled block. That handler is the only place that logs the
            # reason, sets _local_ai_load_failed so the Correct step stops
            # offering the AI method, reads the install lane and calls
            # _release_local_ai_install. Skipping it left the pending flag set
            # for the whole session, which makes every later Fix click return
            # silently ("a setup is already running"), and left nothing on
            # screen, in the log or in telemetry: a silent failure in place of a
            # loud one, and no way to measure whether this fix worked.
            self._on_predictor_loaded(None, why)
            return
        from ..background_workers import PredictorLoadWorker
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("Loading AI model..."))
        self._predictor_worker = PredictorLoadWorker()
        self._predictor_worker.done.connect(self._on_predictor_loaded)
        self._predictor_worker.start()

    def _abandon_local_ai_session(self, err_msg: str, from_install: bool = False) -> None:
        """Give the Automatic review back after the on-device model refused to
        load, and say so.

        Without this the review stays in an armed AI fix session: the other
        polygons are dimmed and unclickable, the help line still invites keep
        points, and every click is swallowed because there is no predictor. The
        run is paid for, so the session is torn down and the resting select
        tool re-armed, which is what makes the Manual fix method usable.

        ``from_install`` covers the lane that has no session yet: the user
        accepted the one-time setup, waited it out, and it failed. The caller
        reads it BEFORE releasing the lock, because the release clears the
        pending flags this would otherwise test. Never raises.
        """
        in_session = bool(getattr(self, "_refine_handoff_active", False)
                          or getattr(self, "_pending_refine_import", False))
        if not (in_session or from_install):
            return
        if in_session:
            # Same order as the healthy exit (_on_reshape_done), because the
            # steps are not commutative: the harvest has to read the session
            # flags, the restore clears them, and the re-arm refuses to run
            # while _refine_handoff_active is still up.
            if getattr(self, "_refine_add_mode_active", False):
                try:
                    self._exit_ai_add_mode()
                except (RuntimeError, AttributeError):
                    pass
            try:
                self.dock_widget.set_refine_handoff_preparing(False)
            except (RuntimeError, AttributeError):
                pass
            # Harvest first, and do NOT clear _pending_refine_import here: that
            # flag is how the harvest tells an edited session from one the user
            # never got into, and clearing it would drop real hand edits.
            for step in (self._collect_manual_refine_into_review,
                         self._restore_auto_review_after_handoff):
                try:
                    step()
                except (RuntimeError, AttributeError):
                    pass
            try:
                self.dock_widget.leave_ai_reshape_state()
            except (RuntimeError, AttributeError):
                pass
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("The on-device AI could not start, so the AI fix is off. "
                   "Your detections are safe: switch the fix method to Manual "
                   "to keep correcting, or save them as they are."),
            )
        except (RuntimeError, AttributeError):
            pass
        QgsMessageLog.logMessage(
            f"Local AI unavailable for this session: {err_msg}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            from ...core.telemetry_errors import report_exception
            report_exception(
                RuntimeError(err_msg or "predictor load failed"),
                stage="local_ai_load", module="env_setup")
        except Exception:  # noqa: BLE001 -- reporting must never re-raise here
            pass  # nosec B110

    def _on_predictor_loaded(self, predictor, err_msg: str):
        if predictor is None:
            QgsMessageLog.logMessage(
                f"Failed to initialize predictor: {err_msg}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            if self.dock_widget:
                self.dock_widget.set_checkpoint_status(
                    False, tr("Model load failed"))
            # The on-device model is out for this session. Remembering it stops
            # the Correct step offering the AI method again on every click.
            self._local_ai_load_failed = True
            # Read the lane BEFORE the release: _release_local_ai_install
            # clears both pending flags, and an install that failed with no
            # session open is exactly the case that would otherwise end in
            # silence (the status line it writes lives in the Manual section,
            # which is hidden in Automatic mode).
            was_install = self._local_ai_install_pending()
            # An install started from the Automatic review can no longer
            # complete: drop both pending lanes and give the review back.
            self._release_local_ai_install()
            self._abandon_local_ai_session(err_msg, from_install=was_install)
            return
        self._local_ai_load_failed = False
        if self._cloud_correct_predictor_active():
            # A load already in flight when a remote route took the slot. The
            # model is loaded and must not be thrown away: park it where the
            # route's fallback and the next session both read it, and leave the
            # slot alone.
            self._local_predictor_held = predictor
        else:
            self.predictor = predictor
        QgsMessageLog.logMessage(
            "SAM predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("AI ready"))
            self.dock_widget.set_install_progress(100, tr("Ready"))
        # Warm the model NOW when Manual use is predictable, so the first
        # session's first click never waits out the model load: right after a
        # fresh install/model download (the user is onboarding, about to try),
        # when Manual mode is on screen at all (_warm_local_ai_for_manual: it
        # is the only mode that runs on this model), or when this machine ran a
        # Manual session recently (regular Manual users reach their
        # resident-model steady state at startup instead of at their first
        # click; the model stays loaded until QGIS closes anyway once a session
        # runs). Automatic-only users trip none of the three, so they never pay
        # the model's RAM. warm_up() is idempotent
        # and cheap on this thread (Popen + one init line; the load happens
        # inside the subprocess). Best-effort: the Start-time warm_up retries.
        try:
            if not self._headless:
                should_warm = getattr(self, "_warm_predictor_on_ready", False)
                should_warm = should_warm or self._manual_used_recently()
                if should_warm:
                    self._warm_predictor_on_ready = False
                    predictor.warm_up()
                    QgsMessageLog.logMessage(
                        "Pre-warming the SAM worker (Manual use predicted)",
                        "AI Segmentation", level=Qgis.MessageLevel.Info
                    )
        except Exception:  # noqa: BLE001 - prediction of intent must never break load
            pass  # nosec B110
        # An install kicked off from the Automatic review has finished. Give the
        # review back, then open the very thing the user asked for on the
        # still-intact review: the fix on the polygon they picked, or the Add
        # lane. If they left the review meanwhile, the release is the whole
        # answer (the AI is ready for next time). Which lane is read BEFORE the
        # release, which clears both flags.
        if self._local_ai_install_pending():
            resume_add = bool(getattr(self, "_ai_add_install_pending", False))
            self._release_local_ai_install()
            if self._auto_review:
                if resume_add:
                    self._on_ai_add_requested()
                else:
                    self._on_reshape_ai_requested()
            return
        # Complete a Refine-in-Manual handoff that arrived before the model was
        # loaded: now that the predictor is up, start the Manual session and
        # import the held detections as editable polygons.
        if self._pending_refine_import and self._refine_handoff_active:
            self._pending_refine_import = False
            layer = getattr(self, "_handoff_source_layer", None)
            review = self._auto_review
            if layer is not None and review:
                if self.dock_widget:
                    try:
                        self.dock_widget.set_refine_handoff_preparing(False)
                    except (RuntimeError, AttributeError):
                        pass
                    try:
                        combo = self.dock_widget.layer_combo
                        combo.blockSignals(True)
                        combo.setLayer(layer)
                        combo.blockSignals(False)
                    except (RuntimeError, AttributeError):
                        pass
                self._on_start_segmentation(layer)
                # Inherit the review's CURRENT refine settings, exactly like the
                # direct (predictor-already-loaded) reshape path does, so a run
                # whose model was still loading is not stranded on the generic
                # Manual defaults.
                self._seed_refine_from_review()
                self._import_review_geoms_as_saved(review)
                # Now the detections are loaded, open the one the user selected
                # for reshape (deferred until the predictor was ready).
                self._open_reshape_target()
            return
        # Nothing was waiting on this load, so the cursor gets the last word: a
        # hover made while the model was still coming up was recorded and not
        # acted on, and the crop under it is the one the next click will want.
        self._replay_hover_warm_when_ready()

    def _manual_used_recently(self, days: int = 14) -> bool:
        """True when this machine ran a Manual session within ``days``.

        The timestamp is written by _on_start_segmentation. Drives the
        predictive model warm-up in _on_predictor_loaded: a machine that
        segmented manually this fortnight will almost surely do it again,
        while an Automatic-only machine never gets the key at all."""
        try:
            import time

            from qgis.PyQt.QtCore import QSettings
            ts = QSettings().value(
                SETTINGS_KEY_LAST_MANUAL_SESSION_TS, 0, type=int)
            return ts > 0 and (time.time() - ts) < days * 86400
        except Exception:  # noqa: BLE001 - heuristic only
            return False

    def _start_device_info_worker(self):
        """Detect and log the compute device off the UI thread (diagnostics)."""
        if self._device_info_worker is not None and self._device_info_worker.isRunning():
            return
        from ..background_workers import DeviceInfoWorker
        self._device_info_worker = DeviceInfoWorker()
        self._device_info_worker.done.connect(self._on_device_info)
        self._device_info_worker.start()

    def _on_device_info(self, ok: bool, info: str):
        if ok:
            QgsMessageLog.logMessage(
                f"Device info: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            return

        # Check if this is a PyTorch DLL error (Windows)
        if "DLL" in info or "shm.dll" in info:
            QgsMessageLog.logMessage(
                f"PyTorch DLL error: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                return
            # Show user-friendly error dialog (same converged helper every
            # other error path here uses: copy-logs + email, not a bare OK box)
            show_error_report(
                self.iface.mainWindow(),
                tr("PyTorch cannot load on Windows"),
                tr("The plugin requires Visual C++ Redistributables to run PyTorch.\n\n"
                   "Please download and install:\n"
                   "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                   "After installation, restart QGIS and try again."),
                error_code="pytorch_dll_error",
            )
        else:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _sam_pipe_busy(self) -> bool:
        """Whether a worker currently owns the SAM subprocess pipe.

        Narrower than is_local_ai_busy() on purpose. That helper also lists
        _startup_check_worker, which is still isRunning() inside its own done
        slot, and that slot is one of the callers that asks for an install, so
        using it here would refuse every automatic update install.
        """
        for attr in ("_manual_encode_worker", "_predictor_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def _install_wants_local_model(self) -> bool:
        """Does the install being asked for have to carry the on-device AI?

        Automatic runs its inference off the machine, so an install asked for
        from that side takes the light packages only and skips the weights.
        Manual is the on-device mode, and so is every review lane that falls
        back to it, so those take everything.

        Answers yes on anything it cannot read: the full install is what every
        version before this one did, and a wrong yes only costs bytes, while a
        wrong no leaves a mode unable to start.
        """
        # A review lane asked for this: the fix and Add lanes fall back to the
        # on-device model when the network route is off, which is the only
        # reason they ever start an install.
        for flag in ("_refine_install_pending", "_ai_add_install_pending",
                     "_refine_handoff_active", "_pending_refine_import"):
            if getattr(self, flag, False):
                return True
        try:
            from ..ai_segmentation_dockwidget import Mode
            return self.dock_widget._mode != Mode.AUTOMATIC
        except (RuntimeError, AttributeError):
            return True

    def _on_install_requested(self, include_local_model: bool | None = None):
        # Which half of the install this is. Read once, here, and remembered
        # for the workers and for every status the run reports: the mode can
        # change while an install runs, and the answer must not change with it.
        if include_local_model is None:
            include_local_model = self._install_wants_local_model()
        self._install_includes_local_model = bool(include_local_model)
        # The install deletes the environment the SAM subprocess runs in, and
        # this method now hands that subprocess to the worker to be shut down.
        # Doing either while a crop encode owns the pipe kills it mid round
        # trip, and set_image then relaunches the process into the tree that
        # is about to be deleted, which is the half-deleted venv this hand-off
        # exists to prevent. Wait for the pipe instead, bounded so a wedged
        # worker cannot postpone the install forever.
        if self._sam_pipe_busy():
            waited = getattr(self, "_install_pipe_waits", 0)
            if waited < _INSTALL_PIPE_WAIT_MAX:
                from functools import partial

                from qgis.PyQt.QtCore import QTimer as _QTimer
                self._install_pipe_waits = waited + 1
                # Carry the decision through the wait: re-deciding on the
                # retry would read whatever mode the dock is on by then.
                _QTimer.singleShot(1500, partial(
                    self._on_install_requested, include_local_model))
                return
            QgsMessageLog.logMessage(
                "Starting the install with the local AI pipe still busy: "
                "waited out the retry budget",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._install_pipe_waits = 0

        # Manual (local) install needs to download/execute a standalone
        # Python and build a multi-GB venv, both unreliable or blocked under
        # Flatpak/Snap confinement. Fail fast with a clear message instead of
        # letting the install grind through a cryptic low-level failure;
        # Automatic mode needs no local install and works in the sandbox.
        from ...core.python_manager import is_sandboxed_linux
        if is_sandboxed_linux():
            QgsMessageLog.logMessage(
                "Manual install blocked: running inside a Flatpak/Snap sandbox",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False,
                tr("Semi-Auto mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Semi-Auto mode is not supported"),
                tr(
                    "Semi-Auto mode needs to install local dependencies, "
                    "which is not supported inside this sandboxed QGIS installation "
                    "(Flatpak or Snap). Please use Automatic mode instead, "
                    "which runs fully in the cloud and needs no local install."
                ),
                error_code="sandboxed_linux_manual_blocked",
            )
            return

        # Intel Macs on QGIS 4 (Python 3.13+) have no installable local runtime
        # (the Intel-Mac fallback has no build for this Python). Fail fast with a
        # clear message instead of a cryptic dependency-resolver error; Automatic
        # mode runs in the cloud and needs no local install.
        # Only the local model has no build here. numpy and rasterio do, so an
        # install that carries neither must not be refused: refusing it is what
        # left Automatic with no environment at all on these machines.
        from ...core.model_config import MACOS_X86_NO_LOCAL_INFERENCE
        if MACOS_X86_NO_LOCAL_INFERENCE and include_local_model:
            QgsMessageLog.logMessage(
                "Manual install blocked: no local inference build for this Mac + Python",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False,
                tr("Semi-Auto mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Semi-Auto mode is not supported"),
                tr(
                    "Semi-Auto mode installs local components that are not "
                    "available for this Mac with this version of QGIS. Please use Automatic "
                    "mode instead, which runs fully in the cloud and needs no "
                    "local install."
                ),
                error_code="macos_x86_no_local_inference",
            )
            return

        # Guard: prevent concurrent installs
        if self.deps_install_worker is not None and self.deps_install_worker.isRunning():
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            return

        # Guard: never install into a tree that is being deleted. Same-process
        # concurrency has to be caught HERE, because the cross-process lock the
        # removal holds records this very process and would be broken as a dead
        # leftover by an install started from this window.
        remover = getattr(self, "_remove_data_worker", None)
        if remover is not None:
            try:
                running = remover.isRunning()
            except RuntimeError:
                running = False
            if running:
                QgsMessageLog.logMessage(
                    "Install refused: the downloaded AI data is being removed",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self.dock_widget.set_dependency_status(
                    False, tr("Removing the downloaded AI data..."))
                return

        from ...core.venv_manager import get_venv_status, local_model_ready

        # This runs on the UI thread, on a button click: no subprocess probe,
        # which could freeze the window for a minute. An environment whose
        # packages are all on disk counts as ready even when the import probe
        # has not run yet, so a healthy install is not re-downloaded from
        # scratch; the background startup check does the real verification.
        is_ready, message = get_venv_status(allow_subprocess_probe=False)
        # get_venv_status answers "can the plugin run", and it says yes without
        # the local model because Automatic is a cloud mode. Taking the
        # shortcut on that alone stranded every environment installed without
        # torch (blocked download, or a disk too small for it): the button that
        # exists to repair Manual mode skipped straight to the model download
        # and never installed the package that mode needs. Clicking Install
        # must reach the installer whenever the local model is missing.
        model_ready, _model_msg = local_model_ready()
        if is_ready and not include_local_model:
            # Automatic asked, and everything Automatic loads is already on
            # disk. The weights below belong to the on-device model, so this
            # stops here rather than downloading a file this mode never opens.
            self.dock_widget.set_dependency_status(
                True, "✓ " + tr("Ready for Automatic mode"))
            self._refresh_activation_async()
            return
        if is_ready and model_ready:
            # Deps already installed, just need model download
            self.dock_widget.set_dependency_status(True, "✓ " + tr("AI ready"))
            self._auto_download_checkpoint()
            return

        QgsMessageLog.logMessage(
            "Starting virtual environment creation and dependency installation...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Platform: {sys.platform}, Python: {sys.version}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        self.dock_widget.set_install_progress(0, "Preparing installation...")

        import time as _time
        self._install_t0 = _time.monotonic()
        self._install_attempt = _bump_install_attempt()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_install_started()
        except Exception:
            pass  # nosec B110

        # Hand the loaded model to the worker so it is shut down BEFORE the
        # install touches the venv, the way _remove_ai_data does it. Its
        # subprocess runs the venv's python.exe with native libraries mapped,
        # and Windows will not delete a running exe or a loaded DLL, so a
        # rebuild would half-delete the tree and write new code over stale
        # binaries. Cleared here on the main thread so nothing else uses it.
        predictor = getattr(self, "predictor", None)
        self.predictor = None
        self.deps_install_worker = DepsInstallWorker(
            predictor, include_local_model=include_local_model)
        self.deps_install_worker.progress.connect(self._on_deps_install_progress)
        self.deps_install_worker.done.connect(self._on_deps_install_finished)
        self.deps_install_worker.start()
        if not (self.deps_install_worker.isRunning() or self.deps_install_worker.isFinished()):
            # start() only warns when the OS refuses the thread: run() never
            # executes, so `done` never fires and the dock would sit on
            # "Preparing installation..." for the rest of the session. Give the
            # predictor back and route this through the normal failure path, so
            # the user is told and the install funnel closes.
            self.predictor = predictor
            self._on_deps_install_finished(
                False, "the installation thread did not start")
            return

        # Surface the sign-in section 2 seconds after install starts
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, self._refresh_activation_async)

    def _refresh_activation_async(self, force: bool = False):
        """Refresh the signed-in state without ever blocking the UI thread.

        Re-validates the stored key in a hidden QgsTask; the UI keeps its
        current state meanwhile (benefit of the doubt). Two safeguards for flaky
        networks: a connectivity failure never signs the user out (see
        _on_key_revalidate_failed), and the check is throttled to once per 15 min
        so a restored-open dock does not re-poll on every visibility event.
        """
        import time

        from ...core.activation_manager import is_plugin_activated
        if not self.dock_widget:
            return
        if not is_plugin_activated():
            if self._report_locked_activation_key():
                return
            if not self.dock_widget.is_activated():
                self.dock_widget._update_full_ui()
            return
        # Activated on startup (stored key): the dock restores straight to its
        # last tab, so if that tab is Automatic, _on_mode_changed never fires and
        # the upsell would stay visible for a Pro account until the user toggles
        # modes. Fetch usage now so the Pro/Free state is correct immediately.
        self._refresh_auto_credits()
        # Throttle: one revalidation per 15 min. Sign-in / sign-out reset the
        # stamp (to 0.0) so a fresh state revalidates immediately.
        if not force and (time.time() - self._last_key_validation_unix) < 900:
            return
        if self._key_revalidate_task is not None and self._key_revalidate_task.is_active():
            return
        from qgis.core import QgsApplication

        # Auth header is read here (main thread); the task only talks HTTP.
        from ...api.terralab_client import TerraLabClient
        from ...core.activation_manager import get_auth_header
        from ...workers.generic_request_task import GenericRequestTask
        auth = get_auth_header()
        client = TerraLabClient()
        self._key_revalidate_task = GenericRequestTask(
            tr("Checking your AI Segmentation subscription"),
            lambda: client.get_usage(auth=auth),
            hidden=True,
        )
        self._key_revalidate_task.succeeded.connect(self._on_key_revalidate_ok)
        self._key_revalidate_task.failed.connect(self._on_key_revalidate_failed)
        QgsApplication.taskManager().addTask(self._key_revalidate_task)

    def _report_locked_activation_key(self) -> bool:
        """True when the key is stored but the auth database is still shut.

        A QGIS master password keeps the key unreadable until the user types
        it, and the plain copy was blanked when the key moved into the auth
        database. Read as a sign-out, that put a paying account back on the
        sign-in page with nothing said. So say it, leave the account signed in
        on screen, and look again in a minute: the key comes back on its own
        once the database opens.
        """
        from ...core.auth_helper import activation_key_is_locked
        if not activation_key_is_locked():
            return False
        self.dock_widget.set_activation_message(
            tr("You are signed in on this computer, but QGIS cannot read your "
               "sign-in until you enter its master password."),
            is_error=False,
        )
        if not getattr(self, "_locked_key_recheck_armed", False):
            self._locked_key_recheck_armed = True
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(60000, self._recheck_locked_activation_key)
        return True

    def _recheck_locked_activation_key(self) -> None:
        """Look again for a key that was unreadable, and stop once it is not.

        A single-shot timer cannot be called off, so this can land after the
        plugin is unloaded: no dock, nothing to do.
        """
        self._locked_key_recheck_armed = False
        if getattr(self, "dock_widget", None) is None:
            return
        self._refresh_activation_async()

    def _on_key_revalidate_ok(self, _usage: object) -> None:
        import time
        self._key_revalidate_task = None
        self._last_key_validation_unix = time.time()

    def _on_key_revalidate_failed(self, message: str, code: str) -> None:
        self._key_revalidate_task = None
        normalized = (code or "").strip().upper()
        # Only a key the server refuses signs the user out. A network blip must
        # NOT clear the stored key (that was the old bug that logged users out
        # offline), and neither must a payment problem: the key is still the
        # user's own and works again as soon as the card does. Deleting it there
        # cost them the whole browser sign-in for a card they had already fixed,
        # and it disagreed with the account window, which keeps the key.
        if normalized == "INVALID_KEY":
            from ...core.activation_manager import clear_auth
            clear_auth()
            _drop_untagged_account_history()
            self._last_key_validation_unix = 0.0
            # The account is gone, so nothing may keep travelling under it.
            self._end_cloud_click_session()
            if self.dock_widget:
                self.dock_widget.set_activated_state(False)
            # A forced sign-out is a churn-grade moment (device limit hit,
            # subscription lapsed, key revoked): make it visible post-hoc.
            try:
                from ...core import telemetry_errors
                telemetry_errors.track_plugin_error(
                    stage="activate",
                    error_code=(code or "key_rejected").lower(),
                    message="stored key rejected on revalidation",
                )
            except Exception:
                pass  # nosec B110
            return
        if normalized == "SUBSCRIPTION_INACTIVE":
            self._notify_billing_problem()
            return
        self._notify_revalidate_failure(normalized, message)

    def _notify_billing_problem(self) -> None:
        """One notice per session for a subscription the server calls inactive.

        The same flag guards the credits refresh, so the two paths cannot tell
        the user twice about one lapsed payment.
        """
        if getattr(self, "_billing_warning_shown", False):
            return
        self._billing_warning_shown = True
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("There's a problem with your subscription. Open Settings "
                   "to update your payment method or review your plan."),
                level=Qgis.MessageLevel.Warning,
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _notify_revalidate_failure(self, code: str, message: str) -> None:
        """Say something for a failure that is neither a refused key nor a
        billing problem.

        A connectivity code keeps its own notice. Everything else used to be
        swallowed, so the dock went on saying "signed in" while the server
        rejected every call, and nothing on screen ever named it.
        """
        if (code or "").strip().upper() in self._CONNECTIVITY_CODES:
            self._notify_connection_issue(code, message)
            return
        import time

        now = time.monotonic()
        last = getattr(self, "_last_auth_notice_monotonic", 0.0)
        if now - last < self._CONN_NOTICE_MIN_GAP_S:
            return
        self._last_auth_notice_monotonic = now
        QgsMessageLog.logMessage(
            f"Key revalidation failed ({code or 'unknown'})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Could not check your AI Segmentation account. If this "
                   "lasts, sign out and sign in again."),
                level=Qgis.MessageLevel.Warning,
                duration=8,
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    # --- One-click connect (browser pairing handoff) ------------------------

    def _on_pairing_requested(self, code: str):
        """Start polling for the key, then open the browser to /connect.

        The poll starts FIRST. On a machine where QGIS cannot open a browser
        the user is handed the address to paste, and that sign-in only lands if
        something here is already listening for the answer.

        Re-entrant: the same code keeps the running poll (the user clicked
        "open the page again"); a different code replaces it.
        """
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        from ...api.terralab_client import TerraLabClient

        client = TerraLabClient()
        self._start_pairing_poll(client, code)
        # Build the connect URL from the client base so .env.local
        # TERRALAB_BASE_URL is honored in dev. Never log the code or full URL.
        url = (
            f"{client.base_url}/connect?code={code}&product=ai-segmentation"
            "&utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
            "&utm_content=connect"
        )
        if not QDesktopServices.openUrl(QUrl(url)):
            self._show_pairing_address(url)

    def _start_pairing_poll(self, client, code: str) -> None:
        """Wait on this code, replacing a poll that waits on an older one.

        cancel() is cooperative, so a cancelled task keeps reading as running
        for seconds. Trusting that alone left the code minted by the next
        Connect click with no listener at all, and the user watched a spinner
        that could never end.
        """
        worker = self._pairing_worker
        if worker is not None and worker.is_active():
            if worker.pairing_code == code:
                return  # same code: the browser was just re-opened
            # Older code: cut its signals before cancelling, so its wind-down
            # cannot report a failure against the sign-in now under way.
            for signal_name in ("pairing_succeeded", "pairing_failed",
                                "pairing_timeout", "pairing_stalled"):
                safe_disconnect(worker, signal_name)
            try:
                worker.cancel()
            except RuntimeError:
                pass  # nosec B110 - already gone
        from qgis.core import QgsApplication

        from ...workers.pairing_poll_task import PairingPollTask
        self._pairing_worker = PairingPollTask(client, code)
        self._pairing_worker.pairing_succeeded.connect(self._on_pairing_succeeded)
        self._pairing_worker.pairing_failed.connect(self._on_pairing_failed)
        self._pairing_worker.pairing_timeout.connect(self._on_pairing_timeout)
        # The task has hinted at a stalled sign-in since it was written, and
        # nothing listened: the user watched the waiting animation for the full
        # ten minutes and was then told it timed out.
        self._pairing_worker.pairing_stalled.connect(self._on_pairing_stalled)
        QgsApplication.taskManager().addTask(self._pairing_worker)
        import time as _time
        self._pairing_t0 = _time.monotonic()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_started()
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing started", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _show_pairing_address(self, url: str) -> None:
        """QGIS could not open a browser: hand the user the address instead.

        Browser pairing is the ONLY way in, so a diagnosis ends the product
        here. On Linux the open goes through xdg-open, which returns False when
        that is absent, when no browser is registered, or when a sandbox has no
        portal, and none of those is a connection problem. The address carries
        a short-lived code and lands on the user's own clipboard, which is no
        wider than the browser we were trying to open it in.

        The panel stays in its waiting state on purpose: the poll started
        before this, and Cancel is the way out. Idle would offer a Connect
        button that mints a second code and retires the one just pasted.
        """
        if not self.dock_widget:
            return
        copied = False
        try:
            from qgis.PyQt.QtWidgets import QApplication

            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(url)
                copied = True
        except (RuntimeError, AttributeError, ImportError):
            copied = False
        try:
            # Selectable, so the address can still be read and copied by
            # hand when the clipboard itself is unavailable.
            from qgis.PyQt.QtCore import Qt

            label = self.dock_widget.activation_message_label
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
        except (RuntimeError, AttributeError, ImportError):
            pass
        if copied:
            message = tr(
                "QGIS could not open a browser. The sign-in address is "
                "copied to your clipboard: paste it into a browser to "
                "finish, then come back here. It works once.")
        else:
            message = tr(
                "QGIS could not open a browser. Open this address to finish "
                "signing in, then come back here. It works once:\n{}").format(url)
        self.dock_widget.set_activation_message(message, is_error=True)

    def _clear_pairing_address(self) -> None:
        """Take a retired sign-in address off the dock.

        The line holds a single-use code. Once that code is cancelled or spent,
        leaving it on screen offers the user an address that cannot work.
        """
        if not self.dock_widget:
            return
        try:
            label = self.dock_widget.activation_message_label
            label.clear()
            label.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 - the dock is being torn down

    def _pairing_elapsed_ms(self) -> int | None:
        """Elapsed ms since the current browser sign-in poll started."""
        try:
            import time as _time
            t0 = getattr(self, "_pairing_t0", None)
            return int((_time.monotonic() - t0) * 1000) if t0 else None
        except Exception:
            return None

    def _on_pairing_succeeded(self, key: str):
        from ...core.activation_manager import save_auth_token
        save_auth_token(key)
        # The code is spent, so the address the fallback may have shown is dead.
        self._clear_pairing_address()
        # Whoever was signed in before must leave nothing behind in the library.
        _drop_untagged_account_history()
        # Fresh sign-in: clear the revalidation throttle so the next dock event
        # re-checks the new key immediately instead of waiting out the window.
        self._last_key_validation_unix = 0.0
        if self.dock_widget:
            self.dock_widget.set_activated_state(True)
            # Reflect the real plan (Pro vs free) in the Automatic panel right
            # away, so the upsell card is hidden immediately for a Pro account
            # instead of waiting for the next mode toggle.
            self._refresh_auto_credits()
        # Bring QGIS back to front so the user sees the activated dock. Windows
        # needs more than activateWindow(), hence the helper.
        try:
            bring_qgis_window_to_front(self.iface.mainWindow(), self.dock_widget)
        except Exception:  # nosec B110
            pass
        try:
            from ...core.telemetry_session_events import track_plugin_activated
            track_plugin_activated(duration_ms=self._pairing_elapsed_ms())
        except Exception:  # nosec B110
            pass
        QgsMessageLog.logMessage(
            "Pairing successful", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _on_pairing_failed(self, message: str, code: str):
        if self.dock_widget:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(message, is_error=True)
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_failed(
                error_code=code or "unknown",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            f"Pairing failed ({code})", "AI Segmentation",
            level=Qgis.MessageLevel.Warning)

    def _on_pairing_stalled(self):
        """The server has not seen the browser after a long wait.

        A hint, not a failure: the poll keeps running and can still succeed, so
        this only replaces the silent waiting animation with the two things the
        user can act on. Never switches the panel back to idle, which would
        take away the Cancel the poll is still honouring."""
        if not self.dock_widget:
            return
        self.dock_widget.set_activation_message(
            tr("Still waiting for the sign-in page. If no browser opened, or "
               "the page shows an error, click Cancel and try again."),
            is_error=False,
        )

    def _on_pairing_timeout(self):
        if self.dock_widget:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(
                tr("Sign-in timed out. Click Connect to try again."),
                is_error=True,
            )
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_failed(
                error_code="timeout",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing timed out", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _cancel_pairing_worker(self):
        """Cancel any in-flight pairing poll (never terminate())."""
        if self._pairing_worker is not None and self._pairing_worker.is_active():
            try:
                self._pairing_worker.cancel()
            except RuntimeError:
                pass

    def _cancel_task(self, attr: str):
        """Cooperatively cancel a GenericRequestTask held on ``self.<attr>``.

        Disconnect its signals so a late result can't fire into torn-down UI,
        then cancel it. We NEVER terminate a network-bound thread: the task
        manager drains run() on its own, and terminate() on a wedged socket
        crashes QGIS.
        """
        task = getattr(self, attr, None)
        if task is None:
            return
        # One guard per signal: batched, a task the manager had already disposed
        # of raised on the first and left `failed` connected, so a late failure
        # still landed on torn-down UI.
        safe_disconnect(task, "succeeded")
        safe_disconnect(task, "failed")
        try:
            if task.is_active():
                task.cancel()
        except Exception:  # nosec B110
            pass
        setattr(self, attr, None)

    # Network error codes (from terralab_client._classify_network_error) that
    # mean "no/unstable connection" rather than an auth or app error.
    _CONNECTIVITY_CODES = frozenset({
        "DNS_ERROR", "CONNECTION_REFUSED", "TIMEOUT",
        "SSL_ERROR", "PROXY_ERROR", "NO_INTERNET",
    })
    _CONN_NOTICE_MIN_GAP_S = 60.0

    def _notify_connection_issue(self, code: str, message: str):
        """Show ONE transient, dismissible 'no connection' notice in the message
        bar. Non-blocking (never freezes), throttled so a flaky network does not
        spam the user, and silent for non-connectivity codes (auth/app errors
        surface through their own paths)."""
        import time

        if (code or "").strip().upper() not in self._CONNECTIVITY_CODES:
            return
        now = time.monotonic()
        if now - self._last_conn_notice_monotonic < self._CONN_NOTICE_MIN_GAP_S:
            return
        self._last_conn_notice_monotonic = now
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                message or tr("Network error. Check your internet connection."),
                level=Qgis.MessageLevel.Warning,
                duration=8,
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _on_cancel_pairing(self, code: str = ""):
        self._cancel_pairing_worker()
        # The code is being retired: the address that carries it goes with it.
        self._clear_pairing_address()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_cancelled(duration_ms=self._pairing_elapsed_ms())
        except Exception:
            pass  # nosec B110
        if code:
            # Retire the code server-side so a later Confirm in the browser
            # shows "expired" instead of binding a key nobody is polling for.
            from qgis.core import QgsApplication, QgsTask

            from ...api.terralab_client import TerraLabClient
            client = TerraLabClient()
            self._pairing_cancel_task = QgsTask.fromFunction(
                tr("Cancelling sign-in"),
                lambda task, c=code: client.cancel_pairing(c),
            )
            QgsApplication.taskManager().addTask(self._pairing_cancel_task)
        QgsMessageLog.logMessage(
            "Pairing cancelled", "AI Segmentation", level=Qgis.MessageLevel.Info)

    # --- Settings / sign out -------------------------------------------------

    def _on_settings_clicked(self):
        from ...core.activation_manager import get_auth_header, get_auth_token, is_plugin_activated
        if not is_plugin_activated():
            return
        from ...api.terralab_client import TerraLabClient
        from ..account_settings_dialog import AccountSettingsDialog
        client = TerraLabClient()
        dlg = AccountSettingsDialog(
            client=client,
            auth=get_auth_header(),
            activation_key=get_auth_token(),
            parent=self.iface.mainWindow(),
            on_remove_ai_data=self.remove_local_ai_data,
            is_busy_check=self.is_local_ai_busy,
        )
        dlg.sign_out_requested.connect(self._on_sign_out_requested)
        # This window reads the balance for its own card; the dock takes the
        # same payload instead of firing a second call and staying behind.
        dlg.usage_loaded.connect(self._on_account_usage_loaded)
        dlg.exec()
        # The window is a child of the QGIS main window, so C++ keeps it alive
        # after this method drops its only Python reference. Without this every
        # Settings open left one more hidden dialog on the main window, each
        # holding its loader task, its avatar reader and its live connections
        # for the rest of the session. done() has already cancelled both, and
        # nothing reads dlg past this point, so the delete is safe here.
        dlg.deleteLater()

    def is_local_ai_busy(self) -> bool:
        """True while any local install/download/verify/model-load worker or a
        detection run is in flight, so removing the downloaded AI data would
        corrupt an operation. A loaded-but-idle predictor is NOT busy: it is
        shut down cleanly by remove_local_ai_data before any deletion. Fail
        safe: if the state cannot be read, report busy so nothing is deleted
        out from under a live operation."""
        try:
            for attr in (
                "deps_install_worker", "download_worker", "_verify_worker",
                "_predictor_worker", "_startup_check_worker", "_auto_worker",
                # The device probe imports torch from the venv; deleting the
                # venv under it locks files on Windows and faults the probe.
                "_device_info_worker",
                # The async crop encode talks to the SAM subprocess; deleting
                # the venv under it kills the process mid-round-trip.
                "_manual_encode_worker",
                # A removal already under way: the tree is disappearing, so
                # nothing may start against it (and no second removal).
                "_remove_data_worker",
            ):
                worker = getattr(self, attr, None)
                if worker is None:
                    continue
                try:
                    if worker.isRunning():
                        return True
                except (RuntimeError, AttributeError):
                    continue
            return False
        except Exception:  # nosec B110
            return True

    def remove_local_ai_data(self, on_progress=None, on_finished=None) -> tuple[bool, str]:
        """Start deleting every trace of the local AI install this plugin
        created: the isolated venv + downloaded model weights (all under
        PLUGIN_CACHE_DIR), the stored activation key (QgsAuthManager + the QSettings
        fallback) and this plugin's own QSettings groups. Signs the user out as
        a side effect.

        Returns (started, message), and the message always says where that
        leaves the user. started=False before the account state is cleared
        means nothing at all was touched: an install or a detection run is in
        flight, or another QGIS window holds the install lock. started=False
        after it (a worker that could not be created or started) means the key
        and settings are gone but the files are not, and the message says so.
        On True the settings and key are already cleared and the multi-GB
        delete runs on a worker thread, so
        the window stays responsive; `on_progress(text)` reports the current
        step and `on_finished(ok, message)` closes it out, both on the GUI
        thread. Never raises: any partial-failure detail is folded into the
        message passed to on_finished."""
        worker = getattr(self, "_remove_data_worker", None)
        if worker is not None:
            try:
                if worker.isRunning():
                    return False, tr("The removal is already running.")
            except RuntimeError:
                pass  # C++ side gone; the attribute is stale
        if self.is_local_ai_busy():
            return False, tr(
                "An install or detection is still running. Wait for it to "
                "finish, then try again.")

        # is_local_ai_busy only sees this process. The cache is shared across
        # QGIS windows and profiles, so the cross-process install lock is what
        # stops this delete from pulling the venv out from under another
        # window's running install. It is held until the worker reports back.
        from ...core.install_lock import InstallLock
        from ...core.venv_manager import INSTALL_LOCK_FILE

        lock = InstallLock(INSTALL_LOCK_FILE)
        if not lock.acquire():
            return False, tr(
                "Another QGIS window is installing the AI engine. Wait for it "
                "to finish, then try again.")

        errors = self._clear_local_ai_account_state()

        predictor = getattr(self, "predictor", None)
        self.predictor = None
        try:
            from ..background_workers import RemoveAiDataWorker
            from .shared import park_orphaned_worker

            worker = RemoveAiDataWorker(predictor)
            self._remove_data_worker = worker
            if on_progress is not None:
                worker.progress.connect(
                    lambda text, cb=on_progress: _notify_ui(cb, text))
            worker.done.connect(
                lambda nothing_left, freed, error, lk=lock, errs=errors, cb=on_finished:
                self._on_remove_ai_data_done(nothing_left, freed, error, lk, errs, cb))
            # Anchor the thread for its whole life: the dialog that started it
            # can be dismissed and the plugin unloaded before the delete is
            # over, and a QThread destroyed while running aborts all of QGIS.
            park_orphaned_worker(worker)
            worker.start()
            # start() does not raise when the OS refuses the thread: it warns
            # and returns, run() never executes and `done` never fires. Treat
            # that as a failed start rather than waiting forever for it.
            if not (worker.isRunning() or worker.isFinished()):
                raise RuntimeError("the removal thread did not start")
        except Exception as err:  # noqa: BLE001 - the lock must never leak
            # Nothing is running, so nothing will ever release the lock or
            # report back. Give everything back here: the lock is shared across
            # QGIS windows, and leaking it refuses every later install and
            # removal until QGIS is restarted.
            QgsMessageLog.logMessage(
                f"Could not start the AI data removal: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical)
            self._remove_data_worker = None
            self.predictor = predictor
            try:
                lock.release()
            except Exception:  # nosec B110 - the release is best-effort
                pass
            return False, tr(
                "The removal could not start. You are signed out, but the "
                "downloaded AI data is still on this computer. Try again.")
        return True, tr("Removing the downloaded AI data...")

    def _clear_local_ai_account_state(self) -> list:
        """Clear the stored key and this plugin's settings, and reflect the
        sign-out in the dock. GUI-thread work only (auth + QSettings + widgets),
        all of it instant, and it runs BEFORE the delete so the account is
        already clean if the file removal turns out to be partial. Returns the
        list of failure details, for the final message."""
        self._env_ready = False
        self._first_time_setup_done = False

        # Clear the activation key (removes the QgsAuthManager config via the
        # authcfg_id, which still lives in QSettings at this point) BEFORE
        # wiping the QSettings groups, else the stored config would leak.
        errors: list[str] = []
        try:
            from ...core.activation_manager import clear_auth
            clear_auth()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])
        _drop_untagged_account_history()
        self._last_key_validation_unix = 0.0
        # The account is gone, so nothing may keep travelling under it.
        self._end_cloud_click_session()

        # Wipe this plugin's own QSettings groups. Leave the shared TerraLab/
        # keys (telemetry opt-out, device seed) untouched: they are shared with
        # the sibling plugin and are not "downloaded AI data".
        try:
            from qgis.PyQt.QtCore import QSettings
            settings = QSettings()
            for group in ("AISegmentation", "AI_Segmentation",
                          "TerraLab/AI_Segmentation"):
                settings.remove(group)
            settings.sync()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])

        if self.dock_widget:
            try:
                self.dock_widget.set_activated_state(False)
            except (RuntimeError, AttributeError):
                pass
        return errors

    def _on_remove_ai_data_done(self, nothing_left, freed, error,
                                lock, errors, on_finished) -> None:
        """The delete worker finished: release the cross-process lock, drop the
        emptied cache directory and report. Runs on the GUI thread."""
        self._remove_data_worker = None
        errors = list(errors or [])
        if error:
            errors.append(str(error)[:80])
        elif not nothing_left:
            # Some files survived (locked / permissions): report it rather
            # than claim a clean removal.
            errors.append("leftover files")

        lock.release()
        # The lock file is the one thing kept alive inside the cache directory
        # while it is being emptied (see purge_cache_dir), so drop the now-empty
        # directory once release() has removed it.
        import os

        from ...core.cache_paths import PLUGIN_CACHE_DIR
        try:
            os.rmdir(PLUGIN_CACHE_DIR)
        except OSError:
            pass  # nosec B110 - not empty (reported) or already gone

        QgsMessageLog.logMessage(
            "Local AI data removed ({}), errors={}".format(
                freed or "0 MB", len(errors)),
            "AI Segmentation", level=Qgis.MessageLevel.Info)

        if errors:
            message = tr(
                "AI data removed, but some items could not be fully cleared. "
                "You can delete the folder manually.")
        else:
            message = tr("Downloaded AI data removed. You have been signed out.")
        _notify_ui(on_finished, True, message)

    def _on_sign_out_requested(self):
        from ...core.activation_manager import clear_auth
        clear_auth()
        _drop_untagged_account_history()
        self._last_key_validation_unix = 0.0
        # Both reads were started under the account that just left. Left alone,
        # a late answer wrote its balance back into the cache, and the next
        # account to sign in opened on the previous one's numbers.
        self._cancel_task("_usage_fetch_task")
        self._cancel_task("_key_revalidate_task")
        # The account is gone, so nothing may keep travelling under it.
        self._end_cloud_click_session()
        # The object ledger belongs to the account that opened it.
        self._end_manual_credit_session()
        if self.dock_widget:
            self.dock_widget.set_activated_state(False)

    def _on_deps_install_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale deps progress to 0-80% (model download gets 80-100%)
        scaled = int(percent * 0.8)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return

        if success:
            # install_completed fires in _on_verify_finished: a venv that fails
            # verification (broken torch DLL) is a FAILED install to the user,
            # so counting it completed here skewed the install funnel.
            # Run verification + device detection off main thread
            self.dock_widget.set_install_progress(80, tr("Verifying installation..."))
            # A verify pass can run for minutes, and a repair install can
            # finish while one is still going. Reassigning the attribute would
            # drop the last reference to a live QThread, which aborts QGIS.
            if self._verify_worker is not None and self._verify_worker.isRunning():
                return
            self._verify_worker = VerifyWorker(
                include_local_model=getattr(
                    self, "_install_includes_local_model", True))
            self._verify_worker.progress.connect(self._on_verify_progress)
            self._verify_worker.done.connect(self._on_verify_finished)
            self._verify_worker.start()
        else:
            # A user cancel reaches this handler through the SAME channel as a
            # real failure (the installer returns (False, "Installation
            # cancelled") from every one of its cancel checks). Left to fall
            # through, it showed an "Installation Failed" dialog with a bug
            # report form to someone who deliberately pressed Cancel, and
            # emitted install_failed on top of the install_cancelled that
            # _on_cancel_install already sent, so the two events double-counted
            # each other and inflated the measured install failure rate.
            # Cancels wind down quietly instead; the dock's own progress
            # handler already renders a "cancel" message as its neutral
            # cancelled state rather than the failed one.
            # The worker's own flag is the authoritative discriminator: WE set
            # it in _on_cancel_install, so it cannot be defeated by wrapping or
            # translation. The string check stays as a fallback for a cancel
            # that did not come through that button, and matches anywhere in
            # the message rather than only at the start: the uv path returns
            # the bare sentinel, but the pip fallback path (uv unavailable, so
            # exactly the antivirus and proxy-restricted population) wraps it
            # as "Failed to install <package>: Installation cancelled", which a
            # prefix test misses.
            _cancelled = bool(
                getattr(self.deps_install_worker, "_cancelled", False))
            _msg_lower_early = (message or "").lower()
            if (_cancelled or "installation cancelled" in _msg_lower_early or "download cancelled" in _msg_lower_early):
                self.dock_widget.set_install_progress(100, "Cancelled")
                self.dock_widget.set_dependency_status(
                    False, tr("Installation cancelled"))
                self._release_local_ai_install()
                # The model was shut down before the install started, and a
                # cancel leaves the environment untouched, so bring it back.
                # Without this a cancelled update install kills a working
                # Manual mode for the rest of the session: nothing else
                # reloads it while the venv still reads as needing an update.
                self._load_predictor()
                return

            self.dock_widget.set_install_progress(100, "Failed")
            error_msg = message[:300] if message else tr("Unknown error")
            self.dock_widget.set_dependency_status(False, tr("Installation failed"))
            # An install from the Automatic review can no longer finish.
            self._release_local_ai_install()

            error_title = tr("Installation Failed")
            error_code = "installation_failed"
            msg_lower = message.lower() if message else ""
            # The classifiers below live in pip_diagnostics; most were already
            # written and battle-tested but only a handful were wired in here.
            # Routing the failure through ALL of them turns the opaque
            # "installation_failed" bucket into a specific error_class (better
            # telemetry) AND gives the user the correct fix instead of a generic
            # one. Order matters: most specific first, and disk/app-control are
            # checked before the antivirus/permission branches (a full disk and
            # a managed policy both surface as access errors on Windows).
            from ...core.cache_paths import PLUGIN_CACHE_DIR
            from ...core.pip_diagnostics import (
                get_app_control_help,
                get_broken_python_runtime_help,
                get_corrupt_venv_help,
                get_crash_help,
                get_dependency_conflict_help,
                get_file_locked_help,
                get_glibc_too_old_help,
                get_invalid_path_help,
                get_macos_intel_help,
                get_pip_antivirus_help,
                get_ssl_error_help,
                get_vcpp_help,
                is_antivirus_error,
                is_app_control_error,
                is_broken_python_runtime,
                is_corrupt_venv,
                is_dependency_conflict,
                is_disk_full,
                is_dll_init_error,
                is_file_locked_error,
                is_glibc_too_old,
                is_index_forbidden_error,
                is_invalid_path_error,
                is_macos_intel_no_wheel,
                is_proxy_auth_error,
                is_rename_or_record_error,
                is_unable_to_create_process,
            )
            from ...core.venv_manager import mark_venv_for_rebuild
            if "another qgis window is installing" in msg_lower:
                # Cross-process lock contention: nothing failed, another live
                # install owns the build. An info box is the right weight; the
                # bug-report dialog made people file reports for waiting. No
                # install_failed here either: counting a wait as a failure is
                # the same metric-inflation trap the user-cancel branch avoids.
                self.dock_widget.set_dependency_status(
                    False, tr("Installation running in another window"))
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.iface.mainWindow(),
                    tr("Installation Already Running"),
                    tr(
                        "Another QGIS window is installing the AI components. "
                        "Wait for it to finish, then try again."
                    ),
                )
                return
            if "not enough free disk space" in msg_lower or is_disk_full(msg_lower):
                # Preflight or mid-install disk exhaustion: the message already
                # carries the free-up / AI_SEGMENTATION_CACHE_DIR guidance.
                error_title = tr("Not Enough Disk Space")
                error_code = "disk_space"
            elif is_broken_python_runtime(msg_lower):
                # The local runtime cannot boot (gutted stdlib). Marking the
                # venv for rebuild + the standalone re-verification at install
                # start make the next attempt self-heal instead of looping on
                # the same wall.
                error_title = tr("AI Environment Damaged")
                error_msg = get_broken_python_runtime_help(PLUGIN_CACHE_DIR)
                error_code = "broken_python_runtime"
                mark_venv_for_rebuild()
            elif is_corrupt_venv(msg_lower):
                # The venv exists but its interpreter or pip is gone; a plain
                # retry reused it forever. The marker forces a from-scratch
                # rebuild on the next install run (worker thread, not UI).
                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "corrupt_venv"
                mark_venv_for_rebuild()
            elif is_invalid_path_error(msg_lower):
                # Windows refused the path itself (cloud-synced placeholder,
                # odd characters, 260-char limit), not a permission problem.
                error_title = tr("Installation Path Problem")
                error_msg = get_invalid_path_help(PLUGIN_CACHE_DIR)
                error_code = "invalid_install_path"
            elif is_dependency_conflict(msg_lower):
                error_title = tr("Package Versions Conflict")
                error_msg = get_dependency_conflict_help()
                error_code = "dependency_conflict"
            elif any(p in msg_lower for p in [
                "ssl", "certificate verify", "sslerror",
                "unable to get local issuer",
                # uv (rustls) wording, no "ssl" substring anywhere
                "invalid peer certificate", "unknownissuer",
            ]):
                # The written guidance already exists and only reached the log;
                # without this the dialog body stayed the raw installer string.
                error_title = tr("SSL Certificate Error")
                error_msg = get_ssl_error_help(msg_lower, PLUGIN_CACHE_DIR)
                error_code = "ssl_certificate_error"
            elif "file in use by qgis" in msg_lower or is_file_locked_error(msg_lower):
                # Native module (.pyd/.dll) locked by the running QGIS process
                # during a torch upgrade - no dialog-splash of AV advice here,
                # the actionable fix is just "restart QGIS".
                error_title = tr("Restart QGIS Required")
                error_msg = get_file_locked_help()
                error_code = "restart_qgis_required"
            elif is_index_forbidden_error(msg_lower):
                # HTTP 403 from the package index. Its own branch because the
                # generic bucket buries it as an unclassified failure, and the
                # fix belongs to whoever runs the network, not to the user.
                error_title = tr("Downloads Blocked by Your Network")
                error_msg = tr(
                    "The package index refused the download (error 403).\n\n"
                    "This is usually a company or campus network filtering "
                    "downloads. Ask your IT administrator to allow "
                    "pypi.org and files.pythonhosted.org, or run the install "
                    "from another network.\n\n"
                    "Automatic (cloud) mode does not need this download."
                )
                error_code = "index_forbidden"
            elif is_app_control_error(msg_lower):
                # AppLocker/WDAC style managed policy: checked BEFORE the
                # generic "blocked" branch below, whose change-path advice is
                # a dead end (the policy blocks any user-writable location).
                error_title = tr("Blocked by IT Security Policy")
                error_msg = get_app_control_help(PLUGIN_CACHE_DIR)
                error_code = "app_control_policy"
            elif is_dll_init_error(msg_lower):
                # A DLL init failure (missing VC++ runtime). is_dll_init_error
                # self-excludes policy/AV blocks, so this is genuinely a missing
                # redistributable, not a quarantine.
                error_title = tr("Missing System Component")
                error_msg = get_vcpp_help()
                error_code = "dll_init_error"
            elif any(p in msg_lower for p in [
                "access is denied", "winerror 5", "winerror 225",
                "permission denied", "blocked",
                "cannot write to install",
                "cannot open the device or file",
            ]):
                error_title = tr("Installation Blocked")
                error_msg = f"{error_msg}\n\n{_get_change_path_instructions()}"
                error_code = "installation_blocked"
            elif is_antivirus_error(msg_lower):
                # Localized "access denied" and antivirus-quarantine wordings the
                # English "blocked" branch above misses (uv surfaces the raw OS
                # message in the system language). Antivirus/exclusion guidance.
                error_title = tr("Blocked by Antivirus or Security Software")
                error_msg = get_pip_antivirus_help(PLUGIN_CACHE_DIR)
                error_code = "antivirus_blocked"
            elif is_proxy_auth_error(msg_lower):
                # HTTP 407 behind a corporate proxy: distinct from a generic
                # network failure - the fix is credentials in QGIS proxy settings.
                error_title = tr("Proxy Authentication Required")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "Your network proxy requires a username and password. "
                        "Enter them in QGIS > Settings > Options > Network, "
                        "then restart QGIS and try again."
                    ),
                )
                error_code = "proxy_auth_required"
            elif any(p in msg_lower for p in [
                "network error", "connection aborted", "connection reset",
                "timed out", "timeout", "network connection failed",
                "connection broken", "could not resolve",
            ]):
                error_title = tr("Network Connection Problem")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "Your connection appears unstable or blocked. "
                        "Check: (1) your internet is working, "
                        "(2) QGIS > Settings > Options > Network has a proxy "
                        "configured if you are on a corporate network, "
                        "(3) your firewall allows connections to pypi.org "
                        "and files.pythonhosted.org."
                    ),
                )
                error_code = "network_connection_problem"
            elif is_unable_to_create_process(msg_lower):
                # Broken Python launcher shim (Windows): rebuilding the venv fixes it.
                error_title = tr("Installation Failed")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "The installer could not start a helper process (a "
                        "damaged Python launcher). Click Reinstall Dependencies "
                        "to rebuild the environment from scratch."
                    ),
                )
                error_code = "broken_pip_shim"
            elif is_rename_or_record_error(msg_lower):
                # dist-info rename/RECORD error mid torch upgrade (Windows): a
                # stale, partially-written package. A clean rebuild resolves it.
                error_title = tr("Restart QGIS Required")
                error_msg = get_crash_help(PLUGIN_CACHE_DIR)
                error_code = "dist_info_rename"
            elif is_glibc_too_old(msg_lower):
                # Linux distro older than the current PyTorch wheels support.
                error_title = tr("Linux System Too Old")
                error_msg = get_glibc_too_old_help()
                error_code = "glibc_too_old"
            elif is_macos_intel_no_wheel(msg_lower):
                # Intel Mac + a Python newer than the last Intel torch wheel.
                error_title = tr("Unsupported Mac and Python Combination")
                error_msg = get_macos_intel_help()
                error_code = "macos_intel_no_wheel"
            # The three branches below read venv_manager's own failure wording
            # rather than a pip or uv stderr pattern. They are the families that
            # reach here with a message no stderr classifier above can match, so
            # without them every one of these lands in the generic bucket and
            # the telemetry says only that the install failed.
            elif "process crashed" in msg_lower:
                # Windows access violation mid-install, which is nearly always
                # antivirus killing pip rather than a broken package.
                error_msg = get_crash_help(PLUGIN_CACHE_DIR)
                error_code = "process_crash"
            elif any(marker in msg_lower for marker in (
                "failed to create venv",
                "failed to bootstrap pip",
                "virtual environment does not exist",
                "virtual environment not found",
            )):
                # The environment could not be built at all, so no package ever
                # got a chance. A plain retry reuses the same half-made venv.
                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "venv_create_failed"
                mark_venv_for_rebuild()
            elif "is broken" in msg_lower:
                # Installed but unimportable: the post-install verification
                # caught it, and only a rebuild clears it.
                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "package_broken"
                mark_venv_for_rebuild()

            show_error_report(
                self.iface.mainWindow(),
                error_title,
                error_msg,
                error_code=error_code,
            )
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                    retry_count=getattr(self, "_install_attempt", 0),
                    detail=message,
                )
            except Exception:
                pass  # nosec B110

    def _on_verify_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale verify progress (0-100%) into the 80-95% range
        scaled = 80 + int(percent * 0.15)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_verify_finished(self, is_valid: bool, message: str):
        if not self.dock_widget:
            return
        if is_valid:
            # The install only counts as completed once the venv VERIFIES:
            # firing at deps-install success counted broken-torch installs
            # (DLL blocked) as completed and understated install_failed.
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                if t0:
                    telemetry_session_events.track_install_completed(
                        duration_ms=int((_time.monotonic() - t0) * 1000),
                        python_minor=sys.version_info.minor,
                        retry_count=getattr(self, "_install_attempt", 0),
                    )
                    self._install_t0 = 0.0
                _clear_install_attempts()
            except Exception:
                pass  # nosec B110
            # An install is allowed to finish without the local model packages
            # (a blocked index, a download that will not complete), because
            # Automatic is a cloud mode and runs without them. Saying
            # "Dependencies ready" there would promise a Manual mode that
            # cannot start, so name what is short instead.
            model_ok = True
            try:
                from ...core.venv_manager import local_model_ready
                model_ok, _why = local_model_ready()
            except Exception:  # noqa: BLE001 -- never block on the probe
                pass  # nosec B110
            if model_ok:
                self.dock_widget.set_dependency_status(
                    True, "✓ " + tr("AI ready"))
            else:
                self.dock_widget.set_dependency_status(
                    True, "✓ " + tr("Ready for Automatic mode"))
                # The warning belongs to the install that TRIED and could not.
                # An install asked for by Automatic left the on-device AI out
                # on purpose, and reporting that as a failure would name a loss
                # the user never took.
                if getattr(self, "_install_includes_local_model", True):
                    try:
                        self.iface.messageBar().pushWarning(
                            "AI Segmentation",
                            tr("Automatic mode is ready. The on-device AI could "
                               "not be installed, so Semi-Auto mode and the "
                               "AI fix are off until it is. Everything else "
                               "works."),
                        )
                    except (RuntimeError, AttributeError):
                        pass
            if message and not message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    f"Device info: {message}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            elif message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    "Could not determine device info: {}".format(
                        message.replace("device_error: ", "")),
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            try:
                self._auto_download_checkpoint()
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Auto-download checkpoint failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self.dock_widget.set_install_progress(100, "Failed")
                self.dock_widget.set_dependency_status(
                    True, tr("Almost ready: the AI file did not download."))
                self._release_local_ai_install()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                False, "{} {}".format(tr("Verification failed:"), message))
            self._release_local_ai_install()
            # A broken native module (torch DLL blocked by a missing VC++
            # runtime) is the dominant verify failure: route it to the
            # actionable fix-it steps instead of a dead-end generic dialog.
            from ...core.pip_diagnostics import (
                get_app_control_help,
                get_vcpp_help,
                is_app_control_error,
                is_dll_init_error,
            )
            error_title = tr("Verification Failed")
            error_code = "verification_failed"
            body = "{}\n{}".format(
                tr("The AI was set up but could not start."),
                message)
            low = (message or "").lower()
            if is_app_control_error(low):
                # A managed policy (AppLocker/WDAC) blocking a DLL also prints
                # "DLL load failed": checked first so the user gets the IT
                # allow-rule guidance, not the VC++ runtime steps.
                from ...core.cache_paths import PLUGIN_CACHE_DIR
                error_title = tr("Blocked by IT Security Policy")
                error_code = "app_control_policy"
                body = f"{message}\n\n{get_app_control_help(PLUGIN_CACHE_DIR)}"
            elif is_dll_init_error(low) or "required dll failed to initialize" in low:
                error_title = tr("A Component Failed to Load")
                error_code = "dll_init_failed"
                body = f"{message}\n\n{get_vcpp_help()}"
            show_error_report(
                self.iface.mainWindow(),
                error_title,
                body,
                error_code=error_code)
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                    retry_count=getattr(self, "_install_attempt", 0),
                    detail=message,
                )
            except Exception:
                pass  # nosec B110

    def _on_cancel_install(self):
        # The model download is the second half of the same install, so Cancel
        # must stop it too: without this the worker kept retrying for up to an
        # hour after the user asked it to stop.
        if self.download_worker is not None and self.download_worker.isRunning():
            self._download_cancelled = True
            try:
                from ...core.checkpoint_manager import request_download_cancel
                request_download_cancel()
            except Exception:
                pass  # nosec B110
            QgsMessageLog.logMessage(
                "Model download cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_cancelled(
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None)
            except Exception:
                pass  # nosec B110
            QgsMessageLog.logMessage(
                "Installation cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _auto_download_checkpoint(self):
        """Auto-download model after deps install if not already present."""
        # Guard: reachable from four paths (install finish, verify finish,
        # dependency-ready shortcut, retry); a second start while one is
        # already running would corrupt the checkpoint download.
        if self.download_worker is not None and self.download_worker.isRunning():
            return

        # The checkpoint is the local model's weights and nothing else loads
        # them. Without the packages that read it the download is a wasted
        # gigabyte, and on the machine that skipped those packages for lack of
        # room it is the download that fills the disk. Stop here and leave
        # Automatic working rather than end onboarding on a failure the user
        # cannot act on.
        from ...core.venv_manager import local_model_ready
        model_ready, _model_msg = local_model_ready()
        if not model_ready:
            QgsMessageLog.logMessage(
                "Skipping the AI model download: the packages that load it are "
                "not installed. Automatic mode is unaffected.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                True, "✓ " + tr("Automatic mode ready"))
            # Nothing else is coming, so a review lane waiting on this install
            # has to be let go here or its banner holds the review for the rest
            # of the session.
            self._release_local_ai_install()
            self._refresh_activation_async()
            return

        # This path only runs at the tail of a fresh install/repair: the user
        # is onboarding and will click Start within moments, so ask
        # _on_predictor_loaded to warm the model as soon as it is ready.
        self._warm_predictor_on_ready = True
        from ...core.checkpoint_manager import checkpoint_exists
        try:
            if checkpoint_exists():
                self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
                self._load_predictor()
                self._refresh_activation_async()
                return
        except Exception:
            pass  # nosec B110

        self.dock_widget.set_install_progress(80, tr("Downloading AI model..."))
        self._download_cancelled = False
        try:
            import time as _time
            self._model_download_t0 = _time.monotonic()
            self.download_worker = DownloadWorker()
            self.download_worker.progress.connect(self._on_download_progress)
            self.download_worker.done.connect(self._on_download_finished)
            self.download_worker.start()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to start model download: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Almost ready: the AI file did not download."))
            self._release_local_ai_install()

    def _on_download_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale model download to 80-100% of the unified progress
        scaled = 80 + int(percent * 0.2)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_download_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
        if success:
            try:
                import time as _time

                from ...core import telemetry_session_events
                from ...core.model_config import USE_SAM2
                t0 = getattr(self, "_model_download_t0", 0.0)
                telemetry_session_events.track_model_download_completed(
                    model="sam2" if USE_SAM2 else "sam1",
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None)
            except Exception:
                pass  # nosec B110
            self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
            self._load_predictor()
            self._refresh_activation_async()
        else:
            # A cancel comes back through the same channel as a real failure.
            # Wind it down quietly instead of accusing the user of a failed
            # download they deliberately stopped.
            if getattr(self, "_download_cancelled", False):
                self._download_cancelled = False
                self.dock_widget.set_install_progress(100, tr("Cancelled"))
                self.dock_widget.set_dependency_status(
                    True, tr("Almost ready: the AI file is still missing."))
                self._release_local_ai_install()
                return

            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Almost ready: the AI file did not download."))
            self._release_local_ai_install()

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(
                    tr("Failed to download model:"),
                    message),
                error_code="download_failed",
            )

    def _recover_corrupt_checkpoint(self, deleted: bool) -> bool:
        """Recover from a corrupt model checkpoint detected during encoding.

        ``deleted`` reports whether the bad file was removed. When it was, we
        kick off a fresh download (which re-verifies the hash and reloads the
        predictor on success). When it could not be removed, we ask the user
        to clear the cache folder manually. Always returns False so the caller
        aborts the current encode. (#65)
        """
        from ...core.checkpoint_manager import get_checkpoints_dir

        if not deleted:
            msg = tr(
                "The AI model file is corrupted but could not be removed "
                "automatically. Please delete this folder and restart QGIS:"
            ) + "\n" + get_checkpoints_dir()
            if self._headless:
                self._headless_error = msg
                return False
            show_error_report(
                self.iface.mainWindow(),
                tr("Model File Corrupted"),
                msg,
                error_code="checkpoint_corrupt",
            )
            return False

        msg = tr(
            "The AI model file was corrupted and is being re-downloaded. "
            "Please try your selection again once it finishes."
        )
        if self._headless:
            self._headless_error = msg
            return False

        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self.iface.mainWindow(), tr("Re-downloading Model"), msg)
        try:
            self._auto_download_checkpoint()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to re-download checkpoint after corruption: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False

    def _recover_broken_venv(self, error: str) -> bool:
        """Recover from a venv that can no longer run.

        Happens when the base Python the venv points at was deleted by a
        cleanup tool or its recorded path got corrupted: site-packages look
        complete, the panel says ready, but every worker spawn fails.
        Reinstalling recreates the runtime in place; the package cache makes
        it a short repair, not a full re-download. Always returns False so
        the caller aborts the current encode. (#64)
        """
        QgsMessageLog.logMessage(
            f"Broken Python runtime detected, starting repair: {error[:200]}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)

        msg = tr(
            "The Python runtime used by the AI engine is damaged "
            "(this can be caused by a disk cleanup tool or antivirus). "
            "It will now be repaired automatically. "
            "Please try your selection again once the repair finishes."
        )
        if self._headless:
            self._headless_error = msg
            return False

        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self.iface.mainWindow(), tr("Repairing Installation"), msg)
        self.dock_widget.set_dependency_status(
            False, tr("Repairing installation..."))
        try:
            self._on_install_requested()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to start repair install: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
