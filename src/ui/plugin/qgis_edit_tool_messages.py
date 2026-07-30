"""QGIS's own map-tool message bar, kept out of the hand-edit bridge.

Every native digitizing tool that refuses a gesture emits ``messageEmitted``,
and QGIS turns it into a message bar item. On the review's selection layer one
of those sentences sends the user hunting for something that is not there:
a refused cut reads "No features were split" followed by "If there are selected
features, the split tool only applies to those [...] clear the selection."
QGIS emits that advice on every refusal, whether or not anything is selected.
The bridge selects nothing (see ``_select_and_frame_bridge_target``) and the
selection layer is not in the Layers panel, so there is no selection to find
and no layer to clear it on. The real rule is the one the dock already writes:
the cut line has to cross the shape completely.

So while the bridge is open, the tool's own bar is popped and its first line
goes to the dock's feedback line instead, where the gesture poll then replaces
it with the rule that was missed. The advice line is dropped, never reworded:
matching QGIS's translated text would break in every locale but English.

Best-effort throughout. A build that emits nothing, or a message bar that has
already moved on, leaves the user with QGIS's own bar, which is what they had
before this module.
"""
from __future__ import annotations


class QgisEditToolMessagesMixin:
    """Relay of the active map tool's refusals into the bridge's feedback line."""

    def _connect_bridge_tool_messages(self) -> None:
        """Follow the canvas tool while the bridge is open and bind each one's
        messages. Bound per tool rather than once: the dock arms four different
        native tools over one session, and each is a fresh QgsMapTool."""
        self._bridge_message_tool = None
        try:
            canvas = self.iface.mapCanvas()
            canvas.mapToolSet.connect(self._on_bridge_map_tool_set)
            self._bridge_message_conn = True
        except (RuntimeError, AttributeError, TypeError):
            self._bridge_message_conn = False
            return
        try:
            self._bind_bridge_tool_messages(canvas.mapTool())
        except (RuntimeError, AttributeError):
            pass

    def _disconnect_bridge_tool_messages(self) -> None:
        """Give QGIS its message bar back. Runs on every bridge teardown."""
        self._bind_bridge_tool_messages(None)
        if not getattr(self, "_bridge_message_conn", False):
            return
        self._bridge_message_conn = False
        try:
            self.iface.mapCanvas().mapToolSet.disconnect(
                self._on_bridge_map_tool_set)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_bridge_map_tool_set(self, *args) -> None:
        """Canvas tool changed: move the message binding onto the new one.

        The signal carries (new, old) on every 3.x build, but the argument count
        has moved before, so the new tool is read from the canvas instead."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        try:
            self._bind_bridge_tool_messages(self.iface.mapCanvas().mapTool())
        except (RuntimeError, AttributeError):
            pass

    def _bind_bridge_tool_messages(self, tool) -> None:
        """Bind ``messageEmitted`` on one tool, unbinding the previous one."""
        previous = getattr(self, "_bridge_message_tool", None)
        if previous is tool:
            return
        if previous is not None:
            try:
                previous.messageEmitted.disconnect(self._on_bridge_tool_message)
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._bridge_message_tool = None
        if tool is None:
            return
        signal = getattr(tool, "messageEmitted", None)
        if signal is None:
            return
        try:
            signal.connect(self._on_bridge_tool_message)
            self._bridge_message_tool = tool
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_bridge_tool_message(self, text, *_args) -> None:
        """A native tool refused a gesture: say it in the dock, not in a bar the
        user has to read across the canvas.

        The bar is popped on the next event-loop turn, because QGIS's own slot
        may not have pushed it yet when this one runs."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        message = str(text or "").strip()
        if not message:
            return
        # QGIS joins the reason and its selection advice with a newline. Keep
        # the reason, drop the advice: on this layer it names a selection the
        # bridge deliberately does not make.
        first_line = message.splitlines()[0].strip()
        if first_line:
            self._bridge_feedback(first_line, "warning")
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(
                0, lambda: self._pop_bridge_tool_message(message))
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _pop_bridge_tool_message(self, message: str) -> None:
        """Pop the bar item QGIS pushed for ``message``, and nothing else.

        The item's text is compared against the string this mixin was handed, so
        an unrelated message that arrived in between (a warning from the plugin
        itself, another QGIS subsystem) is left where it is."""
        try:
            bar = self.iface.messageBar()
            item = bar.currentItem()
            if item is None:
                return
            shown = str(item.text() or "")
            if shown.strip() and shown.strip() in message:
                bar.popWidget(item)
        except (RuntimeError, AttributeError, TypeError):
            pass
