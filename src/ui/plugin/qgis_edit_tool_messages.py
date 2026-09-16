





















from __future__ import annotations


class QgisEditToolMessagesMixin:


    def _connect_bridge_tool_messages(self) -> None:






        if getattr(self, "_bridge_message_conn", False):
            return
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




        if not getattr(self, "_qgis_bridge_active", False):
            return
        try:
            self._bind_bridge_tool_messages(self.iface.mapCanvas().mapTool())
        except (RuntimeError, AttributeError):
            pass

    def _bind_bridge_tool_messages(self, tool) -> None:

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





        if not getattr(self, "_qgis_bridge_active", False):
            return
        message = str(text or "").strip()
        if not message:
            return



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
