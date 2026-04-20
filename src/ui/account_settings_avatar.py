





from __future__ import annotations

from ..core.qt_compat import safe_disconnect


class AccountAvatarMixin:





    def _show_account_picture(self, url, diameter: int) -> None:






        from .account_avatar import (
            AccountAvatarLoader,
            cached_avatar_pixmap,
            is_avatar_url_usable,
        )

        if not is_avatar_url_usable(url):
            return
        pixmap = cached_avatar_pixmap(url, diameter)
        if pixmap is not None:
            self._paint_account_picture(pixmap)
            return
        if self._avatar_requested:
            return
        self._avatar_requested = True


        self._avatar_loader = AccountAvatarLoader(self)
        self._avatar_loader.loaded.connect(self._paint_account_picture)
        self._avatar_loader.fetch(url, diameter)

    def _paint_account_picture(self, pixmap) -> None:

        label = self._avatar_label
        if label is None or pixmap is None or pixmap.isNull():
            return
        try:
            label.setText("")
            label.setStyleSheet("background: transparent; border: none;")
            label.setPixmap(pixmap)
        except RuntimeError:
            self._avatar_label = None

    def _cancel_avatar_load(self) -> None:

        loader = self._avatar_loader
        self._avatar_loader = None
        if loader is None:
            return



        self._avatar_requested = False
        safe_disconnect(loader, "loaded")
        try:
            loader.abort()
        except RuntimeError:
            pass
