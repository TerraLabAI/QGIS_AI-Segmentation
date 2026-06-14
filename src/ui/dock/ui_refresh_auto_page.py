







from __future__ import annotations

from qgis.core import QgsRasterLayer


class DockAutoPageMixin:





    @staticmethod
    def _is_online_layer(layer) -> bool:






        if layer is None or not isinstance(layer, QgsRasterLayer):
            return False
        provider = layer.dataProvider()
        if provider is None:
            return False
        from ...core.raster_provider_kinds import CANVAS_RENDERED_PROVIDERS
        return provider.name() in CANVAS_RENDERED_PROVIDERS

    def _is_layer_georeferenced(self, layer) -> bool:



        from ..plugin.shared import is_layer_georeferenced
        return is_layer_georeferenced(layer)

    def _sync_imagery_hero(self, combo, hero) -> bool:










        from ..layer_tree_combobox import project_raster_presence
        from .widgets import set_hero_variant
        try:
            visible, hidden = project_raster_presence()
        except (RuntimeError, AttributeError):
            visible, hidden = 0, 0




        if combo.count_layers() != visible and not getattr(combo, "_frozen", False):
            try:
                combo._refresh()
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        has_rasters = combo.count_layers() > 0
        variant = "hidden" if (not has_rasters and hidden > 0) else "empty"
        try:
            if getattr(hero, "hero_variant", None) != variant:
                set_hero_variant(hero, variant)
            show_btn = getattr(hero, "hero_show_btn", None)
            if show_btn is not None and not getattr(hero, "_show_btn_wired", False):
                hero._show_btn_wired = True
                show_btn.clicked.connect(self._on_reveal_hidden_imagery)
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        return has_rasters

    def _on_reveal_hidden_imagery(self) -> None:



        from ..layer_tree_combobox import reveal_hidden_rasters
        try:
            reveal_hidden_rasters()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
