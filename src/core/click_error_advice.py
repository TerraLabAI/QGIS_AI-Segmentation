

















from __future__ import annotations

from .i18n import tr
from .server_dials import ServerDialSet, dial_copy






_LAYER_ANSWERED_NOTHING = ServerDialSet(
    "tuning.click.layer_answered_nothing_extra",
    {
        "crop_error_online_blank_tiles",
        "crop_error_online_tiles_refused",
        "crop_error_outside_bounds",
    },
)


def _notice_line(code: str) -> str:










    builders = {
        "crop_error_online_blank_tiles": lambda: dial_copy(
            "click_error.blank_tiles",
            tr("This layer has no imagery at this zoom. Zoom in until you "
               "see it on the map, then click again.")),
        "crop_error_online_tiles_refused": lambda: dial_copy(
            "click_error.tiles_refused",
            tr("This layer's server refused the request. Pick another "
               "basemap at the top of the panel, then click again.")),
        "crop_error_online_fetch_failed": lambda: dial_copy(
            "click_error.fetch_failed",
            tr("Could not reach this layer's server. Check your connection, "
               "then click again.")),
        "crop_error_outside_bounds": lambda: dial_copy(
            "click_error.outside_bounds",
            tr("Your click is outside this layer. Click on the imagery "
               "itself, or pick another layer at the top of the panel.")),
        "crop_error_no_bands": lambda: dial_copy(
            "click_error.no_bands",
            tr("This raster has no bands to read. Pick another layer at the "
               "top of the panel.")),
        "crop_error_file_missing": lambda: dial_copy(
            "click_error.file_missing",
            tr("This layer's file is no longer where QGIS expects it. Reload "
               "it from where the file is now, then start again.")),
        "crop_error_unsupported_format": lambda: dial_copy(
            "click_error.unsupported_format",
            tr("QGIS cannot read this raster format here. Convert it to "
               "GeoTIFF, then start again.")),
        "crop_error_gdal_unavailable": lambda: dial_copy(
            "click_error.gdal_unavailable",
            tr("QGIS cannot read this raster format here. Convert it to "
               "GeoTIFF, then start again.")),
        "crop_error_no_path": lambda: dial_copy(
            "click_error.no_path",
            tr("This layer has no file to read. Pick another layer at the "
               "top of the panel, then start again.")),
    }
    builder = builders.get(code)
    return builder() if builder is not None else ""


def click_error_notice(error_code: str, selected_layer: str = "",
                       visible_layer: str = "") -> str:







    code = (error_code or "").strip()
    if visible_layer and code in _LAYER_ANSWERED_NOTHING:
        if selected_layer:
            return dial_copy("click_error.wrong_layer", tr(
                '"{selected}" has no imagery here. You are looking at '
                '"{other}". Pick it at the top of the panel, then click '
                'again.')).replace("{selected}", selected_layer).replace(
                    "{other}", visible_layer)
        return dial_copy("click_error.wrong_layer_unnamed", tr(
            'The layer you picked has no imagery here. You are looking at '
            '"{other}". Pick it at the top of the panel, then click '
            'again.')).replace("{other}", visible_layer)
    return _notice_line(code)
