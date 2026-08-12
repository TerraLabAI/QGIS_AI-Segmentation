"""One short line for the click failures the user can fix themselves.

A basemap picked in the panel that draws nothing where the user clicked, a
zoom the tile service does not publish, a click off the edge of the raster: none
of these is a defect. They used to open the copy-logs report dialog, three
paragraphs deep, asking the user to mail a bug report for a layer they picked by
mistake. They get one sentence in the panel now, and the report dialog stays for
the failures we really do have to read logs about.

The lines are short on purpose. They name the thing that answered nothing and
the one move that fixes it, and they stop. Nothing here says "please report".

Kept client-side rather than served: every sentence ships through ``tr()`` in
four languages, and a served string would shadow those translations for every
non-English user. The reader is wired all the same, so a line can be retuned
without a release the day one of them turns out to be wrong.
"""

from __future__ import annotations

from .i18n import tr
from .server_dials import dial_copy

# Codes where "the layer you picked answers nothing here" is the whole story,
# so the name of the layer the user is actually LOOKING at is a real answer.
# A network drop is not in here: another raster covering the click proves
# nothing about a connection that went away.
_LAYER_ANSWERED_NOTHING = frozenset({
    "crop_error_online_blank_tiles",
    "crop_error_online_tiles_refused",
    "crop_error_outside_bounds",
})


def _notice_lines() -> dict[str, str]:
    """Error code -> the sentence shown in the panel, built on each call.

    Built per call rather than at import: ``tr()`` resolves against the locale
    loaded at the time, and this module is imported long before the user can
    change the QGIS language.
    """
    return {
        "crop_error_online_blank_tiles": dial_copy(
            "click_error.blank_tiles",
            tr("This layer has no imagery at this zoom. Zoom in until you "
               "see it on the map, then click again.")),
        "crop_error_online_tiles_refused": dial_copy(
            "click_error.tiles_refused",
            tr("This layer's server refused the request. Pick another "
               "basemap at the top of the panel, then click again.")),
        "crop_error_online_fetch_failed": dial_copy(
            "click_error.fetch_failed",
            tr("Could not reach this layer's server. Check your connection, "
               "then click again.")),
        "crop_error_outside_bounds": dial_copy(
            "click_error.outside_bounds",
            tr("Your click is outside this layer. Click on the imagery "
               "itself, or pick another layer at the top of the panel.")),
        "crop_error_no_bands": dial_copy(
            "click_error.no_bands",
            tr("This raster has no bands to read. Pick another layer at the "
               "top of the panel.")),
        "crop_error_file_missing": dial_copy(
            "click_error.file_missing",
            tr("This layer's file is no longer where QGIS expects it. Reload "
               "it from where the file is now, then start again.")),
        "crop_error_unsupported_format": dial_copy(
            "click_error.unsupported_format",
            tr("QGIS cannot read this raster format here. Convert it to "
               "GeoTIFF, then start again.")),
        "crop_error_gdal_unavailable": dial_copy(
            "click_error.gdal_unavailable",
            tr("QGIS cannot read this raster format here. Convert it to "
               "GeoTIFF, then start again.")),
        "crop_error_no_path": dial_copy(
            "click_error.no_path",
            tr("This layer has no file to read. Pick another layer at the "
               "top of the panel, then start again.")),
    }


def click_error_notice(error_code: str, selected_layer: str = "",
                       visible_layer: str = "") -> str:
    """The panel sentence for this failure, or "" when it deserves a report.

    ``visible_layer`` is the topmost visible raster that DOES cover the click,
    when the caller found one. It buys the one diagnosis a user cannot make
    from an error message: the layer picked in the panel answers nothing, while
    the imagery they are looking at comes from another layer underneath.
    """
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
    return _notice_lines().get(code, "")
