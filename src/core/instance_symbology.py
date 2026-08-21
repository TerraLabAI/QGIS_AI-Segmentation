"""One colour per detected object on a saved zone-run layer.

A zone run started from the public API saves itself, and a saved layer wears
the export style: one colour for the whole table. That reads well for a lone
object and badly for a terrace of houses, where every wall is the same hue and
the block turns into one shape. The panel shows the run the other way before
it is saved, one colour per object, and that is what makes neighbours read
apart. This module puts that look on the saved layer, and into the file.

Asked for per run and off by default: the colour is an expression the canvas
evaluates once per feature per render pass, which is a cost worth paying on
request and not by default. It is the same cost at any object count, so no
layer is refused the colours for being large.

Why here and not in a sibling:

- ``class_symbology`` maps a handful of class labels onto the brand ramp. Its
  job is a legend a person reads. This one spreads thousands of anonymous
  objects as far apart as the hue circle allows, and its legend is a
  by-product. Different rule, different module.
- ``layer_conventions`` holds the other renderers, and it is already at the
  size a file in this repo may reach, so a new concern goes beside it rather
  than inside it.
"""
from __future__ import annotations

import colorsys

from qgis.core import Qgis, QgsFillSymbol, QgsMessageLog

# The golden angle, 360 degrees divided by phi squared. Walking the hue circle
# by it puts every new object as far as it can get from the hues already used,
# at every count: three neighbours differ as plainly as three hundred do, and
# the walk never lands twice on the same hue. A simple fraction of 360 repeats
# after a handful of steps, which is how a row of buildings ends up sharing a
# colour again.
GOLDEN_ANGLE_DEGREES = 137.50776405003785

# Saturation and value every instance hue is built at. Fixed, so no object
# reads as louder than its neighbour, and high enough that the outline holds
# over a bright roof and over a shadow.
INSTANCE_SATURATION = 0.68
INSTANCE_VALUE = 0.88

# Perceived-lightness ceiling. At one fixed value the yellows and greens come
# out far lighter than the blues, and a pale outline vanishes over concrete.
# A colour above this is scaled down until it sits inside the band. No hue in
# the walk falls under about 0.32, so there is no floor to enforce.
MAX_RELATIVE_LUMINANCE = 0.62

# Same fill and outline weight the export already uses for a class colour, so
# a coloured layer reads as the same family of output, not a second style.
INSTANCE_FILL_ALPHA = 64
INSTANCE_OUTLINE_WIDTH = "0.66"

# The feature id, as an expression rather than a field, so the renderer needs
# no column of its own and works on a layer written by any plugin version.
# A GeoPackage feature id is the row's primary key, so it survives the file.
INSTANCE_CLASSIFIER = "$id"

# How much darker the outline is drawn than the fill it sits on. Matches
# QColor.darker(115), which divides the colour's value by 1.15.
INSTANCE_OUTLINE_DARKEN = 100.0 / 115.0

# No object count refuses the colours any more. A count ceiling belonged to
# the renderer that built one category per object, where a symbol was cloned on
# every render pass, a legend row was carried for each, and the style XML grew
# with the layer. This one wears a SINGLE symbol whose colour is computed from
# the feature id, so the renderer, the legend and the style stored in the
# GeoPackage are the same size at ten objects and at ten thousand. The number
# below is only the roof the served dial cannot be raised past; it is far
# beyond any layer a run produces.
NO_INSTANCE_COLOR_CEILING = 1_000_000_000


def instance_color_ceiling() -> int:
    """How many objects a layer may hold and still get one colour each.

    No practical limit, for the reason above. The served dial is still read, so
    a ceiling can be put back from the server without waiting for a release,
    and so any renderer built one category at a time still has a number to
    obey; this module no longer has one. One is the floor: below that there is
    nothing to tell apart.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "symbology.instance.max_categories", NO_INSTANCE_COLOR_CEILING,
            1, NO_INSTANCE_COLOR_CEILING))
    except Exception:  # noqa: BLE001 -- symbology is cosmetic  # nosec B110
        return NO_INSTANCE_COLOR_CEILING


def instance_palette_in_force() -> tuple[float, float, float]:
    """The (saturation, value, lightness ceiling) the walk builds hues at.

    Resolve it ONCE per styling call and pass it down: the walk paints up to
    thousands of objects, and each read walks the cached configuration. The
    floor keeps a served value from washing every hue into grey or black.
    """
    try:
        from .server_dials import dial_in_range

        return (
            float(dial_in_range("symbology.instance.saturation", INSTANCE_SATURATION, 0.2, 1.0)),
            float(dial_in_range("symbology.instance.value", INSTANCE_VALUE, 0.2, 1.0)),
            float(dial_in_range(
                "symbology.instance.max_lightness", MAX_RELATIVE_LUMINANCE, 0.2, 1.0)),
        )
    except Exception:  # noqa: BLE001 -- symbology is cosmetic  # nosec B110
        return (INSTANCE_SATURATION, INSTANCE_VALUE, MAX_RELATIVE_LUMINANCE)


def _under_the_lightness_ceiling(
    red: float, green: float, blue: float, ceiling: float = MAX_RELATIVE_LUMINANCE
) -> tuple[float, float, float]:
    """Scale a 0-1 colour down until its perceived lightness fits the band."""
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance <= ceiling or luminance <= 0:
        return red, green, blue
    scale = ceiling / luminance
    return red * scale, green * scale, blue * scale


def instance_color_hex(index: int, palette: tuple[float, float, float] | None = None) -> str:
    """Hex colour for the object at ``index``, counted from 0.

    The same index always gives the same colour, so a layer restyled twice
    looks the same both times. ``palette`` is the triple from
    ``instance_palette_in_force()``; a caller colouring many objects resolves
    it once and passes it, a lone call may leave it None.
    """
    if palette is None:
        palette = instance_palette_in_force()
    saturation, value, ceiling = palette
    hue = (int(index) * GOLDEN_ANGLE_DEGREES) % 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    channels = _under_the_lightness_ceiling(red, green, blue, ceiling)
    return "#" + "".join(
        f"{max(0, min(255, round(channel * 255))):02x}" for channel in channels)


def _stroke_color_property():
    """Data-defined StrokeColor property key for a QgsSymbolLayer.

    Same spelling drift as the FillColor shim in ``core/qt_compat.py``: QGIS 4
    renamed the member to ``QgsSymbolLayer.Property.StrokeColor``, QGIS 3
    spelled it ``PropertyStrokeColor`` flat or scoped. Returns whichever
    resolves, None when none does (never ``or`` chaining: a 0-valued enum
    member is falsy).
    """
    from qgis.core import QgsSymbolLayer

    prop_scope = getattr(QgsSymbolLayer, "Property", None)
    for owner, name in (
        (prop_scope, "StrokeColor"),              # QGIS 4
        (QgsSymbolLayer, "PropertyStrokeColor"),  # QGIS 3 flat
        (prop_scope, "PropertyStrokeColor"),      # QGIS 3 scoped
    ):
        if owner is None:
            continue
        value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def instance_color_expression(
    first_id: int = 0, alpha: int = 255, scale: float = 1.0,
    palette: tuple[float, float, float] | None = None,
) -> str:
    """The per-object colour as ONE QGIS expression over the feature id.

    Same walk as :func:`instance_color_hex`, written for the renderer instead
    of for Python: the hue steps by the golden angle, saturation and value are
    fixed, and the result is scaled down until it fits under the lightness
    ceiling. ``first_id`` is subtracted so a table numbered from 1 gets the
    same colours the per-object walk gave it.

    ``scale`` darkens the whole colour (the outline asks for it), ``alpha`` is
    the opacity to draw at.
    """
    if palette is None:
        palette = instance_palette_in_force()
    saturation, value, ceiling = palette
    hue = f"(({INSTANCE_CLASSIFIER} - {int(first_id)}) * {GOLDEN_ANGLE_DEGREES}) % 360"
    base = (f"color_hsv({hue}, {round(saturation * 100)},"
            f" {round(value * 100)})")
    # color_part, not red()/green()/blue(): those are not expression functions
    # and the whole expression fails to parse with them, which leaves the
    # layer on the grey stand-in below.
    bands = [f"color_part(@c, '{band}')" for band in ("red", "green", "blue")]
    # k is the factor that brings a too-light hue under the ceiling, 1 for
    # every hue already inside it. The max() is the guard against a black
    # colour, whose luminance would divide by zero.
    factor = (f"min(1, {ceiling} * 255 / max(1, 0.2126 * {bands[0]}"
              f" + 0.7152 * {bands[1]} + 0.0722 * {bands[2]}))")
    channels = ", ".join(f"round({band} * @k * {scale})" for band in bands)
    return (f"with_variable('c', {base},"
            f" with_variable('k', {factor},"
            f" color_rgba({channels}, {int(alpha)})))")


def make_instance_renderer(layer, *, feature_ids=None):
    """Renderer giving every object on the layer its own hue.

    ONE fill symbol whose colour is computed per feature from its id, not one
    symbol per object: the renderer, the legend and the style written into the
    GeoPackage all stay the same size whatever the object count, where a
    category each grew with it and was cloned on every render pass. Nothing
    here counts the objects to decide, because nothing here costs per object.

    ``feature_ids`` lets a caller that already walked the layer hand its ids
    over, so the walk happens once per styling call rather than once per
    question asked about the same layer.

    Returns None when the layer holds nothing to colour. The caller then leaves
    the export style in place.
    """
    from qgis.core import QgsProperty, QgsSingleSymbolRenderer

    try:
        ids = layer.allFeatureIds() if feature_ids is None else feature_ids
        ids = sorted(ids)
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not ids:
        return None
    palette = instance_palette_in_force()
    first_id = int(ids[0])
    fill_expr = instance_color_expression(
        first_id, alpha=INSTANCE_FILL_ALPHA, palette=palette)
    stroke_expr = instance_color_expression(
        first_id, alpha=255, scale=INSTANCE_OUTLINE_DARKEN, palette=palette)
    # The plain colours below are what the layer draws if a build cannot take
    # one of the data-defined properties: a neutral outline over a translucent
    # fill, never an invisible layer.
    symbol = QgsFillSymbol.createSimple({
        "color": f"160,160,160,{INSTANCE_FILL_ALPHA}",
        "style": "solid",
        "outline_color": "60,60,60,255",
        "outline_width": INSTANCE_OUTLINE_WIDTH,
        "outline_style": "solid",
    })
    symbol_layer = symbol.symbolLayer(0)
    for key, expression in ((_fill_color_property(), fill_expr),
                            (_stroke_color_property(), stroke_expr)):
        if key is None:
            continue
        symbol_layer.setDataDefinedProperty(
            key, QgsProperty.fromExpression(expression))
    return QgsSingleSymbolRenderer(symbol)


def _fill_color_property():
    """Data-defined FillColor property key, through the shared shim."""
    from .qt_compat import symbol_fill_color_property

    return symbol_fill_color_property()


def _log_symbology_failure(step: str, err: Exception) -> None:
    """One Warning line for a colouring step that did not work.

    Every step here is cosmetic and none of them fails the run, but a layer
    handed back in the wrong style otherwise looks like nothing was tried.
    """
    try:
        QgsMessageLog.logMessage(
            f"Instance colours: {step} failed: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # noqa: BLE001 -- logging must never raise  # nosec B110
        pass


def _store_style_in_the_file(layer) -> None:
    """Write the new style into the GeoPackage beside the objects.

    The export stores its style the moment the table is written, so a renderer
    set afterwards would live only in this QGIS session. Saving under the same
    style name replaces that stored style rather than adding a second one.
    """
    try:
        layer.saveStyleToDatabase(layer.name(), "AI Segmentation", True, "")
    except Exception as err:  # noqa: BLE001 -- style persistence never fails a run
        _log_symbology_failure("saving the style into the file", err)


def paint_instances_apart(layer) -> dict:
    """Give every object on a saved layer its own colour, in the file as well.

    Cosmetic and best-effort from end to end: a layer this cannot colour keeps
    the export style, and the run that produced it is still a run that worked.

    Returns a dict with ``applied`` (bool), ``objects`` (int, -1 when the count
    could not be read), ``ceiling`` (int) and ``note`` (str, a sentence for the
    caller when nothing was applied).

    The count decides nothing: a big layer is coloured like a small one, since
    the renderer is the same size either way.
    """
    ceiling = instance_color_ceiling()
    # The ids ARE the count, and the renderer needs them anyway: reading
    # featureCount() first asked the provider for the same layer twice.
    try:
        ids = list(layer.allFeatureIds())
        objects = len(ids)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        ids = None
        objects = -1
    renderer = None
    try:
        renderer = make_instance_renderer(layer, feature_ids=ids)
    except Exception as err:  # noqa: BLE001 -- colouring never fails a run
        _log_symbology_failure("building the colours", err)
    if renderer is None:
        return {
            "applied": False,
            "objects": objects,
            "ceiling": ceiling,
            "note": (
                "The layer keeps its export style: it holds nothing to colour, "
                "or the colours could not be built."
            ),
        }
    try:
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as err:  # noqa: BLE001 -- colouring never fails a run
        _log_symbology_failure("putting the colours on the layer", err)
        return {
            "applied": False,
            "objects": objects,
            "ceiling": ceiling,
            "note": "The layer keeps its export style: the colours could not be applied.",
        }
    _store_style_in_the_file(layer)
    return {"applied": True, "objects": objects, "ceiling": ceiling, "note": ""}


def report_instance_colors(result: dict, layer) -> dict:
    """Colour ``layer`` one hue per object and write the outcome onto ``result``.

    The entry point a caller of the public zone run uses. It adds
    ``instance_colors`` (bool) and, when that is False, ``instance_colors_note``
    saying why in one sentence, so the ceiling is reported rather than passed
    over in silence. ``result`` is returned either way: this is the last step
    of a run whose objects are already saved, so it never raises and never
    turns a run that worked into a failure.
    """
    if layer is None:
        result["instance_colors"] = False
        result["instance_colors_note"] = (
            "The objects are saved. The layer could not be reached to colour "
            "them, so it keeps its export style.")
        return result
    try:
        report = paint_instances_apart(layer)
    except Exception as err:  # noqa: BLE001 -- colouring never fails a run
        _log_symbology_failure("colouring the saved layer", err)
        result["instance_colors"] = False
        result["instance_colors_note"] = (
            "The objects are saved. Colouring them one by one did not work, "
            "so the layer keeps its export style.")
        return result
    result["instance_colors"] = bool(report.get("applied"))
    if not result["instance_colors"]:
        result["instance_colors_note"] = (
            report.get("note") or "The layer keeps its export style.")
    return result
