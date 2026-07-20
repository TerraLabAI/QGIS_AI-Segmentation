























from __future__ import annotations

import colorsys

from qgis.core import Qgis, QgsFillSymbol, QgsMessageLog
from qgis.PyQt.QtGui import QColor







GOLDEN_ANGLE_DEGREES = 137.50776405003785




INSTANCE_SATURATION = 0.68
INSTANCE_VALUE = 0.88





MAX_RELATIVE_LUMINANCE = 0.62





INSTANCE_HUE_BUCKETS = 64



INSTANCE_FILL_ALPHA = 64
INSTANCE_OUTLINE_WIDTH = "0.66"




INSTANCE_CLASSIFIER = "$id"



INSTANCE_OUTLINE_DARKEN = 100.0 / 115.0









NO_INSTANCE_COLOR_CEILING = 1_000_000_000


def instance_color_ceiling() -> int:








    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "symbology.instance.max_categories", NO_INSTANCE_COLOR_CEILING,
            1, NO_INSTANCE_COLOR_CEILING))
    except Exception:  # noqa: BLE001  # nosec B110
        return NO_INSTANCE_COLOR_CEILING


def instance_palette_in_force() -> tuple[float, float, float]:






    try:
        from .server_dials import dial_in_range

        return (
            float(dial_in_range("symbology.instance.saturation", INSTANCE_SATURATION, 0.2, 1.0)),
            float(dial_in_range("symbology.instance.value", INSTANCE_VALUE, 0.2, 1.0)),
            float(dial_in_range(
                "symbology.instance.max_lightness", MAX_RELATIVE_LUMINANCE, 0.2, 1.0)),
        )
    except Exception:  # noqa: BLE001  # nosec B110
        return (INSTANCE_SATURATION, INSTANCE_VALUE, MAX_RELATIVE_LUMINANCE)


def instance_style_in_force() -> tuple[int, int, str]:





    try:
        from .server_dials import dial_in_range

        buckets = int(dial_in_range(
            "tuning.symbology.instance_hue_buckets", INSTANCE_HUE_BUCKETS, 8, 256))
        alpha = int(dial_in_range(
            "tuning.symbology.instance_fill_alpha", INSTANCE_FILL_ALPHA, 16, 160))
        shipped_width = float(INSTANCE_OUTLINE_WIDTH)
        width = float(dial_in_range(
            "tuning.symbology.instance_outline_width", shipped_width, 0.2, 2.0))
        width_text = INSTANCE_OUTLINE_WIDTH if width == shipped_width else f"{width:g}"
        return buckets, alpha, width_text
    except Exception:  # noqa: BLE001  # nosec B110
        return INSTANCE_HUE_BUCKETS, INSTANCE_FILL_ALPHA, INSTANCE_OUTLINE_WIDTH


def _under_the_lightness_ceiling(
    red: float, green: float, blue: float, ceiling: float = MAX_RELATIVE_LUMINANCE
) -> tuple[float, float, float]:

    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance <= ceiling or luminance <= 0:
        return red, green, blue
    scale = ceiling / luminance
    return red * scale, green * scale, blue * scale


def instance_color_hex(index: int, palette: tuple[float, float, float] | None = None) -> str:







    if palette is None:
        palette = instance_palette_in_force()
    saturation, value, ceiling = palette
    hue = (int(index) * GOLDEN_ANGLE_DEGREES) % 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    channels = _under_the_lightness_ceiling(red, green, blue, ceiling)
    return "#" + "".join(
        f"{max(0, min(255, round(channel * 255))):02x}" for channel in channels)


def _stroke_color_property():








    from qgis.core import QgsSymbolLayer

    prop_scope = getattr(QgsSymbolLayer, "Property", None)
    for owner, name in (
        (prop_scope, "StrokeColor"),
        (QgsSymbolLayer, "PropertyStrokeColor"),
        (prop_scope, "PropertyStrokeColor"),
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











    if palette is None:
        palette = instance_palette_in_force()
    saturation, value, ceiling = palette
    hue = f"(({INSTANCE_CLASSIFIER} - {int(first_id)}) * {GOLDEN_ANGLE_DEGREES}) % 360"
    base = (f"color_hsv({hue}, {round(saturation * 100)},"
            f" {round(value * 100)})")



    bands = [f"color_part(@c, '{band}')" for band in ("red", "green", "blue")]



    factor = (f"min(1, {ceiling} * 255 / max(1, 0.2126 * {bands[0]}"
              f" + 0.7152 * {bands[1]} + 0.0722 * {bands[2]}))")
    channels = ", ".join(f"round({band} * @k * {scale})" for band in bands)
    return (f"with_variable('c', {base},"
            f" with_variable('k', {factor},"
            f" color_rgba({channels}, {int(alpha)})))")


def _bucket_classifier(layer, buckets: int = INSTANCE_HUE_BUCKETS) -> str | None:














    try:
        names = {f.name().lower(): f.name() for f in layer.fields()}
    except (AttributeError, RuntimeError, TypeError):
        return None
    for candidate in ("det_id", "fid"):
        real = names.get(candidate)
        if real:
            return f'(to_int(abs("{real}")) * 67) % {int(buckets)}'
    return None


def _bucketed_instance_renderer(layer, palette, style=None):









    from qgis.core import QgsCategorizedSymbolRenderer, QgsRendererCategory

    buckets, fill_alpha, outline_width = style or instance_style_in_force()
    classifier = _bucket_classifier(layer, buckets)
    if classifier is None:
        return None
    categories = []
    for bucket in range(buckets):
        color = QColor(instance_color_hex(bucket, palette))
        outline = color.darker(115)
        fill = QColor(color)
        fill.setAlpha(fill_alpha)
        symbol = QgsFillSymbol.createSimple({
            "color": f"{fill.red()},{fill.green()},{fill.blue()},{fill.alpha()}",
            "style": "solid",
            "outline_color": f"{outline.red()},{outline.green()},{outline.blue()},255",
            "outline_width": outline_width,
            "outline_style": "solid",
        })


        categories.append(QgsRendererCategory(bucket, symbol, ""))

    fallback = QgsFillSymbol.createSimple({
        "color": f"160,160,160,{fill_alpha}",
        "style": "solid",
        "outline_color": "60,60,60,255",
        "outline_width": outline_width,
        "outline_style": "solid",
    })
    categories.append(QgsRendererCategory(None, fallback, "", True))
    return QgsCategorizedSymbolRenderer(classifier, categories)


def make_instance_renderer(layer, *, feature_ids=None):
















    from qgis.core import QgsProperty, QgsSingleSymbolRenderer

    try:
        ids = layer.allFeatureIds() if feature_ids is None else feature_ids
        ids = sorted(ids)
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not ids:
        return None
    palette = instance_palette_in_force()
    style = instance_style_in_force()
    _buckets, fill_alpha, outline_width = style
    bucketed = _bucketed_instance_renderer(layer, palette, style)
    if bucketed is not None:
        return bucketed
    first_id = int(ids[0])
    fill_expr = instance_color_expression(
        first_id, alpha=fill_alpha, palette=palette)
    stroke_expr = instance_color_expression(
        first_id, alpha=255, scale=INSTANCE_OUTLINE_DARKEN, palette=palette)



    symbol = QgsFillSymbol.createSimple({
        "color": f"160,160,160,{fill_alpha}",
        "style": "solid",
        "outline_color": "60,60,60,255",
        "outline_width": outline_width,
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

    from .qt_compat import symbol_fill_color_property

    return symbol_fill_color_property()


def _log_symbology_failure(step: str, err: Exception) -> None:





    try:
        QgsMessageLog.logMessage(
            f"Instance colours: {step} failed: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _store_style_in_the_file(layer) -> None:










    try:
        from .layer_conventions import persist_layer_to_file_later

        persist_layer_to_file_later(layer, style=True)
    except Exception as err:  # noqa: BLE001
        _log_symbology_failure("saving the style into the file", err)


def _collapse_legend_node(layer) -> None:





    try:
        from qgis.core import QgsProject

        node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
        if node is not None:
            node.setExpanded(False)
    except Exception as err:  # noqa: BLE001
        _log_symbology_failure("folding the legend entry", err)


def paint_instances_apart(layer) -> dict:












    ceiling = instance_color_ceiling()


    try:
        ids = list(layer.allFeatureIds())
        objects = len(ids)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        ids = None
        objects = -1
    renderer = None
    try:
        renderer = make_instance_renderer(layer, feature_ids=ids)
    except Exception as err:  # noqa: BLE001
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
        _collapse_legend_node(layer)
        layer.triggerRepaint()
    except Exception as err:  # noqa: BLE001
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









    if layer is None:
        result["instance_colors"] = False
        result["instance_colors_note"] = (
            "The objects are saved. The layer could not be reached to colour "
            "them, so it keeps its export style.")
        return result
    try:
        report = paint_instances_apart(layer)
    except Exception as err:  # noqa: BLE001
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
