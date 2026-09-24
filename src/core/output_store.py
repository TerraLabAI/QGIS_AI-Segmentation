
















from __future__ import annotations

import os
import re
import tempfile
import time
import zlib
from pathlib import Path
from typing import NamedTuple

from qgis.core import (
    Qgis,
    QgsLayerTree,
    QgsMapLayer,
    QgsMessageLog,
    QgsProject,
    QgsProviderRegistry,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QDate, Qt
from qgis.PyQt.QtGui import QColor

from .i18n import tr
from .output_group_order import keep_group_above_imagery



_COMMITTED_AT_PROP = "ai_segmentation/committed_at"

GPKG_FILENAME = "ai_segmentation.gpkg"

GROUP_NAME = "AI Segmentation"

_LOG_TAG = "AI Segmentation"





_SEMANTIC_COLORS: list[tuple[str, str]] = [
    ("water", "#1f78b4"), ("river", "#1f78b4"), ("lake", "#1f78b4"),
    ("sea", "#1f78b4"), ("pool", "#1f78b4"), ("coast", "#1f78b4"),
    ("tree", "#33a02c"), ("forest", "#33a02c"), ("vegetation", "#33a02c"),
    ("grass", "#33a02c"), ("hedge", "#33a02c"), ("park", "#33a02c"),
    ("crop", "#33a02c"),
    ("road", "#4d4d4d"), ("street", "#4d4d4d"), ("path", "#4d4d4d"),
    ("runway", "#4d4d4d"), ("parking", "#4d4d4d"),
    ("building", "#e6550d"), ("roof", "#e6550d"), ("house", "#e6550d"),
    ("solar", "#6a3d9a"), ("panel", "#6a3d9a"),
    ("car", "#ff7f00"), ("vehicle", "#ff7f00"), ("truck", "#ff7f00"),
    ("boat", "#ff7f00"),
    ("field", "#7f9a2d"), ("parcel", "#7f9a2d"), ("farm", "#7f9a2d"),
]




_FALLBACK_PALETTE: list[str] = [
    "#0d888c",
    "#c2308f",
    "#3949ab",
    "#c62828",
    "#8d5524",
    "#0277bd",
    "#ef6c00",
    "#7b1fa2",
]


_LEGACY_COMMITTED_RED = QColor(220, 0, 0)


class WriteResult(NamedTuple):


    gpkg_path: str
    table_name: str
    layer: QgsVectorLayer
    used_fallback: bool
    error_message: str





    intended_path: str = ""


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

_MAX_SERVED_COLORS = 128


def _served_color_map(path: str) -> dict[str, str]:

    out: dict[str, str] = {}
    try:
        from .server_dials import read_value

        value = read_value(path)
        if not isinstance(value, dict):
            return out
        for key, color in list(value.items())[:_MAX_SERVED_COLORS]:
            if not isinstance(key, str) or not isinstance(color, str):
                continue
            word = key.strip().lower()
            if word and _HEX_COLOR_RE.match(color.strip()):
                out[word] = color.strip().lower()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return out


def semantic_colors() -> list[tuple[str, str]]:









    served = _served_color_map("taxonomy.class_colors")
    if not served:
        return list(_SEMANTIC_COLORS)
    merged = dict(_SEMANTIC_COLORS)
    merged.update(served)
    return list(merged.items())


def fallback_palette() -> list[str]:





    out = list(_FALLBACK_PALETTE)
    try:
        from .server_dials import read_value

        value = read_value("taxonomy.fallback_palette")
        if isinstance(value, (list, tuple)):
            for item in value[:_MAX_SERVED_COLORS]:
                if isinstance(item, str) and _HEX_COLOR_RE.match(item.strip()):
                    color = item.strip().lower()
                    if color not in out:
                        out.append(color)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return out


def committed_color_for_prompt(prompt: str) -> QColor:





    norm = (prompt or "").strip().lower()
    if not norm:
        return QColor(_LEGACY_COMMITTED_RED)
    for keyword, hex_color in sorted(
        semantic_colors(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if keyword in norm:
            return QColor(hex_color)
    palette = fallback_palette()
    index = zlib.crc32(norm.encode("utf-8")) % len(palette)
    return QColor(palette[index])


def _find_output_group(root):










    for child in root.children():
        if QgsLayerTree.isGroup(child) and child.name() == GROUP_NAME:
            return child
    return None


def _output_group_layer_names() -> set[str]:
    names: set[str] = set()
    try:
        root = QgsProject.instance().layerTreeRoot()
        group = _find_output_group(root)
        if group is None:
            return names
        for node in group.findLayers():
            layer = node.layer()
            names.add(layer.name() if layer is not None else node.name())
    except Exception:  # nosec B110
        pass
    return names


def friendly_layer_name(prompt: str, gpkg_path: str = "") -> str:















    base = (prompt or "").strip()
    base = (base[0].upper() + base[1:]) if base else tr("Segmentation")
    date_str = QDate.currentDate().toString(Qt.DateFormat.ISODate)
    existing = {name.lower() for name in _output_group_layer_names()}
    if gpkg_path:
        from .output_gpkg_rollover import layer_identifiers

        existing |= {name.lower() for name in (layer_identifiers(gpkg_path) or set())}
    candidate = f"{base} ({date_str})"
    counter = 2
    while candidate.lower() in existing:
        candidate = f"{base} {counter} ({date_str})"
        counter += 1
    return candidate


def _existing_tables(gpkg_path: str) -> set[str] | None:












    from .output_gpkg_rollover import file_size, table_names

    if file_size(gpkg_path) == 0:
        return set()



    names = table_names(gpkg_path)
    if names:
        return names
    try:
        metadata = QgsProviderRegistry.instance().providerMetadata("ogr")
        if metadata is not None:
            names = {
                details.name()
                for details in metadata.querySublayers(gpkg_path)
                if details.name()
            }
            if names:
                return names
    except Exception:  # nosec B110
        pass
    return None


def _table_exists(gpkg_path: str, table: str, tables: set[str] | None) -> bool:
    if tables is not None:



        folded = table.lower()
        return any(name.lower() == folded for name in tables)

    try:
        probe = QgsVectorLayer(f"{gpkg_path}|layername={table}", "probe", "ogr")
        return probe.isValid()
    except Exception:
        return False



_NAME_MAX_CHARS = 40


def _name_max_chars() -> int:

    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("tuning.export.name_max_chars", _NAME_MAX_CHARS, 20, 80))
    except Exception:  # noqa: BLE001  # nosec B110
        return _NAME_MAX_CHARS


def snake_table_name(prompt: str, gpkg_path: str) -> str:





    base = re.sub(r"[^\w]+", "_", (prompt or "").strip().lower()).strip("_")
    base = _ascii_table_stem(base)[:_name_max_chars()].strip("_") or "segmentation"



    if base[0].isdigit():
        base = f"t_{base}"
    date_str = QDate.currentDate().toString("yyyyMMdd")
    tables = _existing_tables(gpkg_path)
    candidate = f"{base}_{date_str}"
    counter = 2
    while _table_exists(gpkg_path, candidate, tables):
        candidate = f"{base}_{date_str}_{counter}"
        counter += 1
    return candidate


def _ascii_table_stem(base: str) -> str:








    import unicodedata

    folded = unicodedata.normalize("NFKD", base)
    out = []
    for char in folded:
        if unicodedata.combining(char):
            continue
        if char.isascii():
            out.append(char)
        else:
            out.append(f"u{ord(char):04x}")
    return "".join(out)


def _probe_writable(directory: str) -> bool:









    try:
        with tempfile.TemporaryFile(dir=directory, suffix=".tmp") as probe:
            probe.write(b"ok")
    except OSError:
        return False
    return True


def _source_layer_dir(source_layer) -> str:

    try:
        source = (source_layer.source() or "") if source_layer is not None else ""
    except Exception:
        return ""
    path = source.split("|")[0]
    if not path or "://" in path or path.lower().startswith(("http", "type=")):
        return ""
    if os.path.isfile(path):
        return os.path.dirname(path)




    try:
        decoded = QgsProviderRegistry.instance().decodeUri("gdal", source) or {}
        inner = str(decoded.get("path") or "")
    except Exception:  # noqa: BLE001
        return ""
    if inner and inner != path and "://" not in inner and os.path.isfile(inner):
        return os.path.dirname(inner)
    return ""


def output_directory_candidates(source_layer, first_only: bool = False) -> list[str]:















    return writable_directories(
        output_directory_candidate_paths(source_layer), first_only=first_only)


def output_directory_candidate_paths(source_layer) -> list[str]:



    project = QgsProject.instance()
    return [
        project.homePath() or project.absolutePath(),
        _source_layer_dir(source_layer),
        str(Path.home()),
    ]


def writable_directories(ordered: list[str], first_only: bool = False) -> list[str]:


    home = str(Path.home())
    seen: set[str] = set()
    writable: list[str] = []
    for candidate in ordered:
        if not candidate or not os.path.isdir(candidate):
            continue
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if _probe_writable(candidate):
            writable.append(candidate)
            if first_only:
                break
    return writable or [home]


def _output_directory(source_layer) -> str:

    return output_directory_candidates(source_layer, first_only=True)[0]


def project_gpkg_path(source_layer) -> str:





    from .output_gpkg_rollover import next_output_gpkg




    return os.path.normpath(
        next_output_gpkg(_output_directory(source_layer), GPKG_FILENAME))


def _ground_metre_transform(memory_layer, plan: dict | None = None):










    from qgis.core import QgsCoordinateTransform

    from .layer_conventions import pick_output_crs

    try:
        source = memory_layer.crs()
        if plan is None:
            target = pick_output_crs(source, memory_layer.extent())
            if target is None or not target.isValid() or target == source:
                return None
            return QgsCoordinateTransform(source, target, QgsProject.instance())
        target = pick_output_crs(
            source, memory_layer.extent(), project_crs=plan["project_crs"],
            transform_context=plan["context"], ellipsoid=plan["ellipsoid"])
        if target is None or not target.isValid() or target == source:
            return None
        return QgsCoordinateTransform(source, target, plan["context"])
    except (RuntimeError, AttributeError, TypeError, KeyError):
        return None


def _write_gpkg(memory_layer, path: str, table: str, overwrite_file: bool,
                transform=None, identifier: str = "",
                description: str = "", transform_context=None) -> str:




    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = table



    layer_options = []
    if identifier:
        layer_options.append(f"IDENTIFIER={_gpkg_option_value(identifier)}")
    if description:
        layer_options.append(f"DESCRIPTION={_gpkg_option_value(description)}")
    if layer_options:
        options.layerOptions = layer_options
    if transform is not None:
        options.ct = transform
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
        if overwrite_file
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
    )
    if transform_context is None:
        transform_context = QgsProject.instance().transformContext()
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        memory_layer,
        path,
        transform_context,
        options,
    )
    if result[0] == QgsVectorFileWriter.WriterError.NoError:
        return ""
    return str(result[1]) if len(result) > 1 and result[1] else "unknown writer error"




_MAX_GPKG_OPTION_CHARS = 250


def _gpkg_option_value(text: str) -> str:

    flat = " ".join(str(text or "").split())
    return flat[:_MAX_GPKG_OPTION_CHARS]


def _load_table(path: str, table: str, display_name: str) -> QgsVectorLayer | None:















    try:
        layer = QgsVectorLayer(f"{path}|layername={table}", display_name, "ogr")
        if not layer.isValid():
            return None
        try:
            layer.dataProvider().reloadData()
            layer.updateExtents()
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        return layer
    except Exception:  # nosec B110
        pass
    return None


def write_run_table(memory_layer, *, prompt: str, source_layer, fallback_stem: str) -> WriteResult | None:










    plan = plan_run_table(prompt, source_layer, fallback_stem)
    return write_planned_run_table(memory_layer, plan, load=_load_table)


def plan_run_table(prompt: str, source_layer, fallback_stem: str) -> dict:




    gpkg_path = project_gpkg_path(source_layer)
    note_unexpected_output_folder(os.path.dirname(gpkg_path), source_layer)
    project = QgsProject.instance()
    return {
        "gpkg_path": gpkg_path,
        "table": snake_table_name(prompt, gpkg_path),
        "friendly": friendly_layer_name(prompt, gpkg_path),
        "fallback_stem": fallback_stem,
        "context": project.transformContext(),
        "project_crs": project.crs(),
        "ellipsoid": project.ellipsoid(),
        "candidate_dirs": output_directory_candidate_paths(source_layer),
    }


def write_planned_run_table(memory_layer, plan: dict, load=None) -> WriteResult | None:









    from .output_gpkg_rollover import file_size

    gpkg_path = plan["gpkg_path"]
    table = plan["table"]
    friendly = plan["friendly"]
    context = plan["context"]
    transform = _ground_metre_transform(memory_layer, plan)







    description = "Objects detected with AI Segmentation."
    error_message = _write_gpkg(
        memory_layer, gpkg_path, table,
        overwrite_file=file_size(gpkg_path) == 0, transform=transform,
        identifier=friendly, description=description,
        transform_context=context,
    )
    if not error_message:
        if load is None:
            return WriteResult(gpkg_path, table, None, False, "", gpkg_path)
        layer = load(gpkg_path, table, friendly)
        if layer is not None:
            return WriteResult(gpkg_path, table, layer, False, "", gpkg_path)
        error_message = "saved table could not be reloaded"
    QgsMessageLog.logMessage(
        f"Shared GeoPackage write failed ({error_message}), falling back to a per-run file",
        _LOG_TAG, level=Qgis.MessageLevel.Warning,
    )

    stem = re.sub(r"[^\w\- ]", "", plan.get("fallback_stem") or "").strip().replace(" ", "_")
    stem = stem[:_name_max_chars()] or "detection"
    timestamp = time.strftime("%Y%m%d_%H%M%S")




    fallback_error = "no writable output directory"



    failures: list[tuple[str, str]] = []
    for directory in writable_directories(plan.get("candidate_dirs") or []):



        fallback_path = ""
        try:


            with tempfile.NamedTemporaryFile(
                dir=os.path.normpath(directory), prefix=f"{stem}_{timestamp}_", suffix=".gpkg", delete=False,
            ) as target:
                fallback_path = target.name
            fallback_error = _write_gpkg(
                memory_layer, fallback_path, table, overwrite_file=True,
                transform=transform, identifier=friendly, description=description,
                transform_context=context)
        except OSError as exc:
            fallback_error = str(exc)
        if not fallback_error:
            if load is None:
                return WriteResult(fallback_path, table, None, True, error_message, gpkg_path)
            layer = load(fallback_path, table, friendly)
            if layer is not None:
                return WriteResult(fallback_path, table, layer, True, error_message, gpkg_path)
            fallback_error = "saved file could not be reloaded"
        if fallback_path and file_size(fallback_path) == 0:
            try:
                os.unlink(fallback_path)
            except OSError:  # nosec B110
                pass
        failures.append((directory, fallback_error))
        QgsMessageLog.logMessage(
            f"Fallback export failed in this folder ({fallback_error}), trying the next one",
            _LOG_TAG, level=Qgis.MessageLevel.Warning,
        )
    detail = "; ".join(f"{folder}: {reason}" for folder, reason in failures) or fallback_error
    QgsMessageLog.logMessage(
        f"Fallback export failed everywhere: {detail}",
        _LOG_TAG, level=Qgis.MessageLevel.Critical,
    )
    return None


def load_written_run_table(written: WriteResult | None,
                           friendly: str) -> WriteResult | None:



    if written is None:
        return None
    layer = _load_table(written.gpkg_path, written.table_name,
                        friendly or written.table_name)
    if layer is None:
        return None
    return written._replace(layer=layer)




_output_folder_notice = {"shown": False}


def note_unexpected_output_folder(directory: str, source_layer) -> None:







    if _output_folder_notice["shown"] or not directory:
        return
    try:
        project = QgsProject.instance()
        expected = [project.homePath() or project.absolutePath(),
                    _source_layer_dir(source_layer)]
        key = os.path.normcase(os.path.abspath(directory))
        for candidate in expected:
            if candidate and os.path.normcase(os.path.abspath(candidate)) == key:
                return
        _output_folder_notice["shown"] = True
        from qgis.utils import iface

        if iface is None:
            return
        iface.messageBar().pushInfo(
            GROUP_NAME,
            tr("Saved to {folder}. Save the project to keep your results "
               "beside it.").format(folder=directory))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def ensure_output_group():





    root = QgsProject.instance().layerTreeRoot()
    group = _find_output_group(root)
    if group is None:
        group = root.insertGroup(0, GROUP_NAME)
    return group


def _raster_subgroup(parent_group, source_name: str | None):







    name = (source_name or "").strip()
    if not name:
        return parent_group



    sub = None
    for child in parent_group.children():
        if QgsLayerTree.isGroup(child) and child.name() == name:
            sub = child
            break
    if sub is None:


        sub = parent_group.addGroup(name)
    return sub


def add_committed_layer(layer, source_name: str | None = None) -> None:







    if QgsProject.instance().addMapLayer(layer, False) is None:


        QgsMessageLog.logMessage(
            "Output store: the saved layer could not be registered",
            _LOG_TAG, level=Qgis.MessageLevel.Warning)
        return
    try:
        layer.setCustomProperty(_COMMITTED_AT_PROP, time.time())
    except Exception:  # nosec B110
        pass
    top = ensure_output_group()
    group = _raster_subgroup(top, source_name)
    node = group.insertLayer(0, layer)
    if node is not None:
        node.setItemVisibilityChecked(True)
    if group is not top:
        group.setItemVisibilityChecked(True)
        group.setExpanded(True)
    top.setItemVisibilityChecked(True)
    top.setExpanded(True)





    keep_group_above_imagery(top)



    apply_fast_canvas_render(layer)






RENDER_SIMPLIFY_PX = 1.5


def render_simplify_px() -> float:







    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "ui.render_simplify_px", RENDER_SIMPLIFY_PX, 1.0, 5.0))
    except Exception:  # noqa: BLE001  # nosec B110
        return RENDER_SIMPLIFY_PX


def _apply_render_simplify(layer) -> None:

    try:
        from qgis.core import QgsVectorSimplifyMethod

        from .qt_compat import (
            SimplifyDistanceAlgorithm,
            SimplifyFullHint,
            SimplifyGeometryHint,
        )

        method = QgsVectorSimplifyMethod()
        hint = SimplifyFullHint if SimplifyFullHint is not None else SimplifyGeometryHint
        if hint is not None:
            method.setSimplifyHints(hint)
        if SimplifyDistanceAlgorithm is not None:
            method.setSimplifyAlgorithm(SimplifyDistanceAlgorithm)
        method.setThreshold(render_simplify_px())
        method.setForceLocalOptimization(True)
        layer.setSimplifyMethod(method)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def apply_fast_canvas_render(layer) -> None:

























    try:
        from .server_dials import feature_enabled

        simplify = feature_enabled("render_simplify")
    except Exception:  # noqa: BLE001  # nosec B110
        simplify = True

    if simplify:
        _apply_render_simplify(layer)
    try:
        provider = layer.dataProvider()
        if _provider_lacks_spatial_index(provider):
            provider.createSpatialIndex()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _provider_lacks_spatial_index(provider) -> bool:









    try:
        from .qt_compat import resolve_qt_enum

        present = resolve_qt_enum(
            type(provider), "SpatialIndexPresence", "SpatialIndexPresent")
    except AttributeError:
        return True
    try:
        return provider.hasSpatialIndex() != present
    except (AttributeError, TypeError, RuntimeError):
        return True


def mark_temp_layer(layer) -> None:







    try:
        layer.setFlags(layer.flags() | QgsMapLayer.LayerFlag.Private)
    except (AttributeError, TypeError):  # nosec B110
        pass
    try:
        layer.setCustomProperty("skipMemoryLayersCheck", 1)
        layer.setCustomProperty("ai_segmentation/temp", True)
    except Exception:  # nosec B110
        pass


def drop_from_snapping(layer) -> None:








    try:
        project = QgsProject.instance()
        cfg = project.snappingConfig()
        cfg.removeLayers([layer])
        project.setSnappingConfig(cfg)
    except Exception:  # nosec B110
        pass


def sweep_stale_temp_layers() -> None:







    try:
        project = QgsProject.instance()
        stale = [
            layer_id
            for layer_id, layer in project.mapLayers().items()
            if layer.customProperty("ai_segmentation/temp")
        ]
        if stale:
            project.removeMapLayers(stale)
    except Exception:  # nosec B110
        pass
