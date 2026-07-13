"""Project-level output store for committed segmentation runs.

One GeoPackage per project (``ai_segmentation.gpkg`` next to the project
file), one table per run, every committed layer under the single
"AI Segmentation" layer-tree group, inside a per-raster sub-group named
after the source raster (so runs group by the layer they were made on),
with a human-friendly name like "Buildings (3 Jul)". Also owns the
temp-layer hygiene: working memory
layers are flagged Private so they render on canvas without ever
appearing in the layer tree, and stale ones are swept on project load.

All functions are main-thread only (QgsProject / layer tree access) and
best-effort: a failure degrades to a fallback or a no-op, never an
exception at the call site.
"""
from __future__ import annotations

import os
import re
import time
import zlib
from pathlib import Path
from typing import NamedTuple

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsLayerTree,
    QgsMapLayer,
    QgsMessageLog,
    QgsProject,
    QgsProviderRegistry,
    QgsVectorDataProvider,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QDate, QLocale
from qgis.PyQt.QtGui import QColor

from .i18n import tr
from .qt_compat import PolygonGeometry

# Provider capability flag, resolved defensively: the enum is scoped
# (QgsVectorDataProvider.Capability.AddFeatures) on newer QGIS/Qt6 and
# unscoped on QGIS 3.x. Used to keep the append destination list to layers
# that actually accept new features.
try:
    _CAP_ADD_FEATURES = QgsVectorDataProvider.Capability.AddFeatures
except AttributeError:  # pragma: no cover - QGIS 3.x unscoped enum
    _CAP_ADD_FEATURES = QgsVectorDataProvider.AddFeatures

# Custom property stamped on every committed layer with its creation epoch, so
# the append destination list can order our own outputs most-recent-first
# (the just-finished Automatic run is the obvious append target). Survives a
# project save/reload since custom properties serialize into the .qgz.
_COMMITTED_AT_PROP = "ai_segmentation/committed_at"

GPKG_FILENAME = "ai_segmentation.gpkg"
# Brand term, deliberately not translated (product names stay English).
GROUP_NAME = "AI Segmentation"

_LOG_TAG = "AI Segmentation"

# Cartographic conventions first: a geomatician expects water to be blue and
# vegetation to be green whatever tool produced the layer (the SCP model:
# class color is stable and meaningful). Longer keywords are matched first so
# "parking" wins over "park". Prompt tokens are always English upstream.
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

# Deterministic fallback for prompts outside the semantic map: the SAME prompt
# always hashes to the SAME color, across runs and sessions (crc32, not the
# per-process-salted hash()). Saturated but distinct from the semantic hues.
_FALLBACK_PALETTE: list[str] = [
    "#0d888c",  # teal
    "#c2308f",  # magenta
    "#3949ab",  # indigo
    "#c62828",  # crimson
    "#8d5524",  # warm brown
    "#0277bd",  # cyan-blue
    "#ef6c00",  # deep orange
    "#7b1fa2",  # violet
]

# Manual runs have no prompt: keep the legacy committed red.
_LEGACY_COMMITTED_RED = QColor(220, 0, 0)


class WriteResult(NamedTuple):
    """Outcome of write_run_table: where the run landed and how."""

    gpkg_path: str
    table_name: str
    layer: QgsVectorLayer
    used_fallback: bool
    error_message: str


def committed_color_for_prompt(prompt: str) -> QColor:
    """Stable color for a committed run, keyed to the prompt.

    Semantic cartographic colors for common object classes, a deterministic
    hashed palette pick otherwise, legacy red for promptless (Manual) runs.
    """
    norm = (prompt or "").strip().lower()
    if not norm:
        return QColor(_LEGACY_COMMITTED_RED)
    for keyword, hex_color in sorted(
        _SEMANTIC_COLORS, key=lambda kv: len(kv[0]), reverse=True
    ):
        if keyword in norm:
            return QColor(hex_color)
    index = zlib.crc32(norm.encode("utf-8")) % len(_FALLBACK_PALETTE)
    return QColor(_FALLBACK_PALETTE[index])


def _find_output_group(root):
    """Our top-level "AI Segmentation" group, or None. NON-recursive on purpose.

    ``QgsLayerTreeGroup.findGroup`` scans the ENTIRE tree, so it would adopt an
    unrelated user group of the same name nested anywhere in their layer tree
    and start writing committed runs into it. Our group is always created at the
    tree top (see ensure_output_group), so only a direct top-level child is the
    intended reuse target. A same-named top-level group IS reused (it is
    indistinguishable from one an older plugin version created, and that is the
    idempotent behaviour we want).
    """
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


def friendly_layer_name(prompt: str) -> str:
    """Tree name for a committed run: "Buildings (3 Jul)".

    Only the first letter is capitalized (the rest stays as typed), the date
    is locale-short, and a same-prompt-same-day rerun becomes
    "Buildings 2 (3 Jul)" by scanning the AI Segmentation group.
    """
    base = (prompt or "").strip()
    base = (base[0].upper() + base[1:]) if base else tr("Segmentation")
    date_str = QLocale().toString(QDate.currentDate(), "d MMM")
    existing = _output_group_layer_names()
    candidate = f"{base} ({date_str})"
    counter = 2
    while candidate in existing:
        candidate = f"{base} {counter} ({date_str})"
        counter += 1
    return candidate


def _existing_tables(gpkg_path: str) -> set[str] | None:
    """Tables already present in the GeoPackage, or None when unknown."""
    if not os.path.exists(gpkg_path):
        return set()
    try:
        metadata = QgsProviderRegistry.instance().providerMetadata("ogr")
        if metadata is not None:
            return {
                details.name()
                for details in metadata.querySublayers(gpkg_path)
                if details.name()
            }
    except Exception:  # nosec B110
        pass
    return None


def _table_exists(gpkg_path: str, table: str, tables: set[str] | None) -> bool:
    if tables is not None:
        return table in tables
    # querySublayers unavailable: probe the single candidate directly.
    try:
        probe = QgsVectorLayer(f"{gpkg_path}|layername={table}", "probe", "ogr")
        return probe.isValid()
    except Exception:
        return False


def snake_table_name(prompt: str, gpkg_path: str) -> str:
    """GeoPackage table name for a run: "buildings_20260703", deduped "_2"."""
    base = re.sub(r"[^a-z0-9]+", "_", (prompt or "").strip().lower()).strip("_")
    base = base[:40].strip("_") or "segmentation"
    date_str = QDate.currentDate().toString("yyyyMMdd")
    tables = _existing_tables(gpkg_path)
    candidate = f"{base}_{date_str}"
    counter = 2
    while _table_exists(gpkg_path, candidate, tables):
        candidate = f"{base}_{date_str}_{counter}"
        counter += 1
    return candidate


def _probe_writable(directory: str) -> bool:
    probe = os.path.join(directory, f".ai_seg_write_probe_{os.getpid()}")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _source_layer_dir(source_layer) -> str:
    """Directory of a file-backed source raster, '' for XYZ/WMS/web layers."""
    try:
        source = (source_layer.source() or "") if source_layer is not None else ""
    except Exception:
        return ""
    path = source.split("|")[0]
    if not path or "://" in path or path.lower().startswith(("http", "type=")):
        return ""
    if os.path.isfile(path):
        return os.path.dirname(path)
    return ""


def _output_directory(source_layer) -> str:
    """Writable output directory: project folder, raster folder, then home."""
    project = QgsProject.instance()
    candidates = [
        project.homePath() or project.absolutePath(),
        _source_layer_dir(source_layer),
        str(Path.home()),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and _probe_writable(candidate):
            return candidate
    return str(Path.home())


def project_gpkg_path(source_layer) -> str:
    """Full path of the per-project GeoPackage all runs accumulate into."""
    return os.path.join(_output_directory(source_layer), GPKG_FILENAME)


def _write_gpkg(memory_layer, path: str, table: str, overwrite_file: bool) -> str:
    """Run one V3 write. Returns '' on success, the error message otherwise."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = table
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
        if overwrite_file
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
    )
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        memory_layer,
        path,
        QgsProject.instance().transformContext(),
        options,
    )
    if result[0] == QgsVectorFileWriter.WriterError.NoError:
        return ""
    message = str(result[1]) if len(result) > 1 and result[1] else "unknown writer error"
    return message


def _load_table(path: str, table: str, display_name: str) -> QgsVectorLayer | None:
    try:
        layer = QgsVectorLayer(f"{path}|layername={table}", display_name, "ogr")
        if layer.isValid():
            return layer
    except Exception:  # nosec B110
        pass
    return None


def write_run_table(memory_layer, *, prompt: str, source_layer, fallback_stem: str) -> WriteResult | None:
    """Persist one run into the project GeoPackage as a new table.

    On any writer or reload failure the run falls back to a standalone
    per-run file (today's behavior) so a locked/corrupted shared gpkg never
    loses a paid detection. Returns None only when even the fallback fails.
    """
    gpkg_path = project_gpkg_path(source_layer)
    table = snake_table_name(prompt, gpkg_path)
    friendly = friendly_layer_name(prompt)

    error_message = _write_gpkg(
        memory_layer, gpkg_path, table, overwrite_file=not os.path.exists(gpkg_path)
    )
    if not error_message:
        layer = _load_table(gpkg_path, table, friendly)
        if layer is not None:
            return WriteResult(gpkg_path, table, layer, False, "")
        error_message = "saved table could not be reloaded"
    QgsMessageLog.logMessage(
        "Shared GeoPackage write failed ({}), falling back to a per-run file".format(
            error_message
        ),
        _LOG_TAG, level=Qgis.MessageLevel.Warning,
    )

    stem = re.sub(r"[^\w\- ]", "", fallback_stem or "").strip().replace(" ", "_")
    stem = stem[:40] or "detection"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fallback_path = os.path.join(
        _output_directory(source_layer), f"{stem}_{timestamp}.gpkg"
    )
    fallback_error = _write_gpkg(memory_layer, fallback_path, table, overwrite_file=True)
    if not fallback_error:
        layer = _load_table(fallback_path, table, friendly)
        if layer is not None:
            return WriteResult(fallback_path, table, layer, True, error_message)
        fallback_error = "saved file could not be reloaded"
    QgsMessageLog.logMessage(
        "Fallback export failed too: {}".format(fallback_error),
        _LOG_TAG, level=Qgis.MessageLevel.Critical,
    )
    return None


def ensure_output_group():
    """The single stable "AI Segmentation" group, created at the tree top.

    Looks up ONLY a direct top-level child (see _find_output_group), so a
    same-named group nested in the user's own hierarchy is never absorbed.
    """
    root = QgsProject.instance().layerTreeRoot()
    group = _find_output_group(root)
    if group is None:
        group = root.insertGroup(0, GROUP_NAME)
    return group


def _raster_subgroup(parent_group, source_name: str | None):
    """Get-or-create the per-raster sub-group inside the AI Segmentation group.

    Each committed run lands under a sub-group named after the raster it was
    segmented on, so the tree shows both WHERE the outputs live and WHICH layer
    produced them. Falls back to the parent group directly when the raster name
    is unknown (e.g. a run that outlived its source layer).
    """
    name = (source_name or "").strip()
    if not name:
        return parent_group
    # Scan only direct children (findGroup recurses the whole subtree and would
    # adopt an unrelated same-named group nested anywhere), mirroring
    # _find_output_group's non-recursive lookup.
    sub = None
    for child in parent_group.children():
        if QgsLayerTree.isGroup(child) and child.name() == name:
            sub = child
            break
    if sub is None:
        # Insert after any existing sub-groups so new rasters append at the
        # bottom while their own runs still stack newest-first inside.
        sub = parent_group.addGroup(name)
    return sub


def add_committed_layer(layer, source_name: str | None = None) -> None:
    """Register a committed layer and slot it first inside its raster sub-group.

    Committed layers are grouped by the raster they were segmented on: a
    per-raster sub-group under the single top-level "AI Segmentation" group.
    ``source_name`` is that raster's layer name; when absent the layer lands
    directly in the top group (legacy behavior).
    """
    QgsProject.instance().addMapLayer(layer, False)
    try:
        layer.setCustomProperty(_COMMITTED_AT_PROP, time.time())
    except Exception:  # nosec B110 - recency stamp is a convenience, never block
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


def mark_temp_layer(layer) -> None:
    """Flag a working memory layer so it renders without polluting the tree.

    Must run BEFORE the layer is added to the project: the layer-tree proxy
    filters Private layers at row-insert time, so setting the flag first
    avoids a one-frame flash in the Layers panel. skipMemoryLayersCheck
    suppresses the "temporary scratch layers will be lost" close prompt.
    """
    try:
        layer.setFlags(layer.flags() | QgsMapLayer.LayerFlag.Private)
    except (AttributeError, TypeError):  # nosec B110
        pass  # LayerFlag.Private needs QGIS 3.18+; floor is 3.22, belt and braces
    try:
        layer.setCustomProperty("skipMemoryLayersCheck", 1)
        layer.setCustomProperty("ai_segmentation/temp", True)
    except Exception:  # nosec B110
        pass


def drop_from_snapping(layer) -> None:
    """Scrub a scratch layer from the project snapping config, right AFTER it
    is registered (addMapLayer auto-enrolls vector layers there).

    QgsSnappingConfig keeps raw layer pointers: a freed working layer leaves a
    dangling entry that crashes the NEXT project save, typically the
    save-on-exit (upstream qgis/QGIS#37505, #42651). A layer with no entry can
    never dangle, and scratch layers have no business being snap targets.
    """
    try:
        project = QgsProject.instance()
        cfg = project.snappingConfig()
        cfg.removeLayers([layer])
        project.setSnappingConfig(cfg)
    except Exception:  # nosec B110
        pass


def sweep_stale_temp_layers() -> None:
    """Remove temp layers that leaked into a saved project.

    A mid-review project save writes the Private memory layers into the
    .qgz; they come back on load as empty invisible layers. This sweep,
    connected to QgsProject.readProject, deletes anything carrying our
    temp marker.
    """
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


# ---------------------------------------------------------------------------
# Append to an existing layer (incremental digitization). The killer path is a
# user who ran an Automatic detection (which committed a polygon layer) and now
# wants to add a few hand-drawn shapes to that SAME layer instead of a new one.
# ---------------------------------------------------------------------------

class AppendCandidate(NamedTuple):
    """One layer the current run could be appended to."""

    layer_id: str
    display_name: str
    is_ai_output: bool


class AppendResult(NamedTuple):
    """Outcome of append_run_to_layer: how many features landed, and why not."""

    added: int
    error_message: str


def is_appendable_polygon_layer(layer) -> bool:
    """True when ``layer`` is a valid, writable, polygon vector layer a run can
    be appended to. Excludes our own on-screen scratch layers (flagged temp)
    and read-only / non-feature-adding providers. Main-thread only."""
    if not isinstance(layer, QgsVectorLayer) or not layer.isValid():
        return False
    try:
        if layer.geometryType() != PolygonGeometry:
            return False
        if layer.customProperty("ai_segmentation/temp"):
            return False
        if layer.readOnly():
            return False
        provider = layer.dataProvider()
        if provider is None:
            return False
        return bool(provider.capabilities() & _CAP_ADD_FEATURES)
    except Exception:  # nosec B110 - a probing failure just excludes the layer
        return False


def list_appendable_polygon_layers() -> list[AppendCandidate]:
    """Editable polygon layers of the project, ours first (most-recent first),
    then the user's own polygon layers by name. Best-effort, main-thread only:
    any failure yields an empty list so the caller degrades to a new layer.
    """
    try:
        project = QgsProject.instance()
        group = _find_output_group(project.layerTreeRoot())
        ours: set[str] = set()
        if group is not None:
            for node in group.findLayers():
                layer = node.layer()
                if layer is not None:
                    ours.add(layer.id())
        ai_cands: list[tuple[float, AppendCandidate]] = []
        other_cands: list[AppendCandidate] = []
        for layer_id, layer in project.mapLayers().items():
            if not is_appendable_polygon_layer(layer):
                continue
            name = layer.name()
            if layer_id in ours:
                try:
                    stamp = float(layer.customProperty(_COMMITTED_AT_PROP) or 0.0)
                except (TypeError, ValueError):
                    stamp = 0.0
                ai_cands.append((stamp, AppendCandidate(layer_id, name, True)))
            else:
                other_cands.append(AppendCandidate(layer_id, name, False))
        ai_cands.sort(key=lambda item: item[0], reverse=True)
        other_cands.sort(key=lambda cand: cand.display_name.lower())
        return [cand for _, cand in ai_cands] + other_cands
    except Exception:  # nosec B110
        return []


def _coerce_to_layer_type(geom: QgsGeometry, target_layer) -> list[QgsGeometry]:
    """Coerce a repaired polygon geometry to the target layer's WKB type,
    returning one or more non-empty geometries. Uses ``coerceToType`` so a
    single-part Polygon target and a MultiPolygon target both accept the
    result; falls back to a MultiPolygon coercion when unavailable."""
    from .layer_conventions import to_multipolygon

    try:
        parts = geom.coerceToType(target_layer.wkbType())
    except Exception:  # nosec B110 - fall back to the MultiPolygon guard below
        parts = None
    if parts:
        return [g for g in parts if g is not None and not g.isEmpty()]
    coerced = to_multipolygon(geom)
    return [coerced] if coerced is not None and not coerced.isEmpty() else []


def append_run_to_layer(target_layer, memory_layer, *, source_name: str) -> AppendResult:
    """Append every feature of ``memory_layer`` into ``target_layer``.

    Reprojects to the target CRS when it differs, matches the target's schema
    BY FIELD NAME (label / area_m2, plus class / score / perimeter_m when the
    target has them; a hand-drawn shape has no score, so it stays NULL), repairs
    each geometry, and writes through the provider (or the open edit buffer when
    the layer is already being edited). Best-effort and fail-open: any error
    returns ``added=0`` with a message so the caller can create a new layer
    instead of losing the work, and the target's style / metadata are never
    touched. Main-thread only."""
    from .layer_conventions import attribute_values_for_fields, repair_polygon

    try:
        target_fields = target_layer.fields()
        target_crs = target_layer.crs()
        src_crs = memory_layer.crs()
        transform = None
        if src_crs.isValid() and target_crs.isValid() and src_crs != target_crs:
            transform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        features: list[QgsFeature] = []
        for feat in memory_layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            g = QgsGeometry(geom)
            if transform is not None:
                # transform() raises QgsCsException on an unrecoverable CRS
                # failure; caught below so a broken transform never crashes.
                g.transform(transform)
            g = repair_polygon(g) or g
            for coerced in _coerce_to_layer_type(g, target_layer):
                nf = QgsFeature(target_fields)
                nf.setGeometry(coerced)
                nf.setAttributes(attribute_values_for_fields(
                    target_fields, coerced, target_crs, source_name, timestamp))
                features.append(nf)

        if not features:
            return AppendResult(0, "no valid geometries to append")

        # Provider-level write when the layer is idle (persists straight to the
        # file, like the MCP path); the edit buffer when the user has it open,
        # so our features integrate with their session instead of writing behind
        # an unsaved edit buffer.
        if target_layer.isEditable():
            ok = target_layer.addFeatures(features)
        else:
            ok, _ = target_layer.dataProvider().addFeatures(features)
        if not ok:
            return AppendResult(0, "the layer provider rejected the features")

        target_layer.updateExtents()
        target_layer.triggerRepaint()
        return AppendResult(len(features), "")
    except Exception as exc:  # nosec B110 - fail open, caller falls back
        return AppendResult(0, str(exc))
