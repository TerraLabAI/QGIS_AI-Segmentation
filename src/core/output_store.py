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

# Custom property stamped on every committed layer with its creation epoch.
# Survives a project save/reload since custom properties serialize into the .qgz.
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
    #: The shared file the run was meant for. It is NOT always GPKG_FILENAME:
    #: a project past the table ceiling is writing into a rolled-over file, and
    #: naming the wrong one in the fallback warning sends the user to look at a
    #: file that is not the one that refused them. Defaulted so no existing
    #: caller has to change.
    intended_path: str = ""


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# A served palette is a handful of colours, not a catalogue.
_MAX_SERVED_COLORS = 128


def _served_color_map(path: str) -> dict[str, str]:
    """Served ``{keyword: "#rrggbb"}`` entries that are usable. Never raises."""
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
    except Exception:  # noqa: BLE001 -- colour is cosmetic, never break a save  # nosec B110
        pass
    return out


def semantic_colors() -> list[tuple[str, str]]:
    """The keyword-to-colour table in force.

    A served ``taxonomy.class_colors`` map ADDS keywords and may retune a
    shipped one. Naming a new object class is otherwise a three-file change
    plus a release, which is far more ceremony than picking a hue deserves.
    Colour is cosmetic: nothing here can gate, bill or hide anything, so a
    served entry replacing a shipped hue costs at most an ugly map until the
    next deploy.
    """
    served = _served_color_map("taxonomy.class_colors")
    if not served:
        return list(_SEMANTIC_COLORS)
    merged = dict(_SEMANTIC_COLORS)
    merged.update(served)
    return list(merged.items())


def fallback_palette() -> list[str]:
    """The hashed-pick palette in force: the shipped hues plus any served one.

    Additive, so the colour a given prompt already hashes to only moves when
    the palette grows, and never disappears.
    """
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
    except Exception:  # noqa: BLE001 -- colour is cosmetic  # nosec B110
        pass
    return out


def committed_color_for_prompt(prompt: str) -> QColor:
    """Stable color for a committed run, keyed to the prompt.

    Semantic cartographic colors for common object classes, a deterministic
    hashed palette pick otherwise, legacy red for promptless (Manual) runs.
    """
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


def friendly_layer_name(prompt: str, gpkg_path: str = "") -> str:
    """Tree name for a committed run: "Buildings (2026-07-03)".

    Only the first letter is capitalized (the rest stays as typed), the date
    is ISO 8601, and a same-prompt-same-day rerun becomes
    "Buildings 2 (2026-07-03)" by scanning the AI Segmentation group.

    ``gpkg_path`` names the file the run is about to be written into. The
    name doubles as the table's GeoPackage identifier, which the file keeps
    unique, so the scan also covers the identifiers already in that file:
    the tree forgets a layer the user removed, the file does not.

    ISO and not a locale-short date: "3 Jul" carries no year, so two runs a
    year apart read as the same day, and the layer panel sorts them by month
    name. The written form also matches the date in the table name.
    """
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
    """Tables already present in the GeoPackage, or None when unknown.

    A file on disk that reads back with no tables answers "unknown", not
    "empty". A GeoPackage held by another writer, or half written, lists
    nothing without raising, and reading that as an empty file hands
    ``snake_table_name`` a name that is already in use. The write then REPLACES
    an earlier run's table. "Unknown" costs one probe per candidate name and
    keeps the run.

    The empty set is for a file confirmed absent (or confirmed empty), never
    for one the OS refuses to stat: that one is unknown too.
    """
    from .output_gpkg_rollover import file_size, table_names

    if file_size(gpkg_path) == 0:
        return set()
    # gpkg_contents over read-only SQLite, which is one query. querySublayers
    # opens every table through OGR, and on a shared file holding a few
    # hundred runs that was seconds spent on the Export click.
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
        # A GeoPackage table name is case-insensitive, so a case-sensitive
        # match reads "Buildings_20260703" as free for "buildings_20260703".
        # The write then REPLACES that table and an earlier run is gone.
        folded = table.lower()
        return any(name.lower() == folded for name in tables)
    # querySublayers unavailable: probe the single candidate directly.
    try:
        probe = QgsVectorLayer(f"{gpkg_path}|layername={table}", "probe", "ogr")
        return probe.isValid()
    except Exception:
        return False


def snake_table_name(prompt: str, gpkg_path: str) -> str:
    """GeoPackage table name for a run: "buildings_20260703", deduped "_2"."""
    # \w, not [a-z0-9]: an ASCII-only class erases a non-Latin prompt entirely,
    # so every such run would collapse to the fallback name and the user could
    # not tell two runs apart in the file. The fallback stem below already uses
    # \w for the same reason.
    base = re.sub(r"[^\w]+", "_", (prompt or "").strip().lower()).strip("_")
    base = _ascii_table_stem(base)[:40].strip("_") or "segmentation"
    # A name that opens on a digit is not a plain SQL identifier, so every
    # reader that does not quote it (a spreadsheet import, a CAD bridge, an
    # ogr2ogr one-liner) fails on the file rather than on the query.
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
    """ASCII form of a table-name stem, accents folded, the rest hex-escaped.

    A GeoPackage stores UTF-8 happily, but the tools the file travels to do
    not all agree: a table name outside ASCII comes back mangled or unopenable
    in enough of them that the name is worth keeping plain. An accented Latin
    prompt folds to its base letters, and anything that will not fold becomes
    "u" plus its code point, which stays unique so two prompts never collide.
    """
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
    """Whether a file can be created in this directory.

    The probe is a real temp file, so the OS removes it on close (unlinked at
    once on POSIX, delete-on-close on Windows) and nothing we name can be left
    lying next to the user's project. Cleanup is also out of the answer: an
    on-write scanner or a sync client can hold a freshly written file for a
    moment, and reading that as "not writable" would send the GeoPackage to
    the home folder instead of the project folder.
    """
    try:
        with tempfile.TemporaryFile(dir=directory, suffix=".tmp") as probe:
            probe.write(b"ok")
    except OSError:
        return False
    return True


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


def output_directory_candidates(source_layer, first_only: bool = False) -> list[str]:
    """Every writable output directory, best first: project, raster, home.

    A list and not a single answer, because "a file can be created here" is not
    the same question as "a GeoPackage can be written here". A folder on a
    network share, a WSL mount or a synced drive accepts the probe file and
    then refuses the SQLite lock the writer needs, which only shows up as
    "database is locked" once the write runs. The fallback in write_run_table
    walks this list, so a run that cannot be written next to the project still
    lands in the home folder instead of failing twice in the same place.

    ``first_only`` stops at the first writable one. Every probe is a real file
    created and removed, and on a synced or network project folder that is not
    free; the caller that just wants "where does this run go" (see
    _output_directory) asks two questions it never reads the answer to.
    """
    project = QgsProject.instance()
    home = str(Path.home())
    ordered = [
        project.homePath() or project.absolutePath(),
        _source_layer_dir(source_layer),
        home,
    ]
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
    """Writable output directory: project folder, raster folder, then home."""
    return output_directory_candidates(source_layer, first_only=True)[0]


def project_gpkg_path(source_layer) -> str:
    """Full path of the per-project GeoPackage all runs accumulate into.

    Rolls over to a next numbered file once this one holds too many tables, so
    the click never pays for years of runs: see ``output_gpkg_rollover``.
    """
    from .output_gpkg_rollover import next_output_gpkg

    # Normalised: the directory comes from Qt with forward slashes and the
    # join adds the platform's, and that mixed spelling is what the log and
    # the layer source would otherwise show a Windows user.
    return os.path.normpath(
        next_output_gpkg(_output_directory(source_layer), GPKG_FILENAME))


def _ground_metre_transform(memory_layer):
    """Transform onto the CRS the saved layer should be written in, or None.

    None when the layer's own CRS already measures in ground metres, which is
    the common case for a user working off their own projected raster. See
    layer_conventions.pick_output_crs for the choice.
    """
    from qgis.core import QgsCoordinateTransform

    from .layer_conventions import pick_output_crs

    try:
        source = memory_layer.crs()
        target = pick_output_crs(source, memory_layer.extent())
        if target is None or not target.isValid() or target == source:
            return None
        return QgsCoordinateTransform(source, target, QgsProject.instance())
    except (RuntimeError, AttributeError, TypeError):
        return None


def _write_gpkg(memory_layer, path: str, table: str, overwrite_file: bool,
                transform=None, identifier: str = "",
                description: str = "") -> str:
    """Run one V3 write. Returns '' on success, the error message otherwise."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = table
    # gpkg_contents carries a human title and a one-line description beside
    # the table name. Without them every reader that is not this QGIS session
    # lists the run as "buildings_20260703" and nothing else.
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
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        memory_layer,
        path,
        QgsProject.instance().transformContext(),
        options,
    )
    if result[0] == QgsVectorFileWriter.WriterError.NoError:
        return ""
    return str(result[1]) if len(result) > 1 and result[1] else "unknown writer error"


# What one GDAL layer-creation option value may carry. The value travels in a
# "NAME=value" string, so a newline in it would end the option.
_MAX_GPKG_OPTION_CHARS = 250


def _gpkg_option_value(text: str) -> str:
    """One line of plain text, short enough for a layer-creation option."""
    flat = " ".join(str(text or "").split())
    return flat[:_MAX_GPKG_OPTION_CHARS]


def _load_table(path: str, table: str, display_name: str) -> QgsVectorLayer | None:
    """Open a table that was written a moment ago, and make it read its rows.

    The reload is not defensive, it is the difference between a layer that
    draws and a blank one. The writer still holds the file, so a read handle
    opened right behind it sees the table in the header and none of the rows
    that are still only in the journal: featureCount() answers the full count,
    getFeatures() yields nothing, and the user gets an empty map after a run
    they paid for. reloadData drops that half-built handle and opens a fresh
    one.

    It used to work by accident. Storing the style and the metadata into the
    file ran on the same click, right after this, and those writes flushed the
    journal as a side effect. The moment they moved off the click the accident
    stopped happening, so the flush is asked for here, where it belongs.
    """
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
    """Persist one run into the project GeoPackage as a new table.

    On any writer or reload failure the run falls back to a standalone
    per-run file (today's behavior) so a locked/corrupted shared gpkg never
    loses a paid detection. Returns None only when even the fallback fails.
    """
    from .output_gpkg_rollover import file_size

    gpkg_path = project_gpkg_path(source_layer)
    note_unexpected_output_folder(os.path.dirname(gpkg_path), source_layer)
    table = snake_table_name(prompt, gpkg_path)
    friendly = friendly_layer_name(prompt, gpkg_path)
    transform = _ground_metre_transform(memory_layer)

    # Recreate the file only when it is confirmed missing. A file that cannot
    # be stat'ed is unknown, and creating over an unknown shared GeoPackage
    # drops every run already in it. Appending into a file that turns out to be
    # absent costs nothing: the writer creates it.
    # English, like the rest of the file's metadata: it is read by whoever the
    # deliverable is handed to, not by the person who ran the export.
    description = "Objects detected with AI Segmentation."
    error_message = _write_gpkg(
        memory_layer, gpkg_path, table,
        overwrite_file=file_size(gpkg_path) == 0, transform=transform,
        identifier=friendly, description=description,
    )
    if not error_message:
        layer = _load_table(gpkg_path, table, friendly)
        if layer is not None:
            return WriteResult(gpkg_path, table, layer, False, "", gpkg_path)
        error_message = "saved table could not be reloaded"
    QgsMessageLog.logMessage(
        f"Shared GeoPackage write failed ({error_message}), falling back to a per-run file",
        _LOG_TAG, level=Qgis.MessageLevel.Warning,
    )

    stem = re.sub(r"[^\w\- ]", "", fallback_stem or "").strip().replace(" ", "_")
    stem = stem[:40] or "detection"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Walk the directories rather than only renaming the file. When the shared
    # write failed on the folder itself (a locked SQLite on a share or a WSL
    # mount), a new name in that same folder fails for the same reason, and the
    # run is lost with a writable home folder one step away.
    fallback_error = "no writable output directory"
    # One reason per folder, kept together for the final line. Each folder
    # refuses for its own reason (a lock on a share, a full disk at home), and
    # the last one alone sends the reader to the wrong cause.
    failures: list[tuple[str, str]] = []
    for directory in output_directory_candidates(source_layer):
        fallback_path = os.path.join(directory, f"{stem}_{timestamp}.gpkg")
        fallback_error = _write_gpkg(
            memory_layer, fallback_path, table, overwrite_file=True,
            transform=transform, identifier=friendly, description=description)
        if not fallback_error:
            layer = _load_table(fallback_path, table, friendly)
            if layer is not None:
                return WriteResult(fallback_path, table, layer, True, error_message, gpkg_path)
            fallback_error = "saved file could not be reloaded"
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


# Armed once the session has told the user where a run landed, so the notice
# is shown at most once however many runs follow.
_output_folder_notice_shown = False


def note_unexpected_output_folder(directory: str, source_layer) -> None:
    """Name the folder a run was saved in when it is not one the user picked.

    A project that was never saved, over a web basemap, has neither a project
    folder nor a raster folder, so the GeoPackage lands in the home folder.
    Nothing on screen said so, and the user went looking for their file next
    to a project that does not exist yet. Said once per session.
    """
    global _output_folder_notice_shown
    if _output_folder_notice_shown or not directory:
        return
    try:
        project = QgsProject.instance()
        expected = [project.homePath() or project.absolutePath(),
                    _source_layer_dir(source_layer)]
        key = os.path.normcase(os.path.abspath(directory))
        for candidate in expected:
            if candidate and os.path.normcase(os.path.abspath(candidate)) == key:
                return
        _output_folder_notice_shown = True
        from qgis.utils import iface

        if iface is None:
            return
        iface.messageBar().pushInfo(
            GROUP_NAME,
            tr("Saved to {folder}. Save the project to keep your results "
               "beside it.").format(folder=directory))
    except Exception:  # noqa: BLE001 -- a notice never blocks a save  # nosec B110
        pass


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
    if QgsProject.instance().addMapLayer(layer, False) is None:
        # The registry refused the layer (invalid source). Adding a tree node
        # for it would leave a row pointing at nothing in the Layers panel.
        QgsMessageLog.logMessage(
            "Output store: the saved layer could not be registered",
            _LOG_TAG, level=Qgis.MessageLevel.Warning)
        return
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
    # The run is in the tree; now make sure it is on screen. A group under an
    # opaque basemap paints under it, so the user saves a run and the canvas
    # looks unchanged. This DESTROYS the node it moves, so nothing may touch
    # `top` or `group` after this line: the layer object survives, the nodes do
    # not.
    keep_group_above_imagery(top)
    # A committed run holds thousands of dense outlines, and the layer has just
    # landed on the canvas. Do this LAST, so a layer the registry refused never
    # gets it, and so it is on before the first frame is drawn.
    apply_fast_canvas_render(layer)


#: Vertex-drop tolerance for render-time simplification, in SCREEN PIXELS. It
#: bounds how far a drawn outline may sit from the true one at any zoom, and it
#: never reaches stored geometry. Above 1.0 QGIS also takes its
#: antialiasing shortcut for the parts that come out sub-pixel.
RENDER_SIMPLIFY_PX = 1.5


def render_simplify_px() -> float:
    """The render-time simplify tolerance in screen pixels, server-tunable.

    Bounded on both sides: at or below 1.0 QGIS drops the antialiasing shortcut
    along with it, and a large value buys nothing back while drawing outlines
    the user can see are wrong. Cache-only and never raises, so it is safe on
    the layer-add path and offline.
    """
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "ui.render_simplify_px", RENDER_SIMPLIFY_PX, 1.0, 5.0))
    except Exception:  # noqa: BLE001 -- the tolerance is best-effort  # nosec B110
        return RENDER_SIMPLIFY_PX


def apply_fast_canvas_render(layer) -> None:
    """Make a dense result layer cheap to pan and zoom. DISPLAY ONLY: not one
    stored coordinate changes, so an exported file is byte for byte what it
    would have been without this call.

    Two levers, and both have to be set together or QGIS silently ignores the
    simplification:

    - Render-time geometry simplification. QGIS drops sub-pixel vertices as it
      draws, so a zoomed-out canvas paints a fraction of the points a segmented
      outline carries. Full detail is back as soon as you zoom in.
      ``setForceLocalOptimization(True)`` is load-bearing, not a nicety: left
      False, a provider that cannot simplify server-side (memory, OGR, which is
      every layer this plugin makes) simplifies nowhere at all.
    - A provider spatial index, so a pan or a zoom fetches the features in view
      instead of scanning the whole set.

    Two things this deliberately does NOT do. It does not raise the tolerance
    with the object count: past about one screen pixel the extra tolerance stops
    buying frame time and starts showing, because what is left to draw is the
    outline itself. And it cannot help while the layer sits in a QGIS edit
    session: QGIS skips simplification whenever a layer has an edit buffer, so
    the Correct step pays full detail for whatever is in view.

    Best-effort and version-defensive; never raises into the caller.
    """
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
    except Exception:  # noqa: BLE001 - display nicety, never break a run on it  # nosec B110
        pass
    try:
        provider = layer.dataProvider()
        if _provider_lacks_spatial_index(provider):
            provider.createSpatialIndex()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _provider_lacks_spatial_index(provider) -> bool:
    """Whether this provider still needs an index built.

    A GeoPackage comes out of the OGR writer with its rtree already there, so
    asking for one again rebuilds an index the file has: a write into the file
    the user is waiting on, for nothing.

    True on any doubt, which is the behaviour of code that never asked: an
    "unknown" answer, a build with no such enum, a provider that cannot answer.
    """
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
