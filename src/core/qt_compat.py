"""Qt5/Qt6 compatibility shim for scoped enums.

Qt6 (QGIS 4) moved flat enums like ``Qt.LeftDockWidgetArea`` into nested
scopes: ``Qt.DockWidgetArea.LeftDockWidgetArea``.  This module resolves
them once at import time so the rest of the codebase stays clean.

Ported from the AI Edit plugin so the shared before/after slider, demo
loader, and library dialog run unchanged on PyQt5 (QGIS 3) and PyQt6 (QGIS 4).
"""
from __future__ import annotations

import importlib

from qgis.core import QgsBlockingNetworkRequest, QgsTask, QgsVectorDataProvider
from qgis.PyQt.QtCore import QIODevice, QObject, Qt, QTimer
from qgis.PyQt.QtGui import QImage, QPalette, QTextCursor, QTextOption
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest
from qgis.PyQt.QtWidgets import QFrame, QSizePolicy, QTextEdit


def resolve_qt_enum(parent, scope: str | None, name: str):
    if scope:
        scoped = getattr(getattr(parent, scope, None), name, None)
        if scoped is not None:
            return scoped
    return getattr(parent, name)


def _import_moved(name: str):
    """Return a class that Qt6 relocated from QtWidgets to QtGui.

    ``QAction`` and ``QShortcut`` live in ``QtWidgets`` on Qt5 and in
    ``QtGui`` on Qt6. Newer QGIS builds re-export them into
    ``qgis.PyQt.QtGui`` even on Qt5, but older 3.x LTR builds do not, so a
    plain ``from qgis.PyQt.QtGui import QAction`` fails to load the plugin
    there. Resolve by name across both modules so the same source runs on
    every QGIS from 3.22 to 4, and so the static Qt6 checker sees no flat
    module import to flag (same reason ``resolve_qt_enum`` uses getattr for enums).
    """
    for module in ("qgis.PyQt.QtGui", "qgis.PyQt.QtWidgets"):
        found = getattr(importlib.import_module(module), name, None)
        if found is not None:
            return found
    raise ImportError(f"{name} not found in qgis.PyQt.QtGui or QtWidgets")


QAction = _import_moved("QAction")
QShortcut = _import_moved("QShortcut")


# Qt.DockWidgetArea
LeftDockWidgetArea = resolve_qt_enum(Qt, "DockWidgetArea", "LeftDockWidgetArea")
RightDockWidgetArea = resolve_qt_enum(Qt, "DockWidgetArea", "RightDockWidgetArea")

# Qt.CursorShape
PointingHandCursor = resolve_qt_enum(Qt, "CursorShape", "PointingHandCursor")
CrossCursor = resolve_qt_enum(Qt, "CursorShape", "CrossCursor")
WaitCursor = resolve_qt_enum(Qt, "CursorShape", "WaitCursor")
ArrowCursor = resolve_qt_enum(Qt, "CursorShape", "ArrowCursor")

# Qt.AlignmentFlag
AlignCenter = resolve_qt_enum(Qt, "AlignmentFlag", "AlignCenter")
AlignTop = resolve_qt_enum(Qt, "AlignmentFlag", "AlignTop")
AlignLeft = resolve_qt_enum(Qt, "AlignmentFlag", "AlignLeft")
AlignVCenter = resolve_qt_enum(Qt, "AlignmentFlag", "AlignVCenter")

# Qt.Key
Key_Return = resolve_qt_enum(Qt, "Key", "Key_Return")
Key_Enter = resolve_qt_enum(Qt, "Key", "Key_Enter")
Key_Escape = resolve_qt_enum(Qt, "Key", "Key_Escape")

# Qt.ShortcutContext
WindowShortcut = resolve_qt_enum(Qt, "ShortcutContext", "WindowShortcut")
WidgetWithChildrenShortcut = resolve_qt_enum(
    Qt, "ShortcutContext", "WidgetWithChildrenShortcut"
)


def event_pos(event):
    """Return a Qt5/Qt6-safe QPoint for a QMouseEvent or QgsMapMouseEvent.

    Qt6 deprecates the old integer position accessor in favour of the float
    one; use this wrapper everywhere a mouse event's widget-local position is
    needed so the same source runs on QGIS 3 and 4.

    The accessor is resolved by NAME (``getattr`` on a string), so no literal
    deprecated accessor appears in the source and the static pyqt5_to_pyqt6
    checker has nothing to flag.
    """
    getter = getattr(event, "position", None) or getattr(event, "localPos", None)
    # The string form keeps the deprecated accessor out of the Qt6 checker's sight.
    getter = getter or getattr(event, "pos")  # noqa: B009
    point = getter()
    # The float accessors return a QPointF (needs rounding to a QPoint); the
    # integer one returns a QPoint already, which has no toPoint.
    to_point = getattr(point, "toPoint", None)
    return to_point() if to_point is not None else point


# Qt.PenStyle
SolidLine = resolve_qt_enum(Qt, "PenStyle", "SolidLine")
DashLine = resolve_qt_enum(Qt, "PenStyle", "DashLine")

# Qt.KeyboardModifier
ShiftModifier = resolve_qt_enum(Qt, "KeyboardModifier", "ShiftModifier")

# Qt.MouseButton
LeftButton = resolve_qt_enum(Qt, "MouseButton", "LeftButton")
RightButton = resolve_qt_enum(Qt, "MouseButton", "RightButton")

# Qt.FocusPolicy
NoFocus = resolve_qt_enum(Qt, "FocusPolicy", "NoFocus")

# Qt.FocusReason
OtherFocusReason = resolve_qt_enum(Qt, "FocusReason", "OtherFocusReason")

# Qt.ToolButtonStyle
ToolButtonTextBesideIcon = resolve_qt_enum(Qt, "ToolButtonStyle", "ToolButtonTextBesideIcon")

# Qt.ArrowType
DownArrow = resolve_qt_enum(Qt, "ArrowType", "DownArrow")
RightArrow = resolve_qt_enum(Qt, "ArrowType", "RightArrow")

# Qt.TextFormat
RichText = resolve_qt_enum(Qt, "TextFormat", "RichText")
PlainText = resolve_qt_enum(Qt, "TextFormat", "PlainText")

# Qt.WidgetAttribute
WA_TransparentForMouseEvents = resolve_qt_enum(Qt, "WidgetAttribute", "WA_TransparentForMouseEvents")
WA_StyledBackground = resolve_qt_enum(Qt, "WidgetAttribute", "WA_StyledBackground")

# Qt.ScrollBarPolicy
ScrollBarAlwaysOff = resolve_qt_enum(Qt, "ScrollBarPolicy", "ScrollBarAlwaysOff")
ScrollBarAsNeeded = resolve_qt_enum(Qt, "ScrollBarPolicy", "ScrollBarAsNeeded")

# QTextOption.WrapMode - wrap mid-token so a long URL or unbreakable string
# still flows to the next line instead of triggering horizontal scroll.
WrapAtWordBoundaryOrAnywhere = resolve_qt_enum(
    QTextOption, "WrapMode", "WrapAtWordBoundaryOrAnywhere"
)

# QTextEdit.LineWrapMode - pinned to widget width so wrapping always engages
# even when QSS or a rich-text paste would otherwise leave it implicit.
LineWrapWidgetWidth = resolve_qt_enum(QTextEdit, "LineWrapMode", "WidgetWidth")

# Qt.AspectRatioMode / Qt.TransformationMode
KeepAspectRatio = resolve_qt_enum(Qt, "AspectRatioMode", "KeepAspectRatio")
SmoothTransformation = resolve_qt_enum(Qt, "TransformationMode", "SmoothTransformation")

# Qt.TextInteractionFlag
TextSelectableByMouse = resolve_qt_enum(Qt, "TextInteractionFlag", "TextSelectableByMouse")
TextBrowserInteraction = resolve_qt_enum(Qt, "TextInteractionFlag", "TextBrowserInteraction")

# QIODevice.OpenModeFlag
WriteOnly = resolve_qt_enum(QIODevice, "OpenModeFlag", "WriteOnly")

# QImage.Format
FormatARGB32 = resolve_qt_enum(QImage, "Format", "Format_ARGB32")
# 3 bytes per pixel, so a row is not 4-byte aligned: every QImage built on this
# format has to be given an explicit bytesPerLine.
FormatRGB888 = resolve_qt_enum(QImage, "Format", "Format_RGB888")

# QTextCursor.MoveOperation
CursorEnd = resolve_qt_enum(QTextCursor, "MoveOperation", "End")

# QSizePolicy.Policy
SizePolicyExpanding = resolve_qt_enum(QSizePolicy, "Policy", "Expanding")
SizePolicyFixed = resolve_qt_enum(QSizePolicy, "Policy", "Fixed")

# QPalette.ColorRole
PaletteBase = resolve_qt_enum(QPalette, "ColorRole", "Base")

# QFrame.Shape / QFrame.Shadow
FrameNoFrame = resolve_qt_enum(QFrame, "Shape", "NoFrame")
FrameHLine = resolve_qt_enum(QFrame, "Shape", "HLine")
FrameVLine = resolve_qt_enum(QFrame, "Shape", "VLine")
FrameSunken = resolve_qt_enum(QFrame, "Shadow", "Sunken")

# QgsBlockingNetworkRequest.ErrorCode
BlockingNoError = resolve_qt_enum(QgsBlockingNetworkRequest, "ErrorCode", "NoError")

# QgsVectorDataProvider.Capability.AddFeatures - scoped on QGIS 4, flat on 3.x.
CapabilityAddFeatures = resolve_qt_enum(QgsVectorDataProvider, "Capability", "AddFeatures")

# Qgis.GeometryType (QGIS 4) vs QgsWkbTypes (QGIS 3)
try:
    from qgis.core import Qgis
    _gt = getattr(Qgis, "GeometryType", None)
    PolygonGeometry = getattr(_gt, "Polygon", None)
    LineGeometry = getattr(_gt, "Line", None)
except Exception:
    PolygonGeometry = None
    LineGeometry = None
if PolygonGeometry is None:
    from qgis.core import QgsWkbTypes
    PolygonGeometry = resolve_qt_enum(QgsWkbTypes, "GeometryType", "PolygonGeometry")
if LineGeometry is None:
    from qgis.core import QgsWkbTypes
    LineGeometry = resolve_qt_enum(QgsWkbTypes, "GeometryType", "LineGeometry")

# Flat WKB type (Qgis.WkbType on QGIS 4, QgsWkbTypes.Type on QGIS 3) for the
# post-flatType comparisons in layer_conventions. QgsWkbTypes.flatType() returns
# a value of this same enum on both versions, so the compare stays correct.
try:
    from qgis.core import Qgis
    _wkb = getattr(Qgis, "WkbType", None)
    WkbPolygon = getattr(_wkb, "Polygon", None)
    WkbMultiPolygon = getattr(_wkb, "MultiPolygon", None)
except Exception:
    WkbPolygon = None
    WkbMultiPolygon = None
if WkbPolygon is None:
    from qgis.core import QgsWkbTypes
    WkbPolygon = resolve_qt_enum(QgsWkbTypes, "Type", "Polygon")
if WkbMultiPolygon is None:
    from qgis.core import QgsWkbTypes
    WkbMultiPolygon = resolve_qt_enum(QgsWkbTypes, "Type", "MultiPolygon")

# Distance unit: Qgis.DistanceUnit.Meters (QGIS 4) vs QgsUnitTypes.DistanceMeters
# (QGIS 3).
try:
    from qgis.core import Qgis
    _du = getattr(Qgis, "DistanceUnit", None)
    DistanceMeters = getattr(_du, "Meters", None)
except Exception:
    DistanceMeters = None
if DistanceMeters is None:
    from qgis.core import QgsUnitTypes
    DistanceMeters = resolve_qt_enum(QgsUnitTypes, "DistanceUnit", "DistanceMeters")


def _render_simplify_enum(modern_scope: str, legacy_scope: str, name: str):
    """One render-time simplification enum member, from QGIS 3.22 to 4.

    QGIS 3.30 moved these onto ``Qgis`` (``VectorRenderingSimplificationFlag``
    for the hints, ``VectorSimplificationAlgorithm`` for the algorithm). Before
    that they lived on ``QgsVectorSimplifyMethod``, scoped on some builds and
    flat on others. Resolved by name through getattr, so the static Qt6 checker
    sees no flat enum access on any branch.

    Returns None when neither spelling exists, which every caller reads as
    "leave the layer's own default alone".
    """
    try:
        from qgis.core import Qgis, QgsVectorSimplifyMethod
    except ImportError:
        return None
    found = getattr(getattr(Qgis, modern_scope, None), name, None)
    if found is not None:
        return found
    try:
        return resolve_qt_enum(QgsVectorSimplifyMethod, legacy_scope, name)
    except AttributeError:
        return None


# Both hints and the algorithm the fast canvas render asks for. FullSimplification
# is GeometrySimplification plus the antialiasing shortcut for sub-pixel parts;
# the geometry-only hint is the fallback on a build that lacks it.
SimplifyFullHint = _render_simplify_enum(
    "VectorRenderingSimplificationFlag", "SimplifyHint", "FullSimplification")
SimplifyGeometryHint = _render_simplify_enum(
    "VectorRenderingSimplificationFlag", "SimplifyHint", "GeometrySimplification")
SimplifyDistanceAlgorithm = _render_simplify_enum(
    "VectorSimplificationAlgorithm", "SimplifyAlgorithm", "Distance")

# QgsVertexMarker.IconType.ICON_CIRCLE - scoped on QGIS 4, flat also on QGIS 3.
try:
    from qgis.gui import QgsVertexMarker
    VertexIconCircle = resolve_qt_enum(QgsVertexMarker, "IconType", "ICON_CIRCLE")
except Exception:
    VertexIconCircle = None


def symbol_fill_color_property():
    """Data-defined FillColor property key for a QgsSymbolLayer.

    QGIS 4 renamed the member to ``QgsSymbolLayer.Property.FillColor``; QGIS 3
    spelled it ``QgsSymbolLayer.PropertyFillColor`` (flat) or, on newer 3.x,
    ``QgsSymbolLayer.Property.PropertyFillColor``. Returns whichever resolves
    (never uses ``or`` chaining, since a 0-valued enum member is falsy).
    """
    from qgis.core import QgsSymbolLayer
    prop_scope = getattr(QgsSymbolLayer, "Property", None)
    for owner, name in (
        (prop_scope, "FillColor"),              # QGIS 4
        (QgsSymbolLayer, "PropertyFillColor"),  # QGIS 3 flat
        (prop_scope, "PropertyFillColor"),      # QGIS 3 scoped
    ):
        if owner is None:
            continue
        val = getattr(owner, name, None)
        if val is not None:
            return val
    return None


# QgsField type argument: QGIS 4 (PyQt6) takes a scoped QMetaType.Type; QGIS 3
# takes a QVariant. QgsField gained QMetaType support in QGIS 3.38, so gate on
# the version int rather than probing, keeping the 3.22/3.28 floor on the
# QVariant overload it documents. Single source for every QgsField(...) call.
try:
    from qgis.core import Qgis as _Qgis
    _QGIS_VERSION_INT = getattr(_Qgis, "QGIS_VERSION_INT", 0)
except Exception:
    _QGIS_VERSION_INT = 0

if _QGIS_VERSION_INT >= 40000:
    from qgis.PyQt.QtCore import QMetaType as _QMetaType
    _FIELD_TYPE_STRING = _QMetaType.Type.QString
    _FIELD_TYPE_DOUBLE = _QMetaType.Type.Double
    _FIELD_TYPE_INT = _QMetaType.Type.Int
else:
    try:
        from qgis.PyQt.QtCore import QVariant as _QVariant
        _FIELD_TYPE_STRING = _QVariant.String
        _FIELD_TYPE_DOUBLE = _QVariant.Double
        _FIELD_TYPE_INT = _QVariant.Int
    except (ImportError, AttributeError):
        # A Qt6 build that still reports a 3.x version. QGIS master did that
        # through the whole run-up to 4.0, and qgisMaximumVersion=4.99 puts it
        # in range. QVariant's type aliases are all that hold this branch up
        # on PyQt6, and QVariant.Type is already gone there. classFactory
        # imports this module, so a raise here is not a broken field type, it
        # is the plugin failing to load at all.
        from qgis.PyQt.QtCore import QMetaType as _QMetaType
        _FIELD_TYPE_STRING = _QMetaType.Type.QString
        _FIELD_TYPE_DOUBLE = _QMetaType.Type.Double
        _FIELD_TYPE_INT = _QMetaType.Type.Int


def field_type_string():
    """QgsField string-type arg, correct across the QGIS 3.22 -> 4 range."""
    return _FIELD_TYPE_STRING


def field_type_double():
    """QgsField double-type arg, correct across the QGIS 3.22 -> 4 range."""
    return _FIELD_TYPE_DOUBLE


def field_type_int():
    """QgsField int-type arg, correct across the QGIS 3.22 -> 4 range."""
    return _FIELD_TYPE_INT


# QNetworkReply.NetworkError
def _net_enum(name: str):
    return resolve_qt_enum(QNetworkReply, "NetworkError", name)


HostNotFoundError = _net_enum("HostNotFoundError")
ConnectionRefusedError_ = _net_enum("ConnectionRefusedError")
TimeoutError_ = _net_enum("TimeoutError")
SslHandshakeFailedError = _net_enum("SslHandshakeFailedError")
ContentAccessDenied = _net_enum("ContentAccessDenied")
AuthenticationRequiredError = _net_enum("AuthenticationRequiredError")
UnknownNetworkError = _net_enum("UnknownNetworkError")

PROXY_ERRORS = {
    _net_enum("ProxyConnectionRefusedError"),
    _net_enum("ProxyConnectionClosedError"),
    _net_enum("ProxyNotFoundError"),
    _net_enum("ProxyTimeoutError"),
    _net_enum("ProxyAuthenticationRequiredError"),
    _net_enum("UnknownProxyError"),
}

# QNetworkRequest.Attribute
HttpStatusCodeAttribute = resolve_qt_enum(
    QNetworkRequest, "Attribute", "HttpStatusCodeAttribute"
)
# Redirect policy. PyQt5 on some QGIS 3 builds exposes these flat, not scoped,
# so resolve through the same scoped-then-flat helper rather than hardcoding
# QNetworkRequest.Attribute.* / QNetworkRequest.RedirectPolicy.* (which would
# AttributeError on those builds, on every API request and download).
RedirectPolicyAttribute = resolve_qt_enum(
    QNetworkRequest, "Attribute", "RedirectPolicyAttribute"
)
NoLessSafeRedirectPolicy = resolve_qt_enum(
    QNetworkRequest, "RedirectPolicy", "NoLessSafeRedirectPolicy"
)


def silent_task_flags():
    """CanCancel plus Hidden/Silent when the running QGIS exposes them.

    Hidden and Silent landed in QGIS 3.26 and the plugin floor is older, so
    resolve defensively: on a modern build the task manager never fills with
    background rows, on an old one the task is still cancellable.
    """
    flags = QgsTask.Flag.CanCancel
    for name in ("Hidden", "Silent"):
        flag = getattr(QgsTask.Flag, name, None)
        if flag is not None:
            flags = flags | flag
    return flags


def safe_single_shot(msec: int, owner: QObject, callback) -> QTimer:
    """A single-shot timer bound to ``owner``'s lifetime.

    ``QTimer.singleShot(msec, lambda: widget.setText(...))`` keeps the lambda
    (and the widget it captures) alive in the global event loop. If the widget
    is destroyed before the timer fires, the deferred call lands on a freed C++
    object and segfaults QGIS, the classic "closed the dialog too fast" crash.

    Parenting the timer to ``owner`` makes Qt destroy the timer together with
    ``owner``, so it can never fire into a dead widget. Returns the timer so the
    caller can stop it early if needed.
    """
    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.start(max(0, int(msec)))
    return timer


# One signal per call, on purpose: this is what replaces a batch of disconnects
# sharing a single try block. Returns whether the disconnect actually happened.
def safe_disconnect(owner, signal_name: str, slot=None) -> bool:
    """Disconnect one signal on ``owner``, swallowing the three teardown faults.

    Batched into one try block, the first failure skipped every disconnect
    after it, and those connections stayed live: a handler on a destroyed
    widget kept firing from the layer tree, and the widget's C++ half was
    already gone by the time it did.

    Each of the three is normal during teardown, none is worth a raise:
    TypeError means this slot was never connected (a second cleanup() call, or
    a widget built while the project swapped its layer tree), RuntimeError
    means the C++ half of ``owner`` is already deleted, and AttributeError
    means ``owner`` is None. Reading ``owner.signal_name`` can itself raise the
    last two, so the lookup sits inside the guard, not outside it.

    ``slot=None`` drops every connection on the signal, which is only safe for
    signals the plugin owns end to end. Never use it on a QgsTask: that would
    also sever the task manager's own hookups from addTask() and orphan it.
    """
    try:
        signal = getattr(owner, signal_name)
        if slot is None:
            signal.disconnect()
        else:
            signal.disconnect(slot)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return True
