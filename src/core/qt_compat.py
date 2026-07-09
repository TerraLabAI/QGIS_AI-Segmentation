"""Qt5/Qt6 compatibility shim for scoped enums.

Qt6 (QGIS 4) moved flat enums like ``Qt.LeftDockWidgetArea`` into nested
scopes: ``Qt.DockWidgetArea.LeftDockWidgetArea``.  This module resolves
them once at import time so the rest of the codebase stays clean.

Ported from the AI Edit plugin so the shared before/after slider, demo
loader, and library dialog run unchanged on PyQt5 (QGIS 3) and PyQt6 (QGIS 4).
"""
from __future__ import annotations

from qgis.core import QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QIODevice, QObject, Qt, QTimer
from qgis.PyQt.QtGui import QImage, QPalette, QTextCursor, QTextOption
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest
from qgis.PyQt.QtWidgets import QFrame, QSizePolicy, QTextEdit


def _resolve(parent, scope: str | None, name: str):
    if scope:
        scoped = getattr(getattr(parent, scope, None), name, None)
        if scoped is not None:
            return scoped
    return getattr(parent, name)


# Qt.DockWidgetArea
LeftDockWidgetArea = _resolve(Qt, "DockWidgetArea", "LeftDockWidgetArea")
RightDockWidgetArea = _resolve(Qt, "DockWidgetArea", "RightDockWidgetArea")

# Qt.CursorShape
PointingHandCursor = _resolve(Qt, "CursorShape", "PointingHandCursor")
CrossCursor = _resolve(Qt, "CursorShape", "CrossCursor")
WaitCursor = _resolve(Qt, "CursorShape", "WaitCursor")
ArrowCursor = _resolve(Qt, "CursorShape", "ArrowCursor")

# Qt.AlignmentFlag
AlignCenter = _resolve(Qt, "AlignmentFlag", "AlignCenter")
AlignTop = _resolve(Qt, "AlignmentFlag", "AlignTop")
AlignLeft = _resolve(Qt, "AlignmentFlag", "AlignLeft")
AlignVCenter = _resolve(Qt, "AlignmentFlag", "AlignVCenter")

# Qt.Key
Key_Return = _resolve(Qt, "Key", "Key_Return")
Key_Enter = _resolve(Qt, "Key", "Key_Enter")
Key_Escape = _resolve(Qt, "Key", "Key_Escape")

# Qt.ShortcutContext
WindowShortcut = _resolve(Qt, "ShortcutContext", "WindowShortcut")
WidgetWithChildrenShortcut = _resolve(
    Qt, "ShortcutContext", "WidgetWithChildrenShortcut"
)


def event_pos(event):
    """Return a Qt5/Qt6-safe QPoint for a QMouseEvent or QgsMapMouseEvent.

    Qt6 deprecates ``QMouseEvent.pos()`` in favour of
    ``position().toPoint()``; use this wrapper everywhere a mouse event's
    widget-local position is needed so the same source runs on QGIS 3 and 4.
    """
    if hasattr(event, "position"):
        try:
            return event.position().toPoint()
        except (AttributeError, TypeError):
            pass
    return event.pos()


# Qt.KeyboardModifier
ShiftModifier = _resolve(Qt, "KeyboardModifier", "ShiftModifier")

# Qt.MouseButton
LeftButton = _resolve(Qt, "MouseButton", "LeftButton")
RightButton = _resolve(Qt, "MouseButton", "RightButton")

# Qt.FocusPolicy
NoFocus = _resolve(Qt, "FocusPolicy", "NoFocus")

# Qt.FocusReason
OtherFocusReason = _resolve(Qt, "FocusReason", "OtherFocusReason")

# Qt.ToolButtonStyle
ToolButtonTextBesideIcon = _resolve(Qt, "ToolButtonStyle", "ToolButtonTextBesideIcon")

# Qt.ArrowType
DownArrow = _resolve(Qt, "ArrowType", "DownArrow")
RightArrow = _resolve(Qt, "ArrowType", "RightArrow")

# Qt.TextFormat
RichText = _resolve(Qt, "TextFormat", "RichText")
PlainText = _resolve(Qt, "TextFormat", "PlainText")

# Qt.WidgetAttribute
WA_TransparentForMouseEvents = _resolve(Qt, "WidgetAttribute", "WA_TransparentForMouseEvents")
WA_StyledBackground = _resolve(Qt, "WidgetAttribute", "WA_StyledBackground")

# Qt.ScrollBarPolicy
ScrollBarAlwaysOff = _resolve(Qt, "ScrollBarPolicy", "ScrollBarAlwaysOff")
ScrollBarAsNeeded = _resolve(Qt, "ScrollBarPolicy", "ScrollBarAsNeeded")

# QTextOption.WrapMode - wrap mid-token so a long URL or unbreakable string
# still flows to the next line instead of triggering horizontal scroll.
WrapAtWordBoundaryOrAnywhere = _resolve(
    QTextOption, "WrapMode", "WrapAtWordBoundaryOrAnywhere"
)

# QTextEdit.LineWrapMode - pinned to widget width so wrapping always engages
# even when QSS or a rich-text paste would otherwise leave it implicit.
LineWrapWidgetWidth = _resolve(QTextEdit, "LineWrapMode", "WidgetWidth")

# Qt.AspectRatioMode / Qt.TransformationMode
KeepAspectRatio = _resolve(Qt, "AspectRatioMode", "KeepAspectRatio")
SmoothTransformation = _resolve(Qt, "TransformationMode", "SmoothTransformation")

# Qt.TextInteractionFlag
TextSelectableByMouse = _resolve(Qt, "TextInteractionFlag", "TextSelectableByMouse")
TextBrowserInteraction = _resolve(Qt, "TextInteractionFlag", "TextBrowserInteraction")

# QIODevice.OpenModeFlag
WriteOnly = _resolve(QIODevice, "OpenModeFlag", "WriteOnly")

# QImage.Format
FormatARGB32 = _resolve(QImage, "Format", "Format_ARGB32")

# QTextCursor.MoveOperation
CursorEnd = _resolve(QTextCursor, "MoveOperation", "End")

# QSizePolicy.Policy
SizePolicyExpanding = _resolve(QSizePolicy, "Policy", "Expanding")
SizePolicyFixed = _resolve(QSizePolicy, "Policy", "Fixed")

# QPalette.ColorRole
PaletteBase = _resolve(QPalette, "ColorRole", "Base")

# QFrame.Shape / QFrame.Shadow
FrameNoFrame = _resolve(QFrame, "Shape", "NoFrame")
FrameHLine = _resolve(QFrame, "Shape", "HLine")
FrameVLine = _resolve(QFrame, "Shape", "VLine")
FrameSunken = _resolve(QFrame, "Shadow", "Sunken")

# QgsBlockingNetworkRequest.ErrorCode
BlockingNoError = _resolve(QgsBlockingNetworkRequest, "ErrorCode", "NoError")

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
    PolygonGeometry = QgsWkbTypes.PolygonGeometry
if LineGeometry is None:
    from qgis.core import QgsWkbTypes
    LineGeometry = QgsWkbTypes.LineGeometry

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
    WkbPolygon = QgsWkbTypes.Polygon
if WkbMultiPolygon is None:
    from qgis.core import QgsWkbTypes
    WkbMultiPolygon = QgsWkbTypes.MultiPolygon

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
    DistanceMeters = QgsUnitTypes.DistanceMeters

# QgsVertexMarker.IconType.ICON_CIRCLE - scoped on QGIS 4, flat also on QGIS 3.
try:
    from qgis.gui import QgsVertexMarker
    VertexIconCircle = _resolve(QgsVertexMarker, "IconType", "ICON_CIRCLE")
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
    from qgis.PyQt.QtCore import QVariant as _QVariant
    _FIELD_TYPE_STRING = _QVariant.String
    _FIELD_TYPE_DOUBLE = _QVariant.Double
    _FIELD_TYPE_INT = _QVariant.Int


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
    return _resolve(QNetworkReply, "NetworkError", name)


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
HttpStatusCodeAttribute = _resolve(
    QNetworkRequest, "Attribute", "HttpStatusCodeAttribute"
)
# Redirect policy. PyQt5 on some QGIS 3 builds exposes these flat, not scoped,
# so resolve through the same scoped-then-flat helper rather than hardcoding
# QNetworkRequest.Attribute.* / QNetworkRequest.RedirectPolicy.* (which would
# AttributeError on those builds, on every API request and download).
RedirectPolicyAttribute = _resolve(
    QNetworkRequest, "Attribute", "RedirectPolicyAttribute"
)
NoLessSafeRedirectPolicy = _resolve(
    QNetworkRequest, "RedirectPolicy", "NoLessSafeRedirectPolicy"
)


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
