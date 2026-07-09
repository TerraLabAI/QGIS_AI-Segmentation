








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










    for module in ("qgis.PyQt.QtGui", "qgis.PyQt.QtWidgets"):
        found = getattr(importlib.import_module(module), name, None)
        if found is not None:
            return found
    raise ImportError(f"{name} not found in qgis.PyQt.QtGui or QtWidgets")


QAction = _import_moved("QAction")
QShortcut = _import_moved("QShortcut")



LeftDockWidgetArea = resolve_qt_enum(Qt, "DockWidgetArea", "LeftDockWidgetArea")
RightDockWidgetArea = resolve_qt_enum(Qt, "DockWidgetArea", "RightDockWidgetArea")


PointingHandCursor = resolve_qt_enum(Qt, "CursorShape", "PointingHandCursor")
CrossCursor = resolve_qt_enum(Qt, "CursorShape", "CrossCursor")
WaitCursor = resolve_qt_enum(Qt, "CursorShape", "WaitCursor")
ArrowCursor = resolve_qt_enum(Qt, "CursorShape", "ArrowCursor")


AlignCenter = resolve_qt_enum(Qt, "AlignmentFlag", "AlignCenter")
AlignTop = resolve_qt_enum(Qt, "AlignmentFlag", "AlignTop")
AlignLeft = resolve_qt_enum(Qt, "AlignmentFlag", "AlignLeft")
AlignVCenter = resolve_qt_enum(Qt, "AlignmentFlag", "AlignVCenter")


Key_Return = resolve_qt_enum(Qt, "Key", "Key_Return")
Key_Enter = resolve_qt_enum(Qt, "Key", "Key_Enter")
Key_Escape = resolve_qt_enum(Qt, "Key", "Key_Escape")


WindowShortcut = resolve_qt_enum(Qt, "ShortcutContext", "WindowShortcut")
WidgetWithChildrenShortcut = resolve_qt_enum(
    Qt, "ShortcutContext", "WidgetWithChildrenShortcut"
)


def event_pos(event):










    getter = getattr(event, "position", None) or getattr(event, "localPos", None)

    getter = getter or getattr(event, "pos")  # noqa: B009
    point = getter()


    to_point = getattr(point, "toPoint", None)
    return to_point() if to_point is not None else point



SolidLine = resolve_qt_enum(Qt, "PenStyle", "SolidLine")
DashLine = resolve_qt_enum(Qt, "PenStyle", "DashLine")


ShiftModifier = resolve_qt_enum(Qt, "KeyboardModifier", "ShiftModifier")


LeftButton = resolve_qt_enum(Qt, "MouseButton", "LeftButton")
RightButton = resolve_qt_enum(Qt, "MouseButton", "RightButton")


NoFocus = resolve_qt_enum(Qt, "FocusPolicy", "NoFocus")


OtherFocusReason = resolve_qt_enum(Qt, "FocusReason", "OtherFocusReason")


ToolButtonTextBesideIcon = resolve_qt_enum(Qt, "ToolButtonStyle", "ToolButtonTextBesideIcon")


DownArrow = resolve_qt_enum(Qt, "ArrowType", "DownArrow")
RightArrow = resolve_qt_enum(Qt, "ArrowType", "RightArrow")


RichText = resolve_qt_enum(Qt, "TextFormat", "RichText")
PlainText = resolve_qt_enum(Qt, "TextFormat", "PlainText")


WA_TransparentForMouseEvents = resolve_qt_enum(Qt, "WidgetAttribute", "WA_TransparentForMouseEvents")
WA_StyledBackground = resolve_qt_enum(Qt, "WidgetAttribute", "WA_StyledBackground")


ScrollBarAlwaysOff = resolve_qt_enum(Qt, "ScrollBarPolicy", "ScrollBarAlwaysOff")
ScrollBarAsNeeded = resolve_qt_enum(Qt, "ScrollBarPolicy", "ScrollBarAsNeeded")



WrapAtWordBoundaryOrAnywhere = resolve_qt_enum(
    QTextOption, "WrapMode", "WrapAtWordBoundaryOrAnywhere"
)



LineWrapWidgetWidth = resolve_qt_enum(QTextEdit, "LineWrapMode", "WidgetWidth")


KeepAspectRatio = resolve_qt_enum(Qt, "AspectRatioMode", "KeepAspectRatio")
SmoothTransformation = resolve_qt_enum(Qt, "TransformationMode", "SmoothTransformation")


TextSelectableByMouse = resolve_qt_enum(Qt, "TextInteractionFlag", "TextSelectableByMouse")
TextBrowserInteraction = resolve_qt_enum(Qt, "TextInteractionFlag", "TextBrowserInteraction")


WriteOnly = resolve_qt_enum(QIODevice, "OpenModeFlag", "WriteOnly")


FormatARGB32 = resolve_qt_enum(QImage, "Format", "Format_ARGB32")


FormatRGB888 = resolve_qt_enum(QImage, "Format", "Format_RGB888")


CursorEnd = resolve_qt_enum(QTextCursor, "MoveOperation", "End")


SizePolicyExpanding = resolve_qt_enum(QSizePolicy, "Policy", "Expanding")
SizePolicyFixed = resolve_qt_enum(QSizePolicy, "Policy", "Fixed")


PaletteBase = resolve_qt_enum(QPalette, "ColorRole", "Base")


FrameNoFrame = resolve_qt_enum(QFrame, "Shape", "NoFrame")
FrameHLine = resolve_qt_enum(QFrame, "Shape", "HLine")
FrameVLine = resolve_qt_enum(QFrame, "Shape", "VLine")
FrameSunken = resolve_qt_enum(QFrame, "Shadow", "Sunken")


BlockingNoError = resolve_qt_enum(QgsBlockingNetworkRequest, "ErrorCode", "NoError")


CapabilityAddFeatures = resolve_qt_enum(QgsVectorDataProvider, "Capability", "AddFeatures")


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





SimplifyFullHint = _render_simplify_enum(
    "VectorRenderingSimplificationFlag", "SimplifyHint", "FullSimplification")
SimplifyGeometryHint = _render_simplify_enum(
    "VectorRenderingSimplificationFlag", "SimplifyHint", "GeometrySimplification")
SimplifyDistanceAlgorithm = _render_simplify_enum(
    "VectorSimplificationAlgorithm", "SimplifyAlgorithm", "Distance")


try:
    from qgis.gui import QgsVertexMarker
    VertexIconCircle = resolve_qt_enum(QgsVertexMarker, "IconType", "ICON_CIRCLE")
except Exception:
    VertexIconCircle = None


def symbol_fill_color_property():







    from qgis.core import QgsSymbolLayer
    prop_scope = getattr(QgsSymbolLayer, "Property", None)
    for owner, name in (
        (prop_scope, "FillColor"),
        (QgsSymbolLayer, "PropertyFillColor"),
        (prop_scope, "PropertyFillColor"),
    ):
        if owner is None:
            continue
        val = getattr(owner, name, None)
        if val is not None:
            return val
    return None












try:
    from qgis.core import Qgis as _Qgis
    _QGIS_VERSION_INT = getattr(_Qgis, "QGIS_VERSION_INT", 0)
except Exception:
    _QGIS_VERSION_INT = 0

if _QGIS_VERSION_INT >= 33800:
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






        from qgis.PyQt.QtCore import QMetaType as _QMetaType
        _FIELD_TYPE_STRING = _QMetaType.Type.QString
        _FIELD_TYPE_DOUBLE = _QMetaType.Type.Double
        _FIELD_TYPE_INT = _QMetaType.Type.Int


def field_type_string():

    return _FIELD_TYPE_STRING


def field_type_double():

    return _FIELD_TYPE_DOUBLE


def field_type_int():

    return _FIELD_TYPE_INT



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


HttpStatusCodeAttribute = resolve_qt_enum(
    QNetworkRequest, "Attribute", "HttpStatusCodeAttribute"
)




RedirectPolicyAttribute = resolve_qt_enum(
    QNetworkRequest, "Attribute", "RedirectPolicyAttribute"
)
NoLessSafeRedirectPolicy = resolve_qt_enum(
    QNetworkRequest, "RedirectPolicy", "NoLessSafeRedirectPolicy"
)


def silent_task_flags():






    flags = QgsTask.Flag.CanCancel
    for name in ("Hidden", "Silent"):
        flag = getattr(QgsTask.Flag, name, None)
        if flag is not None:
            flags = flags | flag
    return flags





try:
    from qgis.core import Qgis as _QgisGeometryOps
    _GEOMETRY_OP_SUCCESS = getattr(
        getattr(_QgisGeometryOps, "GeometryOperationResult", None),
        "Success", None)
except Exception:
    _GEOMETRY_OP_SUCCESS = None


def geometry_op_succeeded(result) -> bool:













    if isinstance(result, bool):
        return result
    if result is None:
        return False
    if _GEOMETRY_OP_SUCCESS is not None and result == _GEOMETRY_OP_SUCCESS:
        return True
    try:
        return int(result) == 0
    except (TypeError, ValueError):
        return False


def safe_single_shot(msec: int, owner: QObject, callback) -> QTimer:











    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.start(max(0, int(msec)))
    return timer




def safe_disconnect(owner, signal_name: str, slot=None) -> bool:


















    try:
        signal = getattr(owner, signal_name)
        if slot is None:
            signal.disconnect()
        else:
            signal.disconnect(slot)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return True
