








from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING

try:
    from .venv_manager import ensure_venv_packages_available
except ImportError:
    pass
else:
    ensure_venv_packages_available()

from .detection_masks import (  # noqa: E402
    logger,
)
from .shape_policy_dials import (  # noqa: E402
    archive_jpeg_quality,
    imagery_probe_px,
    imagery_probe_timeout_ms,
    render_zone_timeout_ms,
)

if TYPE_CHECKING:
    from qgis.core import QgsRasterLayer, QgsRectangle











_TILE_IMAGE_FORMAT: str = "JPEG"
_TILE_JPEG_QUALITY: int = 90


_ARCHIVE_JPEG_QUALITY: int = 80


def _save_jpeg(image, buf, quality: int) -> None:








    from qgis.PyQt.QtGui import QImageWriter

    writer = QImageWriter(buf, b"JPEG")
    writer.setQuality(int(quality))
    writer.setProgressiveScanWrite(True)
    if not writer.write(image):
        mode = buf.openMode()
        buf.close()
        buf.setData(b"")
        buf.open(mode)
        image.save(buf, _TILE_IMAGE_FORMAT, int(quality))


def _tile_jpeg_quality() -> int:



    try:
        from .detection_policy import tile_jpeg_quality  # noqa: PLC0415
        return tile_jpeg_quality(_TILE_JPEG_QUALITY)
    except Exception:  # noqa: BLE001
        return _TILE_JPEG_QUALITY












_STAMP_PAD: int = 3




_STAMP_MAX: int = 195


def should_paste_stamp(
    run_scale_side: float, cap: int, has_prompt: bool,
    min_scale: float = 0.85,
) -> bool:












    if not has_prompt:
        return True
    if run_scale_side <= 0 or cap <= 0:
        return True
    return cap >= run_scale_side * min_scale


def _stamp_pad() -> int:



    try:
        from .detection_policy import exemplar_stamp_pad_px  # noqa: PLC0415
        return exemplar_stamp_pad_px(_STAMP_PAD)
    except Exception:  # noqa: BLE001
        return _STAMP_PAD


def _stamp_max() -> int:

    try:
        from .detection_policy import exemplar_stamp_max_px  # noqa: PLC0415
        return exemplar_stamp_max_px(_STAMP_MAX)
    except Exception:  # noqa: BLE001
        return _STAMP_MAX


def stamp_size_cap(n: int) -> int:












    from .tile_manager import OVERLAP_FRACTION, TILE_SIZE

    n = max(1, int(n))
    pad = _stamp_pad()
    overlap_px = int(TILE_SIZE * OVERLAP_FRACTION)
    band_budget = overlap_px - 2 * pad
    width_budget = (TILE_SIZE - pad) // n - pad
    return max(1, min(_stamp_max(), band_budget, width_budget))


def in_situ_exemplar_box(
    full_box, tx: int, ty: int, tw: int, th: int,
) -> list[float] | None:













    if not full_box or len(full_box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in full_box)
    except (TypeError, ValueError):
        return None


    if not (x0 >= tx and y0 >= ty and x1 <= tx + tw and y1 <= ty + th):
        return None
    bx0 = min(max(x0 - tx, 0.0), float(tw))
    by0 = min(max(y0 - ty, 0.0), float(th))
    bx1 = min(max(x1 - tx, 0.0), float(tw))
    by1 = min(max(y1 - ty, 0.0), float(th))
    if bx1 - bx0 < 1.0 or by1 - by0 < 1.0:
        return None
    return [bx0, by0, bx1, by1]


def region_exemplar_box(
    full_box, tx: int, ty: int, tw: int, th: int, min_side: float = 8.0,
) -> list[float] | None:









    if not full_box or len(full_box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in full_box)
    except (TypeError, ValueError):
        return None
    bx0 = min(max(x0 - tx, 0.0), float(tw))
    by0 = min(max(y0 - ty, 0.0), float(th))
    bx1 = min(max(x1 - tx, 0.0), float(tw))
    by1 = min(max(y1 - ty, 0.0), float(th))
    if bx1 - bx0 < min_side or by1 - by0 < min_side:
        return None
    return [bx0, by0, bx1, by1]


def top_row_bottom_stamp_ok(
    top_ty: int, top_th: int, next_ty: int, band_content_h: int,
    pad: int = 0,
) -> bool:















    if pad <= 0:
        pad = _stamp_pad()
    return next_ty + pad + band_content_h <= top_ty + top_th - pad - band_content_h


def _set_quality_render_flags(settings) -> None:







    from qgis.core import Qgis, QgsMapSettings

    for name in ("Antialiasing", "HighQualityImageTransforms"):



        flag = getattr(QgsMapSettings, name, None)
        if flag is None:
            scoped = getattr(Qgis, "MapSettingsFlag", None)
            flag = None if scoped is None else getattr(scoped, name, None)
        if flag is None:
            continue
        try:
            settings.setFlag(flag, True)
        except (TypeError, AttributeError):
            pass


def _set_blocking_remote_fetch(settings) -> None:







    from qgis.core import Qgis, QgsMapSettings



    flag = getattr(QgsMapSettings, "RenderBlocking", None)
    if flag is None:
        scoped = getattr(Qgis, "MapSettingsFlag", None)
        flag = None if scoped is None else getattr(scoped, "RenderBlocking", None)
    if flag is None:
        return
    try:
        settings.setFlag(flag, True)
    except (TypeError, AttributeError):
        pass





_RENDER_ZONE_TIMEOUT_MS: int = 25000





_render_loop_depth: int = 0


def render_loop_is_active() -> bool:

    return _render_loop_depth > 0


def _exclude_user_input_flag():

    from qgis.PyQt.QtCore import QEventLoop

    from .qt_compat import resolve_qt_enum

    return resolve_qt_enum(
        QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents")


def render_zone_to_image(
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    width: int,
    height: int,
    timeout_ms: int | None = None,
    resample_local: bool = False,
    render_crs=None,
):






































    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings, QgsProject
    from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer

    if timeout_ms is None:
        timeout_ms = render_zone_timeout_ms(_RENDER_ZONE_TIMEOUT_MS)
    from qgis.PyQt.QtGui import QColor

    if width <= 0 or height <= 0:
        logger.warning("render_zone_to_image: invalid dimensions %dx%d", width, height)
        return None, None

    t0 = time.monotonic()
    render_clone = None
    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)


        settings.setTransformContext(QgsProject.instance().transformContext())




        if resample_local:
            render_clone = _local_raster_render_clone(layer)
        render_layer = render_clone if render_clone is not None else layer
        settings.setLayers([render_layer])
        settings.setDestinationCrs(_render_crs_or_layer(layer, render_crs))
        settings.setBackgroundColor(QColor(0, 0, 0))
        _set_quality_render_flags(settings)


        _set_blocking_remote_fetch(settings)


        actual_extent = settings.visibleExtent()

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)

        QTimer.singleShot(timeout_ms, loop.quit)





        _active_render_jobs.append(job)
        global _render_loop_depth
        _render_loop_depth += 1
        try:
            job.start()
            loop.exec(_exclude_user_input_flag())
        finally:
            _render_loop_depth -= 1
            if job in _active_render_jobs:
                _active_render_jobs.remove(job)

        if not job.isActive():
            img = job.renderedImage()
        else:




            job.cancel()
            logger.warning("render_zone_to_image: render timed out after %d ms", timeout_ms)
            del render_clone
            return None, None


        del render_clone
    except Exception as exc:
        logger.warning("render_zone_to_image: failed for %dx%d: %s", width, height, exc)
        return None, None

    if img is None or img.isNull():
        return None, None

    logger.debug(
        "render_zone_to_image: rendered %dx%d in %d ms",
        width, height, int((time.monotonic() - t0) * 1000),
    )
    return img, actual_extent


def visible_extent_for(extent: QgsRectangle, width: int, height: int):














    from qgis.core import QgsMapSettings
    from qgis.PyQt.QtCore import QSize

    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(int(width), int(height)))
        settings.setExtent(extent)
        return settings.visibleExtent()
    except Exception as exc:  # noqa: BLE001
        logger.warning("visible_extent_for: failed (%dx%d): %s", width, height, exc)
        return extent






_active_render_jobs: list = []



_tile_render_hooks: list = []



_draining_hooks: list = []


def _drop_drain_hook(job) -> None:

    for i, entry in enumerate(_draining_hooks):
        if entry[0] is job:
            del _draining_hooks[i]
            return


def _drop_tile_render_hook(job) -> None:

    for i, entry in enumerate(_tile_render_hooks):
        if entry[0] is job:
            del _tile_render_hooks[i]
            return


def _release_job_later(state: dict) -> None:







    from qgis.PyQt.QtCore import QTimer

    QTimer.singleShot(0, lambda: state.update(job=None))


def cancel_active_tile_render() -> None:








    hooks = list(_tile_render_hooks)
    _tile_render_hooks.clear()
    drains = list(_draining_hooks)
    _draining_hooks.clear()
    jobs = list(_active_render_jobs)
    _active_render_jobs.clear()
    for job in jobs:
        try:
            job.cancel()
        except (RuntimeError, AttributeError):
            pass
    for _job, drained in drains:
        try:
            drained()
        except Exception:  # noqa: BLE001
            logger.warning("cancel_active_tile_render: drain release failed")




    for _job, state, finish in hooks:
        state["cancelled"] = True
        try:
            finish()
        except Exception:  # noqa: BLE001
            logger.warning("cancel_active_tile_render: finish failed")


def _configure_downsample_resampling(layer, qgis_module) -> bool:











    try:
        provider = layer.dataProvider()
    except (AttributeError, RuntimeError):
        provider = None


    try:
        from qgis.core import QgsRasterDataProvider

        rm = getattr(QgsRasterDataProvider, "ResamplingMethod", None)
        stage = getattr(
            getattr(qgis_module, "RasterResamplingStage", None), "Provider", None
        )
        can_resample = provider is not None
        can_resample = can_resample and rm is not None
        can_resample = can_resample and stage is not None
        can_resample = can_resample and hasattr(provider, "enableProviderResampling")
        can_resample = can_resample and hasattr(provider, "setZoomedOutResamplingMethod")
        can_resample = can_resample and hasattr(provider, "setZoomedInResamplingMethod")
        can_resample = can_resample and hasattr(layer, "setResamplingStage")
        if can_resample:



            out_method = getattr(rm, "Average", None)
            if out_method is None:
                out_method = getattr(rm, "Bilinear", None)
            in_method = getattr(rm, "Cubic", None)
            if in_method is None:
                in_method = getattr(rm, "Bilinear", None)
            provider.enableProviderResampling(True)
            if out_method is not None:
                provider.setZoomedOutResamplingMethod(out_method)
            if in_method is not None:
                provider.setZoomedInResamplingMethod(in_method)
            if hasattr(provider, "setMaxOversampling"):
                provider.setMaxOversampling(2.0)
            layer.setResamplingStage(stage)
            return True
    except (AttributeError, RuntimeError, TypeError):
        pass



    try:
        from qgis.core import QgsBilinearRasterResampler, QgsCubicRasterResampler

        resample_filter = layer.resampleFilter() if hasattr(layer, "resampleFilter") else None
        if resample_filter is not None:
            resample_filter.setZoomedOutResampler(QgsBilinearRasterResampler())
            resample_filter.setZoomedInResampler(QgsCubicRasterResampler())
            return True
    except (AttributeError, RuntimeError, TypeError):
        pass

    return False


def _local_raster_render_clone(layer: QgsRasterLayer):


















    from qgis.core import QgsRasterLayer



    try:
        if not isinstance(layer, QgsRasterLayer) or layer.providerType() != "gdal":
            return None
    except (AttributeError, RuntimeError):
        return None

    clone = None
    try:
        clone = layer.clone()
    except (AttributeError, RuntimeError):
        clone = None
    if clone is None:


        try:
            rebuilt = QgsRasterLayer(layer.source(), layer.name(), "gdal")
            if not rebuilt.isValid():
                return None
            renderer = layer.renderer()
            if renderer is not None:
                rebuilt.setRenderer(renderer.clone())
            clone = rebuilt
        except (AttributeError, RuntimeError, TypeError):
            return None

    try:
        if clone is None or not clone.isValid():
            return None
    except (AttributeError, RuntimeError):
        return None

    from qgis.core import Qgis

    if not _configure_downsample_resampling(clone, Qgis):
        return None
    return clone


def _render_crs_or_layer(layer, render_crs):






    try:
        if render_crs is not None and render_crs.isValid():
            return render_crs
    except (RuntimeError, AttributeError):
        pass
    return layer.crs()


def _tile_render_settings(layer, tile_extent, width: int, height: int,
                          render_clone=None, render_crs=None,
                          clone_resolved: bool = False):



















    from qgis.core import QgsMapSettings, QgsProject
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor

    settings = QgsMapSettings()
    settings.setOutputSize(QSize(int(width), int(height)))
    settings.setExtent(tile_extent)





    settings.setTransformContext(QgsProject.instance().transformContext())
    if render_clone is None and not clone_resolved:
        render_clone = _local_raster_render_clone(layer)
    render_layer = render_clone if render_clone is not None else layer
    settings.setLayers([render_layer])
    settings.setDestinationCrs(_render_crs_or_layer(layer, render_crs))
    settings.setBackgroundColor(QColor(0, 0, 0))
    _set_quality_render_flags(settings)




    _set_blocking_remote_fetch(settings)
    return settings, render_clone


def start_tile_render_job(
    layer,
    tile_extent,
    width: int,
    height: int,
    on_done,
    timeout_ms: int = 60000,
    render_clone=None,
    render_crs=None,
    clone_resolved: bool = False,
    report_cancel: bool = False,
) -> bool:













    from qgis.core import QgsMapRendererParallelJob
    from qgis.PyQt.QtCore import QTimer

    if width <= 0 or height <= 0:
        logger.warning("start_tile_render_job: invalid dimensions %dx%d", width, height)
        return False
    try:
        settings, render_clone = _tile_render_settings(
            layer, tile_extent, width, height, render_clone, render_crs,
            clone_resolved)
        job = QgsMapRendererParallelJob(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("start_tile_render_job: failed for %dx%d: %s", width, height, exc)
        return False

    state = {"done": False, "clone": render_clone, "job": job,
             "cancelled": False}

    def _drained() -> None:


        if state.get("drained"):
            return
        state["drained"] = True
        rjob = state["job"]
        if rjob in _active_render_jobs:
            _active_render_jobs.remove(rjob)
        _drop_drain_hook(rjob)
        state["clone"] = None
        _release_job_later(state)

    def _finish(timed_out: bool = False) -> None:
        if state["done"]:
            if state.get("draining") and not timed_out:
                _drained()
            return
        state["done"] = True
        rjob = state["job"]
        _drop_tile_render_hook(rjob)
        img = None
        try:
            if state["cancelled"]:



                img = None
            elif not rjob.isActive():
                img = rjob.renderedImage()
            elif timed_out and hasattr(rjob, "cancelWithoutBlocking"):






                state["draining"] = True
                _draining_hooks.append((rjob, _drained))
                rjob.cancelWithoutBlocking()
                logger.warning(
                    "start_tile_render_job: render timed out after %d ms", timeout_ms)
            else:


                rjob.cancel()
                if timed_out:
                    logger.warning(
                        "start_tile_render_job: render timed out after %d ms", timeout_ms)
        except (RuntimeError, AttributeError):
            img = None
        if state.get("draining"):

            try:
                if not rjob.isActive():
                    _drained()
            except (RuntimeError, AttributeError):
                _drained()
        else:
            if rjob in _active_render_jobs:
                _active_render_jobs.remove(rjob)
            state["clone"] = None



            _release_job_later(state)
        if img is not None and img.isNull():
            img = None
        try:
            if report_cancel:
                on_done(img, bool(state["cancelled"]))
            else:
                on_done(img)
        except Exception:  # noqa: BLE001
            logger.warning("start_tile_render_job: on_done callback failed")

    job.finished.connect(_finish)
    QTimer.singleShot(timeout_ms, lambda: _finish(timed_out=True))
    _active_render_jobs.append(job)
    _tile_render_hooks.append((job, state, _finish))
    try:
        job.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("start_tile_render_job: start failed: %s", exc)
        state["done"] = True
        state["job"] = None
        _drop_tile_render_hook(job)
        if job in _active_render_jobs:
            _active_render_jobs.remove(job)
        return False
    return True







_IMAGERY_PROBE_PX: int = 256
_IMAGERY_PROBE_TIMEOUT_MS: int = 12000


def probe_depth_chain(
    layer,
    extent,
    render_crs=None,
    count: int = 2,
    side_px: int | None = None,
    min_side_px: int = 32,
    timeout_ms: int | None = None,
):
















    images = []
    if side_px is None:
        side_px = imagery_probe_px(_IMAGERY_PROBE_PX)
    if timeout_ms is None:
        timeout_ms = imagery_probe_timeout_ms(_IMAGERY_PROBE_TIMEOUT_MS)
    side = int(side_px)
    for _ in range(max(2, int(count))):
        if side < int(min_side_px):
            break
        img = _probe_render(layer, extent, side, render_crs, int(timeout_ms))
        if img is None:
            break
        images.append(img)
        side //= 2
    return images


def start_probe_depth_chain(
    layer,
    extent,
    on_done,
    render_crs=None,
    count: int = 2,
    side_px: int | None = None,
    min_side_px: int = 32,
    timeout_ms: int | None = None,
) -> bool:

















    if side_px is None:
        side_px = imagery_probe_px(_IMAGERY_PROBE_PX)
    if timeout_ms is None:
        timeout_ms = imagery_probe_timeout_ms(_IMAGERY_PROBE_TIMEOUT_MS)
    sides = []
    side = int(side_px)
    for _ in range(max(2, int(count))):
        if side < int(min_side_px):
            break
        sides.append(side)
        side //= 2
    if not sides:
        return False
    state = {"results": [None] * len(sides), "pending": len(sides),
             "cancelled": False, "reported": False}

    def _report() -> None:
        if state["reported"] or state["pending"] > 0:
            return
        state["reported"] = True
        images = []
        for img in state["results"]:
            if img is None:
                break
            images.append(img)
        state["results"] = []
        try:
            on_done(images, bool(state["cancelled"]))
        except Exception:  # noqa: BLE001
            logger.warning("start_probe_depth_chain: on_done callback failed")

    def _level_done(index: int):
        def _done(img, cancelled) -> None:
            state["results"][index] = img
            state["cancelled"] = state["cancelled"] or bool(cancelled)
            state["pending"] -= 1
            _report()
        return _done

    started = 0
    for index, level_side in enumerate(sides):
        try:
            ok = start_tile_render_job(
                layer, extent, level_side, level_side, _level_done(index),
                timeout_ms=int(timeout_ms), render_clone=None,
                render_crs=render_crs, clone_resolved=True, report_cancel=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("probe render failed to start: %s", exc)
            ok = False
        if ok:
            started += 1
        else:


            state["pending"] -= 1
    if started == 0:
        return False
    _report()
    return True


def _probe_render(layer, extent, side_px: int, render_crs, timeout_ms: int):

    try:
        img, _actual = render_zone_to_image(
            layer, extent, side_px, side_px,
            timeout_ms=timeout_ms, render_crs=render_crs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe render failed: %s", exc)
        return None
    return img


def encode_tile_png(
    img,
    tx: int,
    ty: int,
    tw: int,
    th: int,
) -> tuple[tuple[int, int, int, int], bytes] | None:




















    from qgis.PyQt.QtCore import QBuffer, QRect

    from .qt_compat import WriteOnly



    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if tx < 0 or ty < 0 or cw <= 0 or ch <= 0:
        return None
    sub = (img if tx == ty == 0 and cw == img.width() and ch == img.height()
           else img.copy(QRect(tx, ty, cw, ch)))
    buf = QBuffer()
    buf.open(WriteOnly)
    _save_jpeg(sub, buf, _tile_jpeg_quality())
    data = bytes(buf.data())
    buf.close()
    if not data:
        return None
    return (tx, ty, cw, ch), data


def encode_tile_archive_copy(
    img,
    tx: int,
    ty: int,
    tw: int,
    th: int,
) -> bytes | None:













    from qgis.PyQt.QtCore import QBuffer, QRect, Qt

    from .qt_compat import WriteOnly

    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if tx < 0 or ty < 0 or cw <= 0 or ch <= 0:
        return None
    sub = (img if tx == ty == 0 and cw == img.width() and ch == img.height()
           else img.copy(QRect(tx, ty, cw, ch)))
    small = sub.scaled(
        max(1, cw // 2), max(1, ch // 2),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    buf = QBuffer()
    buf.open(WriteOnly)
    _save_jpeg(small, buf, archive_jpeg_quality(_ARCHIVE_JPEG_QUALITY))
    data = bytes(buf.data())
    buf.close()
    return data or None


def composite_tile_with_stamps(img, tx, ty, tw, th, stamps, bottom=False):




























    from qgis.PyQt.QtCore import QBuffer, QRect
    from qgis.PyQt.QtGui import QPainter

    from .qt_compat import WriteOnly

    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if cw <= 0 or ch <= 0:
        return None
    sub = img.copy(QRect(tx, ty, cw, ch))

    boxes: list[dict] = []
    min_x = min_y = None
    max_x = 0
    max_y = 0
    pad = _stamp_pad()
    if stamps:
        from .tile_manager import OVERLAP_FRACTION, TILE_SIZE










        band_h = min(ch, int(TILE_SIZE * OVERLAP_FRACTION))
        painter = QPainter(sub)
        x = pad
        for stamp in stamps:
            if len(stamp) == 3:
                crop, label, obj_box = stamp
            else:
                crop, label = stamp
                obj_box = None
            if crop is None:
                continue
            sw = crop.width()
            sh = crop.height()

            if x + sw + pad > cw and x > pad:
                break
            if sh + pad > band_h:
                continue


            y = (ch - pad - sh) if bottom else pad
            painter.drawImage(QRect(x, y, sw, sh), crop)



            if obj_box is not None and len(obj_box) == 4:
                bx0 = x + max(0.0, min(float(obj_box[0]), sw))
                by0 = y + max(0.0, min(float(obj_box[1]), sh))
                bx1 = x + max(0.0, min(float(obj_box[2]), sw))
                by1 = y + max(0.0, min(float(obj_box[3]), sh))
                if bx1 - bx0 < 1 or by1 - by0 < 1:
                    bx0, by0, bx1, by1 = x, y, x + sw, y + sh
            else:
                bx0, by0, bx1, by1 = x, y, x + sw, y + sh
            boxes.append({
                "box": [float(bx0), float(by0), float(bx1), float(by1)],
                "label": int(label),
            })
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = max(max_x, x + sw)
            max_y = max(max_y, y + sh)
            x += sw + pad
        painter.end()

    stamp_norm = None
    if boxes:



        stamp_norm = [
            max(0.0, (min_x - pad) / cw),
            max(0.0, (min_y - pad) / ch),
            min(1.0, (max_x + pad) / cw),
            min(1.0, (max_y + pad) / ch),
        ]

    buf = QBuffer()
    buf.open(WriteOnly)
    _save_jpeg(sub, buf, _tile_jpeg_quality())
    data = bytes(buf.data())
    buf.close()
    if not data:
        return None
    return (tx, ty, cw, ch), data, boxes, stamp_norm


def tile_png_to_base64(image_bytes: bytes) -> str:





    return base64.b64encode(image_bytes).decode("ascii")


def encoded_image_size(data: bytes) -> tuple[int, int] | None:









    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]

            if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return (w, h) if w > 0 and h > 0 else None
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            i += 2 + max(seg_len, 2)
    return None
