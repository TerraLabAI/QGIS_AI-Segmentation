








from __future__ import annotations

import re
import time

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .i18n import tr
from .raster_crop_reader import (
    _normalize_to_uint8,
)
from .tile_read_completeness import online_read_is_complete


def _argb_to_rgb(argb, premultiplied: bool):







    if not premultiplied:



        return np.ascontiguousarray(argb[:, :, 2::-1])
    alpha = argb[:, :, 3].astype(np.float32)[:, :, None]
    rgb = argb[:, :, 2::-1].astype(np.float32) * 255.0
    np.divide(rgb, alpha, out=rgb, where=alpha > 0)
    rgb[np.broadcast_to(alpha == 0, rgb.shape)] = 0.0
    return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def _fetch_online_bands(provider, extent, width, height, first_block=None,
                        read_band=None):

























    def read_from_provider(band_index):
        return provider.block(band_index, extent, width, height)

    read = read_band or read_from_provider

    block = first_block if first_block is not None else read(1)
    if block is None or not block.isValid():
        return None, False, "Provider block fetch failed"

    block_w = block.width()
    block_h = block.height()
    if block_w == 0 or block_h == 0:
        return None, False, "Provider returned empty block"

    raw_bytes = block.data()
    if raw_bytes is None or len(raw_bytes) == 0:
        return None, False, "Provider returned empty data"

    raw_data = bytes(raw_bytes)
    dt = block.dataType()


    is_argb32 = dt == Qgis.DataType.ARGB32
    is_argb32_pre = dt == Qgis.DataType.ARGB32_Premultiplied
    is_argb = is_argb32 or is_argb32_pre
    if is_argb:
        arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            block_h, block_w, 4)
        return _argb_to_rgb(arr, is_argb32_pre), True, None


    dtype_map = {
        Qgis.DataType.Byte: np.uint8,
        Qgis.DataType.UInt16: np.uint16,
        Qgis.DataType.Int16: np.int16,
        Qgis.DataType.UInt32: np.uint32,
        Qgis.DataType.Int32: np.int32,
        Qgis.DataType.Float32: np.float32,
        Qgis.DataType.Float64: np.float64,
    }

    np_dtype = dtype_map.get(dt)
    if np_dtype is None:

        if len(raw_data) == block_w * block_h * 4:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                block_h, block_w, 4)
            return _argb_to_rgb(arr, False), True, None
        return None, False, f"Unsupported data type: {dt}"

    band_count = min(provider.bandCount(), 3)
    bands = []


    band1 = np.frombuffer(raw_data, dtype=np_dtype).reshape(
        block_h, block_w).copy()
    bands.append(band1)





    band_bytes = block_w * block_h * np.dtype(np_dtype).itemsize
    for band_idx in range(2, band_count + 1):
        b = read(band_idx)
        if b is None or not b.isValid():
            return None, False, f"Provider band {band_idx} fetch failed"
        b_data = bytes(b.data())
        if len(b_data) != band_bytes:
            return None, False, (
                f"Provider band {band_idx} returned {len(b_data)} bytes, "
                f"expected {band_bytes}")
        bands.append(np.frombuffer(
            b_data, dtype=np_dtype).reshape(block_h, block_w).copy())

    bands_array = np.stack(bands, axis=0)
    return bands_array, False, None





_RENDER_FALLBACK_TIMEOUT_MS: int = 30000


def _apply_render_fallback_flags(settings) -> None:













    try:
        from .cloud_detection import (
            _set_blocking_remote_fetch,
            _set_quality_render_flags,
        )
    except Exception:  # noqa: BLE001
        return
    _set_quality_render_flags(settings)
    _set_blocking_remote_fetch(settings)


def _render_layer_to_image(layer, extent, width, height,
                           timeout_ms=None):



























    try:
        from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
        from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer
        from qgis.PyQt.QtGui import QColor, QImage

        from .qt_compat import resolve_qt_enum
        from .server_dials import dial_in_range

        if timeout_ms is None:
            timeout_ms = dial_in_range(
                "tuning.network.render_fallback_timeout_ms",
                _RENDER_FALLBACK_TIMEOUT_MS, 5000, 60000)

        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)
        settings.setLayers([layer])
        settings.setDestinationCrs(layer.crs())
        settings.setBackgroundColor(QColor(0, 0, 0))
        _apply_render_fallback_flags(settings)
        actual_extent = settings.visibleExtent()

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        QTimer.singleShot(int(timeout_ms), loop.quit)
        job.start()
        loop.exec(resolve_qt_enum(
            QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents"))

        if job.isActive():
            job.cancelWithoutBlocking()
            return (None, None,
                    f"Renderer fallback timed out after {int(timeout_ms)} ms")

        img = job.renderedImage()
        if img is None or img.isNull():
            return None, None, "Renderer fallback produced no image"


        img = img.convertToFormat(QImage.Format.Format_RGB32)
        out_h = img.height()
        out_w = img.width()
        ptr = img.bits()
        ptr.setsize(out_h * out_w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            out_h, out_w, 4)


        image_np = np.ascontiguousarray(arr[:, :, 2::-1])
        return image_np, actual_extent, None

    except Exception as e:
        return None, None, f"Renderer fallback failed: {str(e)}"






_SERVER_REPLIED_ERROR_FLOOR: int = 100



_TILE_NOT_PUBLISHED_ERROR: int = 203


def online_no_coverage_message() -> str:





    return tr(
        "Online layer returned blank tiles for this area. The "
        "current zoom level may be outside the service's range, "
        "or this area has no coverage. Zoom to a level where the "
        "layer is visible on the map, then try again."
    )


def failed_tile_request_error(reply_error: int) -> tuple[str, str]:


    if reply_error < _SERVER_REPLIED_ERROR_FLOOR:
        return (
            tr("Failed to fetch tiles from the online layer. "
               "Check your network connection."),
            "crop_error_online_fetch_failed",
        )
    if reply_error == _TILE_NOT_PUBLISHED_ERROR:
        return online_no_coverage_message(), "crop_error_online_blank_tiles"
    return (
        tr("This online layer returned no imagery for this area. Its "
           "server refused the request. Check the layer's URL in Layer "
           "Properties, or use another basemap."),
        "crop_error_online_tiles_refused",
    )


def online_layer_tile_host(layer) -> str:





    try:
        from qgis.core import QgsDataSourceUri
        from qgis.PyQt.QtCore import QUrl

        uri = QgsDataSourceUri()
        uri.setEncodedUri(layer.source())
        url = uri.param("url") or ""
        if not url:
            return ""
        return QUrl(url).host() or ""
    except Exception:  # noqa: BLE001
        return ""


def online_layer_tile_url_pattern(layer):











    try:
        from qgis.core import QgsDataSourceUri

        uri = QgsDataSourceUri()
        uri.setEncodedUri(layer.source())
        template = uri.param("url") or ""
        if not template or "{" not in template:
            return None
        parts = [re.escape(part) for part in re.split(r"\{[^{}]*\}", template)]
        return re.compile("[^/?&#]*".join(parts))
    except Exception:  # noqa: BLE001
        return None


class TileRequestErrorWatch:













    def __init__(self, host: str, tile_url_pattern=None):
        self._host = host
        self._tile_url_pattern = tile_url_pattern
        self._manager = None
        self._created_signal = None
        self._reply_callbacks = []
        self.failed = False




        self.refusals = 0


        self.reply_error = 0

    def __enter__(self):
        if not self._host:
            return self
        try:
            from qgis.core import QgsNetworkAccessManager
            from qgis.PyQt.QtNetwork import QNetworkReply

            self._manager = QgsNetworkAccessManager.instance()




            try:
                created = self._manager.requestCreated[QNetworkReply]
            except (KeyError, TypeError):
                created = self._manager.requestCreated
            created.connect(self._on_reply_created)
            self._created_signal = created
        except Exception:  # noqa: BLE001
            self._manager = None
            self._created_signal = None
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._created_signal is not None:
            try:
                self._created_signal.disconnect(self._on_reply_created)
            except (TypeError, RuntimeError):
                pass
        self._created_signal = None
        self._manager = None
        for reply, callback in self._reply_callbacks:
            try:
                reply.finished.disconnect(callback)
            except (TypeError, RuntimeError):
                pass
        self._reply_callbacks.clear()
        return False

    def _on_reply_created(self, reply):

        try:
            url = reply.request().url()
            if url.host() != self._host:
                return
            if (self._tile_url_pattern is not None
                    and not self._tile_url_pattern.fullmatch(url.toString())):
                return
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

        def callback(r=reply):
            self._on_reply_finished(r)

        try:
            reply.finished.connect(callback)
            self._reply_callbacks.append((reply, callback))
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _on_reply_finished(self, reply):



        try:



            code = reply.error()
            error = int(getattr(code, "value", code))
            if error == 0:
                return
            url = reply.request().url()
            host = url.host()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        if host != self._host:
            return
        if self._tile_url_pattern is not None:
            try:
                if not self._tile_url_pattern.fullmatch(url.toString()):
                    return
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return
        self.failed = True
        self.refusals += 1
        if not self.reply_error:
            self.reply_error = error


def run_direct_tile_fetch(request, cancel_check=None):











    started = time.monotonic()
    try:
        from .xyz_tile_fetch import fetch_xyz_crop

        image, error_code = fetch_xyz_crop(request, cancel_check=cancel_check)
    except Exception as err:  # noqa: BLE001
        image, error_code = None, str(err)
    return image, error_code, int((time.monotonic() - started) * 1000)


class OnlineCropFetcher:
















    _MAX_RETRIES = 8
    _RETRY_DELAY = 1.0









    _READ_BUDGET_S = 8.0

    def __init__(self, layer, center_x, center_y, canvas_mupp, crop_size=1024):
        from qgis.core import QgsRectangle

        from .layer_conventions import ground_unit_aspect
        from .server_dials import dial_in_range

        self._layer = layer
        self._crop_size = crop_size
        self._max_retries = dial_in_range(
            "tuning.network.online_crop_max_retries", self._MAX_RETRIES, 1, 20)
        self._retry_delay = dial_in_range(
            "tuning.network.online_crop_retry_delay_s", self._RETRY_DELAY, 0.1, 10.0)
        self._read_budget_s = dial_in_range(
            "tuning.network.online_crop_read_budget_s", self._READ_BUDGET_S, 2.0, 60.0)
        self.error = None
        self.error_code = None
        self._mutated = False
        self._orig_in = None
        self._orig_out = None




        self._orig_enabled = False
        self._attempt = 0
        self._prev_data = None


        self._prev_empty = None
        self._reload_pending = False
        self._read_seconds = 0.0




        self._tile_host = online_layer_tile_host(layer)
        self._tile_url_pattern = online_layer_tile_url_pattern(layer)
        self._tile_request_failed = False
        self._tile_error_code = 0
        self._tile_refusals = 0


        self._settled_block = None


        self._direct_request = None
        self._direct_image = None

        provider = layer.dataProvider()
        self._provider = provider
        if provider is None:
            self.error = tr("Layer data provider is not available.")
            self.error_code = "crop_error_online_provider_unavailable"
            return








        half_x = crop_size * canvas_mupp / 2.0
        half_y = half_x / ground_unit_aspect(layer.crs(), center_x, center_y)
        self._extent = QgsRectangle(
            center_x - half_x, center_y - half_y,
            center_x + half_x, center_y + half_y
        )
        QgsMessageLog.logMessage(
            f"Online crop request: center=({center_x:.6f}, {center_y:.6f}), "
            f"mupp={canvas_mupp:.6f}, extent=({self._extent.xMinimum():.2f}, "
            f"{self._extent.yMinimum():.2f}, {self._extent.xMaximum():.2f}, "
            f"{self._extent.yMaximum():.2f}), CRS={layer.crs().authid()}",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )






        self._direct_request = self._build_direct_request()

    def _build_direct_request(self):






        try:
            from .xyz_tile_fetch import (
                direct_tile_fetch_available,
                xyz_crop_request,
            )

            if not direct_tile_fetch_available(self._layer):
                return None
            return xyz_crop_request(self._layer, self._extent, self._crop_size)
        except Exception:  # noqa: BLE001  # nosec B110
            return None

    def direct_tile_request(self):



        return self._direct_request

    def take_direct_tile_request(self):




        request = self._direct_request
        self._direct_request = None
        return request

    def begin(self):



        if self._provider is None or self._direct_request is not None:


            return
        provider = self._provider
        try:
            self._orig_in = provider.zoomedInResamplingMethod()
            self._orig_out = provider.zoomedOutResamplingMethod()
        except (AttributeError, RuntimeError):


            return




        if hasattr(provider, "isProviderResamplingEnabled"):
            try:
                self._orig_enabled = bool(provider.isProviderResamplingEnabled())
            except (AttributeError, RuntimeError):
                self._orig_enabled = False

        self._mutated = True
        try:
            provider.enableProviderResampling(True)
            provider.setZoomedInResamplingMethod(
                provider.ResamplingMethod.Bilinear)
            provider.setZoomedOutResamplingMethod(
                provider.ResamplingMethod.Bilinear)
        except (AttributeError, RuntimeError):
            pass

    def restore(self):



        if not self._mutated:
            return
        self._mutated = False
        provider = self._provider
        try:
            provider.setZoomedInResamplingMethod(self._orig_in)
            provider.setZoomedOutResamplingMethod(self._orig_out)
            provider.enableProviderResampling(self._orig_enabled)
        except (AttributeError, RuntimeError):
            pass

    def read_budget_spent(self) -> bool:



        return self._read_seconds >= self._read_budget_s

    def _timed_block(self, band):



        watch = TileRequestErrorWatch(self._tile_host, self._tile_url_pattern)
        started = time.monotonic()
        try:
            with watch:
                return self._provider.block(
                    band, self._extent, self._crop_size, self._crop_size)
        finally:
            self._read_seconds += time.monotonic() - started
            if watch.failed:
                self._tile_request_failed = True
                self._tile_refusals += watch.refusals
                if not self._tile_error_code:
                    self._tile_error_code = watch.reply_error

    def _budgeted_block(self, band):








        if self.read_budget_spent():
            return None
        return self._timed_block(band)

    def _render_fallback(self):








        remaining_ms = int(
            max(0.0, self._read_budget_s - self._read_seconds) * 1000)
        if remaining_ms <= 0:
            return None, None, "Renderer fallback has no time budget left"
        return _render_layer_to_image(
            self._layer, self._extent, self._crop_size, self._crop_size,
            timeout_ms=remaining_ms)

    def _read_brought_nothing(self, image_np, from_renderer: bool) -> bool:










        block = self._settled_block
        if block is not None and not from_renderer:
            try:
                from .tile_read_completeness import read_alpha_plane

                alpha = read_alpha_plane(
                    bytes(block.data()), block.width(), block.height(),
                    block.dataType())
                if alpha is not None:
                    return not bool(alpha.any())
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        return int(image_np.sum()) == 0

    def step(self):



















        if self._direct_request is not None:
            return self._fetch_tiles_directly()
        provider = self._provider
        if self.read_budget_spent():


            return ("exhausted", 0.0)
        if self._reload_pending:


            provider.reloadData()
            self._reload_pending = False
        attempt = self._attempt
        block = self._timed_block(1)
        if block is not None and block.isValid():
            cur_data = bytes(block.data())







            if cur_data.count(0) != len(cur_data):



                self._prev_empty = None
                if online_read_is_complete(cur_data, block.width(),
                                           block.height(), block.dataType()):




                    return self._accept(block)
                if self._read_has_settled(block, cur_data):
                    return self._accept(block)
                self._prev_data = cur_data
                if attempt == 0:












                    self._attempt = attempt + 1
                    return ("refetch", 0.5)
            elif self._empty_read_has_settled(block, cur_data, attempt):
                return self._accept(block)
        if self._tile_request_failed and self._prev_data is None:




            return ("exhausted", 0.0)
        if attempt < self._max_retries - 1:
            delay = self._retry_delay * (1 + attempt * 0.5)
            QgsMessageLog.logMessage(
                f"Online tile fetch attempt {attempt + 1} - "
                f"retrying in {delay:.1f}s...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            self._reload_pending = True
            self._attempt = attempt + 1
            return ("retry", delay)
        return ("exhausted", 0.0)

    def _fetch_tiles_directly(self):






        request = self.take_direct_tile_request()
        return self.accept_direct_tiles(request, *run_direct_tile_fetch(request))

    def accept_direct_tiles(self, request, image, error_code, elapsed_ms):










        from .xyz_tile_fetch import (
            note_direct_tile_fetch_failed,
            note_direct_tile_fetch_succeeded,
        )




        self._direct_request = None
        if image is not None:
            note_direct_tile_fetch_succeeded(request.source_key)
            QgsMessageLog.logMessage(
                f"Fetched {request.tile_count()} tiles at zoom "
                f"{request.zoom} in {elapsed_ms} ms",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            self._direct_image = image
            return ("stabilized", 0.0)
        if error_code not in ("crop_error_online_blank_tiles",
                              "crop_error_online_cancelled"):



            note_direct_tile_fetch_failed(request.source_key)
        QgsMessageLog.logMessage(
            f"Direct tile fetch brought nothing after {elapsed_ms} ms "
            f"({error_code}), reading the layer instead",
            "AI Segmentation", level=Qgis.MessageLevel.Warning
        )
        self.begin()
        return self.step()

    def _accept(self, block):






        self._settled_block = block
        return ("stabilized", 0.0)

    def _read_has_settled(self, block, cur_data: bytes) -> bool:














        if self._prev_data is None:
            return False
        try:
            from .tile_read_completeness import holes_are_the_same, read_carries_alpha

            if read_carries_alpha(block.dataType()):
                return holes_are_the_same(
                    self._prev_data, cur_data, block.width(), block.height(),
                    block.dataType())
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return cur_data == self._prev_data

    def _empty_read_has_settled(self, block, cur_data: bytes,
                                attempt: int) -> bool:











        try:
            from .tile_read_completeness import read_carries_alpha

            if not read_carries_alpha(block.dataType()):
                return False
        except Exception:  # noqa: BLE001
            return False
        if attempt >= 1 and cur_data == self._prev_empty:
            return True
        self._prev_empty = cur_data
        return False

    def finish(self):



        if self._direct_image is not None:


            image_np = self._direct_image
            return image_np, {
                "bounds": (self._extent.xMinimum(), self._extent.yMinimum(),
                           self._extent.xMaximum(), self._extent.yMaximum()),
                "img_shape": (image_np.shape[0], image_np.shape[1]),
            }, None, None
        provider = self._provider
        settled = self._settled_block
        if settled is None and self._tile_request_failed and self._prev_data is None:









            QgsMessageLog.logMessage(
                f"Online tiles refused ({self._tile_refusals} tile requests, "
                f"first network reply error {self._tile_error_code}), trying "
                "the renderer fallback...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            image_np, render_extent, render_err = self._render_fallback()
            if (render_err is not None or image_np is None
                    or int(image_np.sum()) == 0):
                message, code = failed_tile_request_error(self._tile_error_code)
                return None, None, message, code
            QgsMessageLog.logMessage(
                "Refused tiles rescued by the renderer fallback",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            return image_np, {
                "bounds": (render_extent.xMinimum(), render_extent.yMinimum(),
                           render_extent.xMaximum(), render_extent.yMaximum()),
                "img_shape": (image_np.shape[0], image_np.shape[1]),
            }, None, None
        if settled is None and self.read_budget_spent():





            return None, None, tr(
                "Failed to fetch tiles from the online layer. "
                "Check your network connection."
            ), "crop_error_online_fetch_failed"





        bands_result, is_argb, fetch_err = _fetch_online_bands(
            provider, self._extent, self._crop_size, self._crop_size,
            first_block=settled, read_band=self._budgeted_block)




        crop_extent = self._extent


        from_renderer = False
        if fetch_err is not None:
            QgsMessageLog.logMessage(
                f"Provider fetch failed ({fetch_err}), trying renderer "
                "fallback...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            image_np, render_extent, render_err = self._render_fallback()
            if render_err is not None:
                return None, None, tr(
                    "Failed to fetch tiles from the online layer. "
                    "Check your network connection."
                ), "crop_error_online_fetch_failed"
            crop_extent = render_extent
            from_renderer = True
        elif is_argb:

            image_np = bands_result
        else:

            nodata = None
            try:
                nodata = provider.sourceNoDataValue(1)
            except Exception:
                pass  # nosec B110
            image_np = _normalize_to_uint8(bands_result, nodata_value=nodata)

        height = image_np.shape[0]
        width = image_np.shape[1]

        if self._read_brought_nothing(image_np, from_renderer):






            rendered = None
            if not from_renderer:
                rendered, rendered_extent, render_err = self._render_fallback()
                if render_err is not None:
                    rendered = None
            if rendered is not None and int(rendered.sum()) > 0:
                QgsMessageLog.logMessage(
                    "Blank tiles from provider, rescued by renderer fallback",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning
                )
                image_np = rendered
                crop_extent = rendered_extent
                height = image_np.shape[0]
                width = image_np.shape[1]
            else:
                return (None, None, online_no_coverage_message(),
                        "crop_error_online_blank_tiles")

        crop_info = {
            "bounds": (crop_extent.xMinimum(), crop_extent.yMinimum(),
                       crop_extent.xMaximum(), crop_extent.yMaximum()),
            "img_shape": (height, width),
        }
        return image_np, crop_info, None, None





_SYNC_FETCH_BUDGET_S: float = 25.0


def _blocking_wait(seconds, cancel_check=None):








    from qgis.core import QgsApplication

    deadline = time.monotonic() + seconds
    while True:
        if cancel_check is not None and cancel_check():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return True
        QgsApplication.processEvents()
        time.sleep(min(0.05, remaining))


def extract_crop_from_online_layer(layer, center_x, center_y, canvas_mupp,
                                   crop_size=1024, cancel_check=None):




















    fetcher = OnlineCropFetcher(layer, center_x, center_y, canvas_mupp, crop_size)
    if fetcher.error is not None:
        return None, None, fetcher.error, fetcher.error_code
    try:
        fetcher.begin()
        from .server_dials import dial_in_range
        budget_s = dial_in_range(
            "tuning.network.sync_fetch_budget_s", _SYNC_FETCH_BUDGET_S, 5, 120)
        stop_by = time.monotonic() + budget_s
        while True:
            action, delay = fetcher.step()
            if action in ("stabilized", "exhausted"):
                break
            if not _blocking_wait(delay, cancel_check):
                return (None, None, tr("Crop fetch was cancelled."),
                        "crop_error_online_cancelled")
            if time.monotonic() >= stop_by:
                break
        return fetcher.finish()
    except Exception as e:  # noqa: BLE001
        return None, None, str(e), "crop_error_online_exception"
    finally:
        fetcher.restore()
