from typing import Tuple, Optional
import numpy as np

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsPointXY,
    QgsRectangle,
    QgsRasterLayer,
)


class ImageCRSManager:
    def __init__(self, img_crs) -> None:
        if isinstance(img_crs, str):
            self.img_crs = QgsCoordinateReferenceSystem(img_crs)
        else:
            self.img_crs = img_crs

    def point_to_img_crs(
        self,
        point: QgsPointXY,
        src_crs: QgsCoordinateReferenceSystem
    ) -> QgsPointXY:
        if src_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            src_crs, self.img_crs, QgsProject.instance()
        )
        return transform.transform(point)

    def img_point_to_crs(
        self,
        point: QgsPointXY,
        dst_crs: QgsCoordinateReferenceSystem
    ) -> QgsPointXY:
        if dst_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            self.img_crs, dst_crs, QgsProject.instance()
        )
        return transform.transform(point)

    def extent_to_img_crs(
        self,
        extent: QgsRectangle,
        src_crs: QgsCoordinateReferenceSystem
    ) -> QgsRectangle:
        if src_crs == self.img_crs:
            return extent
        transform = QgsCoordinateTransform(
            src_crs, self.img_crs, QgsProject.instance()
        )
        return transform.transformBoundingBox(extent)


def geo_to_pixel(
    geo_x: float,
    geo_y: float,
    transform: 'rasterio.Affine'
) -> Tuple[int, int]:
    import rasterio
    from rasterio import transform as rio_transform
    from .import_guard import assert_package_isolated
    assert_package_isolated('rasterio', rasterio)
    col, row = rio_transform.rowcol(transform, geo_x, geo_y)
    return int(row), int(col)


def pixel_to_geo(
    row: int,
    col: int,
    transform: 'rasterio.Affine'
) -> Tuple[float, float]:
    geo_x = transform.c + col * transform.a + row * transform.b
    geo_y = transform.f + col * transform.d + row * transform.e
    return geo_x, geo_y


def get_raster_info(layer: QgsRasterLayer) -> dict:
    provider = layer.dataProvider()

    extent = layer.extent()
    width = layer.width()
    height = layer.height()
    crs = layer.crs()

    pixel_size_x = extent.width() / width
    pixel_size_y = extent.height() / height

    return {
        'path': layer.source(),
        'extent': (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()),
        'width': width,
        'height': height,
        'crs': crs,
        'crs_authid': crs.authid(),
        'pixel_size': (pixel_size_x, pixel_size_y),
    }


def map_point_to_image_coords(
    map_point: QgsPointXY,
    extent: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
    map_crs: QgsCoordinateReferenceSystem,
    img_crs: QgsCoordinateReferenceSystem,
) -> Tuple[float, float]:
    if map_crs != img_crs:
        transform = QgsCoordinateTransform(map_crs, img_crs, QgsProject.instance())
        map_point = transform.transform(map_point)

    x_min, y_min, x_max, y_max = extent
    height, width = img_size

    pixel_x = (map_point.x() - x_min) / (x_max - x_min) * width
    pixel_y = (y_max - map_point.y()) / (y_max - y_min) * height

    return pixel_x, pixel_y


def image_coords_to_map_point(
    pixel_x: float,
    pixel_y: float,
    extent: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
) -> QgsPointXY:
    x_min, y_min, x_max, y_max = extent
    height, width = img_size

    map_x = x_min + (pixel_x / width) * (x_max - x_min)
    map_y = y_max - (pixel_y / height) * (y_max - y_min)

    return QgsPointXY(map_x, map_y)
