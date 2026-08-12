"""Which QGIS raster providers must be rendered instead of read from a file.

One list, and nothing heavy to import to reach it. It used to live beside the
crop reader in ``feature_encoder``, which pulls numpy in at module top, so the
dock kept a hand-copied second list rather than pay that on a panel refresh.
The two drifted: the copy left out the fileless local providers, and the panel
called a layer local that the reader treats exactly like a web service.
"""
from __future__ import annotations

# Online/remote raster providers that need rendering before encoding. QGIS
# serves WMTS and XYZ through the "wms" provider, so those two names never
# come back from provider.name(); they are kept because a build or a future
# QGIS could report them, and an extra name here costs nothing.
ONLINE_PROVIDERS = frozenset(["wms", "wmts", "xyz", "arcgismapserver", "wcs"])

# Local providers that serve pixels but have no file behind them, so the
# windowed file read has nothing to open. A PostGIS raster lives in a
# database, a virtual raster is an expression over other layers. Both are
# valid QgsRasterLayers, both are offered in the layer picker, and both used
# to reach the file reader and fail there.
FILELESS_LOCAL_PROVIDERS = frozenset(["postgresraster", "virtualraster"])

# The real question every caller asks: must this layer be rendered through
# QGIS instead of read from its own file? Online or fileless, the answer and
# the code path are the same.
CANVAS_RENDERED_PROVIDERS = ONLINE_PROVIDERS | FILELESS_LOCAL_PROVIDERS
