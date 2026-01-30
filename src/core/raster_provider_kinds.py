







from __future__ import annotations





ONLINE_PROVIDERS = frozenset(["wms", "wmts", "xyz", "arcgismapserver", "wcs"])






FILELESS_LOCAL_PROVIDERS = frozenset(["postgresraster", "virtualraster"])




CANVAS_RENDERED_PROVIDERS = ONLINE_PROVIDERS | FILELESS_LOCAL_PROVIDERS
