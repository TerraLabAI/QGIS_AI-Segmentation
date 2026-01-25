# Architecture Recommendations - Multi-Platform Decoupling

## Overview

This document provides recommendations for refactoring the QGIS AI-Segmentation plugin to achieve **true separation of business logic from QGIS-specific code**, enabling:

1. **Decoupling** - No need to use QGIS's Python, or even Python at all
2. **Multi-platform** - Intranet / SaaS / Local process
3. **Multi-client** - Single codebase for QGIS, ArcGIS, TerraLab3D
4. **Click and play** - Easy installation
5. **Single QGIS plugin** - Unified maintenance for all features

---

## Current State Assessment

### ✅ Strengths

1. **Subprocess Isolation**: Workers (`encoding_worker.py`, `prediction_worker.py`) are well-isolated with no QGIS dependencies
2. **Virtual Environment Management**: Clean dependency isolation strategy
3. **Process Communication**: JSON-based IPC protocol is platform-agnostic

### ❌ Critical Issues

**Current Decoupling Level: ~30%**

The main blocker is **QGIS imports throughout core modules**, preventing reuse across platforms:

| Module | QGIS Dependencies | Impact |
|--------|------------------|--------|
| `core/feature_encoder.py` | `QgsMessageLog, Qgis` | ❌ High - Used for all encoding operations |
| `core/sam_predictor.py` | `QgsMessageLog, Qgis` | ❌ High - Core prediction logic |
| `core/polygon_exporter.py` | `QgsVectorLayer, QgsGeometry, ...` | ❌ Critical - Returns QGIS types |
| `core/feature_dataset.py` | `QgsMessageLog, Qgis` | ❌ High - Feature loading |
| `core/geo_utils.py` | `QgsCoordinateReferenceSystem, ...` | ❌ Critical - All geo operations |

---

## Target Architecture

### Recommended Structure

```
QGIS_AI-Segmentation/
├── core/                          # ✅ PURE BUSINESS LOGIC (no QGIS!)
│   ├── __init__.py
│   ├── abstractions.py            # Interfaces/abstract classes
│   ├── logger.py                  # Logging abstraction
│   ├── segmentation_engine.py     # Core segmentation orchestration
│   ├── feature_processor.py      # Feature encoding/decoding logic
│   ├── prediction_service.py      # SAM prediction service
│   ├── cache_manager.py           # Feature cache management
│   ├── coordinate_math.py         # Pure coordinate transformations
│   └── types.py                   # Platform-agnostic data types
│
├── adapters/                      # ✅ PLATFORM ADAPTERS
│   ├── __init__.py
│   ├── base.py                    # Base adapter interfaces
│   ├── qgis_adapter.py            # QGIS-specific implementations
│   ├── qgis_logger.py             # QGIS logging adapter
│   ├── qgis_geometry.py            # QGIS geometry wrapper
│   ├── arcgis_adapter.py          # ArcGIS implementations (future)
│   ├── terralab3d_adapter.py      # TerraLab3D implementations (future)
│   └── standalone_adapter.py      # CLI/standalone implementations
│
├── workers/                       # ✅ ALREADY GOOD (mostly)
│   ├── __init__.py
│   ├── encoding_worker.py        # ✅ No QGIS deps
│   └── prediction_worker.py      # ✅ No QGIS deps
│
├── qgis_plugin/                   # ✅ QGIS-SPECIFIC LAYER
│   ├── __init__.py
│   ├── plugin.py                  # Thin QGIS plugin wrapper
│   ├── dockwidget.py              # QGIS UI only
│   ├── maptool.py                 # QGIS map tool only
│   └── coordinator.py             # Bridges adapter → QGIS UI
│
└── cli/                           # ✅ STANDALONE CLI (future)
    ├── __init__.py
    └── main.py                    # Uses standalone_adapter
```

---

## Refactoring Steps

### Phase 1: Extract Abstractions (High Priority)

#### Step 1.1: Create Logging Abstraction

**File: `core/logger.py`**

```python
from abc import ABC, abstractmethod
from enum import Enum

class LogLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Logger(ABC):
    """Platform-agnostic logging interface."""
    
    @abstractmethod
    def info(self, message: str, context: str = None):
        """Log an informational message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, context: str = None):
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, context: str = None):
        """Log an error message."""
        pass
    
    @abstractmethod
    def critical(self, message: str, context: str = None):
        """Log a critical error message."""
        pass

class NullLogger(Logger):
    """No-op logger for testing."""
    def info(self, message: str, context: str = None): pass
    def warning(self, message: str, context: str = None): pass
    def error(self, message: str, context: str = None): pass
    def critical(self, message: str, context: str = None): pass
```

**File: `adapters/qgis_logger.py`**

```python
from qgis.core import QgsMessageLog, Qgis
from core.logger import Logger

class QGISLogger(Logger):
    """QGIS-specific logging implementation."""
    
    def __init__(self, context: str = "AI Segmentation"):
        self.context = context
    
    def info(self, message: str, context: str = None):
        QgsMessageLog.logMessage(
            message, 
            context or self.context, 
            level=Qgis.Info
        )
    
    def warning(self, message: str, context: str = None):
        QgsMessageLog.logMessage(
            message, 
            context or self.context, 
            level=Qgis.Warning
        )
    
    def error(self, message: str, context: str = None):
        QgsMessageLog.logMessage(
            message, 
            context or self.context, 
            level=Qgis.Critical
        )
    
    def critical(self, message: str, context: str = None):
        QgsMessageLog.logMessage(
            message, 
            context or self.context, 
            level=Qgis.Critical
        )
```

#### Step 1.2: Create Geometry Abstraction

**File: `core/abstractions.py`**

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

class Geometry(ABC):
    """Platform-agnostic geometry interface."""
    
    @abstractmethod
    def as_wkt(self) -> str:
        """Return geometry as WKT string."""
        pass
    
    @abstractmethod
    def as_geojson(self) -> dict:
        """Return geometry as GeoJSON dict."""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if geometry is empty."""
        pass
    
    @abstractmethod
    def area(self) -> float:
        """Calculate geometry area."""
        pass

class VectorLayer(ABC):
    """Platform-agnostic vector layer interface."""
    
    @abstractmethod
    def add_feature(self, geometry: Geometry, attributes: Dict[str, Any]) -> bool:
        """Add a feature to the layer."""
        pass
    
    @abstractmethod
    def set_crs(self, crs: str):
        """Set the coordinate reference system."""
        pass
    
    @abstractmethod
    def get_crs(self) -> Optional[str]:
        """Get the coordinate reference system."""
        pass

class GISAdapter(ABC):
    """Platform adapter interface for GIS operations."""
    
    @abstractmethod
    def geometry_from_wkt(self, wkt: str) -> Geometry:
        """Create geometry from WKT string."""
        pass
    
    @abstractmethod
    def geometry_from_geojson(self, geojson: dict) -> Geometry:
        """Create geometry from GeoJSON dict."""
        pass
    
    @abstractmethod
    def transform_point(
        self, 
        point: Tuple[float, float], 
        src_crs: str, 
        dst_crs: str
    ) -> Tuple[float, float]:
        """Transform point coordinates between CRS."""
        pass
    
    @abstractmethod
    def transform_geometry(
        self,
        geometry: Geometry,
        src_crs: str,
        dst_crs: str
    ) -> Geometry:
        """Transform geometry between CRS."""
        pass
    
    @abstractmethod
    def create_vector_layer(
        self, 
        name: str, 
        geometry_type: str,
        crs: Optional[str] = None
    ) -> VectorLayer:
        """Create a new vector layer."""
        pass
    
    @abstractmethod
    def get_raster_info(self, raster_path: str) -> Dict[str, Any]:
        """Get raster metadata (extent, CRS, dimensions)."""
        pass
```

**File: `adapters/qgis_geometry.py`**

```python
from typing import Tuple
from qgis.core import QgsGeometry, QgsPointXY, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
from core.abstractions import Geometry, GISAdapter, VectorLayer

class QGISGeometry(Geometry):
    """QGIS geometry wrapper."""
    
    def __init__(self, qgs_geometry: QgsGeometry):
        self._geom = qgs_geometry
    
    def as_wkt(self) -> str:
        return self._geom.asWkt()
    
    def as_geojson(self) -> dict:
        # Convert QgsGeometry to GeoJSON
        # Implementation details...
        pass
    
    def is_empty(self) -> bool:
        return self._geom.isEmpty()
    
    def area(self) -> float:
        return self._geom.area()
    
    @property
    def qgs_geometry(self) -> QgsGeometry:
        """Access underlying QGIS geometry (for QGIS-specific code only)."""
        return self._geom

class QGISAdapter(GISAdapter):
    """QGIS-specific GIS adapter implementation."""
    
    def geometry_from_wkt(self, wkt: str) -> QGISGeometry:
        geom = QgsGeometry.fromWkt(wkt)
        return QGISGeometry(geom)
    
    def transform_point(
        self, 
        point: Tuple[float, float], 
        src_crs: str, 
        dst_crs: str
    ) -> Tuple[float, float]:
        src_crs_obj = QgsCoordinateReferenceSystem(src_crs)
        dst_crs_obj = QgsCoordinateReferenceSystem(dst_crs)
        
        transform = QgsCoordinateTransform(
            src_crs_obj, 
            dst_crs_obj, 
            QgsProject.instance()
        )
        
        qgs_point = QgsPointXY(point[0], point[1])
        transformed = transform.transform(qgs_point)
        return (transformed.x(), transformed.y())
    
    # ... implement other methods
```

### Phase 2: Refactor Core Modules

#### Step 2.1: Refactor `polygon_exporter.py`

**Before (Current):**
```python
from qgis.core import QgsGeometry, QgsVectorLayer, ...

def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0
) -> List[QgsGeometry]:  # ❌ QGIS-specific return type
    # ... uses QgsMessageLog, QgsGeometry directly
```

**After (Refactored):**
```python
from typing import List
from .abstractions import Geometry, GISAdapter
from .logger import Logger

def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    adapter: GISAdapter,      # ✅ Injected dependency
    logger: Logger,            # ✅ Injected dependency
    simplify_tolerance: float = 0.0
) -> List[Geometry]:          # ✅ Returns abstract Geometry
    """Convert mask to polygons using platform adapter."""
    if mask is None or mask.sum() == 0:
        logger.warning("Empty or None mask")
        return []
    
    # Pure business logic - no QGIS imports!
    # Use adapter for geometry operations
    geometries = []
    # ... implementation using adapter methods
    
    return geometries
```

#### Step 2.2: Refactor `feature_encoder.py`

**Before:**
```python
from qgis.core import QgsMessageLog, Qgis

def encode_raster_to_features(...):
    QgsMessageLog.logMessage(...)  # ❌ Direct QGIS logging
```

**After:**
```python
from .logger import Logger

def encode_raster_to_features(
    ...,
    logger: Logger  # ✅ Injected logger
):
    logger.info("Starting encoding worker subprocess...")
    # ... rest of implementation
```

#### Step 2.3: Refactor `sam_predictor.py`

**Before:**
```python
from qgis.core import QgsMessageLog, Qgis

class SamPredictorNoImgEncoder:
    def _start_worker(self):
        QgsMessageLog.logMessage(...)  # ❌ Direct QGIS logging
```

**After:**
```python
from .logger import Logger

class SamPredictorNoImgEncoder:
    def __init__(self, sam_config: dict, logger: Logger):
        self.logger = logger  # ✅ Injected logger
    
    def _start_worker(self):
        self.logger.info("Starting prediction worker...")
```

#### Step 2.4: Refactor `feature_dataset.py`

**Before:**
```python
from qgis.core import QgsMessageLog, Qgis

class FeatureDataset:
    def _load_index(self):
        QgsMessageLog.logMessage(...)  # ❌ Direct QGIS logging
```

**After:**
```python
from .logger import Logger

class FeatureDataset:
    def __init__(self, root: str, cache: bool = True, logger: Logger = None):
        self.logger = logger or NullLogger()  # ✅ Optional logger
    
    def _load_index(self):
        self.logger.info(f"Loaded index from: {csv_path}")
```

### Phase 3: Create Platform Coordinator

**File: `qgis_plugin/coordinator.py`**

```python
from typing import Optional
from qgis.core import QgsRasterLayer, QgsVectorLayer
from adapters.qgis_adapter import QGISAdapter
from adapters.qgis_logger import QGISLogger
from core.segmentation_engine import SegmentationEngine
from core.feature_processor import FeatureProcessor
from core.prediction_service import PredictionService

class QGISCoordinator:
    """Coordinates between QGIS UI and business logic."""
    
    def __init__(self):
        # Initialize adapters
        self.adapter = QGISAdapter()
        self.logger = QGISLogger()
        
        # Initialize business logic services
        self.engine = SegmentationEngine(
            adapter=self.adapter,
            logger=self.logger
        )
        self.feature_processor = FeatureProcessor(
            adapter=self.adapter,
            logger=self.logger
        )
        self.prediction_service = PredictionService(
            adapter=self.adapter,
            logger=self.logger
        )
    
    def process_raster_layer(self, layer: QgsRasterLayer):
        """Process a QGIS raster layer through business logic."""
        # Convert QGIS layer to abstract representation
        raster_info = self.adapter.get_raster_info_from_layer(layer)
        
        # Call business logic
        result = self.engine.process(raster_info)
        
        # Convert result back to QGIS layer
        return self.adapter.create_qgis_vector_layer(result)
```

### Phase 4: Refactor Plugin Class

**File: `qgis_plugin/plugin.py` (Refactored)**

```python
from qgis.gui import QgisInterface
from .coordinator import QGISCoordinator
from .dockwidget import AISegmentationDockWidget
from .maptool import AISegmentationMapTool

class AISegmentationPlugin:
    """Thin QGIS plugin wrapper - delegates to coordinator."""
    
    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.coordinator = QGISCoordinator()  # ✅ Business logic coordinator
        self.dock_widget = None
        self.map_tool = None
    
    def initGui(self):
        # UI setup only - no business logic
        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())
        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        
        # Connect UI signals to coordinator
        self.dock_widget.start_segmentation_requested.connect(
            self._on_start_segmentation
        )
    
    def _on_start_segmentation(self, layer: QgsRasterLayer):
        # Delegate to coordinator
        result = self.coordinator.process_raster_layer(layer)
        # Update UI
        self._update_ui_with_result(result)
```

---

## Migration Strategy

### Priority Order

#### 🔴 **Critical (Do First)**
1. Extract logging abstraction (`core/logger.py`)
2. Replace all `QgsMessageLog` calls with logger abstraction
3. Extract geometry abstraction (`core/abstractions.py`)
4. Refactor `polygon_exporter.py` to use abstractions

#### 🟡 **High Priority**
5. Refactor `feature_encoder.py` to use logger
6. Refactor `sam_predictor.py` to use logger
7. Refactor `feature_dataset.py` to use logger
8. Create `GISAdapter` interface and `QGISAdapter` implementation

#### 🟢 **Medium Priority**
9. Create `SegmentationEngine` class to encapsulate business logic
10. Refactor plugin class to use coordinator pattern
11. Extract coordinate transformation logic to pure math module

#### 🔵 **Low Priority (Future)**
12. Create ArcGIS adapter skeleton
13. Create TerraLab3D adapter skeleton
14. Create standalone CLI using standalone adapter

### Incremental Migration Approach

1. **Week 1**: Extract abstractions (logging, geometry)
2. **Week 2**: Refactor one core module at a time (start with `polygon_exporter.py`)
3. **Week 3**: Create coordinator and refactor plugin
4. **Week 4**: Testing and cleanup

### Backward Compatibility

During migration, maintain backward compatibility:

```python
# core/polygon_exporter.py (Transitional)
def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    adapter: GISAdapter = None,  # Optional during transition
    logger: Logger = None,       # Optional during transition
    simplify_tolerance: float = 0.0
) -> List[Geometry]:
    # Auto-create adapters if not provided (for backward compat)
    if adapter is None:
        adapter = _get_default_adapter()  # Returns QGISAdapter if in QGIS
    if logger is None:
        logger = _get_default_logger()   # Returns QGISLogger if in QGIS
    
    # ... implementation
```

---

## Code Examples

### Example 1: Using Abstractions in Core Logic

```python
# core/segmentation_engine.py
from typing import List
from .abstractions import Geometry, GISAdapter
from .logger import Logger

class SegmentationEngine:
    """Core segmentation business logic - no QGIS dependencies."""
    
    def __init__(self, adapter: GISAdapter, logger: Logger):
        self.adapter = adapter
        self.logger = logger
    
    def process_mask(
        self, 
        mask: np.ndarray, 
        transform_info: dict
    ) -> List[Geometry]:
        """Process mask into geometries using adapter."""
        from .polygon_exporter import mask_to_polygons
        
        geometries = mask_to_polygons(
            mask=mask,
            transform_info=transform_info,
            adapter=self.adapter,
            logger=self.logger
        )
        
        self.logger.info(f"Generated {len(geometries)} polygons")
        return geometries
```

### Example 2: QGIS-Specific Implementation

```python
# qgis_plugin/coordinator.py
from qgis.core import QgsRasterLayer
from adapters.qgis_adapter import QGISAdapter
from adapters.qgis_logger import QGISLogger
from core.segmentation_engine import SegmentationEngine

class QGISCoordinator:
    def __init__(self):
        self.adapter = QGISAdapter()
        self.logger = QGISLogger()
        self.engine = SegmentationEngine(self.adapter, self.logger)
    
    def segment_raster(self, layer: QgsRasterLayer):
        # Convert QGIS layer to abstract representation
        raster_info = {
            'path': layer.source(),
            'extent': layer.extent(),
            'crs': layer.crs().authid(),
            # ... other metadata
        }
        
        # Process through business logic
        result = self.engine.process(raster_info)
        
        # Convert geometries back to QGIS
        qgs_geometries = [
            geom.qgs_geometry for geom in result.geometries
            if isinstance(geom, QGISGeometry)
        ]
        
        return qgs_geometries
```

### Example 3: Future ArcGIS Implementation

```python
# adapters/arcgis_adapter.py
from arcpy import env, Point, Polygon, SpatialReference
from core.abstractions import GISAdapter, Geometry

class ArcGISAdapter(GISAdapter):
    """ArcGIS-specific adapter implementation."""
    
    def geometry_from_wkt(self, wkt: str) -> Geometry:
        # Convert WKT to ArcGIS geometry
        # Implementation...
        pass
    
    def transform_point(
        self, 
        point: Tuple[float, float], 
        src_crs: str, 
        dst_crs: str
    ) -> Tuple[float, float]:
        # Use ArcGIS transformation
        # Implementation...
        pass
```

---

## Testing Strategy

### Unit Tests for Core Logic

```python
# tests/test_segmentation_engine.py
from core.segmentation_engine import SegmentationEngine
from adapters.standalone_adapter import StandaloneAdapter
from core.logger import NullLogger

def test_segmentation_engine():
    """Test business logic without QGIS."""
    adapter = StandaloneAdapter()
    logger = NullLogger()
    engine = SegmentationEngine(adapter, logger)
    
    # Test with mock data
    mask = np.array([[0, 1, 1], [1, 1, 0]])
    result = engine.process_mask(mask, {...})
    
    assert len(result) > 0
```

### Integration Tests for Adapters

```python
# tests/test_qgis_adapter.py
from adapters.qgis_adapter import QGISAdapter

def test_qgis_adapter():
    """Test QGIS adapter implementation."""
    adapter = QGISAdapter()
    geom = adapter.geometry_from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    assert not geom.is_empty()
```

---

## Benefits After Refactoring

### ✅ Achieved Goals

1. **Decoupling**: Core logic has zero QGIS dependencies
2. **Multi-platform**: Same core code works in QGIS, ArcGIS, TerraLab3D
3. **Multi-client**: Single codebase with platform adapters
4. **Easy installation**: Core can be installed as standalone package
5. **Maintenance**: Business logic changes don't require platform-specific updates

### 📊 Metrics

- **Before**: ~30% decoupled (workers only)
- **After**: ~90% decoupled (core + adapters pattern)

### 🔄 Reusability

```python
# Same core code works everywhere:

# In QGIS
coordinator = QGISCoordinator(adapter=QGISAdapter(), logger=QGISLogger())

# In ArcGIS (future)
coordinator = ArcGISCoordinator(adapter=ArcGISAdapter(), logger=ArcGISLogger())

# Standalone CLI
engine = SegmentationEngine(adapter=StandaloneAdapter(), logger=ConsoleLogger())
```

---

## Next Steps

1. **Review** this document with the team
2. **Prioritize** refactoring tasks based on current needs
3. **Create** GitHub issues for each phase
4. **Start** with Phase 1 (abstractions) - highest impact, lowest risk
5. **Test** incrementally after each module refactoring

---

## Questions & Considerations

### Q: Won't this add complexity?

**A**: Initially yes, but it pays off:
- Core logic becomes testable without QGIS
- Platform-specific bugs are isolated to adapters
- New platforms require only adapter implementation

### Q: What about performance?

**A**: Minimal impact:
- Adapter calls are thin wrappers
- Business logic performance unchanged
- Potential for optimization in adapter layer

### Q: How do we handle QGIS-specific features?

**A**: Through adapter methods:
```python
# In adapter
def get_raster_renderer(self, layer):
    # QGIS-specific rendering logic
    pass
```

---

## References

- [Adapter Pattern](https://en.wikipedia.org/wiki/Adapter_pattern)
- [Dependency Injection](https://en.wikipedia.org/wiki/Dependency_injection)
- [Separation of Concerns](https://en.wikipedia.org/wiki/Separation_of_concerns)

---

**Last Updated**: 2024
**Status**: Recommendations - Not yet implemented
