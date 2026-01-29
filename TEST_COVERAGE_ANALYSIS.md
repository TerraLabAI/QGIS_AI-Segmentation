# Test Coverage Analysis

## Executive Summary

**Current Test Coverage: 0%**

This codebase has **no automated tests**. There are no test files, no test configuration (pytest.ini, setup.cfg), no CI/CD pipeline, and no testing dependencies. This represents a significant risk for a plugin that handles complex geometry operations, subprocess communication, and machine learning inference.

---

## Codebase Overview

| Category | Files | Lines of Code |
|----------|-------|---------------|
| Core Logic | 13 modules | ~3,565 lines |
| UI Components | 4 modules | ~1,628 lines |
| Worker Processes | 2 modules | ~568 lines |
| **Total** | **19 modules** | **~5,761 lines** |

---

## Priority Areas for Test Coverage

### Priority 1: Critical - Pure Functions with High Risk

These modules contain pure algorithmic logic that is highly testable and failure-prone:

#### 1. `polygon_exporter.py` (433 lines) - **HIGHEST PRIORITY**

This module converts ML model output (binary masks) to GIS-compatible polygons. Bugs here cause data loss or corruption.

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `geojson_to_wkt()` | 82-103 | High | Easy - pure function |
| `mask_to_polygons()` | 106-163 | Critical | Medium - requires mock rasterio |
| `mask_to_polygons_fallback()` | 165-211 | High | Medium - pure geometry |
| `find_contours()` | 213-252 | High | Easy - pure numpy |
| `trace_contour()` | 255-302 | High | Easy - pure algorithm |
| `pixel_to_map_coords()` | 305-336 | Medium | Easy - pure math |
| `apply_mask_refinement()` | 339-362 | Medium | Easy - pure numpy |
| `_numpy_dilate()` | 365-378 | Low | Easy - pure numpy |
| `_numpy_erode()` | 382-395 | Low | Easy - pure numpy |

**Recommended tests:**
```python
# test_polygon_exporter.py

def test_geojson_to_wkt_polygon():
    """Test simple polygon conversion."""
    geojson = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
    }
    wkt = geojson_to_wkt(geojson)
    assert wkt == "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

def test_geojson_to_wkt_multipolygon():
    """Test multi-polygon conversion."""
    # ...

def test_geojson_to_wkt_empty():
    """Test empty/invalid input handling."""
    assert geojson_to_wkt({}) == ""
    assert geojson_to_wkt({"type": "Point"}) == ""

def test_find_contours_simple_square():
    """Test contour finding on a simple mask."""
    mask = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    contours = find_contours(mask)
    assert len(contours) == 1
    assert len(contours[0]) >= 4  # At least 4 points for a square

def test_find_contours_disconnected_regions():
    """Test multiple disconnected mask regions."""
    # ...

def test_numpy_dilate_expands_mask():
    """Test that dilation expands mask boundaries."""
    mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    dilated = _numpy_dilate(mask, iterations=1)
    # Center pixel plus 4-connected neighbors
    assert dilated.sum() == 5

def test_numpy_erode_shrinks_mask():
    """Test that erosion shrinks mask boundaries."""
    mask = np.ones((5, 5), dtype=np.uint8)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    eroded = _numpy_erode(mask, iterations=1)
    assert eroded.sum() < mask.sum()

def test_pixel_to_map_coords_with_bbox():
    """Test pixel to map coordinate transformation."""
    transform_info = {
        "bbox": [0.0, 100.0, 0.0, 100.0],  # minx, maxx, miny, maxy
        "img_shape": (100, 100)
    }
    mx, my = pixel_to_map_coords(50, 50, transform_info)
    assert mx == 50.0
    assert my == 50.0

def test_mask_to_polygons_empty_mask():
    """Test that empty masks return empty list."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    result = mask_to_polygons(empty_mask, {})
    assert result == []
```

#### 2. `geo_utils.py` (133 lines) - **HIGH PRIORITY**

Coordinate transformation logic. Errors here cause mis-positioned features.

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `ImageCRSManager` class | 14-55 | High | Medium - requires QGIS mocking |
| `geo_to_pixel()` | 58-65 | Medium | Easy with mock transform |
| `pixel_to_geo()` | 68-75 | Medium | Easy - pure math |
| `get_raster_info()` | 78-97 | Medium | Medium - requires QGIS layer |
| `map_point_to_image_coords()` | 100-117 | High | Medium - requires CRS objects |
| `image_coords_to_map_point()` | 120-132 | High | Easy - pure math |

**Recommended tests:**
```python
# test_geo_utils.py

def test_pixel_to_geo_identity_transform():
    """Test pixel to geo with identity-like transform."""
    # Affine transform: scale=1, no rotation, origin at 0,0

def test_map_point_to_image_coords_same_crs():
    """Test coordinate transformation when CRS matches."""

def test_image_coords_to_map_point_corners():
    """Test that image corners map to extent corners."""
    extent = (0.0, 0.0, 100.0, 100.0)  # xmin, ymin, xmax, ymax
    img_size = (100, 100)  # height, width

    # Top-left pixel (0,0) should map to (xmin, ymax)
    point = image_coords_to_map_point(0, 0, extent, img_size)
    assert point.x() == 0.0
    assert point.y() == 100.0
```

---

### Priority 2: High - Subprocess Communication

The plugin uses subprocess workers for ML inference to avoid polluting QGIS's Python environment. This is complex and error-prone.

#### 3. `prediction_worker.py` (300 lines) - **HIGH PRIORITY**

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `encode_numpy_array()` | 212-213 | Medium | Easy - pure function |
| `decode_numpy_array()` | 216-219 | Medium | Easy - pure function |
| `send_response()` | 199-201 | Low | Easy - mock stdout |
| `main()` JSON protocol | 222-296 | Critical | Medium - requires stdin/stdout mocking |

**Recommended tests:**
```python
# test_prediction_worker.py

def test_encode_decode_roundtrip():
    """Test that encoding and decoding preserves array data."""
    original = np.random.rand(256, 256).astype(np.float32)
    encoded = encode_numpy_array(original)
    decoded = decode_numpy_array(encoded, list(original.shape), str(original.dtype))
    np.testing.assert_array_equal(original, decoded)

def test_encode_decode_different_dtypes():
    """Test encoding/decoding works for various dtypes."""
    for dtype in [np.float32, np.float64, np.int32, np.uint8, np.bool_]:
        arr = np.array([[1, 2], [3, 4]], dtype=dtype)
        encoded = encode_numpy_array(arr)
        decoded = decode_numpy_array(encoded, list(arr.shape), str(arr.dtype))
        np.testing.assert_array_equal(arr, decoded)

def test_worker_protocol_init():
    """Test worker initialization protocol."""
    # Mock stdin/stdout and verify JSON protocol
```

#### 4. `sam_predictor.py` (310 lines) - **HIGH PRIORITY**

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `_get_clean_env_for_venv()` | 12-21 | Low | Easy - pure function |
| `_get_subprocess_kwargs()` | 24-32 | Low | Easy - pure function |
| `SamPredictorNoImgEncoder._start_worker()` | 78-146 | Critical | Hard - requires subprocess mocking |
| `SamPredictorNoImgEncoder.cleanup()` | 148-172 | High | Medium |
| `SamPredictorNoImgEncoder.predict()` | 252-310 | Critical | Hard |

**Recommended tests:**
```python
# test_sam_predictor.py

def test_get_clean_env_removes_qgis_vars():
    """Test that QGIS-specific env vars are removed."""
    import os
    os.environ['PYTHONPATH'] = '/some/path'
    os.environ['QGIS_PREFIX_PATH'] = '/qgis'

    clean_env = _get_clean_env_for_venv()

    assert 'PYTHONPATH' not in clean_env
    assert 'QGIS_PREFIX_PATH' not in clean_env
    assert clean_env.get('PYTHONIOENCODING') == 'utf-8'

def test_get_subprocess_kwargs_windows():
    """Test Windows-specific subprocess options."""
    # Mock sys.platform == 'win32'
```

---

### Priority 3: Medium - Device Detection and Package Management

#### 5. `device_manager.py` (229 lines) - **MEDIUM PRIORITY**

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `get_optimal_device()` | 14-62 | Medium | Medium - requires torch mocking |
| `is_mps_available()` | 116-123 | Low | Easy with mocking |
| `is_cuda_available()` | 126-131 | Low | Easy with mocking |
| `get_device_capabilities()` | 134-165 | Medium | Medium |
| `reset_device_cache()` | 168-171 | Low | Easy |

**Recommended tests:**
```python
# test_device_manager.py

def test_reset_device_cache():
    """Test that cache reset clears cached values."""
    global _cached_device, _device_info
    _cached_device = "fake"
    _device_info = "fake info"

    reset_device_cache()

    assert _cached_device is None
    assert _device_info is None

def test_is_cuda_available_when_torch_missing():
    """Test graceful handling when torch not installed."""
    # Mock import failure

def test_get_device_capabilities_structure():
    """Test that capabilities dict has expected keys."""
    caps = get_device_capabilities()
    assert "platform" in caps
    assert "cpu_cores" in caps
    assert "recommended_device" in caps
```

#### 6. `dependency_manager.py` (493 lines) - **MEDIUM PRIORITY**

**Functions to test:**
| Function | Lines | Risk | Testability |
|----------|-------|------|-------------|
| `check_python_version_match()` | 57-78 | Medium | Easy |
| `is_package_installed()` | 132-137 | Low | Easy |
| `get_missing_dependencies()` | 184-198 | Medium | Easy |
| `all_dependencies_installed()` | 201-220 | Medium | Easy - filesystem mocking |
| `get_install_size_warning()` | 223-232 | Low | Easy |

**Recommended tests:**
```python
# test_dependency_manager.py

def test_get_install_size_warning_with_torch():
    """Test warning message when torch needs installation."""
    # Mock get_missing_dependencies to return torch

def test_get_install_size_warning_without_torch():
    """Test no warning when torch is installed."""

def test_all_dependencies_installed_empty_dir(tmp_path):
    """Test detection of empty libs directory."""
```

---

### Priority 4: Lower - Integration Tests

These tests require more setup but provide high confidence:

#### Integration Test Scenarios

1. **Full mask-to-polygon pipeline**
   - Create synthetic mask
   - Convert to polygons
   - Verify geometry validity
   - Check coordinate accuracy

2. **Worker communication protocol**
   - Start worker subprocess
   - Send init request
   - Send features
   - Request prediction
   - Verify response format

3. **Dependency installation flow**
   - Check missing packages
   - Install to temp directory
   - Verify isolation

---

## Recommended Test Infrastructure

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_polygon_exporter.py
│   ├── test_geo_utils.py
│   ├── test_device_manager.py
│   ├── test_dependency_manager.py
│   ├── test_prediction_worker.py
│   └── test_sam_predictor.py
├── integration/
│   ├── __init__.py
│   ├── test_mask_to_polygon_pipeline.py
│   └── test_worker_protocol.py
└── fixtures/
    ├── sample_masks/
    └── sample_geojson/
```

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (slower, may need QGIS)
    slow: Slow tests (ML model loading, etc.)
```

### Required Testing Dependencies

Add to `requirements-dev.txt`:
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
numpy>=1.20.0
```

### conftest.py Fixtures

```python
import pytest
import numpy as np

@pytest.fixture
def simple_square_mask():
    """4x4 mask with 2x2 square in center."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    return mask

@pytest.fixture
def transform_info_simple():
    """Simple transform info for testing."""
    return {
        "bbox": [0.0, 100.0, 0.0, 100.0],
        "img_shape": (100, 100),
        "crs": "EPSG:4326"
    }

@pytest.fixture
def mock_qgis_crs(mocker):
    """Mock QGIS CRS for tests that don't need real QGIS."""
    mock_crs = mocker.MagicMock()
    mock_crs.authid.return_value = "EPSG:4326"
    return mock_crs
```

---

## Risk Assessment

| Module | Current Risk | With Tests |
|--------|--------------|------------|
| polygon_exporter.py | **CRITICAL** - Silent data corruption possible | Low |
| geo_utils.py | **HIGH** - Mis-positioned features | Low |
| prediction_worker.py | **HIGH** - Protocol errors cause crashes | Medium |
| sam_predictor.py | **HIGH** - Subprocess failures | Medium |
| device_manager.py | **MEDIUM** - Wrong device selection | Low |
| dependency_manager.py | **MEDIUM** - Install failures | Low |
| UI modules | **LOW** - User-visible errors | Low |

---

## Implementation Roadmap

### Phase 1: Foundation (Immediate)
1. Create `tests/` directory structure
2. Add `pytest.ini` and `conftest.py`
3. Add `requirements-dev.txt`
4. Write tests for `geojson_to_wkt()` and numpy morphology functions

### Phase 2: Core Algorithms (Week 1-2)
1. Complete `polygon_exporter.py` unit tests
2. Complete `geo_utils.py` unit tests
3. Add encoding/decoding tests for worker protocol

### Phase 3: Subprocess Communication (Week 2-3)
1. Add mocked subprocess tests for `sam_predictor.py`
2. Add protocol tests for `prediction_worker.py`
3. Add device detection tests

### Phase 4: Integration & CI (Week 3-4)
1. Add integration tests for mask-to-polygon pipeline
2. Set up GitHub Actions CI workflow
3. Add code coverage reporting

---

## CI/CD Configuration

### `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov pytest-mock numpy

      - name: Run tests
        run: |
          pytest tests/unit -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

---

## Summary

The codebase has **zero test coverage** which is a significant technical debt. The highest-priority areas are:

1. **`polygon_exporter.py`** - Contains critical geometry algorithms that are highly testable
2. **`geo_utils.py`** - Coordinate transformations that affect spatial accuracy
3. **`prediction_worker.py`** - Subprocess protocol that can fail silently

Starting with pure functions like `geojson_to_wkt()`, `find_contours()`, and the numpy morphology operations would provide immediate value with minimal setup overhead.
