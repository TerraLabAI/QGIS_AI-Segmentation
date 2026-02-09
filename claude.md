# QGIS AI Segmentation Plugin

A QGIS plugin for AI-powered image segmentation using Meta's Segment Anything Model (SAM).

## Architecture Overview

### Core Components

- **`src/ui/ai_segmentation_plugin.py`**: Main plugin class, handles QGIS integration, tool management, and segmentation workflow coordination
- **`src/ui/ai_segmentation_dockwidget.py`**: Qt dock widget with all UI elements (dependency management, model download, segmentation controls, mode switching)
- **`src/ui/ai_segmentation_maptool.py`**: Custom QgsMapTool for handling map clicks (positive/negative points) and keyboard shortcuts

### Segmentation Modes

The plugin has two modes:

1. **Simple Mode (default)**:
   - Segment one object at a time
   - Each export creates a new layer named `{RasterName}_polygon_{counter}`
   - No "Save polygon" button - only "Export to layer"
   - After export, returns to initial state
   - Blue info box: "Simple mode: one element per export. For multiple elements in one layer, use Batch mode."

2. **Batch Mode**:
   - Activated via "Batch mode" checkbox
   - Save multiple polygons, then export all together in one layer
   - "Save polygon" button visible to add current selection to collection
   - Shows polygon count badge
   - Blue info box: "Batch mode: save multiple polygons, then export all in one layer."

### Terminology

The plugin uses specific terminology for clarity:
- **Selection** = temporary working state (the current AI-generated mask before saving)
- **Polygon** = saved items (after clicking "Save polygon" or exporting)

UI strings follow this pattern:
- "Refine selection" (temporary state)
- "Save polygon" (action to save)
- "Export polygon(s) to layer" (final export)

### Key Classes

- `AISegmentationPlugin`: Main plugin entry point
- `AISegmentationDockWidget`: UI widget with signals for all user actions
- `AISegmentationMapTool`: Map interaction tool with point markers

### Signal Flow

1. User clicks "Start AI Segmentation" -> `start_segmentation_requested` signal
2. User clicks on map -> `positive_click` or `negative_click` signal
3. Plugin runs SAM inference -> mask displayed as QgsRubberBand
4. User clicks "Export to layer" -> `export_layer_requested` signal
5. In Simple mode: creates new layer, resets session
6. In Batch mode: "Save polygon" -> `save_polygon_requested`, then "Export to layer" -> exports all saved polygons

### State Variables (DockWidget)

- `_batch_mode`: Boolean, False = Simple mode (default)
- `_segmentation_active`: Whether segmentation session is active
- `_has_mask`: Whether current mask/selection exists
- `_saved_polygon_count`: Number of saved polygons (Batch mode)
- `_refine_expanded`: Collapsed state of refine panel
- `_positive_count`, `_negative_count`: Point counts for UI

### Refine Panel

Collapsible "Refine selection" panel with parameters displayed in a bordered box:
- **Expand/Contract**: Dilate/erode selection boundaries (-30 to +30 pixels)
- **Simplify outline**: Douglas-Peucker simplification tolerance (0-20)
- **Fill holes**: Checkbox to fill interior holes
- **Min. region size**: Remove small disconnected regions (0-10000 px²)

Settings applied in real-time to preview and export.

### Dependencies

- PyTorch >= 2.0
- segment-anything
- rasterio, pandas, numpy
- Installed in isolated venv at `src/venv_py3.12/`

### Model

- SAM ViT-B checkpoint (~375MB)
- Stored in `src/checkpoints/sam_vit_b_01ec64.pth`
- Auto-detects GPU (CUDA/MPS) or falls back to CPU

## Development Notes

- All UI text must be in English in the source code
- Buttons hidden when not in segmentation mode (not disabled-but-visible)
- Use `_update_button_visibility()` to manage button states based on mode and session state
- QSettings used for persisting tutorial flags: `AI_Segmentation/tutorial_simple_shown`, `AI_Segmentation/tutorial_batch_shown`
- Never write comments in french, only write in english

## Refine Panel Defaults (KEEP IN SYNC)

Current defaults: `expand=0, simplify=4, fill_holes=False, min_area=100`

5 locations must match: plugin `__init__`, `_reset_session()`, `_restore_last_saved_mask` fallbacks, dockwidget `_setup_refine_panel`, `reset_refine_sliders`.

## Common Pitfalls

- **Subprocess stderr**: Never use `stderr=subprocess.PIPE` without draining it — use `DEVNULL` or a temp file to avoid deadlocks
- **`os.replace()` not `os.rename()`**: rename fails on Windows if dest exists
- **`os.path.normcase()`** before comparing paths (Windows case-insensitive)
- **`encoding='utf-8'`** on all `open()` calls
- **`Tuple[bool, str]`** from typing, not `tuple[bool, str]` (needs Python 3.9+)
- **`blockSignals(True/False)`** when setting widget values programmatically
- **Disconnect `QgsProject.instance()` signals** in `unload()`

## Internationalization (i18n) - IMPORTANT

The plugin supports multiple languages: French (fr), Portuguese Brazil (pt_BR), Spanish (es).

**When modifying any UI string, you MUST update both the code AND all translation files.**

### Architecture

- `src/core/i18n.py`: Contains the `tr()` function - parses `.ts` XML directly at runtime (no binary needed)
- `i18n/ai_segmentation_fr.ts`: French translation file
- `i18n/ai_segmentation_pt_BR.ts`: Portuguese (Brazil) translation file
- `i18n/ai_segmentation_es.ts`: Spanish translation file

### How to add/modify a UI string

1. **In the Python code**, wrap the string with `tr()`:
   ```python
   from ..core.i18n import tr

   # Instead of:
   button.setText("My button text")

   # Write:
   button.setText(tr("My button text"))
   ```

2. **In ALL translation files** (`fr.ts`, `pt_BR.ts`, `es.ts`), add a new `<message>` block inside `<context><name>AISegmentation</name>`:
   ```xml
   <message>
       <source>My button text</source>
       <translation>Mon texte de bouton</translation>
   </message>
   ```

3. **Commit all .ts files** - no compilation needed, XML is parsed at runtime

### Terms to keep in English (do NOT translate)

- "AI Segmentation" (product name)
- "SAM" (technical term)
- "TerraLab" (company name)
- "Batch mode" (feature name - keep "Batch" in English)
- "Export" (keep as-is, commonly understood)
- "Checkpoint" (technical term)
- Package names: PyTorch, rasterio, pandas, etc.

### String formatting with variables

Use `.format()` for dynamic strings:
```python
# Code:
tr("Export {count} polygon(s)").format(count=5)

# Translation file:
<source>Export {count} polygon(s)</source>
<translation>Exporter {count} polygone(s)</translation>
```

### User experience

- **No binaries**: Complies with QGIS plugin repository rules
- **Automatic language detection**: Plugin reads QGIS locale settings
- **Fallback**: If translation is missing, English text is shown
- **Works on all OS**: Pure Python XML parsing, no Qt tools needed

## Code Quality Rules (PEP8/Flake8)

- **No unused imports (F401)**: Remove any imported modules/classes that are not used in the file. Always verify `sys`, `os`, etc. are actually referenced before importing.
- **No unused variables (F841)**: Don't assign variables that are never referenced; delete them or use `_` prefix only if needed for unpacking. In `except` clauses, if you don't use the exception variable, omit it: write `except RuntimeError:` not `except RuntimeError as e:`
- **Too many blank lines (E303)**: Maximum 2 blank lines between top-level definitions, 1 blank line between methods inside a class
- **Trailing whitespace on blank lines (W291/W293)**: Blank lines must be completely empty — no spaces or tabs. This includes blank lines inside docstrings, between code blocks, etc. Always strip trailing whitespace.
- **Whitespace around operators (E226)**: Always use spaces around arithmetic operators: `y - 1` not `y-1`
- **Import order**: Standard library first, then third-party, then local imports. When an import must come after runtime setup (e.g. `os.environ.setdefault()` before `import numpy`), add `# noqa: E402` to suppress the linter warning
- **Line length**: Keep lines under 120 characters
- **Line breaks with binary operators (W503/W504)**: The linter flags both W503 (break before operator) and W504 (break after operator). Use `.format()` or parentheses instead of line-continuation with operators:
  ```python
  # WRONG (W503 - break before operator):
  message = tr("Line one") + "\n"
      + tr("Line two")

  # WRONG (W504 - break after operator):
  message = tr("Line one") + "\n" +
      tr("Line two")

  # CORRECT (use .format() to avoid line-continuation):
  message = "{}\n{}".format(
      tr("Line one"),
      tr("Line two"))

  # CORRECT (for boolean expressions, use intermediate variables):
  # WRONG (W503):
  return (bounds[0] <= x <= bounds[1]
          and bounds[2] <= y <= bounds[3])
  # CORRECT:
  in_x = bounds[0] <= x <= bounds[1]
  in_y = bounds[2] <= y <= bounds[3]
  return in_x and in_y
  ```
- **Undefined names (F821)**: Never reference a variable before it is defined. Build lists/dicts incrementally using the correct name at each stage:
  ```python
- **Global keyword (F824)**: Only use `global` when reassigning a module-level variable. Not needed when just modifying a dict/list:
  ```python
  # WRONG (F824 - global not needed for dict modification):
  _translations = {}
  def foo():
      global _translations  # NOT needed
      _translations["key"] = "value"  # This modifies, not reassigns

  # CORRECT:
  _loaded = False
  def foo():
      global _loaded  # Needed because we reassign
      _loaded = True
  ```

## Security Rules (Bandit)

- **XML Parsing (B314)**: Never use `xml.etree.ElementTree.parse()` without protection. Use `defusedxml.defuse_stdlib()` to patch the standard library:
  ```python
  # WRONG (vulnerable to XML attacks - Bandit flags this):
  import xml.etree.ElementTree as ET
  tree = ET.parse(file_path)

  # CORRECT (secure - patch stdlib then use normally):
  try:
      import defusedxml
      defusedxml.defuse_stdlib()
  except ImportError:
      pass  # Only acceptable for local trusted plugin files
  import xml.etree.ElementTree as ET  # noqa: E402
  tree = ET.parse(file_path)
  ```
- **Hardcoded credentials**: Never hardcode passwords, API keys, or secrets in code. Also avoid credential-like patterns in docstrings/comments (e.g. `user:pass@host`) — use generic descriptions instead.
- **Shell injection**: Use subprocess with list arguments, not shell=True with string interpolation

