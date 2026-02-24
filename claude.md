# QGIS AI Segmentation Plugin

A QGIS plugin for AI-powered image segmentation using Meta's Segment Anything Model (SAM).

## Architecture Overview

### Core Components

- **`src/core/model_config.py`**: Central version-dependent constants (SAM1 vs SAM2, checkpoint URLs, package versions). Uses `sys.version_info >= (3, 10)` to select SAM2 or SAM1 fallback.
- **`src/ui/ai_segmentation_plugin.py`**: Main plugin class, handles QGIS integration, tool management, and segmentation workflow coordination
- **`src/ui/ai_segmentation_dockwidget.py`**: Qt dock widget with all UI elements (dependency management, model download, segmentation controls, mode switching)
- **`src/ui/ai_segmentation_maptool.py`**: Custom QgsMapTool for handling map clicks (positive/negative points) and keyboard shortcuts

### Dual SAM Model Support

The plugin supports two SAM models depending on Python version:
- **Python 3.10+** (QGIS 3.34+): SAM 2.1 Base Plus via `sam2` package (~323MB checkpoint)
- **Python 3.9** (QGIS 3.22/3.28): SAM ViT-B via `segment-anything` package (~375MB checkpoint)

All version branching is centralized in `src/core/model_config.py`. The `prediction_worker.py` detects at runtime which package is available. The parent-process wrapper (`sam_predictor.py`) handles the optional `input_size` field returned only by SAM1.

### Segmentation Workflow

The plugin uses a single workflow (batch mode is always on):
- Segment elements one by one using positive/negative clicks
- "Save polygon" button to add current selection to collection
- Export all saved polygons together in one layer
- Blue info box: "Segment one element at a time. You must save your polygon before selecting a new element. Export all saved polygons to a layer when finished."

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
4. User clicks "Save polygon" -> `save_polygon_requested` signal (adds to collection)
5. User clicks "Export to layer" -> `export_layer_requested` signal (exports all saved polygons)

### State Variables (DockWidget)

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

Version-dependent (see `src/core/model_config.py`):
- **Python 3.10+**: PyTorch >= 2.5.1, torchvision >= 0.20.1, sam2 >= 1.0
- **Python 3.9**: PyTorch >= 2.0.0, torchvision >= 0.15.0, segment-anything >= 1.0
- Common: rasterio, pandas, numpy
- Installed in isolated venv at `~/.qgis_ai_segmentation/venv_py3.*/`

### Model

- **Python 3.10+**: SAM 2.1 Base Plus checkpoint (~323MB), config `configs/sam2.1/sam2.1_hiera_b+.yaml`
- **Python 3.9**: SAM ViT-B checkpoint (~375MB), uses `sam_model_registry["vit_b"]`
- Stored in `~/.qgis_ai_segmentation/checkpoints/`
- Auto-detects GPU (CUDA/MPS) or falls back to CPU

## Development Notes

- All UI text must be in English in the source code
- Buttons hidden when not in segmentation mode (not disabled-but-visible)
- Use `_update_button_visibility()` to manage button states based on mode and session state
- QSettings used for persisting tutorial flag: `AI_Segmentation/tutorial_simple_shown`
- Never write comments in french, only write in english

## Refine Panel Defaults (KEEP IN SYNC)

Current defaults: `expand=0, simplify=3, fill_holes=False, min_area=100`

5 locations must match: plugin `__init__`, `_reset_session()`, `_restore_last_saved_mask` fallbacks, dockwidget `_setup_refine_panel`, `reset_refine_sliders`.

## Common Pitfalls

- **Subprocess stderr**: Never use `stderr=subprocess.PIPE` without draining it -- use `DEVNULL` or a temp file to avoid deadlocks
- **`os.replace()` not `os.rename()`**: rename fails on Windows if dest exists
- **`os.path.normcase()`** before comparing paths (Windows case-insensitive)
- **`encoding='utf-8'`** on all `open()` calls
- **`Tuple[bool, str]`** from typing, not `tuple[bool, str]` (needs Python 3.9+)
- **`blockSignals(True/False)`** when setting widget values programmatically (including `reset_refine_sliders`)
- **Disconnect `QgsProject.instance()` signals** in `unload()`

## Dark Theme Compatibility - IMPORTANT

QGIS supports light and dark themes. All UI styling MUST work in both.

- **NEVER use `palette(mid)` for text color** -- it is invisible on dark backgrounds. Use `palette(text)` instead for readable text in both themes.
- **For secondary/muted text**: use `palette(text)` with a smaller `font-size` (e.g. 11px). The size alone communicates secondary importance.
- **For disabled elements**: rely on Qt's native `setEnabled(False)` dimming, or use `palette(dark)` for disabled-state overrides.
- **Hardcoded colors** (e.g. `#333333`) are OK for text on elements with their own hardcoded background (e.g. a yellow warning box), but never for text on the default window background.
- **Always test new UI changes in both light and dark theme** before committing.

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

## Plugin Naming Convention

- The GitHub repo is `QGIS_AI-Segmentation` but the QGIS plugin folder is `AI_Segmentation`
- Users only see `AI_Segmentation` - the `QGIS_` prefix is only in the repo name
- Code referencing the plugin key (update check, plugin manager) must use `'AI_Segmentation'`

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



Always output a short line and bullet points resuming the changes that you made when you made a big change like from a plan


never put this character when you write : "—" ca fait trop ia evite les em dashes surtout dans les issues github

don't put too much comments in the code, stay concis clear in english, be minimalist