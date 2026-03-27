# AI Segmentation PRO — UX Redesign Spec

**Date:** 2026-03-26
**Branch:** LA-fal.ai-features
**Status:** Design validated, ready for implementation

---

## Context

The PRO mode dock widget has usability issues:
- Double screen (initial + main) duplicates controls
- Category → Tag hierarchy adds unnecessary clicks
- "Save polygon" is ambiguous in batch mode
- 0% confidence default shows too much noise
- Too many accent colors (green, blue, gray) for a 300px panel

Target users: geomaticians broadly (urbanists, bureau d'etudes, researchers in remote sensing).
Primary workflow: batch detect objects over a zone (or full image), refine, export.

---

## Decisions

| # | Decision | Before | After |
|---|----------|--------|-------|
| 1 | **Merge 2 screens into 1** | Initial screen → Start button → Main screen | Single screen, raster pre-selected |
| 2 | **Autocomplete search** | Category → Tag (2 dropdowns) | Single text field with QCompleter + free-text fallback |
| 3 | **Remove "Save polygon"** | Save + Export = 2 buttons | Only "Export to layer", visible when polygons exist |
| 4 | **Credits in Detect button** | Separate "Estimated credits: 2" text | "Detect objects (2 credits)" in button label |
| 5 | **Confidence default 25%** | Default 0% | Default 25% |
| 6 | **Refine after detection only** | Always visible | Collapsible section, appears after 1st detection |
| 7 | **Right-click to delete** | Not implemented | Right-click polygon on map to remove it |
| 8 | **Zone is optional** | "Select zone" before Start | Zone in main screen, full image by default |
| 9 | **Minimalist color palette** | 3 accent colors | 1 accent color (green for Detect only) |
| 10 | **Click-to-segment** | Not implemented | Deferred (no backend yet) |

---

## Color Palette (dark theme)

| Element | Color | Usage |
|---------|-------|-------|
| Background | `palette(window)` | Dock background |
| Primary accent | `#4CAF50` | Detect button ONLY |
| Secondary accent | `#66BB6A` | Export button (when active) |
| Primary text | `palette(text)` | All text |
| Secondary text | `palette(text)` + 11px | Labels, hints |
| Borders | `rgba(128, 128, 128, 0.2)` | Panels, inputs |
| Neutral buttons | `rgba(128, 128, 128, 0.12)` | Zone, Undo, Stop |
| Info feedback | `rgba(128, 128, 128, 0.08)` | "47 objects detected" |
| Warning | `rgba(255, 180, 50, 0.12)` | Zone too large |

**Rule:** Only the Detect button is green. Everything else is neutral.

---

## Wireframes (ASCII)

### State 1: Initial (plugin opened)

```
+------------------------------------------+
| AI Segmentation PRO by TerraLab     [] X |
|------------------------------------------|
|                                          |
|  +-- Raster -------------------------v-+ |
|  | Ortho_Echantillon_ANPT              | |
|  +--------------------------------------+|
|                                          |
|  +----------------+ +------------------+ |
|  |  Select zone   | |   Full image     | |
|  +----------------+ +------------------+ |
|                                          |
|  +--------------------------------------+|
|  | Search object to detect...           | |
|  +--------------------------------------+|
|                                          |
|  +--------------------------------------+|
|  |                                      | |
|  |    > Detect objects (2 credits)      | |  <-- only green button
|  |                                      | |
|  +--------------------------------------+|
|                                          |
|  Min. confidence             +- 25% --+ |
|                              +---------+ |
|                                          |
|  The larger the zone, the more           |
|  credits are used.                       |
|                                          |
+------------------------------------------+
```

### State 2: After detection

```
+------------------------------------------+
| AI Segmentation PRO by TerraLab     [] X |
|------------------------------------------|
|                                          |
|  +-- Raster -------------------------v-+ |
|  | Ortho_Echantillon_ANPT              | |
|  +--------------------------------------+|
|                                          |
|  +----------------+ +------------------+ |
|  |  Redraw zone   | |   Full image     | |
|  +----------------+ +------------------+ |
|                                          |
|  +--------------------------------------+|
|  | roof                             x   | |
|  +--------------------------------------+|
|                                          |
|  +--------------------------------------+|
|  |    > Detect objects (2 credits)      | |
|  +--------------------------------------+|
|                                          |
|  Min. confidence             +- 25% --+ |
|                              +---------+ |
|                                          |
|  +--------------------------------------+|
|  | 47 objects detected                  | |
|  | Right-click a polygon to remove it   | |
|  +--------------------------------------+|
|                                          |
|  > Refine polygons              (collapsed)
|                                          |
|  +--------------------------------------+|
|  |                                      | |
|  |    Export 47 polygons to layer       | |
|  |                                      | |
|  +--------------------------------------+|
|                                          |
|  +----------------+ +------------------+ |
|  |     Undo       | |      Stop        | |
|  +----------------+ +------------------+ |
|                                          |
|  ================== Tile 8/12            |
|                                          |
+------------------------------------------+
```

### State 2b: Refine expanded

```
|  v Refine polygons                       |
|  +--------------------------------------+|
|  | Expand/Contract        +-- 0 px ---+ ||
|  |                        +------------+ ||
|  | Simplify outline       +-- 3 ------+ ||
|  |                        +------------+ ||
|  | [ ] Fill holes                        ||
|  | Min. region size       +-- 100 px2-+ ||
|  |                        +------------+ ||
|  +--------------------------------------+|
```

### Autocomplete dropdown (while typing)

```
|  +--------------------------------------+|
|  | ro                                   | |
|  |  +----------------------------------+| |
|  |  | Roof                             || |
|  |  | Road                             || |
|  |  +----------------------------------+| |
|  +--------------------------------------+|
```

If user types something not in the list (e.g. "swimming pool"),
the text is used directly as a free-text prompt.

---

## User Flow (after redesign)

```
1. Open plugin
   -> Raster already selected (active layer)
   -> Panel shows: raster, zone buttons, search field, Detect button

2. (Optional) Select zone
   -> Click "Select zone" -> draw rectangle on map
   -> Credits update in Detect button

3. Type object
   -> Autocomplete suggests known objects
   -> If not in list, free text is used as prompt

4. Click "Detect objects (N credits)"
   -> First click: starts segmentation session + runs detection
   -> Progress bar shows tile progress
   -> Polygons appear on map

5. Review results
   -> Status: "47 objects detected"
   -> Right-click a polygon to delete it
   -> Refine panel appears (collapsed)

6. (Optional) Refine
   -> Expand refine panel
   -> Adjust expand/simplify/fill holes/min area
   -> Changes apply in real-time to ALL polygons

7. (Optional) Re-detect
   -> Change search term or zone
   -> Click Detect again (adds to existing polygons)

8. Export
   -> Click "Export 47 polygons to layer"
   -> Creates QGIS vector layer

9. Stop or continue
   -> "Stop" ends the session
   -> "Undo" removes last action
```

---

## Implementation Plan

### Phase 1: Layout restructure (no backend changes)

1. **Remove initial screen / Start button**
   - Remove `start_pro_button` and its show/hide logic
   - Remove `_on_start_pro_clicked` — emit `start_pro_segmentation_requested`
     when user clicks Detect for the first time
   - Show all controls immediately (raster, zone, search, detect)

2. **Replace category + tag combos with autocomplete QLineEdit**
   - Flat list: Roof, Building, Warehouse, Solar panel, Car, Truck, Road,
     Parking, Railway, Greenhouse, Tree, Bush, Grass, Forest, Crop, Field,
     River, Pool, Shadow
   - QCompleter with Qt.MatchContains filter
   - Remove `_TAG_CATEGORIES`, `_category_combo`, `_tag_combo`, `_custom_prompt_edit`
   - `get_pro_text_prompt()` returns the text field value directly

3. **Credits in Detect button**
   - Dynamic text: `tr("Detect objects ({credits} credits)").format(credits=n)`
   - Update when zone/raster changes via `set_credit_estimate()`
   - Remove standalone `credit_label` and `credit_info_label`

4. **Confidence default 25%**
   - `self.score_threshold_spinbox.setValue(25)` in init and `reset_session()`

### Phase 2: Action simplification

5. **Remove Save polygon button**
   - Remove `save_mask_button` from PRO dock
   - Keep `save_polygon_requested` signal only if needed by plugin.py
   - Batch detection auto-saves all polygons to session

6. **Export visible after detection only**
   - `self.export_button.setVisible(self._saved_polygon_count > 0)`
   - Style: `#66BB6A` when active, hidden otherwise

7. **Refine collapsible, post-detection only**
   - `self.refine_group.setVisible(False)` initially
   - Show after first `set_batch_done()` call

### Phase 3: Visual polish

8. **Single accent color**
   - Detect button: `#4CAF50` (only green element)
   - Export: `#66BB6A` (lighter green, secondary)
   - All other buttons: `rgba(128, 128, 128, 0.12)`
   - Remove all `#1976d2` (blue), `#2e7d32` (dark green), `#c8e6c9` (light green)

9. **Post-detection status banner**
   - Replace `instructions_label` with a consistent status area
   - "47 objects detected — right-click to remove"
   - Style: `rgba(128, 128, 128, 0.08)` background

10. **Zone buttons neutral style**
    - `rgba(128, 128, 128, 0.12)` background
    - No bold colors for secondary actions

---

## i18n Impact

New/modified strings to add to .ts files:
- `"Detect objects ({credits} credits)"`
- `"Search object to detect..."`
- `"{count} objects detected"`
- `"Right-click a polygon to remove it"`
- `"Refine polygons"` (replaces "Refine selection")

Removed strings:
- `"Start AI Segmentation PRO"`
- `"Select an object type to detect"`
- `"Select a category..."`
- `"Select an object..."`
- `"Describe the object..."`
- `"Save polygon"`

---

## Future (deferred)

- **Click-to-segment mode**: complementary to batch. User clicks map to add
  individual objects after batch detection. Needs backend implementation first.
- **Right-click to delete**: needs map tool interaction to detect click on polygon
  and remove it from the session.
