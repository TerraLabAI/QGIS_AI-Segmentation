# TerraLab QGIS Plugin Design System

Source of truth for UI conventions shared across all TerraLab QGIS plugins (AI Segmentation, AI Edit). AI Segmentation is the reference implementation — AI Edit must match these patterns for everything it has in common.

## Color Palette (Material Design 2)

### Brand Constants

| Token | Hex | Usage |
|-------|-----|-------|
| `BRAND_GREEN` | `#2e7d32` | Primary CTA buttons (Start, Get code) |
| `BRAND_GREEN_HOVER` | `#1b5e20` | Green button hover |
| `BRAND_GREEN_DISABLED` | `#c8e6c9` | Green button disabled |
| `BRAND_BLUE` | `#1976d2` | Secondary CTA (Activate, Save), links, TerraLab branding |
| `BRAND_BLUE_HOVER` | `#1565c0` | Blue button hover |
| `BRAND_RED` | `#d32f2f` | Destructive actions (Cancel) |
| `BRAND_RED_HOVER` | `#b71c1c` | Red button hover |
| `BRAND_GRAY` | `#757575` | Neutral buttons (Undo, Stop) |
| `BRAND_GRAY_HOVER` | `#616161` | Gray button hover |
| `BRAND_DISABLED` | `#b0bec5` | Universal disabled state |
| `DISABLED_TEXT` | `#666666` | Disabled button text |

### Semantic Text Colors

| Token | Hex | Usage |
|-------|-----|-------|
| Error text | `#ef5350` | Inline error messages |
| Success text | `#66bb6a` | Inline success messages |
| Warning text | `#333333` | Warning box text (on yellow bg) |
| Export enabled | `#4CAF50` | Export button when items exist |
| Export enabled hover | `#388E3C` | Export button hover |

### RGBA Backgrounds

| Purpose | Value |
|---------|-------|
| Info box bg | `rgba(25, 118, 210, 0.08)` |
| Info box border | `rgba(25, 118, 210, 0.2)` |
| Update notification bg | `rgba(25, 118, 210, 0.15)` |
| Update notification border | `rgba(25, 118, 210, 0.4)` |
| Neutral container bg | `rgba(128, 128, 128, 0.12)` |
| Neutral container border | `rgba(128, 128, 128, 0.25)` |
| Subtle panel bg | `rgba(128, 128, 128, 0.08)` |
| Subtle panel border | `rgba(128, 128, 128, 0.2)` |
| Warning box bg | `rgb(255, 230, 150)` |
| Warning box border | `rgba(255, 152, 0, 0.6)` |
| Batch info bg | `rgba(100, 149, 237, 0.15)` |
| Batch info border | `rgba(100, 149, 237, 0.3)` |
| Key badge bg | `rgba(128, 128, 128, 0.18)` |
| Key badge border | `rgba(128, 128, 128, 0.35)` |

## Button Styles (QSS)

### Primary CTA (Green)

```qss
QPushButton { background-color: #2e7d32; color: #000000; padding: 8px 16px; }
QPushButton:hover { background-color: #1b5e20; color: #000000; }
QPushButton:disabled { background-color: #c8e6c9; color: #666666; }
```

### Secondary CTA (Blue)

```qss
QPushButton { background-color: #1976d2; color: #000000; padding: 6px 12px; }
QPushButton:hover { background-color: #1565c0; color: #000000; }
QPushButton:disabled { background-color: #b0bec5; color: #666666; }
```

### Auth Buttons (Green/Blue with white text, bold)

```qss
/* Green auth (Get code, Activate with key) */
QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; }
QPushButton:hover { background-color: #1b5e20; }
QPushButton:disabled { background-color: #b0bec5; }

/* Blue auth (Unlock, Activate with code) */
QPushButton { background-color: #1976d2; color: white; font-weight: bold; border-radius: 4px; }
QPushButton:hover { background-color: #1565c0; }
QPushButton:disabled { background-color: #b0bec5; }
```

### Destructive (Red)

```qss
QPushButton { background-color: #d32f2f; color: #ffffff; }
QPushButton:hover { background-color: #b71c1c; color: #ffffff; }
```

### Neutral (Gray)

```qss
QPushButton { background-color: #757575; color: #000000; padding: 4px 8px; }
QPushButton:hover { background-color: #616161; color: #000000; }
```

## Widget Patterns (QSS)

### Instructions Box

```qss
QLabel {
    background-color: rgba(128, 128, 128, 0.12);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 4px;
    padding: 8px;
    font-size: 12px;
    color: palette(text);
}
```

### Info Box (Blue tint)

```qss
QWidget {
    background-color: rgba(25, 118, 210, 0.08);
    border: 1px solid rgba(25, 118, 210, 0.2);
    border-radius: 4px;
}
QLabel { background: transparent; border: none; }
```

### Warning Box (Yellow)

```qss
QWidget {
    background-color: rgb(255, 230, 150);
    border: 1px solid rgba(255, 152, 0, 0.6);
    border-radius: 4px;
}
QLabel { background: transparent; border: none; color: #333333; }
```

### Subtle Panel (Refine, collapsible sections)

```qss
QWidget {
    background-color: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 4px;
}
QLabel { background: transparent; border: none; }
```

### Update Notification

```qss
QLabel {
    background-color: rgba(25, 118, 210, 0.15);
    border: 2px solid rgba(25, 118, 210, 0.4);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: bold;
    color: palette(text);
}
```

### Section Headers (within panels)

```qss
QLabel {
    font-size: 10px;
    color: palette(text);
    font-weight: bold;
    background: transparent;
    border: none;
    border-bottom: 1px solid palette(mid);
    padding: 8px 0px 4px 0px;
    margin-bottom: 4px;
    letter-spacing: 1px;
}
```

## Typography

| Role | Size | Weight | Color |
|------|------|--------|-------|
| Widget title | 12px | bold | `palette(text)` |
| Dialog title | 14pt | bold | `palette(text)` |
| Section header | 10px | bold | `palette(text)` + `letter-spacing: 1px` |
| Body / Instructions | 12px | normal | `palette(text)` |
| Small / Helper text | 10-11px | normal | `palette(text)` |
| Footer links | 13px | normal | `#1976d2` |
| Error message | 11px | normal | `#ef5350` |
| Success message | 11px | normal | `#66bb6a` |
| Keyboard badges | monospace | normal | inherit |

## Spacing

### Layout Margins (QLayout.setContentsMargins)

| Context | Margins (L, T, R, B) |
|---------|----------------------|
| Main dock layout | `8, 8, 8, 8` |
| Title bar | `0, 0, 0, 0` |
| Title row | `4, 0, 0, 0` |
| Nested sections (activation, start) | `4, 4, 4, 4` |
| Info boxes (welcome, batch) | `10, 10, 10, 10` or `8, 6, 8, 6` |
| Warning boxes | `8, 8, 8, 8` |
| Dialogs | `16, 16, 16, 16` or `24, 24, 24, 24` |
| Footer links | `0, 4, 0, 4` |
| Refine content | `10, 10, 10, 10` |

### Layout Spacing (QLayout.setSpacing)

| Context | Spacing |
|---------|---------|
| Main dock layout | `8` |
| Activation/action groups | `6-8` |
| Tight nested layouts | `0-4` |
| Dialog sections | `14` |
| Footer links | `16` (horizontal) |

## Border Radius

| Context | Radius |
|---------|--------|
| Standard widgets, buttons | `4px` |
| Install path labels, key badges | `3px` |
| Prominent notifications | `6px` |

## Minimum Sizes

| Widget | Min Height | Min Width |
|--------|------------|-----------|
| Dock widget | — | `260px` |
| Email/code input (dock) | `28px` | — |
| Email input (dialog) | `36px` | — |
| Auth buttons (dialog) | `36-40px` | — |
| Activate button (dock) | `28px` | `60px` |
| Dialog | — | `400-420px` (max `500px`) |

## Dark Theme Rules

- **NEVER** use `palette(mid)` for text — invisible on dark backgrounds
- Always use `palette(text)` for text colors — adapts to light/dark
- Use `palette(base)` for input backgrounds that need theme awareness
- Secondary text: `palette(text)` with smaller `font-size` (10-11px), never a lighter color
- All RGBA backgrounds work on both themes because they're semi-transparent overlays
- Warning box text `#333333` is the one exception — works because the yellow background is opaque

## Canvas Selection Rectangle

```python
rb.setColor(QColor(65, 105, 225, 80))         # Fill: cornflower blue @ 31% alpha
rb.setStrokeColor(QColor(65, 105, 225, 200))  # Border: cornflower blue @ 78% alpha
rb.setWidth(2)
```

## Error/Success Message Pattern

Inline validation messages use colored text only (no background):
```python
label.setStyleSheet("color: #ef5350; font-size: 11px;")  # error
label.setStyleSheet("color: #66bb6a; font-size: 11px;")  # success
```

Status boxes with background use the RGBA patterns from the Widget Patterns section above.

## Footer Links Pattern

```python
link_label.setStyleSheet(f"color: {BRAND_BLUE}; font-size: 13px;")
link_label.setOpenExternalLinks(True)
# Use <a> tags with matching color: style="color: #1976d2; text-decoration: none;"
```

Links layout: horizontal, spacing `16`, margins `0, 4, 0, 4`.

## Custom Title Bar Pattern

Every dock widget has a custom title bar with:
- Left: TerraLab link label (`color: #1976d2`)
- Right: Float button + Close button (using `SP_TitleBarNormalButton`, `SP_TitleBarCloseButton`)
- QGIS standard icons via `QApplication.style().standardIcon()`

## Code Conventions

### Imports

```python
from __future__ import annotations  # ALL files with type hints
# Standard library
# Third-party (qgis, PyQt)
# Relative imports
```

### Type Hints

- Modern: `list[x]` not `List[x]`, `x | None` not `Optional[x]`
- `from __future__ import annotations` enables this on Python 3.9

### Linting

- **Ruff** with shared `ruff.toml` (line-length 120, target py39)
- Conventional commits: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`
- No unused imports (F401), no unused variables (F841)
- `raise ... from err` in except clauses (B904)
- `# noqa: W503` for line-break-before-operator (flake8 compat)

### Qt Patterns

- `blockSignals(True/False)` when setting widget values programmatically
- Disconnect `QgsProject.instance()` signals in `unload()`
- Event filters: NEVER install on `QApplication` — use `mainWindow.installEventFilter()`
- All network/generation work runs in QThread workers
- UI updates happen via Qt signals only

### Error Handling

- `ErrorReportDialog` for user-facing errors with diagnostic info
- Path anonymization: `/Users/xxx` → `<USER>`
- Network errors enriched with actionable guidance and links
- `os.replace()` not `os.rename()` (Windows compat)
- `os.path.normcase()` before comparing paths (Windows case-insensitive)
- `encoding='utf-8'` on all `open()` calls

### Logging

- Use `QgsMessageLog` wrapper (never print())
- NEVER log: API URLs, model names, API keys, activation codes
- OK to log: dimensions, extents, CRS, timing, file paths

## Open-Source Confidentiality

Both plugins are GPL. Public repos must only contain clean abstracted code.

**Never mention in code/commits/comments:**
- Backend/inference provider names, model names, weight URLs
- Infrastructure names (Supabase, Stripe, Azure, specific cloud providers)
- API endpoints, activation codes, internal URLs
- Internal architecture decisions or cost/pricing details

**Debug artifacts** (`.debug/`, `.env.local`): gitignored — never commit.

## Dual-Repo Workflow

Both plugins follow the same pattern:
```
origin  → TerraLabAI/QGIS_AI-{Plugin}-Team  (private, default push/pull)
public  → TerraLabAI/QGIS_AI-{Plugin}       (public, release-only)
```

- All commits go to `origin` (private) by default
- Release flow: bump `metadata.txt` → verify → push to public
