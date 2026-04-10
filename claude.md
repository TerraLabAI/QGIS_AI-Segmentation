# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QGIS AI Segmentation plugin by TerraLab — point-and-click AI segmentation on raster data directly in QGIS.

Part of the TerraLab ecosystem:
- **AI Segmentation** (this repo — PRIVATE dev) — local SAM inference
- **AI Edit** — image generation, same shared modules — `/Users/yvann/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Edit`
- **terra-lab.ai** — Next.js website, activation service, billing — `/Users/yvann/Documents/GitHub/terralab-website`

## Dual-Repo Workflow

```
origin  → TerraLabAI/QGIS_AI-Segmentation-Team  (private, default push/pull)
public  → TerraLabAI/QGIS_AI-Segmentation       (public open-source, release-only)
```

- All commits go to `origin` (private) by default
- NEVER `git push public` directly — always use `/release-public`
- Release flow: bump `metadata.txt` → `/verify` → `/release-public`

**What NEVER goes in public commits** (code, comments, commit messages):
- Backend/inference provider names, model names, weight URLs
- Infrastructure names (Supabase, Stripe, Azure, specific cloud providers)
- API endpoints, activation codes, internal URLs
- Internal architecture decisions or cost/pricing details

## Open-Source Confidentiality

Plugin is GPL open-source. The community should only see clean abstracted code.

- **Never mention in code/commits/comments**: inference provider names, Supabase, Stripe, specific model checkpoint names, API endpoint paths
- **Production-safe logging**: NEVER log API URLs, model names, API keys. Only log dimensions, extents, CRS, timing, file paths
- **Debug artifacts** (`.debug/`, `.env.local`): gitignored — never commit

## Shared Modules

- **`src/shared/`** must stay byte-identical with AI Edit's `src/shared/`. Use `/sync-shared` before committing changes to shared modules
- Shared modules use `product_id` from `PRODUCTS` dict in `src/shared/constants.py` — never hardcode plugin-specific behavior in shared code

## Architecture

### Core modules (reusable across plugins)
- `src/core/logging_utils.py` — shared QGIS log wrapper (`log()`)
- `src/core/archive_utils.py` — safe tar/zip extraction with path traversal protection
- `src/core/pip_diagnostics.py` — pip/uv error detection + user-facing help messages
- `src/core/prompt_manager.py` — point annotation with undo history (`PromptManager`, `FrozenCropSession`)
- `src/core/model_config.py` — ALL version-dependent constants (SAM1 vs SAM2). Single source of truth
- `src/core/venv_manager.py` — venv lifecycle, dependency install, verification
- `src/core/python_manager.py` — standalone Python download (astral-sh/python-build-standalone)
- `src/core/uv_manager.py` — uv binary download for fast pip installs
- `src/core/subprocess_utils.py` — clean env + cross-platform subprocess kwargs

### UI modules (reusable across plugins)
- `src/ui/layer_tree_combobox.py` — combo box mirroring QGIS Layer panel tree
- `src/ui/background_workers.py` — QThread workers for install/download/verify
- `src/ui/shortcut_filter.py` — global keyboard shortcut handler
- `src/ui/error_report_dialog.py` — error/bug report dialogs with diagnostic info

### Plugin-specific
- `src/ui/ai_segmentation_plugin.py` — main plugin class, segmentation workflow
- `src/ui/ai_segmentation_dockwidget.py` — dock widget UI
- `src/workers/prediction_worker.py` — SAM subprocess (JSON-RPC protocol)

### Key patterns
- SAM2 (Python 3.10+, QGIS 3.34+) / SAM1 fallback (Python 3.9, QGIS 3.22/3.28)
- Dependencies installed in isolated venv at `~/.qgis_ai_segmentation/venv_py3.*/`
- `prediction_worker.py` runs as subprocess in isolated venv — JSON over stdin/stdout, keep protocol stable
- GPU/CUDA code exists but is NOT user-facing. Never mention GPU/CUDA in UI, issues, or descriptions
- Plugin key in code must be `'AI_Segmentation'` (not the repo name)

## Commands

```bash
ruff check src/                      # lint (config in ruff.toml)
find src/ -name '*.py' -exec python3 -m py_compile {} +  # compile check
pytest tests/                        # run all tests
pytest -k 'test_name'                # run single test
```

## Git Workflow

**Commit after every logical change.** Each commit should be atomic:
- One bug fix = one commit
- One refactor = one commit
- Lint/format fix after a change = part of the same commit

**Before every commit:**
1. Run `ruff check` on changed files
2. Write a conventional commit message (`fix:`, `feat:`, `refactor:`, `test:`, `docs:`)

## Code Quality

- **Ruff** for linting (see `ruff.toml`: line-length 120, target-version py39)
- **`from __future__ import annotations`** in all files with type hints
- Modern type hints: `list[x]` not `List[x]`, `x | None` not `Optional[x]`
- Conventional commits: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`
- No unused imports (F401), no unused variables (F841)
- `raise ... from err` in except clauses (B904)

## Terminology

- **Selection** = temporary AI-generated mask (before saving)
- **Polygon** = saved items (after "Save polygon" or export)

## i18n

Languages: French (fr), Portuguese Brazil (pt_BR), Spanish (es).

**When modifying ANY UI string, you MUST update code AND all 3 .ts files.**

1. Wrap string with `tr()` from `..core.i18n`
2. Add `<message>` block in ALL .ts files inside `<context><name>AISegmentation</name>`
3. Use f-strings for dynamic strings: `tr("Export {count} polygon(s)").format(count=5)`
4. Keep in English: "AI Segmentation", "SAM", "TerraLab", "Batch mode", "Export", "Checkpoint", package names

## Refine Panel Defaults (KEEP IN SYNC)

`expand=0, simplify=3, fill_holes=False, min_area=100`

5 locations must match: plugin `__init__`, `_reset_session()`, `_restore_last_saved_mask` fallbacks, dockwidget `_setup_refine_panel`, `reset_refine_sliders`.

## Dark Theme

- **NEVER** use `palette(mid)` for text — invisible on dark backgrounds. Use `palette(text)`
- Secondary text: `palette(text)` with smaller `font-size` (11px)

## Common Pitfalls

- `stderr=subprocess.PIPE` without draining it = deadlock. Use `DEVNULL` or temp file
- `os.replace()` not `os.rename()` (Windows fails if dest exists)
- `os.path.normcase()` before comparing paths (Windows case-insensitive)
- `encoding='utf-8'` on all `open()` calls
- `blockSignals(True/False)` when setting widget values programmatically
- Disconnect `QgsProject.instance()` signals in `unload()`
- Event filters: NEVER install on `QApplication` — use `mainWindow.installEventFilter()`
- Smooth/Round corners: Use `QgsGeometry.smooth()` (C++ native) not custom Python Chaikin

## Security

- XML parsing: use `defusedxml.defuse_stdlib()` before `ET.parse()` (B314)
- Subprocess: list arguments, never `shell=True` with string interpolation
- No hardcoded credentials or credential-like patterns in code/comments
