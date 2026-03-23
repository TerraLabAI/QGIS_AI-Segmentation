# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QGIS AI Segmentation plugin by TerraLab — point-and-click AI segmentation on raster data directly in QGIS.

Part of the TerraLab ecosystem:
- **AI Segmentation** (this repo — PRIVATE dev) — local SAM inference via inference backend
- **AI Canvas** — image generation, same shared modules — `/Users/yvann/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Canvas`
- **terra-lab.ai** — Next.js website, activation service, billing — `/Users/yvann/Documents/GitHub/terralab-website`

## Dual-Repo Workflow

This is the **PRIVATE development repo** (`TerraLabAI/QGIS_AI-Segmentation-Team`). All development happens here.

```
origin  → TerraLabAI/QGIS_AI-Segmentation-Team  (private, default push/pull)
public  → TerraLabAI/QGIS_AI-Segmentation       (public open-source, release-only)
```

**Rules:**
- All commits go to `origin` (private) by default
- NEVER `git push public` directly — always use `/release-public` skill
- See `docs/workflow/private-public-workflow.md` for full details

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

- **`src/shared/`** must stay byte-identical with AI Canvas's `src/shared/`. Use `/sync-shared` before committing changes

## Architecture (what you can't infer from code)

- `src/core/model_config.py`: ALL version-dependent constants (SAM1 vs SAM2). Single source of truth.
- SAM2 (Python 3.10+, QGIS 3.34+) / SAM1 fallback (Python 3.9, QGIS 3.22/3.28)
- Dependencies installed in isolated venv at `~/.qgis_ai_segmentation/venv_py3.*/`
- GPU/CUDA code exists but is NOT user-facing. Never mention GPU/CUDA in UI, issues, or descriptions.
- Plugin key in code must be `'AI_Segmentation'` (not the repo name `QGIS_AI-Segmentation`)
- Buttons hidden when not in segmentation mode (not disabled-but-visible), use `_update_button_visibility()`

## Terminology

- **Selection** = temporary AI-generated mask (before saving)
- **Polygon** = saved items (after "Save polygon" or export)

## Refine Panel Defaults (KEEP IN SYNC)

`expand=0, simplify=3, fill_holes=False, min_area=100`

5 locations must match: plugin `__init__`, `_reset_session()`, `_restore_last_saved_mask` fallbacks, dockwidget `_setup_refine_panel`, `reset_refine_sliders`.

## i18n - IMPORTANT

Languages: French (fr), Portuguese Brazil (pt_BR), Spanish (es).

**When modifying ANY UI string, you MUST update code AND all 3 .ts files.**

1. Wrap string with `tr()` from `..core.i18n`
2. Add `<message>` block in ALL .ts files inside `<context><name>AISegmentation</name>`
3. Use `.format()` for dynamic strings: `tr("Export {count} polygon(s)").format(count=5)`
4. Keep in English: "AI Segmentation", "SAM", "TerraLab", "Batch mode", "Export", "Checkpoint", package names

## Dark Theme

- **NEVER** use `palette(mid)` for text -- invisible on dark backgrounds. Use `palette(text)`.
- Secondary text: `palette(text)` with smaller `font-size` (11px).

## Common Pitfalls

- `stderr=subprocess.PIPE` without draining it = deadlock. Use `DEVNULL` or temp file.
- `os.replace()` not `os.rename()` (Windows fails if dest exists)
- `os.path.normcase()` before comparing paths (Windows case-insensitive)
- `encoding='utf-8'` on all `open()` calls
- `Tuple[bool, str]` from typing, not `tuple[bool, str]` (Python 3.9 compat)
- `blockSignals(True/False)` when setting widget values programmatically
- Disconnect `QgsProject.instance()` signals in `unload()`

## Code Quality (Ruff)

- Max line length: 88 characters (see `ruff.toml`)
- No unused imports (F401), no unused variables (F841)
- Import order: stdlib, third-party, local

## Security

- XML parsing: use `defusedxml.defuse_stdlib()` before `ET.parse()` (B314)
- Subprocess: list arguments, never `shell=True` with string interpolation
- No hardcoded credentials or credential-like patterns in code/comments

## Commands

```bash
pytest tests/                    # run all tests
pytest -k 'test_name'            # run single test
ruff check src/ tests/           # lint
ruff format src/ tests/          # format
# /verify                        # full check (lint + format + tests)
# /release-public                # release to public repo
# /sync-shared                   # sync src/shared/ with AI Canvas
```

## Release Process

- **Never create GitHub releases or tags without explicit user confirmation**
- Release flow: bump `metadata.txt`, run `/verify`, run `/release-public`
- Public commits must be generic — no provider names, no internal details
