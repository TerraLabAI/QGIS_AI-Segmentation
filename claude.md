# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QGIS AI Segmentation plugin by TerraLab — point-and-click AI segmentation that turns raster objects into vector polygons. Users click on features (buildings, vegetation, any object) in their raster layers and the plugin runs local inference to produce segmentation masks, exported as vector polygons.

Part of the TerraLab ecosystem:
- **AI Segmentation** (this repo) — local inference via on-device model, same shared modules
- **AI Canvas** — image generation via terra-lab.ai backend, path: `/Users/yvann/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Canvas`
- **terra-lab.ai** — Next.js website, API proxy, activation service, path: `/Users/yvann/Documents/GitHub/terralab-website`

## Architecture

3-tier layout in `src/`:
- `ui/` — QGIS UI components (dock widget, map tool, plugin entry point, activation dialog, error report dialog)
- `core/` — Business logic (model config, checkpoint manager, venv/uv/python managers, SAM predictor, geo utils, polygon exporter, device manager, feature encoder)
- `shared/` — Modules shared byte-identical with AI Canvas (activation, branding, menu, constants)
- `workers/` — Subprocess-based prediction worker (`prediction_worker.py` runs in isolated venv with PyTorch/SAM)

Note: this plugin has no `api/` layer — inference is fully local; there is no HTTP client for generation. The `core/activation_manager.py` (plugin-local, legacy) is distinct from `src/shared/activation_manager.py` (shared, canonical).

## Critical Rules

### Open-source confidentiality
This plugin is GPL open-source. The community should only see clean abstracted code — they must NOT be able to infer the backend stack or paid infrastructure from the code, commits, comments, or logs.

- **Never mention in code/commits/comments**: Supabase, Stripe, specific inference provider names, API endpoint paths, or any third-party service names used internally
- **Activation abstraction**: the activation/licensing system is exposed only through generic terms ("activation code", "license", "unlock"). No backend provider details in public code
- **Commit messages**: No provider names, no infrastructure details. Use generic terms: "backend", "API", "authentication", "activation", "inference"
- **Production-safe logging**: NEVER log API URLs, model checkpoint URLs, activation keys, or backend provider names. Only log dimensions, extents, CRS, timing, file paths, device info
- **Debug artifacts** (`.debug/`, `.env.local`): gitignored — never commit

### Shared modules
- **`src/shared/`** must stay byte-identical with AI Canvas's `src/shared/`. Changes require syncing. Use `/sync-shared` to verify.
- **Parametric design**: shared modules use `product_id` from `PRODUCTS` dict — never hardcode plugin-specific behavior in shared code.

### Inference worker isolation
- `src/workers/prediction_worker.py` runs as a subprocess in an isolated venv (managed by `src/core/venv_manager.py`)
- Communication is JSON over stdin/stdout — keep the protocol stable
- The worker must never log checkpoint URLs or model names to stdout (JSON channel); stderr only for diagnostics

## Commands

```bash
# Run all tests
pytest tests/

# Run single test
pytest tests/path/to/test.py -k 'test_name'

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Full verification
# Use /verify skill
```

## Code Style

- Conventional commits: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`
- Python 3.9+ type hints
- SOLID/DRY/KISS — no over-engineering
- Ruff for formatting and linting (see `ruff.toml`)
- Line length 88 (ruff), 120 (flake8 legacy — prefer ruff)

## Dev Environment

- QGIS 3.22+ required, tests mock `qgis.*` modules via `conftest.py`
- Dependencies (PyTorch, SAM) are installed automatically into an isolated venv at `~/.qgis_ai_segmentation/` on first use — not bundled in the repo
- Model checkpoints are downloaded to `~/.qgis_ai_segmentation/checkpoints/` — not in repo (gitignored via `models/`)
- `AI_SEGMENTATION_CACHE_DIR` env var overrides the default cache location for testing
- `pytest tests/` runs without QGIS runtime — `conftest.py` mocks all `qgis.*` imports
- No `.env.local` needed for local development (no remote API calls in this plugin)
