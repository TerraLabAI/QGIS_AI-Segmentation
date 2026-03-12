# QGIS AI Segmentation Plugin

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update memory/MEMORY.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### Task Management
1. Plan First: Write plan to tasks/todo.md with checkable items
2. Verify Plan: Check in before starting implementation
3. Track Progress: Mark items complete as you go
4. Explain Changes: High-level summary at each step
5. Document Results: Add review section to tasks/todo.md
6. Capture Lessons: Update memory/MEMORY.md after corrections

### Core Principles
- Simplicity First: Make every change as simple as possible. Impact minimal code.
- No Laziness: Find root causes. No temporary fixes. Senior developer standards.
- Minimal Impact: Changes should only touch what's necessary. Avoid introducing bugs.
- After big changes: output a short summary with bullet points of what changed.

## Writing Style

- Never use em dashes "--" in text output or GitHub issues
- Minimal comments in code, concise and in English only
- All UI text in English in source code, never in French

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
2. Add `<message>` block in ALL .ts files (`fr.ts`, `pt_BR.ts`, `es.ts`) inside `<context><name>AISegmentation</name>`
3. Use `.format()` for dynamic strings: `tr("Export {count} polygon(s)").format(count=5)`
4. Keep in English: "AI Segmentation", "SAM", "TerraLab", "Batch mode", "Export", "Checkpoint", package names

## Dark Theme

- **NEVER** use `palette(mid)` for text -- invisible on dark backgrounds. Use `palette(text)`.
- Secondary text: `palette(text)` with smaller `font-size` (11px).
- Hardcoded colors OK only on elements with their own hardcoded background.

## Common Pitfalls

- `stderr=subprocess.PIPE` without draining it = deadlock. Use `DEVNULL` or temp file.
- `os.replace()` not `os.rename()` (Windows fails if dest exists)
- `os.path.normcase()` before comparing paths (Windows case-insensitive)
- `encoding='utf-8'` on all `open()` calls
- `Tuple[bool, str]` from typing, not `tuple[bool, str]` (Python 3.9 compat)
- `blockSignals(True/False)` when setting widget values programmatically
- Disconnect `QgsProject.instance()` signals in `unload()`

## Code Quality (Flake8)

- Max line length: 120 characters
- No unused imports (F401), no unused variables (F841)
- Import order: stdlib, third-party, local. Use `# noqa: E402` when import must follow runtime setup.
- W503/W504 (line breaks with operators): use `.format()` or intermediate variables instead
- `global` only needed when reassigning a module-level variable, not when modifying a dict/list

## Security (Bandit)

- XML parsing: use `defusedxml.defuse_stdlib()` before `ET.parse()` (B314)
- Subprocess: list arguments, never `shell=True` with string interpolation
- No hardcoded credentials or credential-like patterns in code/comments

## Release Process

- **Never create GitHub releases or tags without explicit user confirmation**
- Release flow: create a GitHub Release with tag `vX.Y.Z` to trigger `release.yml`
- `release.yml` packages the plugin, attaches the zip to the GitHub Release, and publishes to plugins.qgis.org
- Required repo secrets: `OSGEO_USERNAME` and `OSGEO_PASSWORD` (GitHub Settings > Secrets)
- `.gitattributes` ensures dev files (tests, CI, CLAUDE.md, etc.) are excluded from the release zip
