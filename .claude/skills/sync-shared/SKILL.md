---
name: sync-shared
description: Compare and sync src/shared/ modules between AI Segmentation and AI Canvas plugins. Use before committing shared module changes.
disable-model-invocation: true
---

Compare `src/shared/` between the two plugin repos:
- **AI Segmentation**: `/Users/yvann/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/src/shared/`
- **AI Canvas**: `/Users/yvann/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Canvas/src/shared/`

## Steps

1. **Diff**: Run `diff -rq` between both `src/shared/` directories. Report which files differ and which are missing from either side.

2. **Show diffs**: For each differing file, show the actual diff so the user can review.

3. **Ask direction**: Use AskUserQuestion to ask which direction to sync:
   - Segmentation → Canvas (copy from this repo to AI Canvas)
   - Canvas → Segmentation (copy from AI Canvas to this repo)
   - Skip (just report, don't copy)

4. **Copy**: If the user chose a direction, copy the files. Use `$ARGUMENTS` if provided to limit to specific files (e.g., `/sync-shared constants.py`).

5. **Verify**: Run `diff -rq` again to confirm the directories are now identical.

**Important**: Only sync files in `src/shared/`. Never touch files outside this directory.
