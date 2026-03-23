# Private/Public Repo Workflow

## Repo Structure

```
TerraLabAI/QGIS_AI-Segmentation-Team   ← PRIVATE dev repo (this repo)
TerraLabAI/QGIS_AI-Segmentation       ← PUBLIC open-source (release-only)
```

Local remotes:
```
origin  → TerraLabAI/QGIS_AI-Segmentation-Team  (default push/pull)
public  → TerraLabAI/QGIS_AI-Segmentation      (release target)
```

## Day-to-Day Development

**Always work on the private repo.** Commit freely, use any branch names, reference internals.

```bash
git commit -m "feat: add Supabase activation check"    # OK here — private repo
git push                                                # pushes to private by default
```

## Releasing to Public

Only at release time (version bump). Never push WIP, feature branches, or debug commits.

```bash
# 1. Bump version in metadata.txt
# 2. Run /verify — must be green
# 3. Run /release-public — guided squash merge to public
```

The `/release-public` skill creates a single squash commit on the public repo. The public commit message must be clean and generic.

## What NEVER Goes in Public Commits

Even through squash, the final commit message and all file contents must be clean:

| Category | Examples |
|----------|---------|
| Backend/inference details | Inference API names, model weights URLs, provider names |
| Infrastructure | Supabase, Stripe, Azure, specific cloud providers |
| Internal URLs | API endpoints with provider-specific paths |
| Business logic | Activation codes, pricing, cost per inference |
| Internal notes | TODOs referencing paid features, RECAP files |

**In code:** Use generic terms — "backend", "inference API", "activation service", "model endpoint"

## Two-Dev Coordination

Both devs clone and work from the private repo:

```bash
git clone https://github.com/TerraLabAI/QGIS_AI-Segmentation-Team.git
```

- Default `git push` = private repo — safe to use anytime
- `git push public` = public release — only via `/release-public` skill
- Never push `public` remote manually — the skill ensures the squash is clean

## Shared Modules (`src/shared/`)

Shared byte-identical with AI Canvas. Before committing changes to `src/shared/`:

```bash
/sync-shared  # diffs and syncs with AI Canvas private repo
```

## Setting Up a New Clone

```bash
git clone https://github.com/TerraLabAI/QGIS_AI-Segmentation-Team.git
cd QGIS_AI-Segmentation-Team
git remote add public https://github.com/TerraLabAI/QGIS_AI-Segmentation.git
```
