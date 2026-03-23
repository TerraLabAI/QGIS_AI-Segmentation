---
name: release-public
description: Release AI Segmentation to the public open-source repo. Squash merges private main into a single clean commit and pushes to TerraLabAI/QGIS_AI-Segmentation. Run only when a release is ready.
disable-model-invocation: true
---

## Pre-flight checks

1. Confirm you are on `main` branch of the private repo: `git branch --show-current`
2. Confirm `public` remote exists: `git remote get-url public`
3. Run `/verify` — MUST pass before proceeding. Stop if anything fails.
4. Confirm `metadata.txt` version has been bumped for this release. Read it and show the current version. Ask the user to confirm it's correct.

## Release flow

```bash
# 1. Fetch latest public main
git fetch public

# 2. Create release branch from public/main
git checkout -b release/squash-$(date +%Y%m%d) public/main

# 3. Squash merge all private main changes
git merge --squash origin/main

# 4. Ask user for the release commit message
# Format: "feat: release vX.Y.Z — <one line summary>"
# The message must NOT mention backend providers, inference API names, or internal details

# 5. Commit
git commit -m "<release message from user>"

# 6. Push to public main
git push public HEAD:main

# 7. Tag the release
git tag v<version>
git push public v<version>

# 8. Clean up local release branch
git checkout main
git branch -D release/squash-$(date +%Y%m%d)
```

## Post-release

- Confirm: `gh repo view TerraLabAI/QGIS_AI-Segmentation --web` to verify the public repo looks correct
- Remind: upload the plugin zip to plugins.qgis.org manually
- Remind: create a GitHub Release on the PUBLIC repo with the changelog
