---
name: verify
description: Run full project verification — ruff lint, ruff format check, and pytest. Use before committing or after major changes.
---

Run these commands in sequence. Stop and report on first failure:

1. **Lint check**: `ruff check src/ tests/`
2. **Format check**: `ruff format --check src/ tests/`
3. **Tests**: `pytest tests/ -v`

If ruff reports fixable issues, fix them with:
- `ruff check --fix src/ tests/`
- `ruff format src/ tests/`

Then re-run the full verification.

Report a summary: pass/fail for each step, number of tests run, any issues found.
